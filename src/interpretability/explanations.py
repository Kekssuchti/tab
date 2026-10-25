"""Original-feature marginal SHAP explanations of binary mortality log odds.

Callers supply the same fixed empirical background to every model (normally 32
rows), select explanation rows (normally 100), and pass a coalition budget
(normally 512). This module never subsamples, preprocesses, or fits an estimator.
"""

import numpy as np


def explain_model(
    model_name: str,
    estimator,
    X: np.ndarray,
    background: np.ndarray,
    *,
    budget: int,
    seed: int,
) -> dict[str, np.ndarray | str | float]:
    """Explain a bare fitted estimator, returning only arrays and scalar metadata.

    Columns retain their original order, including numeric Sex; native NaNs are
    passed through unchanged. Classes must be ordered as [0, 1], with mortality
    encoded as 1. The background defines an empirical joint marginal game, not a
    conditional or single-baseline game. ``budget`` controls only TabPFN's
    OddSHAP approximation. A small reconstruction residual is not evidence of
    convergence of its individual feature attributions.
    """
    X = np.asarray(X, dtype=float)
    background = np.asarray(background, dtype=float)
    if np.isinf(X).any() or np.isinf(background).any():
        raise ValueError("Features may contain native NaNs, but not infinities")

    if model_name == "ebm":
        values, bases, output = _explain_ebm(estimator, X, background)
        method = "ebm_exact_marginal_shap"
    elif model_name == "xgboost":
        values, bases, output = _explain_xgboost(estimator, X, background)
        method = "xgboost_interventional_tree_shap"
    else:
        values, bases, output = _explain_tabpfn(estimator, X, background, budget, seed)
        method = "tabpfn_marginal_oddshap"

    arrays = {
        "shap_values": np.asarray(values, dtype=float),
        "base_values": np.asarray(bases, dtype=float),
        "model_output": np.asarray(output, dtype=float),
    }
    expected_shapes = {"shap_values": X.shape, "base_values": (len(X),), "model_output": (len(X),)}
    for name, array in arrays.items():
        if array.shape != expected_shapes[name] or not np.isfinite(array).all():
            raise ValueError(f"{model_name}: {name} must be finite with shape {expected_shapes[name]}")
    residual = arrays["base_values"] + arrays["shap_values"].sum(axis=1) - arrays["model_output"]
    return {
        **arrays,
        "method": method,
        "max_additivity_residual": float(np.max(np.abs(residual))),
    }


def _explain_ebm(estimator, X, background):
    """Center each main-effect term on B, then restore original feature order."""
    if estimator.link_ != "logit":
        raise ValueError("EBM explanations require the binary logit link")
    terms = estimator.term_features_
    d = X.shape[1]
    if len(terms) != d or any(len(term) != 1 for term in terms) or sorted(term[0] for term in terms) != list(range(d)):
        raise ValueError("EBM requires exactly one singleton term per input feature and no interactions")
    local_terms = np.asarray(estimator.eval_terms(X), dtype=float)
    background_terms = np.asarray(estimator.eval_terms(background), dtype=float)

    values = np.empty_like(local_terms)
    values[:, [term[0] for term in terms]] = local_terms - background_terms.mean(axis=0)
    background_output = np.asarray(estimator.decision_function(background), dtype=float)
    if background_output.shape != (len(background),) or not np.isfinite(background_output).all():
        raise ValueError("EBM decision_function must return finite binary logits")
    bases = np.full(len(X), background_output.mean())
    output = estimator.decision_function(X)
    return values, bases, output


def _explain_xgboost(estimator, X, background):
    """Use exact interventional TreeSHAP on the complete supplied background."""
    import shap

    if estimator.get_xgb_params().get("objective") != "binary:logistic":
        raise ValueError("XGBoost explanations require objective='binary:logistic' for log-odds margins")
    # XGBoost 3.4 defaults enable_categorical=True even for numeric inputs.
    # SHAP 0.52 rejects that sklearn-wrapper flag without checking the trees.
    # Explain the fitted Booster instead; genuine categorical features are still
    # rejected by SHAP's model loader. Preserve the wrapper's float32 inputs.
    masker = shap.maskers.Independent(np.asarray(background, dtype=np.float32), max_samples=len(background))
    explainer = shap.TreeExplainer(
        estimator.get_booster(),
        data=masker,
        feature_perturbation="interventional",
        model_output="raw",
    )
    try:
        # Report reconstruction error ourselves rather than treating it as convergence.
        explanation = explainer(np.asarray(X, dtype=np.float32), check_additivity=False)
        values = np.asarray(explanation.values, dtype=float).copy()
        bases = np.asarray(explanation.base_values, dtype=float).copy()
        output = estimator.predict(X, output_margin=True)
    finally:
        del explainer
    return values, bases, output


def _explain_tabpfn(estimator, X, background, budget, seed):
    """Explain log odds of final probabilities, never ensemble/class logits."""
    from shapiq import OddSHAP, TabularExplainer
    from tqdm.auto import tqdm

    def predict_log_odds(data):
        probabilities = np.asarray(estimator.predict_proba(data), dtype=float)
        if probabilities.shape != (len(data), 2):
            raise ValueError("TabPFN predict_proba must return shape (n_rows, 2)")
        if not np.isfinite(probabilities).all() or np.any(probabilities <= 0) or np.any(probabilities > 1):
            raise ValueError(
                "TabPFN log odds require finite, strictly positive probabilities in both classes; "
                "zero/invalid probabilities are not clipped"
            )
        return np.log(probabilities[:, 1]) - np.log(probabilities[:, 0])

    explainer = None
    try:
        explainer = TabularExplainer(
            model=predict_log_odds,
            data=background,
            imputer="marginal",
            sample_size=len(background),
            joint_marginal_distribution=True,
            index="SV",
            max_order=1,
            approximator=OddSHAP(n=X.shape[1], random_state=seed),
            random_state=seed,
        )
        output = predict_log_odds(X)
        values = np.empty_like(X, dtype=float)
        bases = np.empty(len(X), dtype=float)
        for row_index, row in enumerate(tqdm(X, desc="TabPFN SHAP", unit="patient")):
            explanation = explainer.explain(row, budget=budget)
            values[row_index] = [explanation[(feature,)] for feature in range(X.shape[1])]
            bases[row_index] = explanation.baseline_value
    finally:
        # shapiq's imputer holds a bound explainer method, creating a cycle.
        # Empty the callable's estimator cell too, so that cycle cannot retain it.
        del explainer, predict_log_odds
        estimator = None
    return values, bases, output
