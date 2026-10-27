"""Original-feature marginal SHAP explanations of binary mortality log odds.

Callers supply one fixed empirical background to every model (normally 32 rows),
the patients to explain (normally 100), and TabPFN's coalition budget (normally
512). This module never subsamples, preprocesses, or fits an estimator.
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
    """Explain one bare fitted estimator, returning arrays and scalar metadata.

    Columns keep their original order (numeric Sex included) and native NaNs pass
    through unchanged. ``budget`` controls only TabPFN's OddSHAP approximation.
    The background defines an empirical joint marginal game, so the reference is
    the mean prediction over whole background rows, not a conditional or
    single-baseline game. A small reconstruction residual is not evidence of
    convergence of individual feature attributions.
    """
    X = np.asarray(X, dtype=float)
    background = np.asarray(background, dtype=float)
    if model_name == "ebm":
        values, bases, output = _explain_ebm(estimator, X, background)
        method = "ebm_exact_marginal_shap"
    elif model_name == "xgboost":
        values, bases, output = _explain_xgboost(estimator, X, background)
        method = "xgboost_interventional_tree_shap"
    elif model_name in ("tabpfn-3.5", "tabpfn-3.5-fast"):
        values, bases, output = _explain_tabpfn(estimator, X, background, budget, seed)
        method = "tabpfn_marginal_oddshap"
    else:
        raise ValueError(f"Unsupported model for explanations: {model_name}")

    residual = bases + values.sum(axis=1) - output
    return {
        "shap_values": values,
        "base_values": bases,
        "model_output": output,
        "method": method,
        "max_additivity_residual": float(np.max(np.abs(residual))),
    }


def _explain_ebm(estimator, X, background):
    """Center each main-effect term on B, then restore original feature order."""
    local_terms = np.asarray(estimator.eval_terms(X), dtype=float)
    background_terms = np.asarray(estimator.eval_terms(background), dtype=float)
    order = [term[0] for term in estimator.term_features_]
    if sorted(order) != list(range(X.shape[1])):
        raise ValueError("EBM main terms must cover each feature exactly once")
    values = np.empty_like(local_terms)
    values[:, order] = local_terms - background_terms.mean(axis=0)
    bases = np.full(len(X), estimator.decision_function(background).mean())
    return values, bases, np.asarray(estimator.decision_function(X), dtype=float)


def _explain_xgboost(estimator, X, background):
    """Use exact interventional TreeSHAP on the complete supplied background."""
    import shap

    # XGBoost 3.4 defaults enable_categorical=True even for numeric inputs, and
    # SHAP 0.52 rejects that sklearn-wrapper flag without checking the trees.
    # Explain the fitted Booster directly; genuine categorical features are still
    # rejected by SHAP's model loader. Preserve the wrapper's float32 inputs.
    masker = shap.maskers.Independent(np.asarray(background, dtype=np.float32), max_samples=len(background))
    explainer = shap.TreeExplainer(
        estimator.get_booster(),
        data=masker,
        feature_perturbation="interventional",
        model_output="raw",
    )
    # Report reconstruction error ourselves rather than treating it as convergence.
    explanation = explainer(np.asarray(X, dtype=np.float32), check_additivity=False)
    return (
        np.asarray(explanation.values, dtype=float),
        np.asarray(explanation.base_values, dtype=float),
        estimator.predict(X, output_margin=True),
    )


def _explain_tabpfn(estimator, X, background, budget, seed):
    """Explain log odds of final probabilities, never ensemble/class logits."""
    from shapiq import OddSHAP, TabularExplainer
    from tqdm.auto import tqdm

    def predict_log_odds(data):
        probabilities = np.asarray(estimator.predict_proba(data), dtype=float)
        if not np.isfinite(probabilities).all() or np.any(probabilities <= 0):
            raise ValueError("TabPFN log odds require finite, strictly positive class probabilities")
        return np.log(probabilities[:, 1]) - np.log(probabilities[:, 0])

    explainer = TabularExplainer(
        model=predict_log_odds,
        data=background,
        imputer="marginal",
        joint_marginal_distribution=True,
        sample_size=len(background),
        index="SV",
        max_order=1,
        approximator=OddSHAP(n=X.shape[1], random_state=seed),
        random_state=seed,
    )
    try:
        output = predict_log_odds(X)
        values = np.empty_like(X, dtype=float)
        bases = np.empty(len(X), dtype=float)
        for row_index, row in enumerate(tqdm(X, desc="TabPFN SHAP", unit="patient")):
            explanation = explainer.explain(row, budget=budget)
            values[row_index] = [explanation[(feature,)] for feature in range(X.shape[1])]
            bases[row_index] = explanation.baseline_value
    finally:
        # The imputer holds a bound method that closes over ``estimator``; clearing
        # the cell breaks that chain, so the caller can release TabPFN's GPU memory.
        explainer = predict_log_odds = None
        estimator = None
    return values, bases, output
