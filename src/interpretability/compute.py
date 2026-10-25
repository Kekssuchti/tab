"""Mortality interpretation experiment. Run with ``uv run python -m src.interpretability.compute``.

Edit the small settings block below, not pipeline YAML. Each completed NPZ contains
one fitted model's results on BOTH held-out cohorts; no estimator is serialized.
"""

import argparse
import hashlib
import json
import platform
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np

# Experiment settings ---------------------------------------------------------
experiment_name = "mortality"
train_size = 10_000
repeat_seeds = (1337, 1338, 1339)
split_seed = 1337
train_fraction = 0.8
explanation_size = 100
background_tudd = 16
background_mimic = 16
background_seed = 2337
explanation_seed = 5337
explainer_seed = 3337
shap_budget = 512  # Coalitions PER explained patient (TabPFN only).

ebm_kwargs = {}  # Extra kwargs; searched parameters take precedence.
xgboost_kwargs = {}
tabpfn_kwargs = {"n_estimators": 8}  # Fixed parameters, no HPO.
# Match the existing baseline experiment; ranges come from model_registry.
ebm_hpo = {"search_space": "default", "n_trials": 50, "n_startup_trials": 5}
xgboost_hpo = {"search_space": "good", "n_trials": 100, "n_startup_trials": 10}
# EBM interactions are disabled; TabPFN version is fixed to v3.5 by the registry.
# Set estimator-specific seeds via repeat_seeds, not inside these dictionaries.
# -----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
MODELS = ("ebm", "xgboost", "tabpfn-3.5")
SOURCES = ("tudd", "mimic")
SCHEMA_VERSION = 1


def settings() -> dict:
    return {
        "target": "mortality",
        "positive_class": 1,
        "repeat_seeds": list(repeat_seeds),
        "train_size": train_size,
        "split_seed": split_seed,
        "train_fraction": train_fraction,
        "explanation_size": explanation_size,
        "background_tudd": background_tudd,
        "background_mimic": background_mimic,
        "background_seed": background_seed,
        "explanation_seed": explanation_seed,
        "explainer_seed": explainer_seed,
        "shap_budget": shap_budget,
        "preprocessing": "native",
        "output": "binary_mortality_log_odds",
        "reference": "joint_empirical_marginal",
    }


def load_pools():
    """Reuse the core loader, split, and alignment; keep original row indices.

    The core's public get_dataset combines/resets training indices. Composing its
    existing split primitives here preserves source-local row IDs for recreation
    while avoiding a second implementation of clinical filtering or splitting.
    """
    from src.classes.dataset import Dataset
    from src.schemas.dataset_schemas import DatasetConfig, DataSplitConfig, XYDataset

    builder = Dataset(
        DatasetConfig(
            target="mortality",
            random_state=split_seed,
            train_size=train_fraction,
            train_on=(DataSplitConfig(dataset="mimic", fraction=1.0),),
        ),
        sample_seed=split_seed,
    )
    splits = {}
    for source, frame in builder._load_data().items():
        X_train, X_test, y_train, y_test = builder._split_single_df(frame)
        splits[source] = {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}
    builder._align_split_feature_columns(splits)
    train = {s: XYDataset(X=p["X_train"], y=p["y_train"]) for s, p in splits.items()}
    test = {s: XYDataset(X=p["X_test"], y=p["y_test"]) for s, p in splits.items()}
    return train, test


def sample_rows(data, count: int, seed: int):
    """Stratified sampling WITHOUT replacement, with the core's row shuffle."""
    from sklearn.utils import resample

    from src.schemas.dataset_schemas import XYDataset

    rows = resample(data.X.index, replace=False, n_samples=count, random_state=seed, stratify=data.y)
    X = data.X.loc[rows].sample(frac=1, random_state=seed)
    return XYDataset(X=X, y=data.y.loc[X.index])


def _json_value(value):
    """Plain metadata, with explicit tags for e.g. XGBoost's missing=np.nan."""
    if isinstance(value, dict):
        return {key: _json_value(part) for key, part in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(part) for part in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return {"__float__": repr(value)}
    if isinstance(value, Path):
        return str(value)
    return value


def _json(value) -> str:
    return json.dumps(_json_value(value), sort_keys=True, allow_nan=False)


def _data_hash(*parts) -> str:
    """Hash actual selected values, labels, row order, and feature names."""
    import pandas as pd

    digest = hashlib.sha256()
    for part in parts:
        digest.update(pd.util.hash_pandas_object(part, index=True).to_numpy().tobytes())
        if isinstance(part, pd.DataFrame):
            digest.update(_json(list(part.columns)).encode())
    return digest.hexdigest()


def _source_hashes(model_name: str) -> dict[str, str]:
    adapter = "tabpfn_adapter.py" if model_name == "tabpfn-3.5" else "sklearn_adapter.py"
    paths = [
        ROOT / "src/interpretability/explanations.py",
        ROOT / "src/adapter" / adapter,
        ROOT / "src/utils/model_registry.py",
        ROOT / "src/classes/dataset.py",
        ROOT / "src/classes/data_registry.py",
        ROOT / "src/utils/dataset_utils.py",
    ]
    if model_name in ("ebm", "xgboost"):
        paths.extend(
            ROOT / relative
            for relative in (
                "src/interpretability/tuning.py",
                "src/classes/trainer.py",
                "src/classes/preprocessor.py",
                "src/schemas/training_schemas.py",
                "src/schemas/pipeline_schemas.py",
                "src/utils/evaluation_utils.py",
                "src/utils/classification_metrics.py",
                "src/utils/optuna_callbacks.py",
            )
        )
    # Hash computation functions, not the editable kwargs/settings block: changing
    # one model's kwargs should not invalidate the other models' completed fits.
    from inspect import getsource

    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    hashes["experiment_functions"] = hashlib.sha256(
        "".join(getsource(f) for f in (load_pools, sample_rows, run_model)).encode()
    ).hexdigest()
    return hashes


def _model_params(model_name: str, features: list[str]) -> dict:
    if model_name == "ebm":
        params = {
            "interactions": 0,
            "feature_names": features,
            "feature_types": ["nominal" if name == "Sex" else "continuous" for name in features],
            **ebm_kwargs,
        }
        if params["interactions"] != 0:
            raise ValueError("This experiment explains EBM main effects; keep interactions=0")
        return params
    if model_name == "xgboost":
        return dict(xgboost_kwargs)
    if "version" in tabpfn_kwargs:
        raise ValueError("Use tabpfn-3.5; do not override version inside tabpfn_kwargs")
    return {"categorical_features_indices": [i for i, name in enumerate(features) if name == "Sex"], **tabpfn_kwargs}


def run_model(path, model_name, seed, train, test, explained, background, metadata):
    """Fit, evaluate, record, then release exactly one model; never return it."""
    from src.classes.data_registry import dataset_task_for_target
    from src.interpretability.explanations import explain_model
    from src.schemas.dataset_schemas import DatasetBundle
    from src.schemas.training_schemas import ModelConfig
    from src.utils.evaluation import evaluate_trained_model
    from src.utils.model_lifecycle import release_model
    from src.utils.model_registry import get_model_spec

    task_type = dataset_task_for_target("mortality").task_type
    spec = get_model_spec(ModelConfig(name=model_name), task_type)
    model = None
    try:
        # Bare adapters intentionally avoid the Trainer's external preprocessing.
        model = spec.create(task_type, metadata["params"], random_state=seed, inference_state=explainer_seed)
        started = perf_counter()
        model.fit(train.X.to_numpy(dtype=float), train.y.to_numpy())
        metadata["fit_seconds"] = perf_counter() - started
        metadata["adapter_kwargs"] = model.kwargs
        metadata["estimator_params"] = model.model.get_params(deep=False)
        if model_name == "tabpfn-3.5":
            metadata["resolved_n_estimators"] = model.model.n_estimators_
        evaluation = evaluate_trained_model(
            model, task_type, DatasetBundle(train_data=train, test_mimic=test["mimic"], test_tudd=test["tudd"])
        )
        payload = {
            "feature_names": np.asarray(train.X.columns, dtype=str),
            "train_row_ids": train.X.index.to_numpy(),
            "train_y": train.y.to_numpy(),
            "background_X": background.to_numpy(dtype=float),
            "background_row_ids": background.index.get_level_values("row_id").to_numpy(),
            "background_sources": background.index.get_level_values("source").to_numpy(dtype=str),
        }
        metadata["metrics"] = {}
        metadata["prediction_seconds"] = {}
        metadata["explanations"] = {}
        for cohort in SOURCES:
            predictions = getattr(evaluation.test_predictions, cohort)
            metadata["metrics"][cohort] = asdict(getattr(evaluation.metrics, f"{cohort}_test"))
            metadata["prediction_seconds"][cohort] = getattr(evaluation.metrics, f"{cohort}_prediction_time")
            X = explained[cohort].X.to_numpy(dtype=float)
            print(f"  explain {cohort}: {len(X)} patients", flush=True)
            started = perf_counter()
            effects = explain_model(
                model_name, model.model, X, background.to_numpy(dtype=float), budget=shap_budget, seed=explainer_seed
            )
            metadata["explanations"][cohort] = {
                "seconds": perf_counter() - started,
                "method": effects["method"],
                "max_additivity_residual": effects["max_additivity_residual"],
            }
            payload.update(
                {
                    f"{cohort}_X": X,
                    f"{cohort}_row_ids": explained[cohort].X.index.to_numpy(),
                    f"{cohort}_shap_values": effects["shap_values"],
                    f"{cohort}_base_values": effects["base_values"],
                    f"{cohort}_model_output": effects["model_output"],
                    f"{cohort}_importance": np.abs(effects["shap_values"]).mean(axis=0),
                    f"{cohort}_test_row_ids": predictions.test_set_id,
                    f"{cohort}_test_y": predictions.y_true,
                    f"{cohort}_test_probability": predictions.positive_class_probability,
                }
            )
        save_result(path, payload, metadata)
    finally:
        release_model(model)


def save_result(path: Path, payload: dict, metadata: dict) -> None:
    """The final NPZ is the completion marker; interrupted writes are not reused."""
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    text = _json(metadata)
    path.with_suffix(".json").write_text(json.dumps(json.loads(text), indent=2) + "\n")
    records = [
        {"cohort": cohort, "feature": feature, "mean_abs_shap": float(importance)}
        for cohort in SOURCES
        for feature, importance in zip(payload["feature_names"], payload[f"{cohort}_importance"], strict=True)
    ]
    pd.DataFrame(records).to_csv(path.with_suffix(".importance.csv"), index=False)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, metadata_json=np.asarray(text), **payload)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "run_results/interpretability" / experiment_name)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--sources", nargs="+", choices=SOURCES, default=list(SOURCES))
    parser.add_argument("--seeds", nargs="+", type=int, choices=repeat_seeds, default=list(repeat_seeds))
    parser.add_argument("--recompute", action="store_true", help="Explicitly replace selected completed model results")
    parser.add_argument(
        "--dry-run", action="store_true", help="Load/sample data and show reuse status, without fitting"
    )
    args = parser.parse_args()

    import pandas as pd

    if min(train_size, explanation_size, shap_budget) < 1 or min(background_tudd, background_mimic) < 0:
        parser.error("Sample counts/budget must be positive; background source counts may be zero")
    if background_tudd + background_mimic == 0:
        parser.error("At least one background patient is required")
    pools, test = load_pools()
    features = list(pools["mimic"].X.columns)
    explained = {s: sample_rows(test[s], explanation_size, explanation_seed) for s in SOURCES}
    background_parts = {
        s: pools[s].X.sample(n=count, random_state=background_seed, replace=False)
        for s, count in (("tudd", background_tudd), ("mimic", background_mimic))
    }
    background = pd.concat(background_parts, names=["source", "row_id"])
    shared_hash = _data_hash(background, *(part for s in SOURCES for part in (test[s].X, test[s].y, explained[s].X)))
    dependencies = {
        name: version(name)
        for name in ("numpy", "pandas", "scikit-learn", "interpret", "xgboost", "tabpfn", "torch", "shap", "shapiq")
    }
    dependencies["python"] = platform.python_version()
    common = settings()
    print(f"Training pools: { {s: len(pools[s].y) for s in SOURCES} }; selected/source={train_size}")
    print(f"Full test sizes: { {s: len(test[s].y) for s in SOURCES} }; explained/cohort={explanation_size}")
    print(f"Background tudd={background_tudd}, mimic={background_mimic}; results={args.results}")
    failures = 0
    for seed in args.seeds:
        for source in args.sources:
            train = sample_rows(pools[source], train_size, seed)
            training_hash = _data_hash(train.X, train.y)
            for model_name in args.models:
                path = args.results / source / model_name / f"seed_{seed}.npz"
                params = _model_params(model_name, features)
                hpo = None
                if model_name in ("ebm", "xgboost"):
                    hpo = {
                        **(ebm_hpo if model_name == "ebm" else xgboost_hpo),
                        "inference_seed": explainer_seed,
                        "optuna_version": version("optuna"),
                    }
                # Adding another repeat need not invalidate already completed fits.
                identity = {
                    "schema_version": SCHEMA_VERSION,
                    "model": model_name,
                    "train_source": source,
                    "seed": seed,
                    "params": params,
                    "settings": {k: v for k, v in common.items() if k != "repeat_seeds"},
                    "data_hash": shared_hash,
                    "training_hash": training_hash,
                    "dependencies": dependencies,
                    "source_hashes": _source_hashes(model_name),
                }
                if hpo is not None:
                    identity["hpo"] = hpo
                signature = hashlib.sha256(_json(identity).encode()).hexdigest()
                metadata = {**identity, "settings": common, "signature": signature, "experiment_name": experiment_name}
                if path.exists() and not args.recompute:
                    with np.load(path, allow_pickle=False) as stored:
                        previous = json.loads(str(stored["metadata_json"]))
                    if previous["signature"] != signature:
                        raise ValueError(
                            f"Settings/data changed for {path}. Choose a new --results directory or --recompute"
                        )
                    print(f"reuse {source}/{model_name}/seed_{seed}", flush=True)
                    continue
                mode = f"HPO {hpo['search_space']}, up to {hpo['n_trials']} trials" if hpo else "fixed parameters"
                print(
                    f"{'would compute' if args.dry_run else 'compute'} {source}/{model_name}/seed_{seed} ({mode})",
                    flush=True,
                )
                if args.dry_run:
                    continue
                if args.recompute and path.exists():
                    # A failed replacement must not look like a completed run.
                    # Keep one previous result for recovery, outside the loader's pattern.
                    for previous in (path, path.with_suffix(".json"), path.with_suffix(".importance.csv")):
                        if previous.exists():
                            previous.replace(previous.with_name(previous.stem + ".previous" + previous.suffix))
                try:
                    if hpo is not None:
                        from src.interpretability.tuning import select_parameters

                        metadata["base_params"] = params
                        metadata["params"], metadata["tuning"] = select_parameters(model_name, params, train, seed, hpo)
                    run_model(path, model_name, seed, train, test, explained, background, metadata)
                    print(f"saved {source}/{model_name}/seed_{seed}", flush=True)
                except Exception as error:  # noqa: BLE001 — isolate model failures, as in the main pipeline.
                    # No traceback/exception object is retained: it could own the model.
                    failures += 1
                    path.parent.mkdir(parents=True, exist_ok=True)
                    failure = {
                        **identity,
                        "signature": signature,
                        "failure": {"type": type(error).__name__, "message": str(error)},
                    }
                    path.with_suffix(".json").write_text(_json(failure) + "\n")
                    print(
                        f"FAILED {source}/{model_name}/seed_{seed}: {type(error).__name__}; "
                        f"details in {path.with_suffix('.json')} (continuing)",
                        flush=True,
                    )
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
