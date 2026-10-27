"""Mortality interpretation experiment. Run with ``uv run python -m src.interpretability.compute``.

Edit the settings and kwargs blocks below, not pipeline YAML. Each completed NPZ
holds one fitted model's results on BOTH held-out cohorts; no estimator is
serialized.
"""

import argparse
import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from importlib.metadata import version
from inspect import getsource
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.utils import resample

from src.classes.data_registry import dataset_task_for_target
from src.classes.dataset import Dataset
from src.interpretability import DEFAULT_MODELS, MODELS, SOURCES, TARGET
from src.interpretability.explanations import explain_model
from src.interpretability.tuning import select_parameters
from src.schemas.dataset_schemas import DatasetBundle, DatasetConfig, DataSplitConfig, XYDataset
from src.schemas.training_schemas import ModelConfig
from src.utils.evaluation import evaluate_trained_model
from src.utils.model_lifecycle import release_model
from src.utils.model_registry import get_model_spec

SCHEMA_VERSION = 1
TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "interpret",
    "xgboost",
    "tabpfn",
    "torch",
    "shap",
    "shapiq",
)


@dataclass(frozen=True)
class ExperimentSettings:
    """Volumes and seeds; every field enters the result fingerprint."""

    train_size: int
    repeat_seeds: tuple[int, ...]
    split_seed: int
    train_fraction: float
    explanation_size: int
    background_tudd: int
    background_mimic: int
    background_seed: int
    explanation_seed: int
    explainer_seed: int
    shap_budget: int


# The split is fixed as in the core experiment. Background, explanation, and
# explainer seeds stay fixed; training selection and model seeds follow the
# repeat. Background patients come from the full training pools (EUH 16379,
# MIMIC 44454 rows), not from a selected training subset.
SETTINGS = ExperimentSettings(
    train_size=10_000,
    repeat_seeds=(1337, 1338, 1339),
    split_seed=1337,
    train_fraction=0.8,
    explanation_size=250,
    background_tudd=64,
    background_mimic=64,
    background_seed=2337,
    explanation_seed=5337,
    explainer_seed=3337,
    shap_budget=256,  # Coalitions PER explained patient (TabPFN only).
)

# Extra estimator kwargs. EBM interactions stay disabled (the figures show main
# effects) and TabPFN's version comes from the registry; do not override either.
# Set estimator seeds via repeat_seeds, not inside these dictionaries.
ebm_kwargs = {}
xgboost_kwargs = {}
tabpfn_kwargs = {"n_estimators": 8}

# Match the existing baseline experiment; ranges come from model_registry.
ebm_hpo = {"search_space": "default", "n_trials": 50, "n_startup_trials": 5}
xgboost_hpo = {"search_space": "good", "n_trials": 100, "n_startup_trials": 10}

ROOT = Path(__file__).resolve().parents[2]


def settings() -> dict:
    """The run fingerprint settings recorded with every result."""
    return {
        **asdict(SETTINGS),
        "target": TARGET,
        "positive_class": 1,
        "preprocessing": "native",
        "output": "binary_mortality_log_odds",
        "reference": "joint_empirical_marginal",
    }


def load_pools() -> tuple[dict[str, XYDataset], dict[str, XYDataset]]:
    """Reuse the core loader, split, and alignment; keep source-local row IDs.

    The core's public get_dataset combines/resets training indices. Composing its
    existing split primitives preserves row IDs for recreation without a second
    implementation of clinical filtering or splitting.
    """
    builder = Dataset(
        DatasetConfig(
            target=TARGET,
            random_state=SETTINGS.split_seed,
            train_size=SETTINGS.train_fraction,
            train_on=(DataSplitConfig(dataset="mimic", fraction=1.0),),
        ),
        sample_seed=SETTINGS.split_seed,
    )
    split_frames = {}
    for source, frame in builder._load_data().items():
        X_train, X_test, y_train, y_test = builder._split_single_df(frame)
        split_frames[source] = {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}
    builder._align_split_feature_columns(split_frames)
    train = {source: XYDataset(X=split["X_train"], y=split["y_train"]) for source, split in split_frames.items()}
    test = {source: XYDataset(X=split["X_test"], y=split["y_test"]) for source, split in split_frames.items()}
    return train, test


def sample_rows(data: XYDataset, count: int, seed: int) -> XYDataset:
    """Stratified sample without replacement, then the core's row shuffle."""
    rows = resample(data.X.index, replace=False, n_samples=count, random_state=seed, stratify=data.y)
    X = data.X.loc[rows].sample(frac=1, random_state=seed)
    return XYDataset(X=X, y=data.y.loc[X.index])


def _model_params(model_name: str, features: list[str]) -> dict:
    """Adapter kwargs for one model; the registry pins the TabPFN version."""
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
        raise ValueError("Use the registry model name; do not set version in tabpfn_kwargs")
    return {
        "categorical_features_indices": [index for index, name in enumerate(features) if name == "Sex"],
        **tabpfn_kwargs,
    }


def _record_fit(model, model_name: str, train: XYDataset, metadata: dict) -> None:
    """Fit in place and record values derived from the fitted estimator."""
    started = perf_counter()
    model.fit(train.X.to_numpy(dtype=float), train.y.to_numpy())
    metadata["fit_seconds"] = perf_counter() - started
    metadata["adapter_kwargs"] = model.kwargs
    metadata["estimator_params"] = model.model.get_params(deep=False)
    if model_name in ("tabpfn-3.5", "tabpfn-3.5-fast"):
        metadata["resolved_n_estimators"] = model.model.n_estimators_


def _fit_payload(train: XYDataset, background: pd.DataFrame) -> dict:
    """Arrays shared by both cohorts: the selected training set and the background."""
    return {
        "feature_names": np.asarray(train.X.columns, dtype=str),
        "train_row_ids": train.X.index.to_numpy(),
        "train_y": train.y.to_numpy(),
        "background_X": background.to_numpy(dtype=float),
        "background_row_ids": background.index.get_level_values("row_id").to_numpy(),
        "background_sources": background.index.get_level_values("source").to_numpy(dtype=str),
    }


def _evaluate_model(model, task_type, train: XYDataset, test: dict, metadata: dict) -> dict:
    """Evaluate both held-out cohorts once; record point metrics and timings."""
    evaluation = evaluate_trained_model(
        model,
        task_type,
        DatasetBundle(train_data=train, test_mimic=test["mimic"], test_tudd=test["tudd"]),
    )
    metadata["metrics"] = {
        "tudd": asdict(evaluation.metrics.tudd_test),
        "mimic": asdict(evaluation.metrics.mimic_test),
    }
    metadata["prediction_seconds"] = {
        "tudd": evaluation.metrics.tudd_prediction_time,
        "mimic": evaluation.metrics.mimic_prediction_time,
    }
    metadata["explanations"] = {}
    return {"tudd": evaluation.test_predictions.tudd, "mimic": evaluation.test_predictions.mimic}


def _explain_cohort(
    model_name: str,
    estimator,
    cohort: str,
    explained: XYDataset,
    background: pd.DataFrame,
    prediction,
    metadata: dict,
) -> dict:
    """Explain one cohort's fixed patients and return its NPZ arrays."""
    X = explained.X.to_numpy(dtype=float)
    print(f"  explain {cohort}: {len(X)} patients", flush=True)
    started = perf_counter()
    effects = explain_model(
        model_name,
        estimator,
        X,
        background.to_numpy(dtype=float),
        budget=SETTINGS.shap_budget,
        seed=SETTINGS.explainer_seed,
    )
    metadata["explanations"][cohort] = {
        "seconds": perf_counter() - started,
        "method": effects["method"],
        "max_additivity_residual": effects["max_additivity_residual"],
    }
    return {
        f"{cohort}_X": X,
        f"{cohort}_row_ids": explained.X.index.to_numpy(),
        f"{cohort}_shap_values": effects["shap_values"],
        f"{cohort}_base_values": effects["base_values"],
        f"{cohort}_model_output": effects["model_output"],
        f"{cohort}_importance": np.abs(effects["shap_values"]).mean(axis=0),
        f"{cohort}_test_row_ids": prediction.test_set_id,
        f"{cohort}_test_y": prediction.y_true,
        f"{cohort}_test_probability": prediction.positive_class_probability,
    }


def run_model(path, model_name, seed, train, test, explained, background, metadata) -> None:
    """Fit, evaluate, explain, record, then release exactly one model."""
    task_type = dataset_task_for_target(TARGET).task_type
    model = None
    try:
        # Bare adapters intentionally avoid the Trainer's external preprocessing.
        model = get_model_spec(ModelConfig(name=model_name), task_type).create(
            task_type, metadata["params"], random_state=seed, inference_state=SETTINGS.explainer_seed
        )
        _record_fit(model, model_name, train, metadata)
        predictions = _evaluate_model(model, task_type, train, test, metadata)
        payload = _fit_payload(train, background)
        for cohort in SOURCES:
            payload.update(
                _explain_cohort(
                    model_name, model.model, cohort, explained[cohort], background, predictions[cohort], metadata
                )
            )
        save_result(path, payload, metadata)
    finally:
        release_model(model)


def save_result(path: Path, payload: dict, metadata: dict) -> None:
    """Write the sidecars and the atomic NPZ that marks a completed unit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = _json(metadata)
    path.with_suffix(".json").write_text(json.dumps(json.loads(text), indent=2) + "\n")
    importance = [
        {"cohort": cohort, "feature": feature, "mean_abs_shap": float(value)}
        for cohort in SOURCES
        for feature, value in zip(payload["feature_names"], payload[f"{cohort}_importance"], strict=True)
    ]
    pd.DataFrame(importance).to_csv(path.with_suffix(".importance.csv"), index=False)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, metadata_json=np.asarray(text), **payload)
    temporary.replace(path)


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
    digest = hashlib.sha256()
    for part in parts:
        digest.update(pd.util.hash_pandas_object(part, index=True).to_numpy().tobytes())
        if isinstance(part, pd.DataFrame):
            digest.update(_json(list(part.columns)).encode())
    return digest.hexdigest()


# Code that can change one model's saved result. Function sources, not compute.py
# itself, are hashed so editing a comment or one model's kwargs does not
# invalidate another model's completed fits.
_SHARED_SOURCES = (
    "src/interpretability/explanations.py",
    "src/utils/model_registry.py",
    "src/classes/dataset.py",
    "src/classes/data_registry.py",
    "src/utils/dataset_utils.py",
)
_TUNING_SOURCES = (
    "src/interpretability/tuning.py",
    "src/classes/trainer.py",
    "src/classes/preprocessor.py",
    "src/schemas/training_schemas.py",
    "src/schemas/pipeline_schemas.py",
    "src/utils/evaluation_utils.py",
    "src/utils/classification_metrics.py",
    "src/utils/optuna_callbacks.py",
)
_ADAPTERS = {
    "ebm": "sklearn_adapter.py",
    "xgboost": "sklearn_adapter.py",
    "tabpfn-3.5": "tabpfn_adapter.py",
    "tabpfn-3.5-fast": "tabpfn_adapter.py",
}
_FINGERPRINTED_FUNCTIONS = (
    load_pools,
    sample_rows,
    _record_fit,
    _evaluate_model,
    _explain_cohort,
    _fit_payload,
    run_model,
)


def _source_hashes(model_name: str) -> dict[str, str]:
    """Fingerprint the files and functions that produced one model's result."""
    relatives = (*_SHARED_SOURCES, f"src/adapter/{_ADAPTERS[model_name]}")
    if model_name in ("ebm", "xgboost"):
        relatives += _TUNING_SOURCES
    hashes = {relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() for relative in relatives}
    hashes["experiment_functions"] = hashlib.sha256(
        "".join(getsource(function) for function in _FINGERPRINTED_FUNCTIONS).encode()
    ).hexdigest()
    return hashes


def _hpo_record(model_name: str) -> dict | None:
    """HPO recipe with its record fields; TabPFN keeps fixed parameters."""
    if model_name not in ("ebm", "xgboost"):
        return None
    config = ebm_hpo if model_name == "ebm" else xgboost_hpo
    return {**config, "inference_seed": SETTINGS.explainer_seed, "optuna_version": version("optuna")}


def _stored_signature(path: Path) -> str:
    """Signature of a completed unit; the NPZ only exists after a full write."""
    with np.load(path, allow_pickle=False) as stored:
        return json.loads(str(stored["metadata_json"]))["signature"]


def _backup_previous(path: Path) -> None:
    """Move a completed unit aside so a failed replacement cannot pass as it."""
    for suffix in (".npz", ".json", ".importance.csv"):
        previous = path.with_suffix(suffix)
        if previous.exists():
            previous.replace(previous.with_name(f"{previous.stem}.previous{previous.suffix}"))


def _record_failure(path: Path, identity: dict, signature: str, error: Exception) -> None:
    """Record a failure without retaining a traceback, which could own the model."""
    path.parent.mkdir(parents=True, exist_ok=True)
    failure = {**identity, "signature": signature, "failure": {"type": type(error).__name__, "message": str(error)}}
    path.with_suffix(".json").write_text(_json(failure) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "run_results/interpretability" / TARGET)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(DEFAULT_MODELS))
    parser.add_argument("--sources", nargs="+", choices=SOURCES, default=list(SOURCES))
    parser.add_argument(
        "--seeds", nargs="+", type=int, choices=SETTINGS.repeat_seeds, default=list(SETTINGS.repeat_seeds)
    )
    parser.add_argument("--recompute", action="store_true", help="Explicitly replace selected completed model results")
    parser.add_argument(
        "--dry-run", action="store_true", help="Load/sample data and show reuse status, without fitting"
    )
    args = parser.parse_args()

    if min(SETTINGS.train_size, SETTINGS.explanation_size, SETTINGS.shap_budget) < 1:
        parser.error("train_size, explanation_size and shap_budget must be positive")
    background_counts = (SETTINGS.background_tudd, SETTINGS.background_mimic)
    if min(background_counts) < 0 or sum(background_counts) == 0:
        parser.error("background counts must be non-negative, with at least one background patient")

    pools, test = load_pools()
    features = list(pools["mimic"].X.columns)
    explained = {
        source: sample_rows(test[source], SETTINGS.explanation_size, SETTINGS.explanation_seed) for source in SOURCES
    }
    background_parts = {
        "tudd": pools["tudd"].X.sample(n=SETTINGS.background_tudd, random_state=SETTINGS.background_seed),
        "mimic": pools["mimic"].X.sample(n=SETTINGS.background_mimic, random_state=SETTINGS.background_seed),
    }
    background = pd.concat(background_parts, names=["source", "row_id"])
    shared_hash = _data_hash(
        background, *(part for source in SOURCES for part in (test[source].X, test[source].y, explained[source].X))
    )
    dependencies = {name: version(name) for name in TRACKED_PACKAGES}
    dependencies["python"] = platform.python_version()
    common = settings()
    pools_text = ", ".join(f"{source}={len(pools[source].y)}" for source in SOURCES)
    tests_text = ", ".join(f"{source}={len(test[source].y)}" for source in SOURCES)
    print(f"Training pools: {pools_text}; selected={SETTINGS.train_size}/source")
    print(f"Full test sizes: {tests_text}; explained={SETTINGS.explanation_size}/cohort")
    print(f"Background tudd={SETTINGS.background_tudd}, mimic={SETTINGS.background_mimic}; results={args.results}")

    failures = 0
    for seed in args.seeds:
        for source in args.sources:
            train = sample_rows(pools[source], SETTINGS.train_size, seed)
            training_hash = _data_hash(train.X, train.y)
            for model_name in args.models:
                path = args.results / source / model_name / f"seed_{seed}.npz"
                params = _model_params(model_name, features)
                hpo = _hpo_record(model_name)
                identity = {
                    "schema_version": SCHEMA_VERSION,
                    "model": model_name,
                    "train_source": source,
                    "seed": seed,
                    "params": params,
                    # Adding another repeat need not invalidate already completed fits.
                    "settings": {key: value for key, value in common.items() if key != "repeat_seeds"},
                    "data_hash": shared_hash,
                    "training_hash": training_hash,
                    "dependencies": dependencies,
                    "source_hashes": _source_hashes(model_name),
                }
                if hpo is not None:
                    identity["hpo"] = hpo
                signature = hashlib.sha256(_json(identity).encode()).hexdigest()
                metadata = {**identity, "settings": common, "signature": signature}
                if path.exists() and not args.recompute:
                    if _stored_signature(path) != signature:
                        raise ValueError(
                            f"Settings/data changed for {path}. Choose a new --results directory or --recompute"
                        )
                    print(f"reuse {source}/{model_name}/seed_{seed}", flush=True)
                    continue
                mode = f"HPO {hpo['search_space']}, up to {hpo['n_trials']} trials" if hpo else "fixed parameters"
                verb = "would compute" if args.dry_run else "compute"
                print(f"{verb} {source}/{model_name}/seed_{seed} ({mode})", flush=True)
                if args.dry_run:
                    continue
                if args.recompute and path.exists():
                    _backup_previous(path)
                try:
                    if hpo is not None:
                        metadata["base_params"] = params
                        metadata["params"], metadata["tuning"] = select_parameters(
                            model_name, params, train, seed, hpo=hpo, inference_seed=SETTINGS.explainer_seed
                        )
                    run_model(path, model_name, seed, train, test, explained, background, metadata)
                    print(f"saved {source}/{model_name}/seed_{seed}", flush=True)
                except Exception as error:  # noqa: BLE001 — isolate model failures, as in the main pipeline.
                    failures += 1
                    _record_failure(path, identity, signature, error)
                    print(
                        f"FAILED {source}/{model_name}/seed_{seed}: {type(error).__name__}; "
                        f"details in {path.with_suffix('.json')} (continuing)",
                        flush=True,
                    )
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
