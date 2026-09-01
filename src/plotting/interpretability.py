"""Train and compare EBM native effects with SHAP feature effects."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from src.classes.trainer import Trainer
from src.plotting.defaults import FEATURE_ALIASES, MODEL_STYLES, SMALL_FEATURE_NAMES, set_plot_style
from src.schemas.dataset_schemas import DatasetBundle
from src.schemas.training_schemas import ModelConfig, ModelPreprocessingConfig
from src.utils.model_lifecycle import release_model
from src.utils.model_registry import get_model_spec

_MODEL_NAMES = ("ebm", "xgboost", "tabpfn-3")


@dataclass(frozen=True)
class PointEffects:
    """Per-row feature values and corresponding SHAP values."""

    feature_names: tuple[str, ...]
    feature_values: np.ndarray
    effects: np.ndarray


@dataclass(frozen=True)
class EBMTermEffect:
    """One EBM main-effect curve from its native global explanation."""

    feature_name: str
    feature_values: np.ndarray
    effects: np.ndarray


@dataclass(frozen=True)
class InterpretabilitySeeds:
    """Independent random states for one interpretability run."""

    model: int
    background: int
    explanation: int
    explainer: int


@dataclass(frozen=True)
class InterpretabilityComparison:
    """Model effects and run metadata for one three-model comparison."""

    feature_names: tuple[str, ...]
    ebm: tuple[EBMTermEffect, ...]
    ebm_points: PointEffects
    xgboost: PointEffects
    tabpfn: PointEffects
    fit_times: Mapping[str, float]
    seeds: InterpretabilitySeeds
    tabpfn_cache_hit: bool = False


@dataclass(frozen=True)
class _CachedTabPFN:
    effects: PointEffects
    fit_time: float


def compute_interpretability_comparison(
    *,
    trainer: Trainer,
    data: DatasetBundle,
    model_params: Mapping[str, Mapping[str, Any]],
    test_source: str = "tudd",
    background_rows: int = 256,
    explanation_rows: int = 1000,
    class_index: int = 1,
    tabpfn_shap_budget: int = 2048,
    model_random_state: int = 1337,
    background_random_state: int = 2337,
    explanation_random_state: int = 3337,
    explainer_random_state: int = 4337,
    model_preprocessing: ModelPreprocessingConfig | None = None,
    tabpfn_cache_path: str | Path | None = None,
    recompute_tabpfn: bool = False,
) -> InterpretabilityComparison:
    """Train EBM, XGBoost, and TabPFNv3 and calculate one run of feature effects.

    EBM contributes native global main-effect curves and per-row native term
    contributions. XGBoost contributes TreeSHAP values, and TabPFNv3 contributes
    first-order ShapIQ SHAP values. Separate random states control model fitting,
    the TabPFN background sample, the fixed explanation sample, and the TabPFN
    explainer. Models are released immediately after their effects have been
    copied. When ``tabpfn_cache_path`` is set, compatible TabPFN effects are
    reused from disk.
    """
    missing = [name for name in _MODEL_NAMES if name not in model_params]
    if missing:
        raise ValueError(f"model_params is missing: {', '.join(missing)}")
    if background_rows < 1 or explanation_rows < 1:
        raise ValueError("background_rows and explanation_rows must be positive")
    if class_index != 1:
        raise ValueError("This binary-classification comparison currently supports class_index=1 only")

    test_sets = {"mimic": data.test_mimic, "tudd": data.test_tudd}
    try:
        test_data = test_sets[test_source]
    except KeyError as exc:
        raise ValueError("test_source must be 'mimic' or 'tudd'") from exc

    seeds = InterpretabilitySeeds(
        model=model_random_state,
        background=background_random_state,
        explanation=explanation_random_state,
        explainer=explainer_random_state,
    )
    effective_model_params = {
        name: {**dict(model_params[name]), "random_state": model_random_state} for name in _MODEL_NAMES
    }
    background_raw = _sample_rows(data.train_data.X, background_rows, background_random_state)
    explain_raw = _sample_rows(test_data.X, explanation_rows, explanation_random_state)
    y_train = data.train_data.y.to_numpy()
    preprocessing_settings = {
        "model_preprocessing": (None if model_preprocessing is None else model_preprocessing.model_dump(mode="json")),
        "default_imputer": trainer.default_imputer.model_dump(mode="json"),
        "default_scaler": trainer.default_scaler.model_dump(mode="json"),
        "log_transform_target": trainer.log_transform_target,
    }
    cache_path = Path(tabpfn_cache_path) if tabpfn_cache_path is not None else None
    cache_key = _tabpfn_cache_key(
        X_train=data.train_data.X,
        y_train=data.train_data.y,
        background_raw=background_raw,
        explain_raw=explain_raw,
        model_params=effective_model_params["tabpfn-3"],
        task_type=trainer.task_type,
        test_source=test_source,
        background_rows=background_rows,
        explanation_rows=explanation_rows,
        class_index=class_index,
        budget=tabpfn_shap_budget,
        seeds=seeds,
        preprocessing_settings=preprocessing_settings,
    )

    fit_times: dict[str, float] = {}
    feature_names: tuple[str, ...] | None = None
    ebm_effects: tuple[EBMTermEffect, ...] | None = None
    ebm_point_effects: PointEffects | None = None
    xgboost_effects: PointEffects | None = None
    tabpfn_effects: PointEffects | None = None
    tabpfn_cache_hit = False

    for model_name in _MODEL_NAMES:
        if model_name == "tabpfn-3" and cache_path is not None and cache_path.exists() and not recompute_tabpfn:
            cached = _load_tabpfn_cache(cache_path, cache_key)
            if cached is not None and cached.effects.feature_names == feature_names:
                tabpfn_effects, cached_fit_time = cached.effects, cached.fit_time
                fit_times[model_name] = cached_fit_time
                tabpfn_cache_hit = True
                continue

        trained_model = None
        try:
            model_config = ModelConfig(name=model_name, preprocessing=model_preprocessing)
            spec = get_model_spec(model_config, trainer.task_type)
            trained_model, fit_time = trainer._fit_model(
                model_config,
                spec,
                effective_model_params[model_name],
                data.train_data.X,
                y_train,
            )
            fit_times[model_name] = fit_time

            pipeline = trained_model.preprocess_pipeline
            estimator = trained_model.adapter.model
            background = np.asarray(pipeline.transform(background_raw))
            explained = np.asarray(pipeline.transform(explain_raw))
            current_names = tuple(str(name) for name in pipeline.get_feature_names_out())
            if feature_names is None:
                feature_names = current_names
            elif current_names != feature_names:
                raise ValueError("The models produced different transformed feature spaces")

            if model_name == "ebm":
                ebm_effects = _extract_ebm_effects(estimator, current_names)
                ebm_point_effects = PointEffects(
                    feature_names=current_names,
                    feature_values=explained.copy(),
                    effects=_ebm_term_contributions(estimator, explained, class_index),
                )
            elif model_name == "xgboost":
                xgboost_effects = PointEffects(
                    feature_names=current_names,
                    feature_values=explained.copy(),
                    effects=_xgboost_shap_values(estimator, explained, class_index),
                )
            else:
                tabpfn_effects = PointEffects(
                    feature_names=current_names,
                    feature_values=explained.copy(),
                    effects=_tabpfn_shap_values(
                        estimator=estimator,
                        background=background,
                        explained=explained,
                        feature_names=current_names,
                        class_index=class_index,
                        budget=tabpfn_shap_budget,
                        random_state=explainer_random_state,
                    ),
                )
                if cache_path is not None:
                    _save_tabpfn_cache(cache_path, cache_key, tabpfn_effects, fit_time)
        finally:
            release_model(trained_model)

    assert feature_names is not None
    assert ebm_effects is not None
    assert ebm_point_effects is not None
    assert xgboost_effects is not None
    assert tabpfn_effects is not None
    return InterpretabilityComparison(
        feature_names=feature_names,
        ebm=ebm_effects,
        ebm_points=ebm_point_effects,
        xgboost=xgboost_effects,
        tabpfn=tabpfn_effects,
        fit_times=fit_times,
        seeds=seeds,
        tabpfn_cache_hit=tabpfn_cache_hit,
    )


def plot_interpretability_comparison(
    comparison: InterpretabilityComparison,
    *,
    features: Sequence[str] | None = None,
    output_dir: str | Path | None = None,
    figsize: tuple[float, float] = (10.0, 6.0),
    point_size: float = 12.0,
    point_alpha: float = 0.35,
) -> tuple[tuple[str, plt.Figure], ...]:
    """Create one feature plot containing EBM, XGBoost, and TabPFNv3 effects.

    XGBoost and TabPFN are rendered as single-color point clouds. In
    particular, points are never colored by a secondary interaction feature.
    """
    set_plot_style()
    selected = tuple(features) if features is not None else comparison.feature_names
    unknown = [name for name in selected if name not in comparison.feature_names]
    if unknown:
        raise ValueError(f"Unknown features: {', '.join(unknown)}")

    target_dir = Path(output_dir) if output_dir is not None else None
    if target_dir is not None:
        target_dir.mkdir(parents=True, exist_ok=True)

    ebm_by_name = {effect.feature_name: effect for effect in comparison.ebm}
    feature_indexes = {name: index for index, name in enumerate(comparison.feature_names)}
    figures: list[tuple[str, plt.Figure]] = []

    for feature_name in selected:
        index = feature_indexes[feature_name]
        ebm_effect = ebm_by_name[feature_name]
        figure, axis = plt.subplots(figsize=figsize, constrained_layout=True)

        plotted_feature_values = np.concatenate(
            (
                comparison.xgboost.feature_values[:, index],
                comparison.tabpfn.feature_values[:, index],
            )
        )
        finite_feature_values = plotted_feature_values[np.isfinite(plotted_feature_values)]
        plotted_range = (
            (float(finite_feature_values.min()), float(finite_feature_values.max()))
            if finite_feature_values.size
            else None
        )
        _plot_ebm_line(axis, ebm_effect, plotted_range=plotted_range)
        axis.scatter(
            comparison.xgboost.feature_values[:, index],
            comparison.xgboost.effects[:, index],
            s=point_size,
            alpha=point_alpha,
            color=MODEL_STYLES["tabswift"].color,
            marker=MODEL_STYLES["xgboost"].marker,
            edgecolors="none",
            label=MODEL_STYLES["xgboost"].label,
            rasterized=True,
        )
        axis.scatter(
            comparison.tabpfn.feature_values[:, index],
            comparison.tabpfn.effects[:, index],
            s=point_size,
            alpha=point_alpha,
            color=MODEL_STYLES["tabpfn-3"].color,
            marker=MODEL_STYLES["tabpfn-3"].marker,
            edgecolors="none",
            label=MODEL_STYLES["tabpfn-3"].label,
            rasterized=True,
        )
        axis.axhline(0.0, color="#777777", linewidth=0.8, alpha=0.55)
        axis.set_xlabel(FEATURE_ALIASES.get(feature_name, feature_name))
        axis.set_ylabel("Feature effect")
        axis.legend()

        if target_dir is not None:
            slug = "".join(
                character if character.isalnum() or character in "-_." else "_" for character in feature_name
            )
            figure.savefig(target_dir / f"{index:02d}_{slug}.svg", bbox_inches="tight", pad_inches=0.2)
        figures.append((feature_name, figure))

    return tuple(figures)


def comparison_summary(comparison: InterpretabilityComparison) -> pd.DataFrame:
    """Return a compact table of fit times and explanation types."""
    explanation_types = {
        "ebm": "native EBM main effect",
        "xgboost": "TreeSHAP",
        "tabpfn-3": "first-order ShapIQ SHAP",
    }
    return pd.DataFrame(
        [
            {
                "model": MODEL_STYLES[name].label,
                "explanation": explanation_types[name],
                "fit_time_s": comparison.fit_times[name],
                "cache": "hit" if name == "tabpfn-3" and comparison.tabpfn_cache_hit else "computed",
                "model_seed": comparison.seeds.model,
                "background_seed": comparison.seeds.background,
                "explanation_seed": comparison.seeds.explanation,
                "explainer_seed": comparison.seeds.explainer,
            }
            for name in _MODEL_NAMES
        ]
    )


def global_feature_importance(comparisons: Sequence[InterpretabilityComparison]) -> pd.DataFrame:
    """Return per-run mean absolute feature effects and within-model ranks.

    The absolute magnitudes are only comparable within the same explanation
    method. Spearman correlations of the resulting ranks are used for comparisons
    across models and runs because the explanation scales can differ.
    """
    if not comparisons:
        raise ValueError("comparisons must contain at least one run")

    expected_names = comparisons[0].feature_names
    rows: list[dict[str, Any]] = []
    for run, comparison in enumerate(comparisons, start=1):
        if comparison.feature_names != expected_names:
            raise ValueError("All runs must use the same transformed feature space")

        effects_by_model = {
            "ebm": comparison.ebm_points.effects,
            "xgboost": comparison.xgboost.effects,
            "tabpfn-3": comparison.tabpfn.effects,
        }
        for model_name, effects in effects_by_model.items():
            if effects.ndim != 2 or effects.shape[1] != len(expected_names):
                raise ValueError(f"Unexpected {model_name} effect shape in run {run}: {effects.shape}")
            if not np.isfinite(effects).all():
                invalid_count = int(effects.size - np.isfinite(effects).sum())
                raise ValueError(f"{model_name} run {run} contains {invalid_count} non-finite feature effects")
            importance = np.mean(np.abs(effects), axis=0)
            ranks = pd.Series(importance).rank(method="average", ascending=False).to_numpy()
            rows.extend(
                {
                    "run": run,
                    "model_seed": comparison.seeds.model,
                    "background_seed": comparison.seeds.background,
                    "explanation_seed": comparison.seeds.explanation,
                    "explainer_seed": comparison.seeds.explainer,
                    "model": model_name,
                    "model_label": MODEL_STYLES[model_name].label,
                    "feature": feature_name,
                    "feature_label": FEATURE_ALIASES.get(feature_name, feature_name),
                    "mean_absolute_effect": float(importance[index]),
                    "rank": float(ranks[index]),
                }
                for index, feature_name in enumerate(expected_names)
            )

    return pd.DataFrame(rows).sort_values(["run", "model", "rank", "feature"], ignore_index=True)


def global_ranking_correlation_matrix(
    comparisons: Sequence[InterpretabilityComparison],
) -> pd.DataFrame:
    """Return the median between-model ranking correlation across runs.

    The diagonal is one by definition. Each off-diagonal cell is the median of
    the run-specific Spearman correlations for that pair of model families.
    Within-model stability across reruns remains available separately through
    :func:`ranking_correlation_table`.
    """
    details = ranking_correlation_table(comparisons)
    agreement = details[details["scope"] == "model agreement"]
    labels = [MODEL_STYLES[model_name].label for model_name in _MODEL_NAMES]
    correlations = pd.DataFrame(np.eye(len(_MODEL_NAMES)), index=labels, columns=labels)

    for model_a, model_b in combinations(_MODEL_NAMES, 2):
        label_a = MODEL_STYLES[model_a].label
        label_b = MODEL_STYLES[model_b].label
        comparison_label = f"{label_a} vs {label_b}"
        values = agreement.loc[agreement["comparison"] == comparison_label, "spearman_rho"]
        if values.empty:
            raise ValueError(f"No model-agreement correlations available for {comparison_label}")
        median_rho = float(values.median())
        if not np.isfinite(median_rho):
            raise ValueError(f"Median feature-ranking correlation is undefined for {comparison_label}")
        correlations.loc[label_a, label_b] = median_rho
        correlations.loc[label_b, label_a] = median_rho

    return correlations


def ranking_correlation_table(comparisons: Sequence[InterpretabilityComparison]) -> pd.DataFrame:
    """Summarize within-model run stability and within-run model agreement."""
    importance = global_feature_importance(comparisons)
    indexed = {
        (model_name, run): group.set_index("feature")["mean_absolute_effect"]
        for (model_name, run), group in importance.groupby(["model", "run"], sort=False)
    }
    rows: list[dict[str, Any]] = []
    run_numbers = tuple(range(1, len(comparisons) + 1))

    def _correlation(left_key: tuple[str, int], right_key: tuple[str, int]) -> float:
        rho = indexed[left_key].corr(indexed[right_key], method="spearman")
        if not np.isfinite(rho):
            raise ValueError(f"Feature-ranking correlation is undefined for {left_key} and {right_key}")
        return float(rho)

    for model_name in _MODEL_NAMES:
        for run_a, run_b in combinations(run_numbers, 2):
            rho = _correlation((model_name, run_a), (model_name, run_b))
            rows.append(
                {
                    "scope": "run stability",
                    "comparison": MODEL_STYLES[model_name].label,
                    "context": f"run {run_a} vs run {run_b}",
                    "spearman_rho": float(rho),
                }
            )

    for run in run_numbers:
        for model_a, model_b in combinations(_MODEL_NAMES, 2):
            rho = _correlation((model_a, run), (model_b, run))
            rows.append(
                {
                    "scope": "model agreement",
                    "comparison": f"{MODEL_STYLES[model_a].label} vs {MODEL_STYLES[model_b].label}",
                    "context": f"run {run}",
                    "spearman_rho": float(rho),
                }
            )

    return pd.DataFrame(rows)


def plot_global_feature_rankings(
    rankings: pd.DataFrame,
    *,
    top_n: int = 10,
    output_path: str | Path | None = None,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Plot median within-model ranks and their run-to-run ranges.

    Each model is shown in a separate panel because its top features may differ.
    Features are selected by median rank across runs; horizontal intervals show
    the minimum and maximum ranks observed across those runs. No absolute effect
    magnitudes are compared across explanation methods.
    """
    required_columns = {"run", "model", "feature", "feature_label", "rank"}
    missing_columns = required_columns.difference(rankings.columns)
    if missing_columns:
        raise ValueError(f"rankings is missing columns: {', '.join(sorted(missing_columns))}")
    if top_n < 1:
        raise ValueError("top_n must be positive")
    if rankings.empty:
        raise ValueError("rankings must contain at least one row")
    if rankings.duplicated(["run", "model", "feature"]).any():
        raise ValueError("rankings must contain one row per run, model, and feature")
    rank_values = rankings["rank"].to_numpy(dtype=float)
    if not np.isfinite(rank_values).all() or (rank_values < 1).any():
        raise ValueError("rankings contains invalid rank values")

    available_models = set(rankings["model"])
    missing_models = [model_name for model_name in _MODEL_NAMES if model_name not in available_models]
    if missing_models:
        raise ValueError(f"rankings is missing models: {', '.join(missing_models)}")

    summaries: list[pd.DataFrame] = []
    for model_name in _MODEL_NAMES:
        model_rankings = rankings.loc[rankings["model"] == model_name]
        label_counts = model_rankings.groupby("feature")["feature_label"].nunique()
        if (label_counts != 1).any():
            inconsistent = ", ".join(label_counts[label_counts != 1].index.astype(str))
            raise ValueError(f"Features have inconsistent labels: {inconsistent}")
        summary = (
            model_rankings.groupby(["feature", "feature_label"], as_index=False)
            .agg(
                median_rank=("rank", "median"),
                minimum_rank=("rank", "min"),
                maximum_rank=("rank", "max"),
                run_count=("run", "nunique"),
            )
            .sort_values(["median_rank", "feature"], kind="stable")
            .head(top_n)
            .assign(model=model_name, model_label=MODEL_STYLES[model_name].label)
        )
        summaries.append(summary)
    top_rankings = pd.concat(summaries, ignore_index=True)

    set_plot_style()
    figure, axes = plt.subplots(
        1,
        len(_MODEL_NAMES),
        figsize=(12.0, 5.2),
        sharex=True,
        constrained_layout=True,
    )
    maximum_displayed_rank = float(top_rankings["maximum_rank"].max())
    for axis, model_name in zip(axes, _MODEL_NAMES, strict=True):
        model_summary = top_rankings.loc[top_rankings["model"] == model_name].reset_index(drop=True)
        positions = np.arange(len(model_summary))
        style = MODEL_STYLES[model_name]
        lower_errors = model_summary["median_rank"] - model_summary["minimum_rank"]
        upper_errors = model_summary["maximum_rank"] - model_summary["median_rank"]
        axis.errorbar(
            model_summary["median_rank"],
            positions,
            xerr=np.vstack((lower_errors, upper_errors)),
            fmt=style.marker,
            markersize=6.0,
            color=style.color,
            markeredgecolor="#222222",
            markeredgewidth=0.6,
            elinewidth=1.5,
            capsize=2.5,
        )
        axis.set_yticks(
            positions,
            model_summary["feature_label"].map(_small_feature_names),
        )
        axis.invert_yaxis()
        axis.set_title(style.label)
        axis.set_xlim(0.5, maximum_displayed_rank + 0.5)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(axis="y", visible=False)
        axis.tick_params(axis="y", length=0)

    figure.supxlabel("Within-model feature-importance rank (1 = highest)")
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(target, bbox_inches="tight", pad_inches=0.2)
    return figure, top_rankings


def plot_global_ranking_correlations(
    comparisons: Sequence[InterpretabilityComparison],
    *,
    output_path: str | Path | None = None,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Plot median between-model Spearman correlations of global rankings."""
    set_plot_style()
    correlations = global_ranking_correlation_matrix(comparisons)
    figure, axis = plt.subplots(figsize=(5.0, 5.0), constrained_layout=True)
    image = axis.imshow(correlations.to_numpy(), cmap="coolwarm", vmin=-1.0, vmax=1.0)
    tick_positions = np.arange(len(correlations.columns))
    axis.set_xticks(tick_positions, correlations.columns)
    axis.set_yticks(tick_positions, correlations.index)
    axis.tick_params(axis="both", which="both", length=0)
    axis.set_xlabel("")
    axis.set_ylabel("")
    axis.grid(False)

    for row_index, column_index in np.ndindex(correlations.shape):
        value = correlations.iat[row_index, column_index]
        label = "NA" if not np.isfinite(value) else f"{value:.2f}"
        text_color = "white" if np.isfinite(value) and abs(value) >= 0.55 else "black"
        axis.text(column_index, row_index, label, ha="center", va="center", color=text_color, fontsize=8)

    colorbar = figure.colorbar(image, ax=axis, shrink=0.8)
    colorbar.set_label("Spearman $\\rho$")
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(target, bbox_inches="tight", pad_inches=0.2)
    return figure, correlations


def _sample_rows(frame, requested_rows: int, random_state: int):
    return frame.sample(n=min(requested_rows, len(frame)), random_state=random_state)


def _tabpfn_cache_key(
    *,
    X_train,
    y_train,
    background_raw,
    explain_raw,
    model_params: Mapping[str, Any],
    task_type: str,
    test_source: str,
    background_rows: int,
    explanation_rows: int,
    class_index: int,
    budget: int,
    seeds: InterpretabilitySeeds,
    preprocessing_settings: Mapping[str, Any],
) -> str:
    settings = {
        "cache_schema": 2,
        "model_params": model_params,
        "task_type": task_type,
        "test_source": test_source,
        "background_rows": background_rows,
        "explanation_rows": explanation_rows,
        "class_index": class_index,
        "budget": budget,
        "seeds": {
            "model": seeds.model,
            "background": seeds.background,
            "explanation": seeds.explanation,
            "explainer": seeds.explainer,
        },
        "preprocessing": preprocessing_settings,
        "shapiq": {"index": "SV", "max_order": 1, "imputer": "baseline"},
    }
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True, default=repr).encode())
    for frame in (X_train, y_train, background_raw, explain_raw):
        row_hashes = pd.util.hash_pandas_object(frame, index=True).to_numpy()
        digest.update(row_hashes.tobytes())
    return digest.hexdigest()


def _load_tabpfn_cache(path: Path, expected_key: str) -> _CachedTabPFN | None:
    try:
        with np.load(path, allow_pickle=False) as cached:
            if str(cached["cache_key"].item()) != expected_key:
                return None
            feature_names = tuple(str(name) for name in cached["feature_names"].tolist())
            feature_values = np.asarray(cached["feature_values"])
            effects = np.asarray(cached["effects"])
            if (
                feature_values.ndim != 2
                or feature_values.shape != effects.shape
                or feature_values.shape[1] != len(feature_names)
            ):
                return None
            return _CachedTabPFN(
                effects=PointEffects(
                    feature_names=feature_names,
                    feature_values=feature_values,
                    effects=effects,
                ),
                fit_time=float(cached["fit_time"].item()),
            )
    except (KeyError, OSError, ValueError):
        return None


def _save_tabpfn_cache(path: Path, cache_key: str, effects: PointEffects, fit_time: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp.npz")
    np.savez_compressed(
        temporary_path,
        cache_key=np.asarray(cache_key),
        feature_names=np.asarray(effects.feature_names),
        feature_values=effects.feature_values,
        effects=effects.effects,
        fit_time=np.asarray(fit_time),
    )
    temporary_path.replace(path)


def _extract_ebm_effects(estimator, feature_names: tuple[str, ...]) -> tuple[EBMTermEffect, ...]:
    if getattr(estimator, "interactions", 0) != 0:
        raise ValueError("EBM interactions must be disabled for this comparison")

    explanation = estimator.explain_global()
    effects = []
    for index, feature_name in enumerate(feature_names):
        term = explanation.data(index)
        effects.append(
            EBMTermEffect(
                feature_name=feature_name,
                feature_values=np.asarray(term["names"]),
                effects=np.asarray(term["scores"], dtype=float),
            )
        )
    return tuple(effects)


def _ebm_term_contributions(estimator, explained: np.ndarray, class_index: int) -> np.ndarray:
    expected_terms = tuple((index,) for index in range(explained.shape[1]))
    if tuple(getattr(estimator, "term_features_", ())) != expected_terms:
        raise ValueError("EBM must contain exactly one main-effect term per transformed feature")
    values = np.asarray(estimator.eval_terms(explained))
    return _select_output(values, explained.shape, class_index, "EBM")


def _xgboost_shap_values(estimator, explained: np.ndarray, class_index: int) -> np.ndarray:
    import shap

    values = np.asarray(shap.TreeExplainer(estimator)(explained).values)
    return _select_output(values, explained.shape, class_index, "XGBoost")


def _tabpfn_shap_values(
    *,
    estimator,
    background: np.ndarray,
    explained: np.ndarray,
    feature_names: tuple[str, ...],
    class_index: int,
    budget: int,
    random_state: int,
) -> np.ndarray:
    from tabpfn_extensions.interpretability import shapiq as tabpfn_shapiq
    from tabpfn_extensions.interpretability import shapiq_to_shap_explanation

    explainer = tabpfn_shapiq.get_tabpfn_imputation_explainer(
        model=estimator,
        data=background,
        index="SV",
        max_order=1,
        imputer="baseline",
        class_index=class_index,
        random_state=random_state,
    )
    explanation = shapiq_to_shap_explanation(
        explainer,
        explained,
        budget=budget,
        feature_names=list(feature_names),
    )
    values = np.asarray(explanation.values)
    return _select_output(values, explained.shape, class_index, "TabPFN")


def _select_output(
    values: np.ndarray, expected_shape: tuple[int, int], class_index: int, model_name: str
) -> np.ndarray:
    if values.shape == expected_shape:
        return values.copy()
    if values.ndim == 3 and values.shape[:2] == expected_shape:
        if not 0 <= class_index < values.shape[2]:
            raise ValueError(f"class_index {class_index} is unavailable for {model_name}")
        return values[:, :, class_index].copy()
    raise ValueError(f"Unexpected {model_name} SHAP shape {values.shape}; expected {expected_shape}")


def _plot_ebm_line(
    axis,
    effect: EBMTermEffect,
    *,
    plotted_range: tuple[float, float] | None = None,
) -> None:
    style = MODEL_STYLES["ebm"]
    x_values = effect.feature_values
    line_options = {
        "color": MODEL_STYLES["xgboost"].color,
        "linestyle": style.linestyle,
        "linewidth": 2.2,
        "label": style.label,
        "zorder": 3,
    }
    if np.issubdtype(x_values.dtype, np.number) and len(x_values) == len(effect.effects) + 1:
        # EBM assigns out-of-range values to its first or last bin. Extend those
        # boundary bins across the plotted samples rather than dropping to zero.
        edges = x_values.astype(float).copy()
        if plotted_range is not None:
            edges[0] = min(edges[0], plotted_range[0])
            edges[-1] = max(edges[-1], plotted_range[1])
        axis.stairs(effect.effects, edges, baseline=None, **line_options)
    else:
        axis.plot(x_values, effect.effects, **line_options)


def _small_feature_names(feature_name: str) -> str:
    return SMALL_FEATURE_NAMES.get(feature_name, feature_name)
