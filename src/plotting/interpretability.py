"""Train and compare EBM native effects with SHAP feature effects."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.classes.trainer import Trainer
from src.plotting.defaults import FEATURE_ALIASES, MODEL_STYLES, set_plot_style
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
class InterpretabilityComparison:
    """Model effects needed to render the three-model feature plots."""

    feature_names: tuple[str, ...]
    ebm: tuple[EBMTermEffect, ...]
    xgboost: PointEffects
    tabpfn: PointEffects
    fit_times: Mapping[str, float]


def compute_interpretability_comparison(
    *,
    trainer: Trainer,
    data: DatasetBundle,
    model_params: Mapping[str, Mapping[str, Any]],
    test_source: str = "tudd",
    background_rows: int = 256,
    explanation_rows: int = 1_000,
    class_index: int = 1,
    tabpfn_shap_budget: int = 2_048,
    random_state: int = 1_337,
    model_preprocessing: ModelPreprocessingConfig | None = None,
) -> InterpretabilityComparison:
    """Train EBM, XGBoost, and TabPFNv3 sequentially and calculate feature effects.

    EBM contributes its native global main-effect curves. XGBoost contributes
    TreeSHAP values, and TabPFNv3 contributes first-order ShapIQ SHAP values.
    Models are released immediately after their effects have been copied.
    """
    missing = [name for name in _MODEL_NAMES if name not in model_params]
    if missing:
        raise ValueError(f"model_params is missing: {', '.join(missing)}")
    if background_rows < 1 or explanation_rows < 1:
        raise ValueError("background_rows and explanation_rows must be positive")

    test_sets = {"mimic": data.test_mimic, "tudd": data.test_tudd}
    try:
        test_data = test_sets[test_source]
    except KeyError as exc:
        raise ValueError("test_source must be 'mimic' or 'tudd'") from exc

    background_raw = _sample_rows(data.train_data.X, background_rows, random_state)
    explain_raw = _sample_rows(test_data.X, explanation_rows, random_state + 1)
    y_train = data.train_data.y.to_numpy()

    fit_times: dict[str, float] = {}
    feature_names: tuple[str, ...] | None = None
    ebm_effects: tuple[EBMTermEffect, ...] | None = None
    xgboost_effects: PointEffects | None = None
    tabpfn_effects: PointEffects | None = None

    for model_name in _MODEL_NAMES:
        trained_model = None
        try:
            model_config = ModelConfig(name=model_name, preprocessing=model_preprocessing)
            spec = get_model_spec(model_config, trainer.task_type)
            trained_model, fit_time = trainer._fit_model(
                model_config,
                spec,
                dict(model_params[model_name]),
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
                        random_state=random_state,
                    ),
                )
        finally:
            release_model(trained_model)

    assert feature_names is not None
    assert ebm_effects is not None
    assert xgboost_effects is not None
    assert tabpfn_effects is not None
    return InterpretabilityComparison(
        feature_names=feature_names,
        ebm=ebm_effects,
        xgboost=xgboost_effects,
        tabpfn=tabpfn_effects,
        fit_times=fit_times,
    )


def plot_interpretability_comparison(
    comparison: InterpretabilityComparison,
    *,
    features: Sequence[str] | None = None,
    output_dir: str | Path | None = None,
    figsize: tuple[float, float] = (8.0, 5.5),
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

        _plot_ebm_line(axis, ebm_effect)
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
        axis.set_ylabel("Feature effect / SHAP value")
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
            }
            for name in _MODEL_NAMES
        ]
    )


def _sample_rows(frame, requested_rows: int, random_state: int):
    return frame.sample(n=min(requested_rows, len(frame)), random_state=random_state)


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


def _plot_ebm_line(axis, effect: EBMTermEffect) -> None:
    style = MODEL_STYLES["ebm"]
    x_values = effect.feature_values
    line_options = {
        "color": "#B3B3B3",
        "linestyle": style.linestyle,
        "linewidth": 2.2,
        "label": style.label,
        "zorder": 3,
    }
    if np.issubdtype(x_values.dtype, np.number) and len(x_values) == len(effect.effects) + 1:
        # EBM exposes continuous terms as one score per interval and its edges.
        # A stair plot preserves that native step function.
        axis.stairs(effect.effects, x_values.astype(float), **line_options)
    else:
        axis.plot(x_values, effect.effects, **line_options)
