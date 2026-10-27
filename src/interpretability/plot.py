"""Plot saved SHAP results only; never import fitting or explanation code.

Run from the repository root::

    uv run python -m src.interpretability.plot
    uv run python -m src.interpretability.plot --no-bands --features Age Sex

XGBoost/TabPFN patients are drawn once, using mean SHAP across saved repeats.
EBM alone has a line: its saved centered main effects, without binning or
smoothing. The line connects observed values; native EBM bin boundaries are not
cached. EBM bands are repeat min/max ranges, NOT confidence intervals. Histogram
bins use pooled observed values from the two explanation cohorts with identical
edges. Missing original feature values have a separate strip on the same SHAP
scale. Only PDF outputs are written under the supplied results directory/figures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

# Support both `python -m src.interpretability.plot` and direct script execution.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# These modules contain display defaults only, with no training/model imports.
from src.interpretability import DEFAULT_MODELS, MODELS, SOURCES, TARGET
from src.plotting.defaults import DATASET_NAMES, FEATURE_ALIASES, MODEL_STYLES, SMALL_FEATURE_NAMES
from src.plotting.scientific_figstyle import BASELINE, PALETTE, TEXT, save, use_style


@dataclass(frozen=True)
class PlotSettings:
    """Editable publication settings; CLI options override these defaults."""

    show_bands: bool = True
    n_bins: int = 10
    selected_features: tuple[str, ...] | None = None
    width: float = TEXT
    height: float = 5.4
    scatter_size: float = 5.0
    scatter_alpha: float = 0.20
    band_alpha: float = 0.18
    line_width: float = 1.25
    discrete_features: tuple[str, ...] = ("Sex",)


PLOT_SETTINGS = PlotSettings()
DEFAULT_RESULTS = Path(__file__).resolve().parents[2] / "run_results/interpretability" / TARGET
# Interpretability-only encodings; do not change other evaluation figures.
MODEL_COLORS = {
    "ebm": "#000000",
    "xgboost": PALETTE["orange"],
    "tabpfn-3.5": MODEL_STYLES["tabpfn-3.5"].color,
    "tabpfn-3.5-fast": MODEL_STYLES["tabpfn-3.5-fast"].color,
}
_RESULT_FILE = re.compile(r"seed_(\d+)\.npz")


@dataclass(frozen=True)
class CohortResults:
    """One cohort's explained patients and their saved SHAP values."""

    X: np.ndarray
    shap_values: np.ndarray


@dataclass(frozen=True)
class SavedResult:
    path: Path
    metadata: dict[str, Any]
    feature_names: tuple[str, ...]
    cohorts: dict[str, CohortResults]


@dataclass(frozen=True)
class ResultSet:
    units: dict[tuple[str, str, int], SavedResult]
    seeds: tuple[int, ...]
    feature_names: tuple[str, ...]
    settings: dict[str, Any]

    @property
    def models(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(model for _, model, _ in self.units))

    def reference(self, cohort: str) -> CohortResults:
        """The explained patients; identical in all units, so any unit can supply them."""
        return self.units[(SOURCES[0], self.models[0], self.seeds[0])].cohorts[cohort]

    def effects(self, train_source: str, model: str, cohort: str, index: int) -> np.ndarray:
        """(repeats, patients) SHAP values of one feature, in saved patient order."""
        return np.stack(
            [self.units[(train_source, model, seed)].cohorts[cohort].shap_values[:, index] for seed in self.seeds]
        )


def _load_result(path: Path) -> SavedResult:
    """Read one compute.py result, keeping only the arrays the figures use."""
    with np.load(path, allow_pickle=False) as archive:
        feature_names = tuple(archive["feature_names"].tolist())
        cohorts = {}
        for cohort in SOURCES:
            X = archive[f"{cohort}_X"]
            shap_values = archive[f"{cohort}_shap_values"]
            if shap_values.shape != X.shape or not np.isfinite(shap_values).all():
                raise ValueError(f"{path}: {cohort} SHAP values must be finite and match the explained features")
            cohorts[cohort] = CohortResults(X=X, shap_values=shap_values)
        return SavedResult(
            path=path,
            metadata=json.loads(archive["metadata_json"].item()),
            feature_names=feature_names,
            cohorts=cohorts,
        )


def load_results(results: str | Path, models: tuple[str, ...] = DEFAULT_MODELS) -> ResultSet:
    """Load saved repeats with complete model/source coverage.

    Planned but not yet saved repeats do not block plotting; one saved repeat is
    enough. Every saved seed must cover all models and both training sources, and
    all units must describe the same run. Never drop results to force alignment.
    """
    if not models or set(models) - set(MODELS):
        raise ValueError(f"Select at least one supported model from {MODELS}")
    root = Path(results)
    units = {}
    for source in SOURCES:
        for model in models:
            for path in sorted((root / source / model).glob("seed_*.npz")):
                match = _RESULT_FILE.fullmatch(path.name)
                if match is None:  # Ignore seed_<seed>.previous.npz backups from --recompute.
                    continue
                seed = int(match.group(1))
                unit = _load_result(path)
                identity = (unit.metadata["train_source"], unit.metadata["model"], unit.metadata["seed"])
                if identity != (source, model, seed):
                    raise ValueError(f"Result identity disagrees with its path: {path}")
                units[(source, model, seed)] = unit
    if not units:
        raise ValueError(f"No saved SHAP results under {root}")
    seeds = tuple(sorted({seed for _, _, seed in units}))
    missing = [
        root / source / model / f"seed_{seed}.npz"
        for source in SOURCES
        for model in models
        for seed in seeds
        if (source, model, seed) not in units
    ]
    if missing:
        raise ValueError("Missing result combinations; no figures generated:\n" + "\n".join(map(str, missing)))
    _validate_units(units, seeds)
    reference = next(iter(units.values()))
    return ResultSet(units, seeds, reference.feature_names, _common_settings(reference))


def _validate_units(units: dict[tuple[str, str, int], SavedResult], seeds: tuple[int, ...]) -> None:
    """Reject mixed experiments: all units must share run, model, and training inputs.

    The data hash covers every explained feature value and row, so equal hashes
    also guarantee the strict patient alignment the figures rely on.
    """
    reference = next(iter(units.values()))
    common = _common_settings(reference)
    for (source, model, seed), unit in units.items():
        where = f"{unit.path} versus {reference.path}"
        if seed not in unit.metadata["settings"]["repeat_seeds"]:
            raise ValueError(f"Saved repeat seed {seed} is not declared in settings.repeat_seeds: {unit.path}")
        if _common_settings(unit) != common:
            raise ValueError(f"Common settings mismatch: {where}")
        for key in ("data_hash", "dependencies"):
            if unit.metadata[key] != reference.metadata[key]:
                raise ValueError(f"{key} mismatch: {where}")
        model_unit = units[(SOURCES[0], model, seeds[0])]
        for key in ("source_hashes", "hpo"):
            if unit.metadata.get(key) != model_unit.metadata.get(key):
                raise ValueError(f"Model {key} mismatch: {where}")
        # HPO may select different winners per source/repeat; compare the input recipe.
        if _base_params(unit) != _base_params(model_unit):
            raise ValueError(f"Model base parameters mismatch: {where}")
        training_unit = units[(source, reference.metadata["model"], seed)]
        if unit.metadata["training_hash"] != training_unit.metadata["training_hash"]:
            raise ValueError(f"Training sample mismatch: {where}")


def _common_settings(unit: SavedResult) -> dict[str, Any]:
    """Settings shared by every repeat; the planned repeat list may grow."""
    return {key: value for key, value in unit.metadata["settings"].items() if key != "repeat_seeds"}


def _base_params(unit: SavedResult) -> dict:
    """Input kwargs; tuned winners may legitimately differ between repeats."""
    return unit.metadata.get("base_params", unit.metadata["params"])


def feature_label(name: str) -> str:
    # Quick's clinical interpretation and Urea's underlying measurement convention
    # are not inferred from a legacy alias. Do not guess units for any feature.
    if name.split("+", 1)[0] in ("Quick", "Urea"):
        return name.split("+", 1)[0]
    label = FEATURE_ALIASES.get(name, name.removesuffix("+100%mean"))
    return SMALL_FEATURE_NAMES.get(label, label)


@dataclass(frozen=True)
class FeatureBins:
    centers: np.ndarray
    edges: np.ndarray | None
    discrete: bool

    def membership(self, x: np.ndarray) -> np.ndarray:
        if self.discrete or self.edges is None:
            return np.searchsorted(self.centers, x)
        return np.clip(np.searchsorted(self.edges, x, side="right") - 1, 0, self.centers.size - 1)


def _feature_bins(name: str, pooled: np.ndarray, settings: PlotSettings) -> FeatureBins:
    """Histogram bins from pooled observed values, shared by all four panels."""
    observed = pooled[np.isfinite(pooled)]
    discrete = name in settings.discrete_features
    unique = np.unique(observed)
    if discrete or unique.size <= 1:
        return FeatureBins(unique, None, discrete)
    # Near-constant features can produce duplicate floating-point edges.
    edges = np.unique(np.linspace(observed.min(), observed.max(), settings.n_bins + 1))
    return FeatureBins(edges[:-1] + np.diff(edges) / 2, edges, False)


def _limits(values: np.ndarray) -> tuple[float, float]:
    low, high = float(values.min()), float(values.max())
    pad = (high - low) * 0.06 if high > low else max(abs(low) * 0.06, 0.05)
    return low - pad, high + pad


def _scatter(ax: Axes, x: np.ndarray, y: np.ndarray, color: str, settings: PlotSettings) -> None:
    """Draw patient attributions; SHAP values and encodings are unchanged."""
    ax.scatter(
        x,
        y,
        s=settings.scatter_size,
        alpha=settings.scatter_alpha,
        color=color,
        marker="o",
        linewidths=0,
        zorder=2,
    )


def _draw_ebm(ax: Axes, x: np.ndarray, repeats: np.ndarray, discrete: bool, settings: PlotSettings) -> None:
    """Connect cached main effects, not binned SHAP means or guessed EBM cuts."""
    observed = np.isfinite(x)
    values, first = np.unique(x[observed], return_index=True)
    if not values.size:
        return
    # With interactions disabled, equal feature values have identical effects.
    # Both explanation cohorts are pooled so the same fitted EBM has the same line
    # in both columns; no cohort-dependent averaging changes its main-effect shape.
    curves = repeats[:, observed][:, first]
    mean, lower, upper = curves.mean(axis=0), curves.min(axis=0), curves.max(axis=0)
    color = MODEL_COLORS["ebm"]
    if discrete or values.size == 1:
        half_width = 0.16 if discrete else max(abs(values[0]) * 0.02, 0.02)
        ax.hlines(mean, values - half_width, values + half_width, color=color, linewidth=settings.line_width, zorder=4)
        if settings.show_bands and repeats.shape[0] > 1:
            ax.vlines(values, lower, upper, color=color, alpha=0.5, linewidth=2, zorder=3)
    else:
        ax.plot(values, mean, color=color, linewidth=settings.line_width, marker=None, zorder=4)
        if settings.show_bands and repeats.shape[0] > 1:
            ax.fill_between(values, lower, upper, color=color, alpha=settings.band_alpha, linewidth=0, zorder=1)


def _draw_missing(ax: Axes, x: np.ndarray, effects: dict[str, np.ndarray], settings: PlotSettings) -> None:
    """Patient effects where the feature is missing, on the same SHAP scale."""
    missing = ~np.isfinite(x)
    ax.axhline(0, color=BASELINE, linewidth=0.6, zorder=0)
    if not missing.any():
        ax.text(0.5, 0.5, "None", transform=ax.transAxes, ha="center", va="center", fontsize=6)
    for position, model in enumerate(effects):
        color = MODEL_COLORS[model]
        if model == "ebm":
            if not missing.any():
                continue
            # One centered main effect per repeat for the missing-value group.
            repeat_means = effects[model][:, missing].mean(axis=1)
            ax.hlines(
                repeat_means.mean(),
                position - 0.22,
                position + 0.22,
                color=color,
                linewidth=settings.line_width,
                zorder=4,
            )
            if settings.show_bands and effects[model].shape[0] > 1:
                ax.vlines(
                    position,
                    repeat_means.min(),
                    repeat_means.max(),
                    color=color,
                    alpha=0.5,
                    linewidth=2.0,
                    zorder=3,
                )
        elif missing.any():
            # Deterministic categorical spread only; SHAP values are unchanged.
            patient_means = effects[model].mean(axis=0)[missing]
            spread = np.linspace(-0.20, 0.20, len(patient_means)) if len(patient_means) > 1 else np.zeros(1)
            _scatter(ax, position + spread, patient_means, color, settings)
    ax.set_xlim(-0.5, len(effects) - 0.5)
    ax.set_xticks([])
    ax.set_xlabel(f"Missing\nn={missing.sum()}", fontsize=6)
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.spines["left"].set_visible(False)
    ax.grid(False)


def _draw_histogram(ax: Axes, bins: FeatureBins, counts: np.ndarray) -> None:
    """Observed-value support for this cohort, with common pooled bin edges."""
    if bins.centers.size:
        if bins.edges is not None:
            widths = np.diff(bins.edges)
        else:
            widths = np.full(bins.centers.size, 0.5) if bins.discrete else np.maximum(np.abs(bins.centers) * 0.04, 0.04)
        ax.bar(bins.centers, counts, width=widths, color=BASELINE, alpha=0.55, linewidth=0)
    ax.set_ylabel("Count", fontsize=6)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=2, integer=True))
    ax.tick_params(labelsize=6)
    ax.grid(False)


def _annotate_cohort(ax: Axes, observed_values: np.ndarray) -> None:
    if not observed_values.size:
        ax.text(
            0.5,
            0.5,
            "All values missing",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=7,
            backgroundcolor="white",
        )
    elif np.unique(observed_values).size == 1:
        ax.text(
            0.03,
            0.96,
            "Constant observed value",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6,
            backgroundcolor="white",
        )


def _draw_cohort(
    ax: Axes,
    missing_ax: Axes,
    support_ax: Axes,
    x: np.ndarray,
    effects: dict[str, np.ndarray],
    bins: FeatureBins,
    settings: PlotSettings,
) -> int:
    """Draw patient effects, the missing-value strip, and the support histogram."""
    observed = np.isfinite(x)
    counts = np.zeros(bins.centers.size, dtype=int)
    ax.axhline(0, color=BASELINE, linewidth=0.6, zorder=0)
    if observed.any():
        counts = np.bincount(bins.membership(x[observed]), minlength=bins.centers.size)
        for model in effects:
            if model == "ebm":
                continue  # EBM is one sampled main-effect line per fitted model.
            _scatter(ax, x[observed], effects[model].mean(axis=0)[observed], MODEL_COLORS[model], settings)
    _annotate_cohort(ax, x[observed])
    _draw_missing(missing_ax, x, effects, settings)
    _draw_histogram(support_ax, bins, counts)
    ax.tick_params(axis="x", labelbottom=False)
    return int(counts.max()) if counts.size else 0


def _draw_legends(fig, data: ResultSet, settings: PlotSettings) -> None:
    """Describe encodings only; no generated caption, title, or claim."""
    model_handles = [
        Line2D(
            [],
            [],
            color=MODEL_COLORS[model],
            marker=None if model == "ebm" else "o",
            linestyle="-" if model == "ebm" else "none",
            markersize=3,
            label=MODEL_STYLES[model].label,
        )
        for model in data.models
    ]
    fig.legend(handles=model_handles, loc="lower center", bbox_to_anchor=(0.5, 0.075), ncol=len(data.models))
    repeat_key = "single repeat" if len(data.seeds) == 1 else f"{len(data.seeds)} repeats"
    encoding_handles = [
        Line2D(
            [],
            [],
            color=BASELINE,
            marker="o",
            linestyle="none",
            markersize=3,
            label="Patient SHAP" if len(data.seeds) == 1 else "Patient mean",
        ),
    ]
    if "ebm" in data.models:
        encoding_handles.append(
            Line2D([], [], color=MODEL_COLORS["ebm"], label=f"EBM sampled main effect ({repeat_key})")
        )
    if "ebm" in data.models and settings.show_bands and len(data.seeds) > 1:
        encoding_handles.append(
            Patch(facecolor=BASELINE, alpha=settings.band_alpha, label="EBM repeat min–max (not CI)")
        )
    fig.legend(
        handles=encoding_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=len(encoding_handles),
        fontsize=5.5,
    )


def plot_feature(data: ResultSet, index: int, settings: PlotSettings = PLOT_SETTINGS) -> Any:
    """Build one 2x2 figure; the caller owns saving/closing it."""
    x_by_cohort = {cohort: data.reference(cohort).X[:, index] for cohort in SOURCES}
    pooled_x = np.concatenate(list(x_by_cohort.values()))
    bins = _feature_bins(data.feature_names[index], pooled_x, settings)
    fig = plt.figure(figsize=(settings.width, settings.height))
    outer = fig.add_gridspec(2, 2, left=0.115, right=0.985, bottom=0.19, top=0.925, wspace=0.27, hspace=0.72)
    main_axes: list[Axes] = []
    support_axes: list[Axes] = []
    all_effects = []
    max_count = 0
    shap_label = "SHAP value (log-odds)" if data.settings["output"] == "binary_mortality_log_odds" else "SHAP value"
    for row, train_source in enumerate(SOURCES):
        # EBM main effects depend only on the fitted model, so both explanation
        # cohorts supply observed values and both columns show the same line.
        ebm_repeats = (
            np.concatenate([data.effects(train_source, "ebm", cohort, index) for cohort in SOURCES], axis=1)
            if "ebm" in data.models
            else None
        )
        for column, cohort in enumerate(SOURCES):
            inner = outer[row, column].subgridspec(
                2, 2, width_ratios=(5, 1.05), height_ratios=(4.5, 1), wspace=0.12, hspace=0.12
            )
            share = {"sharex": main_axes[0] if main_axes else None, "sharey": main_axes[0] if main_axes else None}
            ax = fig.add_subplot(inner[0, 0], **share)
            support_ax = fig.add_subplot(inner[1, 0], sharex=ax, sharey=support_axes[0] if support_axes else None)
            missing_ax = fig.add_subplot(inner[0, 1], sharey=ax)
            main_axes.append(ax)
            support_axes.append(support_ax)
            effects = {model: data.effects(train_source, model, cohort, index) for model in data.models}
            all_effects.extend(effects.values())
            coordinate = row * 2 + column
            max_count = max(
                max_count, _draw_cohort(ax, missing_ax, support_ax, x_by_cohort[cohort], effects, bins, settings)
            )
            if ebm_repeats is not None:
                _draw_ebm(ax, pooled_x, ebm_repeats, bins.discrete, settings)
            train_name, test_name = DATASET_NAMES[train_source], DATASET_NAMES[cohort]
            ax.text(
                0,
                1.08,
                f"({chr(97 + coordinate)})  Train: {train_name}  /  Test: {test_name}",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=6.5,
            )
            support_ax.set_xlabel(feature_label(data.feature_names[index]), fontsize=7)
            if column == 0:
                ax.set_ylabel(shap_label)
            else:
                ax.tick_params(axis="y", labelleft=False)
                support_ax.tick_params(axis="y", labelleft=False)
    y_values = np.concatenate([values.ravel() for values in all_effects] + [np.array([0.0])])
    main_axes[0].set_ylim(*_limits(y_values))
    observed = pooled_x[np.isfinite(pooled_x)]
    if bins.centers.size:
        if bins.discrete:
            main_axes[0].set_xticks(bins.centers)
            main_axes[0].set_xlim(float(bins.centers.min()) - 0.5, float(bins.centers.max()) + 0.5)
        else:
            main_axes[0].xaxis.set_major_locator(MaxNLocator(nbins=4))
            main_axes[0].set_xlim(*_limits(observed))
    else:
        main_axes[0].set_xlim(0, 1)
        main_axes[0].set_xticks([])
    support_axes[0].set_ylim(0, max(1, max_count) * 1.2)
    _draw_legends(fig, data, settings)
    return fig


def _figure_stem(name: str) -> str:
    """Stable, collision-free file stem for one original feature name."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "feature"
    return f"dependence_{safe}_{hashlib.sha256(name.encode('utf-8')).hexdigest()[:8]}"


def make_figures(
    results: str | Path = DEFAULT_RESULTS,
    settings: PlotSettings = PLOT_SETTINGS,
    models: tuple[str, ...] = DEFAULT_MODELS,
) -> list[Path]:
    """Validate every input before writing one PDF per selected original feature."""
    if settings.n_bins < 1:
        raise ValueError("n_bins must be positive")
    data = load_results(results, models)
    selected = settings.selected_features or data.feature_names
    unknown = sorted(set(selected) - set(data.feature_names))
    if unknown:
        raise ValueError(f"Unknown original feature names: {unknown}; use exact saved feature_names")
    use_style()
    output = Path(results) / "figures"
    paths = []
    for name in selected:
        fig = plot_feature(data, data.feature_names.index(name), settings)
        paths.extend(Path(path) for path in save(fig, str(output / _figure_stem(name)), formats=("pdf",)))
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--results", type=Path, default=DEFAULT_RESULTS, help="Root of saved train_source/model/seed_<seed>.npz results"
    )
    parser.add_argument(
        "--no-bands", action="store_true", help="Hide repeat min/max ranges (never confidence intervals)"
    )
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(DEFAULT_MODELS))
    parser.add_argument("--features", nargs="+", help="Exact original feature names, e.g. Age Sex")
    args = parser.parse_args(argv)
    settings = replace(
        PLOT_SETTINGS,
        show_bands=PLOT_SETTINGS.show_bands and not args.no_bands,
        selected_features=tuple(args.features) if args.features is not None else PLOT_SETTINGS.selected_features,
    )
    try:
        paths = make_figures(args.results, settings, tuple(args.models))
    except (ValueError, OSError) as exc:
        print(f"SHAP plotting failed: {exc}", file=sys.stderr)
        return 1
    print(f"Saved {len(paths)} feature figures (PDF) to {args.results / 'figures'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
