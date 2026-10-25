"""Plot saved SHAP results only; never import fitting or explanation code.

Run from the repository root::

    uv run python -m src.interpretability.plot
    uv run python -m src.interpretability.plot --no-bands --features Age Sex

XGBoost/TabPFN patients are drawn once, using mean SHAP across saved repeats.
EBM alone has a line: its saved centered main effects, without binning/smoothing.
The line connects observed values; native EBM bin boundaries are not cached.
EBM bands are repeat min/max ranges, NOT confidence intervals. Histogram bins use
pooled observed values from the two explanation cohorts with identical edges.
Missing original feature values have a separate strip on the same SHAP scale.
Only PDF outputs are written under the supplied results directory / figures.
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

# Support both `python -m src.interpretability.plot` and direct script execution.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# These modules contain display defaults only, with no training/model imports.
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
DEFAULT_RESULTS = Path("run_results/interpretability/mortality")
SOURCES = ("tudd", "mimic")
MODELS = ("ebm", "xgboost", "tabpfn-3.5")
# Interpretability-only encodings; do not change other evaluation figures.
MODEL_COLORS = {"ebm": "#000000", "xgboost": PALETTE["orange"], "tabpfn-3.5": MODEL_STYLES["tabpfn-3.5"].color}


@dataclass(frozen=True)
class CohortResults:
    X: np.ndarray
    row_ids: np.ndarray
    shap_values: np.ndarray
    test_row_ids: np.ndarray
    test_y: np.ndarray


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

    def reference(self, cohort: str) -> CohortResults:
        return self.units[(SOURCES[0], MODELS[0], self.seeds[0])].cohorts[cohort]

    def effects(self, train_source: str, model: str, cohort: str, index: int) -> np.ndarray:
        """Return repeats x patients, after strict global patient alignment."""
        return np.stack(
            [self.units[(train_source, model, seed)].cohorts[cohort].shap_values[:, index] for seed in self.seeds]
        )


def _load_result(path: Path) -> SavedResult:
    """Read the schema written by compute.py; retain only arrays used in plots."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(archive["metadata_json"].item())
            names = tuple(archive["feature_names"].tolist())
            cohorts = {}
            for cohort in SOURCES:
                values = CohortResults(
                    X=archive[f"{cohort}_X"],
                    row_ids=archive[f"{cohort}_row_ids"],
                    shap_values=archive[f"{cohort}_shap_values"],
                    test_row_ids=archive[f"{cohort}_test_row_ids"],
                    test_y=archive[f"{cohort}_test_y"],
                )
                if values.X.ndim != 2 or not len(values.X) or values.X.shape[1] != len(names):
                    raise ValueError(f"{cohort}: expected nonempty patients × features")
                if values.shap_values.shape != values.X.shape or not np.isfinite(values.shap_values).all():
                    raise ValueError(f"{cohort}: SHAP values must be finite and match X's shape")
                if np.isinf(values.X).any():
                    raise ValueError(f"{cohort}: X contains infinities; only NaN denotes missing values")
                cohorts[cohort] = values
        return SavedResult(path, metadata, names, cohorts)
    except (KeyError, ValueError, OSError) as exc:
        raise ValueError(f"Invalid result {path}: {exc}") from exc


def load_results(results: str | Path) -> ResultSet:
    """Plot saved repeats with complete model/source coverage and aligned patients.

    Planned but not yet saved repeats do not block plotting. Every seed present
    must cover all models/sources; never drop results or patients to force alignment.
    """
    root = Path(results)
    units = {}
    for source in SOURCES:
        for model in MODELS:
            for path in sorted((root / source / model).glob("seed_*.npz")):
                seed_text = path.stem.removeprefix("seed_")
                if not seed_text.isdecimal():  # Ignore incomplete seed_<n>.tmp.npz files.
                    continue
                seed = int(seed_text)
                unit = _load_result(path)
                if (unit.metadata["train_source"], unit.metadata["model"], unit.metadata["seed"]) != (
                    source,
                    model,
                    seed,
                ):
                    raise ValueError(f"Result identity disagrees with its path: {path}")
                units[(source, model, seed)] = unit
    if not units:
        missing = [root / source / model / "seed_<seed>.npz" for source in SOURCES for model in MODELS]
        raise ValueError("No saved SHAP results. Missing combinations:\n" + "\n".join(map(str, missing)))
    seeds = tuple(sorted({seed for _, _, seed in units}))
    if any(seed not in unit.metadata["settings"]["repeat_seeds"] for (_, _, seed), unit in units.items()):
        raise ValueError("Saved repeat seeds disagree with declared settings.repeat_seeds")
    missing = [
        root / source / model / f"seed_{seed}.npz"
        for source in SOURCES
        for model in MODELS
        for seed in seeds
        if (source, model, seed) not in units
    ]
    if missing:
        raise ValueError("Missing result combinations; no figures generated:\n" + "\n".join(map(str, missing)))
    reference = units[(SOURCES[0], MODELS[0], seeds[0])]
    common_settings = {key: value for key, value in reference.metadata["settings"].items() if key != "repeat_seeds"}
    for (source, model, seed), unit in units.items():
        current_settings = {key: value for key, value in unit.metadata["settings"].items() if key != "repeat_seeds"}
        if current_settings != common_settings:
            raise ValueError(f"Common settings mismatch: {unit.path} versus {reference.path}")
        for key in ("data_hash", "dependencies"):
            if unit.metadata[key] != reference.metadata[key]:
                raise ValueError(f"{key} mismatch: {unit.path} versus {reference.path}")
        model_reference = units[(SOURCES[0], model, seeds[0])]
        for key in ("source_hashes", "hpo"):
            if unit.metadata.get(key) != model_reference.metadata.get(key):
                raise ValueError(f"Model {key} mismatch: {unit.path} versus {model_reference.path}")
        # HPO may select different winners per source/repeat; compare the input
        # recipe, not the fitted models' selected hyperparameters.
        base_params = unit.metadata.get("base_params", unit.metadata["params"])
        reference_params = model_reference.metadata.get("base_params", model_reference.metadata["params"])
        if base_params != reference_params:
            raise ValueError(f"Model base parameters mismatch: {unit.path} versus {model_reference.path}")
        training_reference = units[(source, MODELS[0], seed)]
        if unit.metadata["training_hash"] != training_reference.metadata["training_hash"]:
            raise ValueError(f"Training sample mismatch: {unit.path} versus {training_reference.path}")
        if unit.feature_names != reference.feature_names:
            raise ValueError(f"Feature names/order mismatch: {unit.path} versus {reference.path}")
        for cohort in SOURCES:
            current, baseline = unit.cohorts[cohort], reference.cohorts[cohort]
            for name in ("row_ids", "X", "test_row_ids", "test_y"):
                if not np.array_equal(getattr(current, name), getattr(baseline, name), equal_nan=True):
                    raise ValueError(f"{cohort} {name} mismatch: {unit.path} versus {reference.path}")
    return ResultSet(units, seeds, reference.feature_names)


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


def _feature_bins(data: ResultSet, index: int, settings: PlotSettings) -> FeatureBins:
    pooled = np.concatenate([data.reference(cohort).X[:, index] for cohort in SOURCES])
    observed = pooled[np.isfinite(pooled)]
    discrete = data.feature_names[index] in settings.discrete_features
    unique = np.unique(observed)
    if discrete or unique.size <= 1:
        return FeatureBins(unique, None, discrete)
    edges = np.linspace(observed.min(), observed.max(), settings.n_bins + 1)
    # Duplicated floating-point edges can occur for near-constant values.
    edges = np.unique(edges)
    return FeatureBins(edges[:-1] + np.diff(edges) / 2, edges, False)


def _draw_ebm(ax: Any, x: np.ndarray, runs: np.ndarray, discrete: bool, settings: PlotSettings) -> None:
    """Connect cached main effects, not binned SHAP means or guessed EBM cuts."""
    observed = np.isfinite(x)
    values, first = np.unique(x[observed], return_index=True)
    if not values.size:
        return
    # With interactions disabled, equal feature values have identical effects.
    # Pool both explanation cohorts so the same fitted EBM has the same line in
    # both columns; no cohort-dependent averaging changes its main-effect shape.
    curves = runs[:, observed][:, first]
    mean, lower, upper = curves.mean(axis=0), curves.min(axis=0), curves.max(axis=0)
    color = MODEL_COLORS["ebm"]
    if discrete or values.size == 1:
        half_width = 0.16 if discrete else max(abs(values[0]) * 0.02, 0.02)
        ax.hlines(mean, values - half_width, values + half_width, color=color, linewidth=settings.line_width, zorder=4)
        if settings.show_bands and runs.shape[0] > 1:
            ax.vlines(values, lower, upper, color=color, alpha=0.5, linewidth=2, zorder=3)
    else:
        ax.plot(values, mean, color=color, linewidth=settings.line_width, marker=None, zorder=4)
        if settings.show_bands and runs.shape[0] > 1:
            ax.fill_between(values, lower, upper, color=color, alpha=settings.band_alpha, linewidth=0, zorder=1)


def _limits(values: np.ndarray) -> tuple[float, float]:
    low, high = float(values.min()), float(values.max())
    pad = (high - low) * 0.06 if high > low else max(abs(low) * 0.06, 0.05)
    return low - pad, high + pad


def _draw_panel(
    ax: Any,
    missing_ax: Any,
    support_ax: Any,
    x: np.ndarray,
    effects: dict[str, np.ndarray],
    bins: FeatureBins,
    settings: PlotSettings,
) -> int:
    observed = np.isfinite(x)
    missing = ~observed
    counts = np.zeros(bins.centers.size, dtype=int)
    ax.axhline(0, color=BASELINE, linewidth=0.6, zorder=0)
    missing_ax.axhline(0, color=BASELINE, linewidth=0.6, zorder=0)
    if observed.any():
        counts = np.bincount(bins.membership(x[observed]), minlength=bins.centers.size)
    for model_index, model in enumerate(MODELS):
        color = MODEL_COLORS[model]
        runs = effects[model]
        patient_means = runs.mean(axis=0)
        if model != "ebm":
            ax.scatter(
                x[observed],
                patient_means[observed],
                s=settings.scatter_size,
                alpha=settings.scatter_alpha,
                color=color,
                marker="o",
                linewidths=0,
                zorder=2,
            )
        if missing.any():
            position = model_index
            if model == "ebm":
                repeat_means = runs[:, missing].mean(axis=1)
                missing_ax.hlines(
                    repeat_means.mean(),
                    position - 0.22,
                    position + 0.22,
                    color=color,
                    linewidth=settings.line_width,
                    zorder=4,
                )
                if settings.show_bands and runs.shape[0] > 1:
                    missing_ax.vlines(
                        position,
                        repeat_means.min(),
                        repeat_means.max(),
                        color=color,
                        alpha=0.5,
                        linewidth=2.0,
                        zorder=3,
                    )
            else:
                # Deterministic categorical spread only; SHAP values are unchanged.
                spread = np.linspace(-0.20, 0.20, missing.sum()) if missing.sum() > 1 else np.zeros(1)
                missing_ax.scatter(
                    position + spread,
                    patient_means[missing],
                    s=settings.scatter_size,
                    alpha=settings.scatter_alpha,
                    color=color,
                    marker="o",
                    linewidths=0,
                    zorder=2,
                )
    if not observed.any():
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
    elif np.unique(x[observed]).size == 1:
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
    if not missing.any():
        missing_ax.text(0.5, 0.5, "None", transform=missing_ax.transAxes, ha="center", va="center", fontsize=6)
    missing_ax.set_xlim(-0.5, 2.5)
    missing_ax.set_xticks([])
    missing_ax.set_xlabel(f"Missing\nn={missing.sum()}", fontsize=6)
    missing_ax.tick_params(axis="y", left=False, labelleft=False)
    missing_ax.spines["left"].set_visible(False)
    missing_ax.grid(False)
    if bins.centers.size:
        if bins.edges is not None:
            widths = np.diff(bins.edges)
        else:
            widths = np.full(bins.centers.size, 0.5) if bins.discrete else np.maximum(np.abs(bins.centers) * 0.04, 0.04)
        support_ax.bar(bins.centers, counts, width=widths, color=BASELINE, alpha=0.55, linewidth=0)
    support_ax.set_ylabel("Count", fontsize=6)
    support_ax.yaxis.set_major_locator(MaxNLocator(nbins=2, integer=True))
    support_ax.tick_params(labelsize=6)
    support_ax.grid(False)
    ax.tick_params(axis="x", labelbottom=False)
    return int(counts.max()) if counts.size else 0


def plot_feature(data: ResultSet, index: int, settings: PlotSettings = PLOT_SETTINGS) -> Any:
    """Build one 2x2 figure; caller owns saving/closing it."""
    bins = _feature_bins(data, index, settings)
    fig = plt.figure(figsize=(settings.width, settings.height))
    outer = fig.add_gridspec(2, 2, left=0.115, right=0.985, bottom=0.19, top=0.925, wspace=0.27, hspace=0.72)
    main_axes, support_axes = [], []
    max_count = 0
    all_effects = []
    output_scale = next(iter(data.units.values())).metadata["settings"].get("output")
    shap_label = "SHAP value (log-odds)" if output_scale == "binary_mortality_log_odds" else "SHAP value"
    ebm_x = np.concatenate([data.reference(cohort).X[:, index] for cohort in SOURCES])
    for row, train_source in enumerate(SOURCES):
        ebm_runs = np.concatenate([data.effects(train_source, "ebm", cohort, index) for cohort in SOURCES], axis=1)
        for column, cohort in enumerate(SOURCES):
            inner = outer[row, column].subgridspec(
                2, 2, width_ratios=(5, 1.05), height_ratios=(4.5, 1), wspace=0.12, hspace=0.12
            )
            ax = fig.add_subplot(
                inner[0, 0], sharex=main_axes[0] if main_axes else None, sharey=main_axes[0] if main_axes else None
            )
            support_ax = fig.add_subplot(inner[1, 0], sharex=ax, sharey=support_axes[0] if support_axes else None)
            missing_ax = fig.add_subplot(inner[0, 1], sharey=ax)
            main_axes.append(ax)
            support_axes.append(support_ax)
            x = data.reference(cohort).X[:, index]
            effects = {model: data.effects(train_source, model, cohort, index) for model in MODELS}
            all_effects.extend(effects.values())
            max_count = max(max_count, _draw_panel(ax, missing_ax, support_ax, x, effects, bins, settings))
            _draw_ebm(ax, ebm_x, ebm_runs, bins.discrete, settings)
            header = f"({chr(97 + row * 2 + column)})  Train: {DATASET_NAMES[train_source]}  /  Test: {DATASET_NAMES[cohort]}"
            ax.text(0, 1.08, header, transform=ax.transAxes, ha="left", va="bottom", fontsize=6.5)
            support_ax.set_xlabel(feature_label(data.feature_names[index]), fontsize=7)
            if column == 0:
                ax.set_ylabel(shap_label)
            else:
                ax.tick_params(axis="y", labelleft=False)
                support_ax.tick_params(axis="y", labelleft=False)
    y_values = np.concatenate([values.ravel() for values in all_effects] + [np.array([0.0])])
    main_axes[0].set_ylim(*_limits(y_values))
    if bins.centers.size:
        pooled = np.concatenate([data.reference(cohort).X[:, index] for cohort in SOURCES])
        observed = pooled[np.isfinite(pooled)]
        x_limits = _limits(observed)
        if bins.discrete:
            x_limits = (float(bins.centers.min()) - 0.5, float(bins.centers.max()) + 0.5)
            main_axes[0].set_xticks(bins.centers)
        else:
            main_axes[0].xaxis.set_major_locator(MaxNLocator(nbins=4))
        main_axes[0].set_xlim(*x_limits)
    else:
        main_axes[0].set_xlim(0, 1)
        main_axes[0].set_xticks([])
    support_axes[0].set_ylim(0, max(1, max_count) * 1.2)
    handles = [
        Line2D(
            [],
            [],
            color=MODEL_COLORS[model],
            marker=None if model == "ebm" else "o",
            linestyle="-" if model == "ebm" else "none",
            markersize=3,
            label=MODEL_STYLES[model].label,
        )
        for model in MODELS
    ]
    repeat_key = "single repeat" if len(data.seeds) == 1 else f"{len(data.seeds)} repeats"
    # Figure legends describe encodings; no generated caption, title or claim.
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.075), ncol=3)
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
        Line2D([], [], color=MODEL_COLORS["ebm"], label=f"EBM sampled main effect ({repeat_key})"),
    ]
    if settings.show_bands and len(data.seeds) > 1:
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
    return fig


def make_figures(results: str | Path = DEFAULT_RESULTS, settings: PlotSettings = PLOT_SETTINGS) -> list[Path]:
    """Validate every input before writing one PDF per original feature."""
    if settings.n_bins < 1:
        raise ValueError("n_bins must be positive")
    data = load_results(results)
    selected = settings.selected_features if settings.selected_features is not None else data.feature_names
    unknown = sorted(set(selected) - set(data.feature_names))
    if unknown:
        raise ValueError(f"Unknown original feature names: {unknown}; use exact saved feature_names")
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("selected_features must be nonempty and unique")
    use_style()
    output = Path(results) / "figures"
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in selected:
        fig = plot_feature(data, data.feature_names.index(name), settings)
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "feature"
        # Stable suffix prevents collisions between different original names.
        digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
        paths.extend(Path(path) for path in save(fig, str(output / f"dependence_{safe}_{digest}"), formats=("pdf",)))
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--results", type=Path, default=DEFAULT_RESULTS, help="Root of saved train_source/model/seed_<seed>.npz results"
    )
    parser.add_argument(
        "--no-bands", action="store_true", help="Hide repeat min/max ranges (never confidence intervals)"
    )
    parser.add_argument("--features", nargs="+", help="Exact original feature names, e.g. Age Sex")
    args = parser.parse_args(argv)
    settings = replace(
        PLOT_SETTINGS,
        show_bands=PLOT_SETTINGS.show_bands and not args.no_bands,
        selected_features=tuple(args.features) if args.features is not None else PLOT_SETTINGS.selected_features,
    )
    try:
        paths = make_figures(args.results, settings)
    except (ValueError, OSError) as exc:
        print(f"SHAP plotting failed: {exc}", file=sys.stderr)
        return 1
    print(f"Saved {len(paths)} feature figures (PDF) to {args.results / 'figures'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
