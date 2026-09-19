"""Feature-distribution figures for the filtered MIMIC-IV and EUH cohorts.

Edit ``DATA`` to choose datasets/features and ``VISUAL`` for presentation, then
run:

    uv run python -m src.plotting.feature_distributions
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import config
from src.plotting.defaults import FEATURE_ALIASES, set_plot_style
from src.plotting.scientific_figstyle import COLUMN, figure, save
from src.plotting.utils.distributions import draw_feature_distribution

# ---------------------------------------------------------------------------
# EDIT THESE SETTINGS
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DataSettings:
    """Input cohorts, feature selection, and output location."""

    dataset_files: tuple[tuple[str, Path], ...] = (
        ("mimic", config.dir_data / "filtered" / "mimic4_mean_100_full.csv"),
        ("tudd", config.dir_data / "filtered" / "tudd_mean_100_full.csv"),
    )
    features: tuple[str, ...] | None = None
    exclude_features: tuple[str, ...] = ("mortality", "LOS", "hours_to_readmit")
    output_dir: Path = config.dir_plots / "feature_distributions"


@dataclass(frozen=True)
class VisualSettings:
    """All locally editable presentation choices for these figures."""

    bins: int = 50
    kde: bool = True
    alpha: float = 0.4
    figure_width: float = COLUMN
    figure_height_ratio: float = 0.76
    density_axis_label: str = "Density"
    show_density_ticks: bool = False
    legend_title: str = "Dataset"
    output_formats: tuple[str, ...] = ("pdf",)


DATA = DataSettings()
VISUAL = VisualSettings()


def make_figure(
    frames: dict[str, pd.DataFrame],
    feature: str,
    visual: VisualSettings = VISUAL,
):
    """Draw one feature distribution figure."""
    set_plot_style()
    fig, ax = figure(width=visual.figure_width, ratio=visual.figure_height_ratio)
    draw_feature_distribution(
        ax,
        frames,
        feature,
        bins=visual.bins,
        kde=visual.kde,
        alpha=visual.alpha,
    )
    ax.set_xlabel(FEATURE_ALIASES.get(feature, feature))
    ax.set_ylabel(visual.density_axis_label)
    if not visual.show_density_ticks:
        ax.set_yticks([])
    ax.legend(title=visual.legend_title)
    return fig


def main() -> None:
    frames = {dataset: pd.read_csv(path) for dataset, path in DATA.dataset_files}
    if DATA.features is None:
        common = set.intersection(*(set(frame.columns) for frame in frames.values()))
        features = tuple(column for column in next(iter(frames.values())).columns if column in common)
    else:
        features = DATA.features
    features = tuple(feature for feature in features if feature not in DATA.exclude_features)
    if not features:
        raise ValueError("No features remain after applying the feature selection")

    DATA.output_dir.mkdir(parents=True, exist_ok=True)
    for feature in features:
        stem = DATA.output_dir / _filename(feature)
        outputs = save(make_figure(frames, feature), str(stem), formats=VISUAL.output_formats)
        for output in outputs:
            print("figure: " + str(Path(output).relative_to(config.dir_root)))


def _filename(feature: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", FEATURE_ALIASES.get(feature, feature)).strip("_").lower()
    return slug or "feature"


if __name__ == "__main__":
    main()
