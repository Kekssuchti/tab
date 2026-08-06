"""Reusable density plots for comparing numeric feature distributions."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from numbers import Integral

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from src.plotting.defaults import DATASET_COLORS, dataset_label, ordered_datasets, set_plot_style

__all__ = ["plot_feature_distribution", "plot_feature_distributions"]

type _BinEdges = Sequence[float] | np.ndarray
type _BinSpec = int | str | _BinEdges
type _Bins = _BinSpec | Mapping[str, _BinSpec]

_HISTOGRAM_RESERVED_KWARGS = frozenset(
    {
        "ax",
        "bins",
        "binrange",
        "binwidth",
        "color",
        "data",
        "discrete",
        "hue",
        "kde",
        "kde_kws",
        "label",
        "line_kws",
        "stat",
        "weights",
        "x",
        "y",
    }
)
_KDE_RESERVED_KWARGS = frozenset({"ax", "color", "data", "hue", "label", "weights", "x", "y"})


def plot_feature_distribution(
    feature: str,
    *,
    mimic: pd.DataFrame | None = None,
    tudd: pd.DataFrame | None = None,
    bins: _Bins = "auto",
    kde: bool = True,
    figsize: tuple[float, float] = (8.0, 5.0),
    hist_kwargs: Mapping[str, object] | None = None,
    kde_kwargs: Mapping[str, object] | None = None,
) -> Figure:
    """Return an overlaid density histogram for one numeric feature.

    NA and infinite observations are discarded without changing the input
    frames. A supplied dataset with no finite observations is omitted; an
    error is raised if neither dataset has observations. Integer and NumPy
    strategy bins are calculated from the combined finite values, so compared
    datasets always share edges. Explicit edges must be finite and strictly
    increasing, and a bins mapping defaults missing feature keys to ``"auto"``.

    KDE curves are omitted for datasets with fewer than two observations or
    zero variance. ``hist_kwargs`` and ``kde_kwargs`` may customize ordinary
    seaborn controls, but controlled data, bins, axes, colors, and labels
    cannot be overridden. The returned figure is neither shown nor saved.
    """
    datasets = _validate_datasets(mimic=mimic, tudd=tudd)
    feature_names = _normalize_feature_collection((feature,), argument="feature")
    _validate_selected_features(feature_names, datasets)
    values = _finite_values(feature, datasets)
    edges = _bin_edges(feature, bins, values)
    histogram_options = _plot_options(
        hist_kwargs,
        argument="hist_kwargs",
        reserved=_HISTOGRAM_RESERVED_KWARGS,
        defaults={"alpha": 0.4},
    )
    kde_options = _plot_options(
        kde_kwargs,
        argument="kde_kwargs",
        reserved=_KDE_RESERVED_KWARGS,
        defaults={"linewidth": 1.6},
    )

    set_plot_style()
    return _draw_distribution(feature, values, edges, kde, figsize, histogram_options, kde_options)


def plot_feature_distributions(
    *,
    mimic: pd.DataFrame | None = None,
    tudd: pd.DataFrame | None = None,
    include_features: Sequence[str] | None = None,
    exclude_features: Collection[str] | None = None,
    bins: _Bins = "auto",
    kde: bool = True,
    figsize: tuple[float, float] = (8.0, 5.0),
    hist_kwargs: Mapping[str, object] | None = None,
    kde_kwargs: Mapping[str, object] | None = None,
) -> dict[str, Figure]:
    """Return one distribution figure per selected numeric feature.

    By default, all numeric columns from one frame are selected; with two
    frames, only their common numeric columns are selected. Explicit include
    order is retained, then exclusions are applied. Strings are not accepted
    as feature collections, and explicit features must exist and be numeric in
    every supplied frame. A bins mapping configures features individually and
    uses ``"auto"`` for missing keys.

    Each figure applies the same finite-value filtering, joint binning, KDE
    safeguards, and controlled-keyword rules as
    :func:`plot_feature_distribution`. Figures are returned in selection order
    and are neither shown nor saved.
    """
    datasets = _validate_datasets(mimic=mimic, tudd=tudd)
    features = _select_features(datasets, include_features, exclude_features)
    histogram_options = _plot_options(
        hist_kwargs,
        argument="hist_kwargs",
        reserved=_HISTOGRAM_RESERVED_KWARGS,
        defaults={"alpha": 0.4},
    )
    kde_options = _plot_options(
        kde_kwargs,
        argument="kde_kwargs",
        reserved=_KDE_RESERVED_KWARGS,
        defaults={"linewidth": 1.6},
    )

    prepared: dict[str, tuple[dict[str, np.ndarray], np.ndarray]] = {}
    for feature in features:
        values = _finite_values(feature, datasets)
        prepared[feature] = values, _bin_edges(feature, bins, values)

    set_plot_style()
    return {
        feature: _draw_distribution(
            feature,
            values,
            edges,
            kde,
            figsize,
            histogram_options,
            kde_options,
        )
        for feature, (values, edges) in prepared.items()
    }


def _validate_datasets(
    *,
    mimic: pd.DataFrame | None,
    tudd: pd.DataFrame | None,
) -> dict[str, pd.DataFrame]:
    supplied = {"mimic": mimic, "tudd": tudd}
    if not any(frame is not None for frame in supplied.values()):
        raise ValueError("At least one of mimic or tudd must be provided")

    datasets: dict[str, pd.DataFrame] = {}
    for name in ordered_datasets(list(supplied)):
        frame = supplied[name]
        if frame is None:
            continue
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{name} must be a pandas DataFrame or None")
        datasets[name] = frame
    return datasets


def _select_features(
    datasets: Mapping[str, pd.DataFrame],
    include_features: Sequence[str] | None,
    exclude_features: Collection[str] | None,
) -> list[str]:
    if include_features is None:
        first_frame = next(iter(datasets.values()))
        features = [
            column
            for column in dict.fromkeys(first_frame.columns)
            if isinstance(column, str) and all(_is_numeric_column(frame, column) for frame in datasets.values())
        ]
    else:
        features = _normalize_feature_collection(include_features, argument="include_features")
        duplicates = _duplicates(features)
        if duplicates:
            names = ", ".join(repr(name) for name in duplicates)
            raise ValueError(f"include_features contains duplicate feature names: {names}")

    _validate_selected_features(features, datasets)
    excluded = set(
        _normalize_feature_collection(exclude_features, argument="exclude_features")
        if exclude_features is not None
        else ()
    )
    features = [feature for feature in features if feature not in excluded]
    if not features:
        raise ValueError("No numeric features remain after applying feature filters")
    return features


def _normalize_feature_collection(features: Collection[object], *, argument: str) -> list[str]:
    if isinstance(features, (str, bytes)):
        raise TypeError(f"{argument} must be a collection of feature names, not a string")
    try:
        names = list(features)
    except TypeError:
        raise TypeError(f"{argument} must be a collection of feature names") from None
    if not all(isinstance(name, str) for name in names):
        raise TypeError(f"Every entry in {argument} must be a string")
    return names


def _duplicates(features: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for feature in features:
        if feature in seen and feature not in duplicates:
            duplicates.append(feature)
        seen.add(feature)
    return duplicates


def _validate_selected_features(features: Sequence[str], datasets: Mapping[str, pd.DataFrame]) -> None:
    for feature in features:
        for dataset, frame in datasets.items():
            if feature not in frame.columns:
                raise ValueError(f"Feature {feature!r} is missing from {dataset_label(dataset)}")
            column = frame[feature]
            if not isinstance(column, pd.Series):
                raise ValueError(f"Feature {feature!r} must identify a unique column in {dataset_label(dataset)}")
            if not _is_numeric_series(column):
                raise ValueError(f"Feature {feature!r} must be numeric in {dataset_label(dataset)}")


def _is_numeric_column(frame: pd.DataFrame, feature: str) -> bool:
    if feature not in frame.columns:
        return False
    column = frame[feature]
    return isinstance(column, pd.Series) and _is_numeric_series(column)


def _is_numeric_series(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series.dtype) and not pd.api.types.is_complex_dtype(series.dtype)


def _finite_values(feature: str, datasets: Mapping[str, pd.DataFrame]) -> dict[str, np.ndarray]:
    finite_values: dict[str, np.ndarray] = {}
    for dataset, frame in datasets.items():
        values = frame[feature].to_numpy(dtype=float, na_value=np.nan)
        values = values[np.isfinite(values)]
        if values.size:
            finite_values[dataset] = values
    if not finite_values:
        raise ValueError(f"Feature {feature!r} has no finite observations in any supplied dataset")
    return finite_values


def _bin_edges(feature: str, bins: _Bins, values: Mapping[str, np.ndarray]) -> np.ndarray:
    if isinstance(bins, Mapping):
        invalid_keys = [key for key in bins if not isinstance(key, str)]
        if invalid_keys:
            raise TypeError("Every key in a bins mapping must be a feature name string")
        specification: object = bins.get(feature, "auto")
    else:
        specification = bins

    if isinstance(specification, (bool, np.bool_)):
        raise TypeError(f"Bins for feature {feature!r} must not be boolean")
    if isinstance(specification, Integral):
        if specification <= 0:
            raise ValueError(f"Bins for feature {feature!r} must be a positive integer")
        specification = int(specification)
    elif not isinstance(specification, str):
        try:
            edges = np.asarray(specification, dtype=float)
        except (TypeError, ValueError):
            raise TypeError(
                f"Bins for feature {feature!r} must be a positive integer, NumPy strategy string, "
                "or sequence of edges"
            ) from None
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError(
                f"Bin edges for feature {feature!r} must be a one-dimensional sequence of at least two values"
            )
        if not np.isfinite(edges).all():
            raise ValueError(f"Bin edges for feature {feature!r} must all be finite")
        if not np.all(np.diff(edges) > 0):
            raise ValueError(f"Bin edges for feature {feature!r} must be strictly increasing")
        return edges

    combined = np.concatenate(tuple(values.values()))
    try:
        return np.histogram_bin_edges(combined, bins=specification)
    except (TypeError, ValueError):
        if isinstance(specification, str):
            raise ValueError(f"Unsupported NumPy bin strategy {specification!r} for feature {feature!r}") from None
        raise ValueError(f"Could not calculate {specification} bins for feature {feature!r}") from None


def _plot_options(
    options: Mapping[str, object] | None,
    *,
    argument: str,
    reserved: Collection[str],
    defaults: Mapping[str, object],
) -> dict[str, object]:
    if options is None:
        return dict(defaults)
    if not isinstance(options, Mapping):
        raise TypeError(f"{argument} must be a mapping or None")
    if not all(isinstance(key, str) for key in options):
        raise TypeError(f"Every key in {argument} must be a string")
    controlled = sorted(set(options) & set(reserved))
    if controlled:
        names = ", ".join(controlled)
        raise ValueError(f"{argument} cannot override controlled plotting keys: {names}")
    return {**defaults, **options}


def _draw_distribution(
    feature: str,
    values_by_dataset: Mapping[str, np.ndarray],
    edges: np.ndarray,
    kde: bool,
    figsize: tuple[float, float],
    histogram_options: Mapping[str, object],
    kde_options: Mapping[str, object],
) -> Figure:
    figure, axis = plt.subplots(figsize=figsize)
    for dataset, values in values_by_dataset.items():
        color = DATASET_COLORS[dataset]
        sns.histplot(
            x=values,
            bins=edges,
            stat="density",
            color=color,
            label=dataset_label(dataset),
            ax=axis,
            **histogram_options,
        )
        if kde and values.size >= 2 and np.any(values != values[0]):
            sns.kdeplot(x=values, color=color, ax=axis, **kde_options)

    axis.set(
        xlabel=feature,
        ylabel="Density",
        title=f"Distribution of {_feature_label(feature)}",
    )
    axis.legend(title="Dataset")
    figure.tight_layout()
    return figure


def _feature_label(feature: str) -> str:
    label = " ".join(feature.replace("_", " ").replace("+", " ").split())
    return label[:1].upper() + label[1:]
