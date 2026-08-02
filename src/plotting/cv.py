"""Reusable interactive plots for cross-validation tuning candidates."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_parallel_coordinates(
    df: pd.DataFrame,
    param_cols: list[str],
    metric_col: str,
    title: str = "CV Candidates — Parallel Coordinates",
) -> go.Figure:
    """Plot every candidate across parameter axes, colored by score."""
    dimensions = []
    for col in param_cols:
        vals = df[col]
        if pd.api.types.is_numeric_dtype(vals) and vals.notna().all():
            dimensions.append(go.parcoords.Dimension(label=col, values=vals, tickvals=sorted(vals.unique())))
        else:
            filled = vals.fillna("None")
            codes, uniques = pd.factorize(filled)
            dimensions.append(
                go.parcoords.Dimension(
                    label=col,
                    values=codes,
                    tickvals=list(range(len(uniques))),
                    ticktext=[str(value) for value in uniques],
                )
            )

    fig = go.Figure(
        go.Parcoords(
            line=dict(
                color=df[metric_col],
                colorscale="viridis",
                showscale=True,
                colorbar=dict(title=metric_col),
            ),
            dimensions=dimensions,
        )
    )
    fig.update_layout(title=f"{title}<br><sup>Colored by {metric_col}</sup>", height=500 + 30 * len(param_cols))
    return fig


def plot_param_marginal_effects(
    df: pd.DataFrame,
    param_cols: list[str],
    metric_col: str,
    title: str = "Parameter Marginal Effects",
) -> go.Figure:
    """Plot score distributions for each parameter value."""
    n_params = len(param_cols)
    n_cols = min(3, n_params)
    n_rows = int(np.ceil(n_params / n_cols))
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=param_cols,
        vertical_spacing=0.15,
        horizontal_spacing=0.08,
    )

    for idx, param in enumerate(param_cols):
        row, col = idx // n_cols + 1, idx % n_cols + 1
        for param_value in sorted(df[param].dropna().unique()):
            subset = df[df[param] == param_value][metric_col]
            fig.add_trace(
                go.Box(
                    y=subset,
                    name=str(param_value),
                    boxpoints="outliers",
                    marker=dict(size=3, opacity=0.5),
                    line=dict(width=1),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
        fig.update_xaxes(title_text=param, row=row, col=col)
        fig.update_yaxes(title_text=metric_col, row=row, col=col)

    fig.update_layout(
        title=f"{title}<br><sup>Metric: {metric_col}</sup>",
        height=300 * n_rows,
        showlegend=False,
    )
    return fig


def plot_pairwise_heatmap(
    df: pd.DataFrame,
    param_cols: list[str],
    metric_col: str,
    title: str = "Pairwise Parameter Heatmaps",
) -> go.Figure:
    """Plot mean score heatmaps for every parameter pair."""
    numeric_params = [column for column in param_cols if pd.api.types.is_numeric_dtype(df[column])] or param_cols
    pairs = [
        (numeric_params[i], numeric_params[j])
        for i in range(len(numeric_params))
        for j in range(i + 1, len(numeric_params))
    ]
    if not pairs:
        return go.Figure().update_layout(title="Need at least 2 parameters for heatmaps")

    n_cols = min(3, len(pairs))
    n_rows = int(np.ceil(len(pairs) / n_cols))
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f"{x} vs {y}" for x, y in pairs],
        vertical_spacing=0.14,
        horizontal_spacing=0.1,
    )
    pivots = [
        (
            x_name,
            y_name,
            df.pivot_table(index=y_name, columns=x_name, values=metric_col, aggfunc="mean"),
        )
        for x_name, y_name in pairs
    ]
    all_values = np.concatenate([pivot.values.ravel() for _, _, pivot in pivots])
    zmin, zmax = float(all_values.min()), float(all_values.max())

    for idx, (x_name, y_name, pivot) in enumerate(pivots):
        row, col = idx // n_cols + 1, idx % n_cols + 1
        fig.add_trace(
            go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                coloraxis="coloraxis",
                zmin=zmin,
                zmax=zmax,
                hovertemplate=(
                    f"{x_name}=%{{x}}<br>{y_name}=%{{y}}<br>{metric_col}=%{{z:.4f}}<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )
        fig.update_xaxes(title_text=x_name, row=row, col=col)
        fig.update_yaxes(title_text=y_name, row=row, col=col)

    fig.update_layout(
        title=f"{title}<br><sup>Metric: {metric_col}</sup>",
        height=350 * n_rows,
        coloraxis=dict(
            colorscale="viridis",
            cmin=zmin,
            cmax=zmax,
            colorbar=dict(title=metric_col),
        ),
    )
    return fig
