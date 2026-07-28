import argparse
import json
from pathlib import Path
from statistics import pstdev

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import config
from src.schemas.metrics import ClassificationMetrics
from src.schemas.run_records import FoldRecord, TuningRecord
from src.utils.evaluation_utils import (
    classification_score,
    mean_classification_metrics,
)


def load_tuning_result(path: Path) -> TuningRecord:
    """Load a TuningRecord from JSON, explicitly converting nested dicts
    to their dataclass instances."""
    raw = json.loads(path.read_text())

    raw["fold_results"] = [
        FoldRecord(
            candidate_index=f["candidate_index"],
            fold_index=f["fold_index"],
            metrics=ClassificationMetrics(**f["metrics"]),
            time=f["time"],
            model_params=f["model_params"],
        )
        for f in raw.get("fold_results", [])
    ]

    return TuningRecord(**raw)


# ── Data wrangling ──────────────────────────────────────────────────────────


def _build_df(tuning: TuningRecord) -> pd.DataFrame:
    """Flat DataFrame: one row per candidate with params, scores, and metrics."""
    rows = []
    candidate_indices = sorted({fold.candidate_index for fold in tuning.fold_results})
    for candidate_index in candidate_indices:
        folds = [fold for fold in tuning.fold_results if fold.candidate_index == candidate_index]
        row = dict(folds[0].model_params)
        row["candidate_index"] = candidate_index
        scores = [classification_score(fold.metrics, tuning.scoring) for fold in folds]
        row["mean_score"] = float(np.mean(scores))
        row["std_score"] = float(pstdev(scores))
        metrics = mean_classification_metrics([fold.metrics for fold in folds])
        for name in [
            "roc_auc",
            "prc_auc",
            "f1",
            "accuracy",
            "sensitivity",
            "precision",
        ]:
            row[f"mean_{name}"] = getattr(metrics, name)
        rows.append(row)
    return pd.DataFrame(rows)


def _param_columns(df: pd.DataFrame) -> list[str]:
    """Hyperparameter column names (exclude derived/metric columns)."""
    exclude = {
        "candidate_index",
        "mean_score",
        "std_score",
        "mean_roc_auc",
        "mean_prc_auc",
        "mean_f1",
        "mean_accuracy",
        "mean_sensitivity",
        "mean_precision",
    }
    return [c for c in df.columns if c not in exclude]


# ── Plots ───────────────────────────────────────────────────────────────────


def plot_parallel_coordinates(
    df: pd.DataFrame,
    param_cols: list[str],
    metric_col: str,
    title: str = "CV Candidates — Parallel Coordinates",
) -> go.Figure:
    """Every candidate as a polyline across parameter axes, colored by score."""
    dimensions = []
    for col in param_cols:
        vals = df[col]
        if pd.api.types.is_numeric_dtype(vals) and vals.notna().all():
            dimensions.append(go.parcoords.Dimension(label=col, values=vals, tickvals=sorted(vals.unique())))
        else:
            # Fill NaN with a sentinel so factorize produces contiguous codes
            filled = vals.fillna("None")
            codes, uniques = pd.factorize(filled)
            dimensions.append(
                go.parcoords.Dimension(
                    label=col,
                    values=codes,
                    tickvals=list(range(len(uniques))),
                    ticktext=[str(u) for u in uniques],
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
    fig.update_layout(
        title=f"{title}<br><sup>Colored by {metric_col}</sup>",
        height=500 + 30 * len(param_cols),
    )
    return fig


def plot_param_marginal_effects(
    df: pd.DataFrame,
    param_cols: list[str],
    metric_col: str,
    title: str = "Parameter Marginal Effects",
) -> go.Figure:
    """Box plots of score distributions for each parameter value."""
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
        for pv in sorted(df[param].dropna().unique()):
            subset = df[df[param] == pv][metric_col]
            fig.add_trace(
                go.Box(
                    y=subset,
                    name=str(pv),
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
    """2D heatmaps for every parameter pair, averaging over remaining params."""
    numeric_params = [c for c in param_cols if pd.api.types.is_numeric_dtype(df[c])] or param_cols
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

    # Collect all pivot tables first to compute shared color range
    pivots = []
    for px_name, py_name in pairs:
        pivots.append(
            (
                px_name,
                py_name,
                df.pivot_table(index=py_name, columns=px_name, values=metric_col, aggfunc="mean"),
            )
        )
    z_all = np.concatenate([p.values.ravel() for _, _, p in pivots])
    zmin, zmax = float(z_all.min()), float(z_all.max())

    for idx, (px_name, py_name, pivot) in enumerate(pivots):
        row, col = idx // n_cols + 1, idx % n_cols + 1
        fig.add_trace(
            go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                coloraxis="coloraxis",
                zmin=zmin,
                zmax=zmax,
                hovertemplate=f"{px_name}=%{{x}}<br>{py_name}=%{{y}}<br>{metric_col}=%{{z:.4f}}<extra></extra>",
            ),
            row=row,
            col=col,
        )
        fig.update_xaxes(title_text=px_name, row=row, col=col)
        fig.update_yaxes(title_text=py_name, row=row, col=col)

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


# ── Output helpers ──────────────────────────────────────────────────────────


# ── Terminal output helpers ────────────────────────────────────────────────
def _print_summary_table(df: pd.DataFrame, param_cols: list[str], metric_col: str, top_n: int = 15):
    """Terminal table of top candidates plus spread stats."""
    top = df.nlargest(top_n, metric_col)
    display_cols = param_cols + [metric_col, "std_score"]
    available = [c for c in display_cols if c in top.columns]
    fmt_top = top[available].copy()
    for c in available:
        if pd.api.types.is_float_dtype(fmt_top[c]):
            fmt_top[c] = fmt_top[c].apply(lambda x: f"{x:.4f}" if x == x else "")
    print(f"\n{'=' * 80}\nTop {top_n} candidates by {metric_col}\n{'=' * 80}")
    print(fmt_top.to_string(index=False))
    print(f"\nBest  {metric_col}: {df[metric_col].max():.4f}")
    print(f"Worst {metric_col}: {df[metric_col].min():.4f}")
    print(f"Spread: {df[metric_col].max() - df[metric_col].min():.4f}")
    print(f"Std across candidates: {df[metric_col].std():.4f}")


def _print_param_importance(df: pd.DataFrame, param_cols: list[str], metric_col: str):
    """Per-parameter marginal effects table."""
    print(f"\n{'=' * 80}\nParameter Importance (mean {metric_col} per value)\n{'=' * 80}")
    for param in param_cols:
        grouped = df.groupby(param)[metric_col].agg(["mean", "std", "count"])
        rng = grouped["mean"].max() - grouped["mean"].min()
        best = grouped["mean"].idxmax()
        print(f"\n--- {param} (range: {rng:.4f}, best: {best}) ---")
        print(grouped.round(4).to_string())


# ── Main ────────────────────────────────────────────────────────────────────

PLOT_OPTIONS = ["parallel", "marginal", "heatmap"]


def _analyze_model(
    json_path: Path,
    target_metric: str | None,
    top_n: int,
    plot_names: list[str],
) -> tuple[str, list[go.Figure], str]:
    """Analyze a single model's CV results. Returns (model_name, figures, summary)."""
    model = json_path.stem
    tuning = load_tuning_result(json_path)
    df = _build_df(tuning)
    param_cols = _param_columns(df)

    metric_col = target_metric
    if metric_col is None:
        metric_col = f"mean_{tuning.scoring}"
    elif not metric_col.startswith("mean_"):
        metric_col = f"mean_{metric_col}"
    if metric_col not in df.columns:
        available = [c for c in df.columns if c.startswith("mean_")]
        raise ValueError(f"Metric '{metric_col}' not found. Available: {available}")

    n_folds = max((fold.fold_index for fold in tuning.fold_results), default=-1) + 1
    lines = [
        f"\n{'=' * 80}",
        f"Model: {model}",
        f"{'=' * 80}",
        f"Candidates: {len(df)} x {n_folds} folds",
        f"Parameters: {param_cols}",
        f"Scoring: {tuning.scoring}  |  Target metric: {metric_col}",
        f"Best params: {tuning.best_params}",
        f"Best {tuning.scoring}: {df['mean_score'].max():.4f}",
    ]
    print("\n".join(lines))

    available_plots = {
        "parallel": lambda: plot_parallel_coordinates(df, param_cols, metric_col),
        "marginal": lambda: plot_param_marginal_effects(df, param_cols, metric_col),
        "heatmap": lambda: plot_pairwise_heatmap(df, param_cols, metric_col),
    }

    figures = []
    for name in plot_names:
        if name in available_plots:
            print(f"  Plot: {name} ...")
            try:
                figures.append(available_plots[name]())
            except Exception as e:
                print(f"    Skipped '{name}': {e}")
        else:
            print(f"  Unknown plot: '{name}' (options: {list(available_plots)})")

    _print_summary_table(df, param_cols, metric_col, top_n=top_n)
    _print_param_importance(df, param_cols, metric_col)

    return model, figures, "\n".join(lines)


def _html_page_multi(sections: list[tuple[str, list[go.Figure]]], title: str) -> str:
    """Build one HTML page with multiple model sections, each with a heading."""
    plots_cdn = None
    parts = []
    for heading, figures in sections:
        parts.append(f'<h2 style="margin-top:40px;border-bottom:2px solid #ccc;padding-bottom:4px;">{heading}</h2>')
        for fig in figures:
            if plots_cdn is None:
                html = fig.to_html(full_html=False, include_plotlyjs="cdn")
                plots_cdn = True
            else:
                html = fig.to_html(full_html=False, include_plotlyjs=False)
            parts.append(f'<div style="margin-bottom:30px;">{html}</div>')
    body = "\n".join(parts)
    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>{title}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         margin: 20px; background: #f8f8f8; }}
  h1 {{ color: #333; }}
  h2 {{ color: #444; }}
</style>
</head>
<body><h1>{title}</h1>{body}</body></html>"""


def main():
    parser = argparse.ArgumentParser(
        description="Analyze CV tuning results with interactive visualizations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  uv run python -m src.exploration.cv_analyzer -id 8dd4ef36 -m xgboost
  uv run python -m src.exploration.cv_analyzer -id 8dd4ef36
  uv run python -m src.exploration.cv_analyzer -id 8dd4ef36 -t accuracy -n 20
        """,
    )
    parser.add_argument("-id", type=str, required=True, help="Pipeline ID (MLflow run ID).")
    parser.add_argument(
        "-m",
        type=str,
        default=None,
        help="Model name (e.g. 'xgboost'). Omit to analyze all models.",
    )
    parser.add_argument(
        "-t",
        "--target-metric",
        type=str,
        default=None,
        help="Metric for ranking/coloring. Default: the CV scoring metric.",
    )
    parser.add_argument(
        "-n",
        "--top-n",
        type=int,
        default=15,
        help="Number of top candidates in terminal table. Default: 15",
    )
    parser.add_argument(
        "-p",
        "--plots",
        type=str,
        default=",".join(PLOT_OPTIONS),
        help=f"Plot types: {', '.join(PLOT_OPTIONS)}, all.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output HTML path. Default: plots/cv_analysis/<id>_<models>.html",
    )

    args = parser.parse_args()
    pipeline_id = args.id.lower()

    cv_results_dir = Path(config.dir_mlflow_artifacts) / pipeline_id / "artifacts" / "cv_results"
    if not cv_results_dir.is_dir():
        print(f"Error: directory not found: {cv_results_dir}")
        return

    if args.m:
        json_paths = [cv_results_dir / f"{args.m.lower()}.json"]
        if not json_paths[0].exists():
            print(f"Error: file not found: {json_paths[0]}")
            return
    else:
        json_paths = sorted(cv_results_dir.glob("*.json"))
        if not json_paths:
            print(f"No JSON files in: {cv_results_dir}")
            return
        print(f"Found {len(json_paths)} model(s): {[p.stem for p in json_paths]}")

    plot_names = [p.strip().lower() for p in args.plots.split(",")]
    if "all" in plot_names:
        plot_names = list(PLOT_OPTIONS)

    sections = []
    for json_path in json_paths:
        try:
            model, figures, _summary = _analyze_model(json_path, args.target_metric, args.top_n, plot_names)
        except Exception as e:
            print(f"\nError analyzing {json_path.stem}: {e}")
            continue
        if figures:
            sections.append((model, figures))

    if not sections:
        print("\nNo figures generated.")
        return

    output_path = args.output
    if output_path is None:
        output_dir = config.dir_plots / "cv_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        models_str = "_".join(m for m, _ in sections)
        output_path = output_dir / f"{pipeline_id}_{models_str}.html"
    else:
        output_path = Path(output_path)

    title = f"CV Analysis — pipeline {pipeline_id}"
    with open(output_path, "w") as f:
        f.write(_html_page_multi(sections, title))

    print(f"\nWrote interactive report to: {output_path}")


if __name__ == "__main__":
    main()
