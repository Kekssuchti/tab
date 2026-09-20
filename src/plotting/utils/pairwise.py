"""Pairwise model contrasts from aligned paired-bootstrap scores.

Every pair of models is scored on the same cohort resample, so a win share is a
paired statistic: it says how often the row model came out ahead *within one
resample*, not how often it would win on new data. The mean difference of the
pair is what carries the effect size, which is why these summaries always travel
with a difference in metric units.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.plotting.defaults import (
    PAIRWISE_TABLE_COLOR_NAMES,
    dataset_label,
    latex_color_definitions,
    metric_label,
    model_label,
    pairwise_decision_bounds,
    pairwise_table_colors,
)
from src.plotting.utils.aggregation import AggregatedEvaluation
from src.schemas.training_schemas import LOWER_IS_BETTER_SCORING
from src.utils.prediction_metrics import pairwise_win_matrices

STATE_COLORS = {"row": "win", "column": "loss", "none": "neutral"}


@dataclass(frozen=True)
class PairwiseSummary:
    """Win shares, paired differences, cross-cohort reversals, and contrasts.

    ``matrix_cells`` holds one row per unordered model pair and test dataset.
    ``win_share`` is a fraction in [0, 1] and ``delta`` is in the metric's own
    units, oriented so that positive values favor the row model. ``cross_cells``
    holds one row per unordered model pair and metric, with the pair oriented so
    that ``row_instance`` is the model that ranks better in-domain, and one win
    share per cohort. ``forest`` holds one row per model and test dataset, with
    ``estimate`` oriented so that positive values mean better than the reference
    model. Setting ``ci_level`` to None skips every interval.
    """

    matrix_cells: pd.DataFrame
    cross_cells: pd.DataFrame
    forest: pd.DataFrame
    ordering: pd.DataFrame
    aggregated: AggregatedEvaluation
    external_dataset: str
    reference_instance: str
    alpha: float

    @property
    def model_metadata(self) -> pd.DataFrame:
        return self.aggregated.model_metadata

    @property
    def model_instances(self) -> tuple[str, ...]:
        return self.aggregated.model_instances

    @property
    def metrics(self) -> tuple[str, ...]:
        return self.aggregated.metrics

    @property
    def datasets(self) -> tuple[str, ...]:
        return self.aggregated.datasets

    @property
    def trained_on(self) -> str:
        return self.aggregated.trained_on

    @property
    def target(self) -> str:
        return self.aggregated.target

    @property
    def run_ids(self) -> tuple[str, ...]:
        return self.aggregated.run_ids

    @property
    def run_count(self) -> int:
        return self.aggregated.run_count

    @property
    def bootstrap_count(self) -> int:
        return self.aggregated.bootstrap_count

    @property
    def ci_level(self) -> float | None:
        return self.aggregated.ci_level

    def order_for(self, dataset: str, metric: str) -> pd.DataFrame:
        """Return the best-first model ordering for one dataset and metric."""
        return self.ordering.loc[self.ordering["dataset"].eq(dataset) & self.ordering["metric"].eq(metric)].sort_values(
            "rank", kind="stable"
        )


def prepare_pairwise_summary(
    aggregated: AggregatedEvaluation,
    *,
    reference_model: str | None = None,
    alpha: float = 0.05,
) -> PairwiseSummary:
    """Calculate every pairwise contrast used by the comparison figures.

    The reference model anchors the difference forest, because a difference
    against one anchor is the only way to read effect size at a common scale.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    external_datasets = [dataset for dataset in aggregated.datasets if dataset != aggregated.trained_on]
    if aggregated.trained_on not in aggregated.datasets or len(external_datasets) != 1:
        raise ValueError(
            "Pairwise summaries require exactly one in-domain and one external test dataset; "
            f"trained_on={aggregated.trained_on!r}, datasets={list(aggregated.datasets)}"
        )
    external_dataset = external_datasets[0]

    ordering = _model_ordering(aggregated)
    matrix_cells = _matrix_cells(aggregated, ordering, alpha)
    cross_cells = _cross_cells(matrix_cells, ordering, aggregated.trained_on, external_dataset)
    reference_instance = _resolve_reference(aggregated, reference_model, external_dataset)
    forest = _forest(aggregated, ordering, reference_instance, alpha)

    return PairwiseSummary(
        matrix_cells=matrix_cells,
        cross_cells=cross_cells,
        forest=forest,
        ordering=ordering,
        aggregated=aggregated,
        external_dataset=external_dataset,
        reference_instance=reference_instance,
        alpha=alpha,
    )


def _interval_tail(ci_level: float | None) -> float | None:
    """Return the two-sided tail probability, or None when no interval is drawn."""
    return None if ci_level is None else (1.0 - ci_level) / 2.0


def _quantile_interval(draws: pd.Series, tail: float | None) -> tuple[float, float]:
    """Return the percentile interval of a paired difference, or two NaNs."""
    if tail is None:
        return float("nan"), float("nan")
    lower, upper = draws.quantile([tail, 1.0 - tail])
    return float(lower), float(upper)


def metric_direction(metric: str) -> float:
    """Return +1 when a larger metric value is better and -1 when it is not."""
    return -1.0 if metric in LOWER_IS_BETTER_SCORING else 1.0


def win_matrix(
    aggregated: AggregatedEvaluation,
    dataset: str,
    metric: str,
) -> pd.DataFrame:
    """Return row-model wins over column-model, counted in bootstrap draws.

    The counts come from the routine that defines the MLflow ``pairwise_wins``
    artifacts, so a figure and its run artifact can never disagree. For a
    lower-is-better metric the comparison is inverted, which keeps ties split
    evenly in both directions.
    """
    scores = aggregated.scores(dataset, metric).reset_index()
    scores.insert(0, "metric", metric)
    scores.insert(0, "dataset", dataset)
    matrix = pairwise_win_matrices(scores)[f"{dataset}_{metric}"]
    if metric in LOWER_IS_BETTER_SCORING:
        matrix = float(aggregated.bootstrap_count) - matrix
    return matrix


def _model_ordering(aggregated: AggregatedEvaluation) -> pd.DataFrame:
    """Rank models within each test dataset and metric by point estimate."""
    rows: list[dict[str, object]] = []
    performance = aggregated.performance
    for metric in aggregated.metrics:
        for dataset in aggregated.datasets:
            subset = performance.loc[performance["dataset"].eq(dataset) & performance["metric"].eq(metric)]
            indexed = subset.set_index("model_instance")
            estimates = indexed["estimate"].astype(float)
            ranks = estimates.rank(method="average", ascending=metric in LOWER_IS_BETTER_SCORING)
            for instance in aggregated.model_instances:
                rows.append(
                    {
                        "dataset": dataset,
                        "metric": metric,
                        "model_instance": instance,
                        "model_name": str(indexed.loc[instance, "model_name"]),
                        "estimate": float(estimates.loc[instance]),
                        "rank": float(ranks.loc[instance]),
                    }
                )
    return pd.DataFrame(rows)


def _matrix_cells(
    aggregated: AggregatedEvaluation,
    ordering: pd.DataFrame,
    alpha: float,
) -> pd.DataFrame:
    """Describe every unordered model pair on every test dataset."""
    lower_bound, upper_bound = pairwise_decision_bounds(alpha)
    alpha_tail = _interval_tail(aggregated.ci_level)
    rows: list[dict[str, object]] = []

    for metric in aggregated.metrics:
        direction = metric_direction(metric)
        for dataset in aggregated.datasets:
            matrix = win_matrix(aggregated, dataset, metric)
            scores = aggregated.scores(dataset, metric)
            ordered = ordering.loc[ordering["dataset"].eq(dataset) & ordering["metric"].eq(metric)].sort_values(
                "rank", kind="stable"
            )
            instances = list(ordered["model_instance"])
            estimates = ordered.set_index("model_instance")["estimate"]
            names = ordered.set_index("model_instance")["model_name"]

            for index, row_instance in enumerate(instances):
                for column_instance in instances[index + 1 :]:
                    share = float(matrix.loc[row_instance, column_instance]) / aggregated.bootstrap_count
                    draws = direction * (scores[row_instance] - scores[column_instance])
                    delta_lower, delta_upper = _quantile_interval(draws, alpha_tail)
                    if share > upper_bound:
                        state = "row"
                    elif share < lower_bound:
                        state = "column"
                    else:
                        state = "none"
                    rows.append(
                        {
                            "dataset": dataset,
                            "metric": metric,
                            "row_instance": row_instance,
                            "column_instance": column_instance,
                            "row_name": str(names.loc[row_instance]),
                            "column_name": str(names.loc[column_instance]),
                            "win_share": share,
                            "delta": direction * float(estimates.loc[row_instance] - estimates.loc[column_instance]),
                            "delta_lower": float(delta_lower),
                            "delta_upper": float(delta_upper),
                            "state": state,
                            "decided": state != "none",
                        }
                    )
    return pd.DataFrame(rows)


def _cross_cells(
    matrix_cells: pd.DataFrame,
    ordering: pd.DataFrame,
    internal_dataset: str,
    external_dataset: str,
) -> pd.DataFrame:
    """Put every pair's in-domain and external win share side by side.

    The two cohorts order the models differently, so each pair is re-keyed in a
    fixed orientation and each cohort's share is oriented onto that key. Joining
    on the per-cohort row and column order instead would silently drop exactly the
    reversed pairs this table exists to show.
    """
    ordered = (
        ordering.loc[ordering["dataset"].eq(internal_dataset)]
        .sort_values(["metric", "rank"], kind="stable")
        .set_index(["metric", "model_instance"])["rank"]
    )
    rows: list[dict[str, object]] = []
    for metric, metric_cells in matrix_cells.groupby("metric", sort=False):
        shares = _pair_shares(metric_cells, internal_dataset, external_dataset)
        metric_order = [instance for instance in ordered.loc[metric].sort_values(kind="stable").index.astype(str)]
        names = (
            ordering.loc[ordering["metric"].eq(metric)].set_index("model_instance")["model_name"].astype(str).to_dict()
        )
        for index, row_instance in enumerate(metric_order):
            for column_instance in metric_order[index + 1 :]:
                share_internal = shares[(row_instance, column_instance, internal_dataset)]
                share_external = shares[(row_instance, column_instance, external_dataset)]
                rows.append(
                    {
                        "metric": metric,
                        "row_instance": row_instance,
                        "column_instance": column_instance,
                        "row_name": names[row_instance],
                        "column_name": names[column_instance],
                        "share_internal": share_internal,
                        "share_external": share_external,
                        "share_change": share_internal - share_external,
                        "reversed": (share_internal - 0.5) * (share_external - 0.5) < 0,
                    }
                )
    return pd.DataFrame(rows)


def _pair_shares(
    metric_cells: pd.DataFrame,
    internal_dataset: str,
    external_dataset: str,
) -> dict[tuple[str, str, str], float]:
    """Map every ordered model pair and cohort to the first model's win share."""
    shares: dict[tuple[str, str, str], float] = {}
    for row in metric_cells.itertuples():
        share = float(row.win_share)
        shares[(row.row_instance, row.column_instance, row.dataset)] = share
        shares[(row.column_instance, row.row_instance, row.dataset)] = 1.0 - share
    return shares


def _resolve_reference(
    aggregated: AggregatedEvaluation,
    reference_model: str | None,
    external_dataset: str,
) -> str:
    """Pick the model that every paired difference is measured against."""
    names = aggregated.model_metadata.set_index("model_instance")["model_name"]
    if reference_model is not None:
        matches = [instance for instance in aggregated.model_instances if names.loc[instance] == reference_model]
        if not matches:
            available = sorted(set(names.astype(str)))
            raise ValueError(
                f"Reference model {reference_model!r} is not selected; available: {available}. "
                "Set reference_model to None to anchor on the strongest external model instead."
            )
        return matches[0]
    if "xgboost" in set(names.astype(str)):
        return next(instance for instance in aggregated.model_instances if names.loc[instance] == "xgboost")

    metric = aggregated.metrics[0]
    external_points = aggregated.performance.loc[
        aggregated.performance["dataset"].eq(external_dataset) & aggregated.performance["metric"].eq(metric)
    ].set_index("model_instance")["estimate"]
    direction = metric_direction(metric)
    return min(aggregated.model_instances, key=lambda instance: -direction * float(external_points.loc[instance]))


def _forest(
    aggregated: AggregatedEvaluation,
    ordering: pd.DataFrame,
    reference_instance: str,
    alpha: float,
) -> pd.DataFrame:
    """Measure every model against the reference on every test dataset."""
    alpha_tail = _interval_tail(aggregated.ci_level)
    rows: list[dict[str, object]] = []

    for metric in aggregated.metrics:
        direction = metric_direction(metric)
        for dataset in aggregated.datasets:
            matrix = win_matrix(aggregated, dataset, metric)
            scores = aggregated.scores(dataset, metric)
            ordered = ordering.loc[ordering["dataset"].eq(dataset) & ordering["metric"].eq(metric)]
            estimates = ordered.set_index("model_instance")["estimate"]
            names = ordered.set_index("model_instance")["model_name"]
            for instance in aggregated.model_instances:
                is_reference = instance == reference_instance
                # Oriented like a "difference from the baseline" plot: positive is
                # the model doing better than the reference, so the reader cannot
                # mistake a positive bar for a loss.
                draws = direction * (scores[instance] - scores[reference_instance])
                lower, upper = _quantile_interval(draws, alpha_tail)
                rows.append(
                    {
                        "dataset": dataset,
                        "metric": metric,
                        "model_instance": instance,
                        "model_name": str(names.loc[instance]),
                        "estimate": direction * float(estimates.loc[instance] - estimates.loc[reference_instance]),
                        "lower": float(lower),
                        "upper": float(upper),
                        "win_share": np.nan
                        if is_reference
                        else float(matrix.loc[instance, reference_instance]) / aggregated.bootstrap_count,
                        "is_reference": bool(is_reference),
                    }
                )
    return pd.DataFrame(rows)


# --- LaTeX tables ---------------------------------------------------------


def pairwise_matrix_to_latex(
    summary: PairwiseSummary,
    *,
    dataset: str,
    metric: str,
    headline: str,
    label: str,
    scale: float = 100.0,
    win_digits: int = 1,
    delta_digits: int = 2,
) -> str:
    """Return a split-encoded LaTeX matrix for one dataset and metric.

    Above the diagonal each cell gives the share of paired resamples in which the
    row model beat the column model, shaded and bold when the pair is decided.
    Below the diagonal each cell gives the mean metric difference, row minus
    column, which keeps a magnitude-blind win share from hiding its effect size.
    Requires ``booktabs``, ``graphicx``, and ``colortbl``, plus the
    ``\\definecolor`` lines printed at the top of the returned string.
    """
    ordered = summary.order_for(dataset, metric)
    if ordered.empty:
        raise ValueError(f"No pairwise cells for dataset {dataset!r} and metric {metric!r}")
    instances = list(ordered["model_instance"])
    cells = summary.matrix_cells
    selected = cells.loc[cells["dataset"].eq(dataset) & cells["metric"].eq(metric)]
    lookup: dict[tuple[str, str], object] = {}
    for row in selected.itertuples():
        lookup[(row.row_instance, row.column_instance)] = row
        lookup[(row.column_instance, row.row_instance)] = row

    colors = pairwise_table_colors(summary.alpha)
    columns = " ".join(["l", *["c"] * len(instances)])
    header = " & ".join(["\\textbf{Row model}", *[f"\\textbf{{{index}}}" for index in range(1, len(instances) + 1)]])
    lines = [
        "% Requires \\usepackage{booktabs, graphicx, colortbl} and these preamble colors:",
        *[f"% {line}" for line in latex_color_definitions(colors)],
        "\\begin{table}[htbp]",
        "    \\centering",
        "    \\resizebox{\\linewidth}{!}{%",
        f"    \\begin{{tabular}}{{{columns}}}",
        "        \\toprule",
        f"        & \\multicolumn{{{len(instances)}}}{{c}}{{\\textbf{{Column model}}}} \\\\",
        f"        \\cmidrule(lr){{2-{len(instances) + 1}}}",
        f"        {header} \\\\",
        "        \\midrule",
    ]

    for row_index, row_instance in enumerate(instances):
        entries = []
        for column_index, column_instance in enumerate(instances):
            if column_index == row_index:
                entries.append("---")
            elif column_index > row_index:
                cell = lookup[(row_instance, column_instance)]
                state = cell.state if cell.row_instance == row_instance else _opposite(cell.state)
                share = scale * float(cell.win_share)
                text = f"{share:.{win_digits}f}"
                if cell.decided:
                    text = f"\\textbf{{{text}}}"
                color_name = PAIRWISE_TABLE_COLOR_NAMES[STATE_COLORS[state]]
                entries.append(f"\\cellcolor{{{color_name}}} {text}")
            else:
                cell = lookup[(row_instance, column_instance)]
                signed = float(cell.delta) * (1.0 if cell.row_instance == row_instance else -1.0)
                entries.append(f"{scale * signed:+.{delta_digits}f}")
        lines.append(f"        {_row_label(ordered, row_instance, row_index + 1)} & " + " & ".join(entries) + " \\\\")

    lines.extend(
        [
            "        \\bottomrule",
            "    \\end{tabular}%",
            "    }",
            f"    \\caption{{{headline} {_matrix_notes(summary, ordered, dataset, metric, scale, win_digits)}}}",
            f"    \\label{{{label}}}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines)


def rank_table_to_latex(
    ranks,
    *,
    headline: str,
    label: str,
    setting_label: str,
    setting_display: dict[str, str] | None = None,
    digits: int = 2,
    cd_column: str = "experiment",
) -> str:
    """Return a LaTeX table of average rank per model, setting, and experiment."""
    display = setting_display or {}
    settings = tuple(ranks.settings)
    by_setting = ranks.by_setting
    overall = ranks.overall
    tests = ranks.tests

    def critical_difference(scope: str, setting: str | None) -> float:
        rows = tests.loc[tests["scope"].eq(scope)]
        if setting is not None:
            rows = rows.loc[rows["setting"].eq(setting)]
        return float(rows["critical_difference"].iloc[0])

    columns = " ".join(["l", *["c"] * len(settings), "c"])
    header = " & ".join(
        [
            "\\textbf{Model}",
            *[f"\\textbf{{{display.get(setting, setting)}}}" for setting in settings],
            "\\textbf{Overall}",
        ]
    )
    lines = [
        "\\begin{table}[htbp]",
        "    \\centering",
        "    \\resizebox{\\linewidth}{!}{%",
        f"    \\begin{{tabular}}{{{columns}}}",
        "        \\toprule",
        f"        & \\multicolumn{{{len(settings)}}}{{c}}{{\\textbf{{{setting_label}}}}} & \\\\",
        f"        \\cmidrule(lr){{2-{len(settings) + 1}}}",
        f"        {header} \\\\",
        "        \\midrule",
    ]

    ordered_models = list(overall.sort_values("mean_rank", kind="stable")["model_instance"])
    best_overall = float(overall["mean_rank"].min())
    for instance in ordered_models:
        entries = []
        for setting in settings:
            row = by_setting.loc[by_setting["setting"].eq(setting) & by_setting["model_instance"].eq(instance)]
            value = float(row["mean_rank"].iloc[0])
            best = float(by_setting.loc[by_setting["setting"].eq(setting), "mean_rank"].min())
            text = f"{value:.{digits}f}"
            entries.append(f"\\textbf{{{text}}}" if np.isclose(value, best) else text)
        overall_row = overall.loc[overall["model_instance"].eq(instance)]
        overall_value = float(overall_row["mean_rank"].iloc[0])
        overall_text = f"{overall_value:.{digits}f}"
        entries.append(f"\\textbf{{{overall_text}}}" if np.isclose(overall_value, best_overall) else overall_text)
        name = str(overall_row["model_name"].iloc[0])
        lines.append(f"        {_latex_escape(model_label(name))} & " + " & ".join(entries) + " \\\\")

    block_counts = [
        f"{int(tests.loc[tests['scope'].eq('setting') & tests['setting'].eq(setting), 'n_blocks'].iloc[0])}"
        for setting in settings
    ]
    block_counts.append(str(int(tests.loc[tests["scope"].eq(cd_column), "n_blocks"].iloc[0])))
    lines.extend(
        [
            "        \\midrule",
            "        \\textit{Blocks} & " + " & ".join(block_counts) + " \\\\",
            "        \\textit{CD ($\\alpha$ = "
            f"{ranks.alpha:.2f}"
            + ")} & "
            + " & ".join(
                [f"{critical_difference('setting', setting):.{digits}f}" for setting in settings]
                + [f"{critical_difference(cd_column, None):.{digits}f}"]
            )
            + " \\\\",
            "        \\bottomrule",
            "    \\end{tabular}%",
            "    }",
            f"    \\caption{{{headline} {_rank_table_notes(ranks, setting_label, digits)}}}",
            f"    \\label{{{label}}}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines)


def _rank_table_notes(ranks, setting_label: str, digits: int) -> str:
    del digits
    run_text = "one pipeline run" if ranks.run_count == 1 else f"{ranks.run_count} pipeline runs"
    rank_span = len(ranks.model_instances) - 1
    unresolvable = ranks.tests.loc[ranks.tests["critical_difference"].gt(rank_span), "scope"].tolist()
    caveat = (
        ""
        if not unresolvable
        else (
            f" A CD of {rank_span} or more rank positions separates nothing, which is the case for "
            f"{len(unresolvable)} of the listed columns: too few evaluation blocks per setting leave the post-hoc "
            f"test without power."
        )
    )
    return (
        f"Each cell is the average rank of a model over {ranks.block_count} evaluation blocks "
        f"({setting_label.lower()} x test cohort x metric) taken from {run_text}; rank 1 is the best model in a block. "
        f"The Overall column pools every block, so it reports how consistently a model ranked where it ranked. "
        f"Bold entries mark the best average rank within a column. "
        f"Blocks is the number of blocks behind each column, and CD is the Nemenyi critical difference for that "
        f"column at the stated level: two models whose average ranks differ by less than the CD are not separated by "
        f"the Friedman--Nemenyi test.{caveat}"
    )


def _row_label(ordered: pd.DataFrame, instance: str, position: int) -> str:
    name = str(ordered.set_index("model_instance").loc[instance, "model_name"])
    return f"\\textbf{{{position}. {_latex_escape(model_label(name))}}}"


def _opposite(state: str) -> str:
    return {"row": "column", "column": "row", "none": "none"}[state]


def _matrix_notes(
    summary: PairwiseSummary,
    ordered: pd.DataFrame,
    dataset: str,
    metric: str,
    scale: float,
    win_digits: int,
) -> str:
    legend = ", ".join(
        f"{index} {_latex_escape(model_label(str(row.model_name)))}"
        for index, row in enumerate(ordered.itertuples(), start=1)
    )
    lower_bound, upper_bound = pairwise_decision_bounds(summary.alpha)
    unit = "percentage points" if scale == 100.0 else "metric units"
    decision = (
        f"Bold cells and a shaded background mark comparisons whose paired "
        f"{round(100 * summary.ci_level)}\\% percentile interval excludes zero, which states the same thing as a win "
        f"share outside {scale * lower_bound:.{win_digits}f}--{scale * upper_bound:.{win_digits}f}."
        if summary.ci_level is not None
        else f"Bold cells and a shaded background mark comparisons decided at "
        f"$\\alpha$ = {summary.alpha:g}, that is a win share outside "
        f"{scale * lower_bound:.{win_digits}f}--{scale * upper_bound:.{win_digits}f}."
    )
    return (
        f"Rows and columns are ordered by the point estimate on {dataset_label(dataset)} "
        f"({metric_label(metric)}): {legend}. Above the diagonal, each cell is the share of "
        f"{summary.bootstrap_count:,} paired cohort-bootstrap draws in which the row model scored higher than the "
        f"column model, so the mirror cell is 100 minus this value. Below the diagonal, each cell is the mean "
        f"{metric_label(metric)} difference in {unit}, row minus column, oriented so that a positive value favors the "
        f"row model. {decision} "
        f"Win shares describe resamples of the test cohort, not the probability that a model generalizes better."
    )


def _latex_escape(text: str) -> str:
    for source, target in (("\\", "\\textbackslash{}"), ("_", "\\_"), ("%", "\\%"), ("&", "\\&"), ("#", "\\#")):
        text = text.replace(source, target)
    return text
