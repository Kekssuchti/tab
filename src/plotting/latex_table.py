"""Create LaTeX performance tables from pipeline evaluation results."""

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import pandas as pd

from src.plotting.defaults import (
    DATASET_NAMES,
    DATASET_ORDER,
    MODEL_LABELS,
    MODEL_ORDER,
)

CAPTION = r"Model performance (AUROC [95\% CI]) for the three ICU classification tasks across train and test set combinations. Gray fields show scenarios with external test sets. $\Delta_{\mathrm{spec}}$ shows model-specific and and $\Delta_{\mathrm{comp}}$ comparative generalizability loss"


def performance_table_to_latex(
    results: pd.DataFrame,
    metric: str = "roc_auc",
    *,
    filters: Mapping[str, object] | None = None,
    dataset_order: Sequence[str] = DATASET_ORDER,
    model_order: Sequence[str] = MODEL_ORDER,
    dataset_labels: Mapping[str, str] = DATASET_NAMES,
    model_labels: Mapping[str, str] = MODEL_LABELS,
    caption: str = CAPTION,
    label: str | None = None,
    include_caption: bool = True,
    decimals: int = 2,
    percentage: bool = True,
    include_ci: bool = True,
    include_generalizability: bool = True,
    bold_best: bool = True,
    shade_external: bool = True,
    external_color: str = "black!8",
    run_aggregation: Literal["average", "highest", "lowest"] | None = None,
    line_width_percent: float = 1.0,
) -> str:
    """Return a LaTeX table for models trained once and tested on each dataset.

    The input must contain ``scope == "test"`` and ``statistic == "point"``
    rows. Model identities come from ``model_instance`` while ``model_name`` is
    used for canonical ordering and display labels. By default, exactly one row
    per model instance and test dataset is required. ``run_aggregation`` can
    average repeated pipeline runs—including arithmetic means of the lower and
    upper confidence bounds—or select each group's complete highest- or
    lowest-metric row. Confidence levels themselves must be consistent and are
    never averaged. By default, metric values are displayed as
    percentages with their confidence interval, for example
    ``82.22 [81.41, 83.03]``. The two generalizability columns use the
    metric-specific ``generalizability_loss_*`` and
    ``comparative_generalizability_loss_*`` values. The direction-aware best
    value in each result column is bolded; for losses this is the highest value
    (normally the value closest to zero).

    Use ``filters`` when the DataFrame contains multiple targets, experiments,
    pipeline runs, or training configurations. Ambiguous or incomplete model x
    dataset combinations raise ``ValueError`` unless aggregation is requested.

    Set ``include_caption`` to False to suppress the caption and label, for
    example when embedding the table into a larger assembly via
    :func:`multiple_latex_tables`.
    """
    ci_lower = f"{metric}_ci_lower"
    ci_upper = f"{metric}_ci_upper"
    generalizability_loss = f"generalizability_loss_{metric}"
    comparative_loss = f"comparative_generalizability_loss_{metric}"
    required = {
        "model_name",
        "model_instance",
        "trained_on",
        "training_size",
        "scope",
        "dataset",
        "statistic",
        "test_row_count",
        metric,
    }
    if include_ci:
        required.update({"ci_level", ci_lower, ci_upper})
    if include_generalizability:
        required.update({generalizability_loss, comparative_loss})

    missing_columns = sorted(required - set(results.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    table_data = results.copy()
    for column, value in (filters or {}).items():
        if column not in table_data.columns:
            raise ValueError(f"Unknown filter column: {column}")
        table_data = table_data.loc[table_data[column].eq(value)]

    table_data = table_data.loc[table_data["scope"].eq("test") & table_data["statistic"].eq("point")].copy()
    if table_data.empty:
        raise ValueError("No scope='test', statistic='point' rows match the filters")

    if run_aggregation not in {None, "average", "highest", "lowest"}:
        raise ValueError("run_aggregation must be one of: average, highest, lowest")

    _validate_instance_metadata(table_data)

    trained_on_values = table_data["trained_on"].dropna().unique().tolist()
    if len(trained_on_values) != 1:
        raise ValueError(f"Expected exactly one trained_on value after filtering; found {trained_on_values}")
    trained_on = str(trained_on_values[0])

    available_datasets = table_data["dataset"].dropna().astype(str).unique().tolist()
    if dataset_order is None:
        datasets = ([trained_on] if trained_on in available_datasets else []) + [
            dataset for dataset in available_datasets if dataset != trained_on
        ]
    else:
        datasets = list(dataset_order)
        unknown_datasets = sorted(set(datasets) - set(available_datasets))
        if unknown_datasets:
            raise ValueError(f"Datasets not present after filtering: {', '.join(unknown_datasets)}")
    table_data = table_data.loc[table_data["dataset"].astype(str).isin(datasets)]

    if include_ci:
        _validate_ci_level(table_data["ci_level"])

    for dataset in datasets:
        row_counts = table_data.loc[table_data["dataset"].astype(str).eq(dataset), "test_row_count"].dropna()
        _validate_numeric_values(row_counts, "test_row_count", allow_missing=False)
        if len(row_counts.unique()) > 1:
            raise ValueError(f"Inconsistent test_row_count values for {dataset}")

    table_data = _aggregate_pipeline_runs(
        table_data,
        metric=metric,
        numeric_columns=[
            *([ci_lower, ci_upper] if include_ci else []),
            *([generalizability_loss, comparative_loss] if include_generalizability else []),
        ],
        run_aggregation=run_aggregation,
    )

    instance_metadata = table_data.drop_duplicates("model_instance", keep="first")
    instance_to_model = dict(
        zip(
            instance_metadata["model_instance"].astype(str),
            instance_metadata["model_name"].astype(str),
            strict=True,
        )
    )
    available_models = list(dict.fromkeys(instance_to_model.values()))
    ordered_names = _ordered_model_names(available_models, model_order)
    model_rank = {name: index for index, name in enumerate(ordered_names)}
    instances = list(instance_to_model)
    instances.sort(key=lambda instance: model_rank[instance_to_model[instance]])

    counts = table_data.groupby(["model_instance", "dataset"], dropna=False).size()
    duplicates = counts[counts > 1]
    if not duplicates.empty:
        combinations = ", ".join(f"{instance}/{dataset}" for instance, dataset in duplicates.index.tolist())
        raise ValueError(
            "Multiple rows exist for these model_instance/dataset combinations; "
            "add filters or set run_aggregation: " + combinations
        )

    observed = set(zip(table_data["model_instance"].astype(str), table_data["dataset"].astype(str)))
    missing_pairs = [
        f"{instance}/{dataset}" for instance in instances for dataset in datasets if (instance, dataset) not in observed
    ]
    if missing_pairs:
        raise ValueError("Missing model/dataset evaluation rows: " + ", ".join(missing_pairs))

    indexed = table_data.assign(
        model_instance=table_data["model_instance"].astype(str),
        dataset=table_data["dataset"].astype(str),
    ).set_index(["model_instance", "dataset"])
    scale = 100 if percentage else 1
    dataset_labels = dict(dataset_labels or {})
    model_labels = dict(model_labels or {})

    def display_dataset(dataset: str) -> str:
        return _escape_latex(dataset_labels.get(dataset, dataset.upper()))

    model_name_counts = pd.Series(instance_to_model.values()).value_counts().to_dict()

    def display_model(instance: str) -> str:
        model_name = instance_to_model[instance]
        label = instance if model_name_counts[model_name] > 1 else model_labels.get(model_name, model_name)
        return _escape_latex(label)

    metric_values = {
        (instance, dataset): _as_number(indexed.loc[(instance, dataset), metric], metric, instance, dataset) * scale
        for instance in instances
        for dataset in datasets
    }
    best = max if metric != "rmse" else min
    best_metric_values = {
        dataset: best(metric_values[(instance, dataset)] for instance in instances) for dataset in datasets
    }

    def format_cell(instance: str, dataset: str) -> str:
        row = indexed.loc[(instance, dataset)]
        value = metric_values[(instance, dataset)]

        value_format = f"{value:.{decimals}f}"
        if bold_best and value == best_metric_values[dataset]:
            value_format = f"\\textbf{{{value_format}}}"

        if not include_ci:
            return value_format

        lower = _as_number(row[ci_lower], ci_lower, instance, dataset) * scale
        upper = _as_number(row[ci_upper], ci_upper, instance, dataset) * scale

        formatted = f"{value_format} [{lower:.{decimals}f}, {upper:.{decimals}f}]"
        return formatted

    delta_values: dict[str, dict[str, float]] = {}
    best_delta_values: dict[str, float] = {}
    if include_generalizability:
        for column in (generalizability_loss, comparative_loss):
            delta_values[column] = {}
            for instance in instances:
                raw_values = pd.to_numeric(
                    table_data.loc[table_data["model_instance"].astype(str).eq(instance), column],
                    errors="coerce",
                ).dropna()
                unique_values = raw_values.unique().tolist()
                if len(unique_values) != 1:
                    raise ValueError(f"Expected one non-null {column} value for {instance}; found {unique_values}")
                delta_values[column][instance] = float(unique_values[0]) * scale
            best_delta_values[column] = max(delta_values[column].values())

    def format_delta(instance: str, column: str) -> str:
        value = delta_values[column][instance]
        formatted = f"{value:.{decimals}f}"
        if bold_best and value == best_delta_values[column]:
            formatted = f"\\textbf{{{formatted}}}"
        return formatted

    headers = []
    for dataset in datasets:
        row_counts = table_data.loc[table_data["dataset"].astype(str).eq(dataset), "test_row_count"].dropna()
        unique_counts = row_counts.unique().tolist()
        if len(unique_counts) > 1:
            raise ValueError(f"Inconsistent test_row_count values for {dataset}")
        count_suffix = f" \\\\ ($N = {_format_count(unique_counts[0])}$)" if unique_counts else ""
        headers.append(f"\\makecell{{{display_dataset(dataset)}{count_suffix}}}")

    extra_column_count = 2 if include_generalizability else 0
    numeric_column_count = len(datasets) + extra_column_count
    column_spec = "@{}cl" + ("c" * numeric_column_count) + "@{}"
    first_header = (
        r"    \multirow{2}{*}{\rotatebox[origin=c]{90}{\makecell{\textbf{Train} \\ \textbf{Set}}}}"
        r" & \multirow{4}{*}{\textbf{Model}}"
        f" & \\multicolumn{{{len(datasets)}}}{{c}}{{\\textbf{{Test Set}}}}"
    )
    second_header = "    & & " + " & ".join(headers)
    if include_generalizability:
        second_header += r" & {$\Delta_{\mathrm{spec}}$} & {$\Delta_{\mathrm{comp}}$}"
    first_header += r" \\"
    second_header += r" \\"

    lines = [
        r"\begin{table}[h]",
        r"    \centering",
        r"    \resizebox{" + f"{line_width_percent}" + r"\linewidth}{!}{%",
        f"    \\begin{{tabular}}{{{column_spec}}}",
        first_header,
    ]
    lines.extend(
        [
            second_header,
            # r"    \midrule",
        ]
    )

    training_sizes = pd.to_numeric(table_data["training_size"], errors="coerce").dropna()
    unique_training_sizes = training_sizes.unique().tolist()
    if len(unique_training_sizes) != 1:
        raise ValueError(f"Expected one training_size after filtering; found {unique_training_sizes}")
    train_label = f"\\makecell{{{display_dataset(trained_on)} \\\\ ($N = {_format_count(unique_training_sizes[0])}$)}}"
    for index, instance in enumerate(instances):
        # offset by 1 to not fk up midrule
        train_cell = (
            f"\\multirow{{{len(instances) + 1}}}{{*}}{{\\rotatebox[origin=c]{{90}}{{{train_label}}}}}"
            if index == 0
            else ""
        )
        cells = []
        for dataset in datasets:
            cell = format_cell(instance, dataset)
            if shade_external and dataset != trained_on:
                cell = f"\\cellcolor{{{external_color}}} {cell}"
            cells.append(cell)
        if include_generalizability:
            cells.extend(
                [
                    format_delta(instance, generalizability_loss),
                    format_delta(instance, comparative_loss),
                ]
            )
        lines.append(f"    {train_cell} & {display_model(instance)} & " + " & ".join(cells) + r" \\")
        if instance_to_model[instance] == "xgboost":
            lines.append(r"    \cmidrule(lr){2-3}")

    lines.extend(
        [
            r"    \bottomrule",
            r"    \end{tabular}%",
            r"    }",
        ]
    )
    if include_caption:
        metric_label = _escape_latex(metric.replace("_", " ").upper())
        if caption is None:
            interval = " with confidence intervals" if include_ci else ""
            caption = f"{metric_label} performance{interval} for models trained on {display_dataset(trained_on)}."
        if include_ci is False:
            caption = caption.replace(r" [95\% CI]", "")
        lines.append(f"    \\caption{{{caption}}}")
        if label:
            lines.append(f"    \\label{{{_escape_latex(label)}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def _as_number(value: object, column: str, model: str, dataset: str) -> float:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        raise ValueError(f"Invalid {column} value for {model}/{dataset}: {value!r}")
    return float(number)


def _validate_instance_metadata(table_data: pd.DataFrame) -> None:
    """Reject instance identities that span incompatible table settings."""
    for column in ("model_instance", "model_name"):
        if table_data[column].isna().any():
            raise ValueError(f"{column} must not contain missing values")

    invariant_columns = ["model_name", "trained_on", "training_size"]
    invariant_columns.extend(
        column for column in ("target", "task_type", "train_sources") if column in table_data.columns
    )
    for instance, rows in table_data.groupby("model_instance", sort=False, dropna=False):
        conflicts = [column for column in invariant_columns if rows[column].nunique(dropna=False) > 1]
        if conflicts:
            raise ValueError(f"Conflicting metadata for model_instance {instance!r}: {', '.join(conflicts)}")


def _aggregate_pipeline_runs(
    table_data: pd.DataFrame,
    *,
    metric: str,
    numeric_columns: Sequence[str],
    run_aggregation: Literal["average", "highest", "lowest"] | None,
) -> pd.DataFrame:
    """Apply the requested aggregation independently to each test dataset."""
    table_data = table_data.reset_index(drop=True).copy()
    _validate_numeric_values(table_data[metric], metric, allow_missing=False)
    table_data[metric] = pd.to_numeric(table_data[metric], errors="coerce")

    if run_aggregation is None:
        _convert_requested_columns(table_data, numeric_columns)
        return table_data

    group_columns = ["model_instance", "dataset"]
    if run_aggregation == "average":
        _convert_requested_columns(table_data, numeric_columns)
        averaged_columns = [metric, *numeric_columns]
        means = table_data.groupby(group_columns, sort=False, dropna=False)[averaged_columns].mean()
        aggregated = table_data.drop_duplicates(group_columns, keep="first").set_index(group_columns)
        aggregated.loc[:, averaged_columns] = means
        return aggregated.reset_index()

    groups = table_data.groupby(group_columns, sort=False, dropna=False)[metric]
    # idxmax/idxmin preserve the first input row on ties; do not sort by run name.
    selected_indices = groups.idxmax() if run_aggregation == "highest" else groups.idxmin()
    selected = table_data.loc[selected_indices.to_list()].reset_index(drop=True)
    _convert_requested_columns(selected, numeric_columns)
    return selected


def _convert_requested_columns(table_data: pd.DataFrame, numeric_columns: Sequence[str]) -> None:
    for column in numeric_columns:
        allow_missing = column.startswith(("generalizability_loss_", "comparative_generalizability_loss_"))
        _validate_numeric_values(table_data[column], column, allow_missing=allow_missing)
        table_data[column] = pd.to_numeric(table_data[column], errors="coerce")


def _validate_numeric_values(values: pd.Series, column: str, *, allow_missing: bool) -> None:
    converted = pd.to_numeric(values, errors="coerce")
    invalid = values.notna() & converted.isna()
    if invalid.any():
        bad_value = values.loc[invalid].iloc[0]
        raise ValueError(f"Invalid numeric value in {column}: {bad_value!r}")
    if not allow_missing and converted.isna().any():
        raise ValueError(f"Missing numeric value in {column}")


def _validate_ci_level(values: pd.Series) -> None:
    _validate_numeric_values(values, "ci_level", allow_missing=True)
    unique_levels = pd.to_numeric(values, errors="coerce").dropna().unique().tolist()
    if len(unique_levels) != 1:
        raise ValueError(f"Expected one non-null ci_level after filtering; found {unique_levels}")


def _ordered_model_names(available: Sequence[str], model_order: Sequence[str] | None) -> list[str]:
    """Apply canonical/configured ordering and append unknown names stably."""
    if model_order is None:
        return list(dict.fromkeys(available))
    present = set(available)
    ordered = [name for name in model_order if name in present]
    ordered_set = set(ordered)
    ordered.extend(name for name in dict.fromkeys(available) if name not in ordered_set)
    return ordered


def _format_count(value: object) -> str:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return _escape_latex(str(value))
    return f"{int(number):,}"


def _escape_latex(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in str(value))


def multiple_latex_tables(results: list[pd.DataFrame], names: list[str], kwargs: dict[str, Any]) -> str:
    """Combine several performance tables into one shared table assembly.

    Each DataFrame is rendered through :func:`performance_table_to_latex`
    with the same ``kwargs``, so every table shares the same metric,
    formatting, and toggles (confidence intervals, generalizability columns,
    external-test shading, run aggregation, ...). The rendered tables stay
    separate ``table`` environments and are concatenated in input order, each
    with a dedicated subcaption ``(a) name1``, ``(b) name2``, ... taken from
    ``names``.

    Only one main caption is emitted: it is placed below the last table.
    ``kwargs["caption"]`` and ``kwargs["label"]`` are hoisted from the shared
    options to that single caption; when ``caption`` is omitted it is
    generated from ``names``.
    """
    if not results:
        raise ValueError("At least one results DataFrame is required")
    if len(names) != len(results):
        raise ValueError(
            f"Expected one name per results DataFrame; got {len(names)} names for {len(results)} DataFrames"
        )
    if len(names) != len(set(names)):
        raise ValueError("names must be unique")
    if len(results) > 26:
        raise ValueError("At most 26 tables are supported")
    for index, table in enumerate(results):
        if not isinstance(table, pd.DataFrame):
            raise ValueError(f"results[{index}] must be a pandas DataFrame")

    shared = dict(kwargs)
    if "include_caption" in shared:
        raise ValueError("'include_caption' is not supported in multiple_latex_tables; it is managed by the assembly")
    caption = shared.pop("caption", None)
    label = shared.pop("label", None)
    if caption is None:
        caption = "Model performance across: " + ", ".join(_escape_latex(name) for name in names) + "."

    tables = []
    for index, (table, name) in enumerate(zip(results, names, strict=True)):
        last = index == len(results) - 1
        rendered = performance_table_to_latex(
            table,
            include_caption=last,
            caption=caption,
            label=label,
            **shared,
        )
        lines = rendered.splitlines()
        subcaption = f"    \\textbf{{({chr(ord('a') + index)}) {_escape_latex(name)}}}"
        if last:
            insert_at = next(i for i, line in enumerate(lines) if line.lstrip().startswith(r"\caption"))
        else:
            insert_at = lines.index(r"\end{table}")
        lines[insert_at:insert_at] = [r"    \par\vspace{0.25em}", subcaption]
        tables.append("\n".join(lines))

    return "\n\n".join(tables)
