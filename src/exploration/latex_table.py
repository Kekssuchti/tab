"""Create LaTeX performance tables from pipeline evaluation results."""

from collections.abc import Mapping, Sequence

import pandas as pd

CAPTION = "Model performance (AUROC [95\% CI]) for the three ICU classification tasks across train and test set combinations. Gray fields show scenarios with external test sets. $\Delta_{\mathrm{spec}}$ shows model-specific and and $\Delta_{\mathrm{comp}}$ comparative generalizability loss"
DATASET_ORDER = ["tudd", "mimic"]

MODEL_ORDER = [
    "logistic-regression",
    "ebm",
    "xgboost",
    "tabpfn-2.5",
    "tabpfn-2.6",
    "tabpfn-3",
    "tabicl-2",
    "limix-2m",
    "limix-16m",
    "mitra",
    "orion-msp",
    "orion-bix",
    "tabswift",
    "tabfm",
]

DATASET_MAP = {"tudd": "EUH", "mimic": "MIMIC-IV"}

MODEL_MAP = {
    "logistic-regression": "LR",
    "ebm": "EBM",
    "xgboost": "XGBoost",
    "tabpfn-2.5": "TabPFNv2.5",
    "tabpfn-2.6": "TabPFNv2.6",
    "tabpfn-3": "TabPFNv3",
    "tabicl-2": "TabICLv2",
    "tabswift": "TabSwift",
    "limix-16m": "LimiX16M",
    "limix-2m": "LimiX2M",
    "mitra": "Mitra",
    "tabfm": "TabFM",
    "orion-msp": "OrionMSP",
    "orion-bix": "OrionBIX",
}


def performance_table_to_latex(
    results: pd.DataFrame,
    metric: str = "roc_auc",
    *,
    filters: Mapping[str, object] | None = None,
    dataset_order: Sequence[str] = DATASET_ORDER,
    model_order: Sequence[str] = MODEL_ORDER,
    dataset_labels: Mapping[str, str] = DATASET_MAP,
    model_labels: Mapping[str, str] = MODEL_MAP,
    caption: str = CAPTION,
    label: str | None = None,
    decimals: int = 2,
    percentage: bool = True,
    include_ci: bool = True,
    include_generalizability: bool = True,
    bold_best: bool = True,
    shade_external: bool = True,
    external_color: str = "black!8",
) -> str:
    """Return a LaTeX table for models trained once and tested on each dataset.

    The input must contain one ``scope == "test"`` and ``statistic == "mean"``
    row per model and test dataset. By default, metric values are displayed as
    percentages with their confidence interval, for example
    ``82.22 [81.41, 83.03]``. The two generalizability columns use the
    metric-specific ``generalizability_loss_*`` and
    ``comparative_generalizability_loss_*`` values. The maximum value in each
    result column is bolded; for losses this is the value closest to zero.

    Use ``filters`` when the DataFrame contains multiple targets, experiments,
    pipeline runs, or training configurations. Ambiguous or incomplete model x
    dataset combinations raise ``ValueError`` rather than being aggregated.
    """
    ci_lower = f"{metric}_ci_lower"
    ci_upper = f"{metric}_ci_upper"
    generalizability_loss = f"generalizability_loss_{metric}"
    comparative_loss = f"comparative_generalizability_loss_{metric}"
    required = {
        "model_name",
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

    table_data = table_data.loc[table_data["scope"].eq("test") & table_data["statistic"].eq("mean")].copy()
    if table_data.empty:
        raise ValueError("No scope='test', statistic='mean' rows match the filters")

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
    if not datasets:
        raise ValueError("No test datasets are available")
    table_data = table_data.loc[table_data["dataset"].astype(str).isin(datasets)]

    available_models = table_data["model_name"].dropna().astype(str).unique().tolist()
    models = list(model_order) if model_order is not None else available_models
    unknown_models = sorted(set(models) - set(available_models))
    if unknown_models:
        # remove not available models but keep order
        models = [model for model in models if model in available_models]
    table_data = table_data.loc[table_data["model_name"].astype(str).isin(models)]

    counts = table_data.groupby(["model_name", "dataset"], dropna=False).size()
    duplicates = counts[counts > 1]
    if not duplicates.empty:
        combinations = ", ".join(f"{model}/{dataset}" for model, dataset in duplicates.index.tolist())
        raise ValueError(
            "Multiple rows exist for these model/dataset combinations; add filters "
            f"such as pipeline_id or target: {combinations}"
        )

    observed = set(zip(table_data["model_name"], table_data["dataset"]))
    missing_pairs = [
        f"{model}/{dataset}" for model in models for dataset in datasets if (model, dataset) not in observed
    ]
    if missing_pairs:
        raise ValueError("Missing model/dataset evaluation rows: " + ", ".join(missing_pairs))

    indexed = table_data.set_index(["model_name", "dataset"])
    scale = 100 if percentage else 1
    dataset_labels = dict(dataset_labels or {})
    model_labels = dict(model_labels or {})

    def display_dataset(dataset: str) -> str:
        return _escape_latex(dataset_labels.get(dataset, dataset.upper()))

    def display_model(model: str) -> str:
        return _escape_latex(model_labels.get(model, model))

    metric_values = {
        (model, dataset): _as_number(indexed.loc[(model, dataset), metric], metric, model, dataset) * scale
        for model in models
        for dataset in datasets
    }
    best_metric_values = {dataset: max(metric_values[(model, dataset)] for model in models) for dataset in datasets}

    def format_cell(model: str, dataset: str) -> str:
        row = indexed.loc[(model, dataset)]
        value = metric_values[(model, dataset)]

        value_format = f"{value:.{decimals}f}"
        if bold_best and value == best_metric_values[dataset]:
            value_format = f"\\textbf{{{value_format}}}"

        if not include_ci:
            return value_format

        lower = _as_number(row[ci_lower], ci_lower, model, dataset) * scale
        upper = _as_number(row[ci_upper], ci_upper, model, dataset) * scale

        formatted = f"{value_format} [{lower:.{decimals}f}, {upper:.{decimals}f}]"
        return formatted

    delta_values: dict[str, dict[str, float]] = {}
    best_delta_values: dict[str, float] = {}
    if include_generalizability:
        for column in (generalizability_loss, comparative_loss):
            delta_values[column] = {}
            for model in models:
                raw_values = pd.to_numeric(
                    table_data.loc[table_data["model_name"].astype(str).eq(model), column],
                    errors="coerce",
                ).dropna()
                unique_values = raw_values.unique().tolist()
                if len(unique_values) != 1:
                    raise ValueError(f"Expected one non-null {column} value for {model}; found {unique_values}")
                delta_values[column][model] = float(unique_values[0]) * scale
            best_delta_values[column] = max(delta_values[column].values())

    def format_delta(model: str, column: str) -> str:
        value = delta_values[column][model]
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
        r"    \resizebox{\linewidth}{!}{%",
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
    for index, model in enumerate(models):
        # offset by 1 to not fk up midrule
        train_cell = (
            f"\\multirow{{{len(models) + 1}}}{{*}}{{\\rotatebox[origin=c]{{90}}{{{train_label}}}}}"
            if index == 0
            else ""
        )
        cells = []
        for dataset in datasets:
            cell = format_cell(model, dataset)
            if shade_external and dataset != trained_on:
                cell = f"\\cellcolor{{{external_color}}} {cell}"
            cells.append(cell)
        if include_generalizability:
            cells.extend(
                [
                    format_delta(model, generalizability_loss),
                    format_delta(model, comparative_loss),
                ]
            )
        lines.append(f"    {train_cell} & {display_model(model)} & " + " & ".join(cells) + r" \\")
        if model == "xgboost":
            lines.append(r"    \midrule")

    metric_label = _escape_latex(metric.replace("_", " ").upper())
    if caption is None:
        interval = " with confidence intervals" if include_ci else ""
        caption = f"{metric_label} performance{interval} for models trained on {display_dataset(trained_on)}."
    if include_ci is False:
        caption = caption.replace(" [95\% CI]", "")
    lines.extend(
        [
            r"    \bottomrule",
            r"    \end{tabular}%",
            r"    }",
            f"    \\caption{{{caption}}}",
        ]
    )
    if label:
        lines.append(f"    \\label{{{_escape_latex(label)}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def _as_number(value: object, column: str, model: str, dataset: str) -> float:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        raise ValueError(f"Invalid {column} value for {model}/{dataset}: {value!r}")
    return float(number)


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
