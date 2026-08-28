"""Create a LaTeX descriptive-statistics table from filtered cohort data."""

from pathlib import Path
from typing import Literal

import pandas as pd

from src.config import config

_FEATURES = (
    ("Age", "Age"),
    ("Weight+100%mean", "Weight"),
    ("Height+100%mean", "Height"),
    ("BMI", "BMI"),
    ("Temp+100%mean", "Temperature"),
    ("RR+100%mean", "Respiratory Rate"),
    ("HR+100%mean", "Heart Rate"),
    ("GLU+100%mean", "Glucose"),
    ("MBP+100%mean", "Mean Blood Pressure"),
    ("Ph+100%mean", "Potential Hydrogen"),
    ("GCST+100%mean", "Glasgow Coma Scale Total"),
    ("Sex", "Gender (Female \\%)"),
    ("PaO2+100%mean", "Partial Pressure of O2"),
    ("FiO2+100%mean", "Fraction of Inspired O2"),
    ("Kalium+100%mean", "Potassium"),
    ("Natrium+100%mean", "Sodium"),
    ("Leukocyten+100%mean", "Leukocytes"),
    ("Thrombocyten+100%mean", "Thrombocytes (Platelets)"),
    ("Bilirubin+100%mean", "Bilirubin"),
    ("HCO3+100%mean", "Bicarbonate"),
    ("Hb+100%mean", "Hemoglobin"),
    ("Quick+100%mean", "Prothrombin Time"),
    ("ASAT+100%mean", "Aspartate Aminotransferase"),
    ("ALAT+100%mean", "Alanine Aminotransferase"),
    ("PaCO2+100%mean", "Partial Pressure of CO2"),
    ("Albumin+100%mean", "Albumin"),
    ("AnionGAP+100%mean", "Anion Gap"),
    ("Lactate+100%mean", "Lactate"),
    ("Urea+100%mean", "Urea Nitrogen"),
    ("Kreatinin+100%mean", "Creatinine"),
)

readmission_extra_feature = (("LOS", "Length of Stay (h)"),)


def descriptive_statistics_table_to_latex(
    filtered_dir: str | Path | None = None,
    type: Literal["normal", "readmission"] = "normal",
) -> str:
    """Return a ready-to-paste LaTeX table for the filtered MIMIC-IV and EUH data.

    Continuous features are reported as mean, sample standard deviation, and
    percentage missing. ``Sex == 1`` is reported as the female percentage.
    Outcome percentages are mortality, length of stay over seven days, and
    readmission within 72 hours. Readmission is calculated from the separately
    filtered readmission cohorts.

    The table uses the LaTeX ``booktabs`` and ``graphicx`` packages.
    """
    data_dir = Path(filtered_dir) if filtered_dir is not None else config.dir_data / "filtered"
    files = {
        "mimic": data_dir / "mimic4_mean_100_full.csv",
        "tudd": data_dir / "tudd_mean_100_full.csv",
        "mimic_readmission": data_dir / "mimic4_readmission.csv",
        "tudd_readmission": data_dir / "tudd_readmission.csv",
    }
    data = {name: pd.read_csv(path) for name, path in files.items()}

    data = _add_bmi(data)

    required_features = {column for column, _ in _FEATURES} | {"mortality", "LOS"}
    for dataset in ("mimic", "tudd"):
        missing = sorted(required_features - set(data[dataset].columns))
        if missing:
            raise ValueError(f"{files[dataset]} is missing required columns: {', '.join(missing)}")

    def format_number(value: float) -> str:
        return "--" if pd.isna(value) else f"{value:.2f}"

    def feature_cells(dataset: str, column: str) -> list[str]:
        values = pd.to_numeric(data[dataset][column], errors="coerce")
        missing_percentage = values.isna().mean() * 100
        if column == "Sex":
            return [
                format_number(values.mean() * 100),
                "--",
                f"{missing_percentage:.2f}",
            ]
        return [
            format_number(values.mean()),
            format_number(values.std()),
            f"{missing_percentage:.2f}",
        ]

    rows = []
    if type == "normal":
        data_tudd_name = "tudd"
        data_mimic_name = "mimic"
    else:
        data_tudd_name = "tudd_readmission"
        data_mimic_name = "mimic_readmission"

    if type == "readmission":
        features = _FEATURES + readmission_extra_feature
    else:
        features = _FEATURES

    for column, label in features:
        cells = feature_cells(data_mimic_name, column) + feature_cells(data_tudd_name, column)
        rows.append(f"        {label} & " + " & ".join(cells) + r" \\")

    outcomes = (
        (
            "Mortality (\\%)",
            data[data_mimic_name]["mortality"].mean() * 100,
            data[data_tudd_name]["mortality"].mean() * 100,
        ),
        (
            "Length of stay \\textgreater 7 days (\\%)",
            data[data_mimic_name]["LOS"].gt(7 * 24).mean() * 100,
            data[data_tudd_name]["LOS"].gt(7 * 24).mean() * 100,
        ),
        (
            "Readmission (\\%)",
            data["mimic_readmission"]["hours_to_readmit"].le(3 * 24).mean() * 100,
            data["tudd_readmission"]["hours_to_readmit"].le(3 * 24).mean() * 100,
        ),
    )
    for label, mimic_value, tudd_value in outcomes:
        rows.append(f"        {label} & {mimic_value:.2f} & -- & 0.00 & {tudd_value:.2f} & -- & 0.00" + r" \\")

    lines = [
        r"\begin{table}[htbp]",
        r"    \centering",
        r"    \resizebox{\linewidth}{!}{%",
        r"    \begin{tabular}{lrrrrrr}",
        r"        \toprule",
        r"        & \multicolumn{3}{c}{\textbf{MIMIC-IV}} & \multicolumn{3}{c}{\textbf{EUH}} \\",
        r"        \cmidrule(lr){2-4} \cmidrule(lr){5-7}",
        r"        \textbf{Feature} & Mean & Std & Missing (\%) & Mean & Std & Missing (\%) \\",
        r"        \midrule",
        *rows,
        r"        \bottomrule",
        r"    \end{tabular}%",
        r"    }",
        r"    \caption{Descriptive statistics of the filtered MIMIC-IV and EUH cohorts. Length of stay denotes stays longer than seven days; readmission denotes readmission within 72 hours.}",
        r"    \label{tab:cohort-descriptive-statistics}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def single_df_descriptive_statistics_table_to_latex(
    name: Literal["mimic", "tudd"],
    filtered_dir: str | Path | None = None,
    type: Literal["normal", "readmission"] = "normal",
) -> str:
    """Return a LaTeX descriptive-statistics table for one filtered cohort.

    Feature and mortality/length-of-stay statistics use the selected cohort
    variant. Readmission is calculated from the source's separately filtered
    readmission data. The table requires the LaTeX ``booktabs`` and
    ``graphicx`` packages.
    """
    data_dir = Path(filtered_dir) if filtered_dir is not None else config.dir_data / "filtered"
    if name == "mimic":
        source_label = "MIMIC-IV"
        files = {
            "normal": data_dir / "mimic4_mean_100_full.csv",
            "readmission": data_dir / "mimic4_readmission.csv",
        }
    else:
        source_label = "EUH"
        files = {
            "normal": data_dir / "tudd_mean_100_full.csv",
            "readmission": data_dir / "tudd_readmission.csv",
        }

    data = {name: pd.read_csv(path) for name, path in files.items()}

    data = _add_bmi(data)

    def format_number(value: float) -> str:
        return "--" if pd.isna(value) else f"{value:.2f}"

    def feature_cells(dataset: str, column: str) -> list[str]:
        values = pd.to_numeric(data[dataset][column], errors="coerce")
        missing_percentage = values.isna().mean() * 100
        if column == "Sex":
            return [
                format_number(values.mean() * 100),
                "--",
                f"{missing_percentage:.2f}",
            ]
        return [
            format_number(values.mean()),
            format_number(values.std()),
            f"{missing_percentage:.2f}",
        ]

    rows = []

    if type == "readmission":
        features = _FEATURES + readmission_extra_feature
    else:
        features = _FEATURES

    for column, label in features:
        cells = feature_cells(type, column)
        rows.append(f"        {label} & " + " & ".join(cells) + r" \\")

    if type == "readmission":
        outcomes = (
            (
                "Readmission (\\%)",
                data["readmission"]["hours_to_readmit"].le(3 * 24).mean() * 100,
            ),
        )
    else:
        outcomes = (
            (
                "Mortality (\\%)",
                data[type]["mortality"].mean() * 100,
            ),
            (
                "Length of stay \\textgreater 7 days (\\%)",
                data[type]["LOS"].gt(7 * 24).mean() * 100,
            ),
            (
                "Readmission (\\%)",
                data["readmission"]["hours_to_readmit"].le(3 * 24).mean() * 100,
            ),
        )

    for label, value in outcomes:
        rows.append(f"        {label} & {value:.2f} & -- & 0.00" + r" \\")

    lines = [
        r"\begin{table}[htbp]",
        r"    \centering",
        r"    \resizebox{\linewidth}{!}{%",
        r"    \begin{tabular}{lrrr}",
        r"        \toprule",
        f"        & \\multicolumn{{3}}{{c}}{{\\textbf{{{source_label}}}}} " + r"\\",
        r"        \cmidrule(lr){2-4}",
        r"        \textbf{Feature} & Mean & Std & Missing (\%) \\",
        r"        \midrule",
        *rows,
        r"        \bottomrule",
        r"    \end{tabular}%",
        r"    }",
        f"    \\caption{{Descriptive statistics of the filtered {source_label} {type} cohort. Length of stay denotes stays longer than seven days; readmission denotes readmission within 72 hours.}}",
        f"    \\label{{tab:cohort-descriptive-statistics-{name}-{type}}}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def _add_bmi(data: dict) -> dict:
    for name, df in data.items():
        if "Height+100%mean" in df.columns and "Weight+100%mean" in df.columns:
            df["BMI"] = df["Weight+100%mean"] / (df["Height+100%mean"] / 100) ** 2
    return data


if __name__ == "__main__":
    print(descriptive_statistics_table_to_latex())
    print(single_df_descriptive_statistics_table_to_latex("tudd", type="normal"))
