"""Create a LaTeX descriptive-statistics table from filtered cohort data."""

from pathlib import Path

import pandas as pd

from src.config import config

_FEATURES = (
    ("Age", "Age"),
    ("Weight+100%mean", "Weight"),
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


def descriptive_statistics_table_to_latex(
    filtered_dir: str | Path | None = None,
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
    for column, label in _FEATURES:
        cells = feature_cells("mimic", column) + feature_cells("tudd", column)
        rows.append(f"        {label} & " + " & ".join(cells) + r" \\")

    outcomes = (
        (
            "Mortality (\\%)",
            data["mimic"]["mortality"].mean() * 100,
            data["tudd"]["mortality"].mean() * 100,
        ),
        (
            "Length of stay \\textgreater 7 days (\\%)",
            data["mimic"]["LOS"].gt(7 * 24).mean() * 100,
            data["tudd"]["LOS"].gt(7 * 24).mean() * 100,
        ),
        (
            "Readmission (\\%)",
            (1 - data["mimic_readmission"]["hours_to_readmit"].isna().mean()) * 100,
            (1 - data["tudd_readmission"]["hours_to_readmit"].isna().mean()) * 100,
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


if __name__ == "__main__":
    print(descriptive_statistics_table_to_latex())
