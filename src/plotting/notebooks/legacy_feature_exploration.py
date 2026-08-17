import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import pathlib
    import sys

    import numpy as np
    import pandas as pd

    project_root = pathlib.Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    import matplotlib.pyplot as plt
    import seaborn as sns

    from src.config import config
    from src.utils.dataset_utils import remove_impossible_values

    return config, np, pathlib, pd, plt, remove_impossible_values, sns


@app.cell
def _(config, pd):
    df_tudd = pd.read_csv(config.dir_data / "filtered" / "tudd_mean_100_full.csv")

    df_mimic = pd.read_csv(config.dir_data / "filtered" / "mimic4_mean_100_full.csv")

    df_tudd_read = pd.read_csv(config.dir_data / "filtered" / "tudd_readmission.csv")

    df_mimic_read = pd.read_csv(config.dir_data / "filtered" / "mimic4_readmission.csv")

    outlier_json = config.dir_configs / "data_limits.json"
    return df_mimic, df_mimic_read, df_tudd, df_tudd_read, outlier_json


@app.cell
def _(df_mimic, df_mimic_read, df_tudd, df_tudd_read, np, outlier_json, remove_impossible_values):
    cleaned = {}
    for name, frame in {
        "mimic": df_mimic,
        "mimic_read": df_mimic_read,
        "tudd": df_tudd,
        "tudd_read": df_tudd_read,
    }.items():
        cleaned[name], removed_counts = remove_impossible_values(frame, outlier_json)
        print(removed_counts)
        cleaned[name].replace([np.inf, -np.inf], np.nan, inplace=True)
    df_mimic_cleaned = cleaned["mimic"]
    df_mimic_read_cleaned = cleaned["mimic_read"]
    df_tudd_cleaned = cleaned["tudd"]
    df_tudd_read_cleaned = cleaned["tudd_read"]
    return df_mimic_cleaned, df_mimic_read_cleaned, df_tudd_cleaned, df_tudd_read_cleaned


@app.cell
def _(config, pathlib, plt, sns):
    def plot_feature_comparision(df_mimic, df_tudd, feature, save=False, path_addition="features"):
        print(feature)
        plt.figure(figsize=(8, 5))

        for frame, color, label, alpha in (
            (df_mimic, "red", "mimic", 0.5),
            (df_tudd, "blue", "tudd", 0.3),
        ):
            sns.histplot(
                frame[feature].dropna(), bins=50, color=color, label=label, kde=True, stat="density", alpha=alpha
            )

        plt.xlabel("")
        plt.ylabel("Density")
        plt.yticks([])
        if save:
            save_dir = pathlib.Path(f"{config.dir_plots}/{path_addition}")
            save_dir.mkdir(exist_ok=True)
            plt.savefig(f"{save_dir}/{feature}_histogram.png", dpi=500)
        plt.show()

    return (plot_feature_comparision,)


@app.cell
def _(df_mimic_cleaned, df_tudd_cleaned, plot_feature_comparision):
    # "LOS3",
    # "LOS7",
    # "read48",
    # "read72",
    features = [
        "LOS",
        "mortality",
        "Sex",
        "Age",
        "Weight+100%mean",
        "Height+100%mean",
        "Ph+100%mean",
        "Temp+100%mean",
        "RR+100%mean",
        "HR+100%mean",
        "GLU+100%mean",
        "MBP+100%mean",
        "GCST+100%mean",
        "PaO2+100%mean",
        "Kreatinin+100%mean",
        "FiO2+100%mean",
        "Kalium+100%mean",
        "Natrium+100%mean",
        "Leukocyten+100%mean",
        "Thrombocyten+100%mean",
        "Bilirubin+100%mean",
        "HCO3+100%mean",
        "Lactate+100%mean",
        "Hb+100%mean",
        "Quick+100%mean",
        "PaCO2+100%mean",
        "ALAT+100%mean",
        "ASAT+100%mean",
        "Albumin+100%mean",
        "AnionGAP+100%mean",
        "Urea+100%mean",
    ]

    for feature in features:
        plot_feature_comparision(df_mimic_cleaned, df_tudd_cleaned, feature)
    return (features,)


@app.cell
def _(df_mimic_read):
    df_mimic_read.columns


@app.cell
def _(df_mimic_read_cleaned, df_tudd_read_cleaned, plot_feature_comparision):
    # "read48",
    # "read72",
    features_read = [
        "hours_to_readmit",
        "LOS",
        "Sex",
        "Age",
        "Weight+100%mean",
        "Height+100%mean",
        "Ph+100%mean",
        "Temp+100%mean",
        "RR+100%mean",
        "HR+100%mean",
        "GLU+100%mean",
        "MBP+100%mean",
        "GCST+100%mean",
        "PaO2+100%mean",
        "Kreatinin+100%mean",
        "FiO2+100%mean",
        "Kalium+100%mean",
        "Natrium+100%mean",
        "Leukocyten+100%mean",
        "Thrombocyten+100%mean",
        "Bilirubin+100%mean",
        "HCO3+100%mean",
        "Lactate+100%mean",
        "Hb+100%mean",
        "Quick+100%mean",
        "PaCO2+100%mean",
        "ALAT+100%mean",
        "ASAT+100%mean",
        "Albumin+100%mean",
        "AnionGAP+100%mean",
        "Urea+100%mean",
    ]

    for feature_read in features_read:
        plot_feature_comparision(
            df_mimic_read_cleaned,
            df_tudd_read_cleaned,
            feature_read,
            save=True,
            path_addition="features_read",
        )


@app.cell
def _(features, pd):
    def describe_features(df, dataset_name):
        print(f"\n===== {dataset_name} Dataset =====")
        print(f"{'Feature':<30} {'Missing %':>10} {'Mean':>15} {'Std Dev':>15}")
        print("-" * 75)

        for feature in features:
            if feature not in df.columns:
                print(f"{feature:<30} {'(missing from dataset)'}")
                continue

            total = len(df)
            missing = df[feature].isna().sum()
            missing_pct = 100 * missing / total

            # Handle numeric and non-numeric differently
            if pd.api.types.is_numeric_dtype(df[feature]):
                mean_val = df[feature].mean()
                std_val = df[feature].std()
                print(f"{feature:<30} {missing_pct:10.2f} {mean_val:15.3f} {std_val:15.3f}")
            else:
                print(f"{feature:<30} {missing_pct:10.2f} {'(categorical)':>15} {'':>15}")
                value_counts = df[feature].value_counts(dropna=False)
                for val, count in value_counts.items():
                    print(f"{'':<30} {val!s:<15} {count} instances")

    return (describe_features,)


@app.cell
def _(describe_features, df_mimic):
    describe_features(df_mimic, "mimic")


if __name__ == "__main__":
    app.run()
