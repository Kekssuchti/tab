import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell
def _():
    import sys

    import pandas as pd

    sys.path.append("/var/home/keks/projects/tab")

    import dtale

    from src.config import config
    from src.utils.dataset_utils import standard_preprocessing

    return config, dtale, pd, standard_preprocessing


@app.cell
def _(config, pd):
    df_tudd = pd.read_csv(config.dir_data / "extracted" / "tudd_mean_100_full.csv")

    df_mimic = pd.read_csv(config.dir_data / "extracted" / "mimic4_mean_100_full.csv")

    df_tudd_read = pd.read_csv(config.dir_data / "extracted" / "tudd_readmission.csv")

    df_mimic_read = pd.read_csv(config.dir_data / "extracted" / "mimic4_readmission.csv")
    return df_mimic, df_mimic_read, df_tudd, df_tudd_read


@app.cell
def _(df_tudd, dtale):
    dtale.show(df_tudd)
    return


@app.cell
def _(df_mimic_read, dtale):
    dtale.show(df_mimic_read)
    return


@app.cell
def _(df_tudd_read, dtale):
    dtale.show(df_tudd_read)
    return


@app.cell
def _(config, df_tudd, pd, standard_preprocessing):
    df_tudd_filter = standard_preprocessing(df_tudd, "tudd", readmission=False)

    df_tudd_filt = pd.read_csv(config.dir_data / "filtered" / "tudd_mean_100_full.csv")

    print("original:\t\t", len(df_tudd))
    print("filter paper:\t", len(df_tudd_filt))
    print("my filter:\t\t", len(df_tudd_filter))

    print(df_tudd_filt.shape)
    print(df_tudd_filter.shape)
    return df_tudd_filt, df_tudd_filter


@app.cell
def _(df_tudd):
    df_tudd["Urea+100%mean"].describe()
    return


@app.cell
def _(df_tudd_filt):
    df_tudd_filt["Urea+100%mean"].describe()
    return


@app.cell
def _(df_tudd_filter):
    df_tudd_filter["Urea+100%mean"].describe()
    return


@app.cell
def _(df_mimic):
    df_mimic["Urea+100%mean"].describe()
    return


@app.cell
def _(df_tudd_filt):
    df_tudd_filt.columns
    return


@app.cell
def _(config, df_tudd_read, pd, standard_preprocessing):
    df_tudd_read_filter = standard_preprocessing(df_tudd_read, "tudd", readmission=True)

    df_tudd_read_filt = pd.read_csv(config.dir_data / "filtered" / "tudd_readmission.csv")

    print("original: \t\t", len(df_tudd_read))
    print("filter paper: \t", len(df_tudd_read_filt))
    print("my filter: \t\t", len(df_tudd_read_filter))

    print(df_tudd_read_filt.shape)
    print(df_tudd_read_filter.shape)
    return (df_tudd_read_filter,)


@app.cell
def _(df_tudd_read_filter, dtale):
    dtale.show(df_tudd_read_filter)
    return


@app.cell
def _(df_tudd_filt, df_tudd_filter):
    missing_cols = df_tudd_filt.columns[~df_tudd_filt.columns.isin(df_tudd_filter.columns)]
    missing_cols
    return


@app.cell
def _(df_tudd_filt):
    df_tudd_filt["Sex"].unique()
    df_tudd_filt["Sex"] = (df_tudd_filt["Sex"] == "F").astype(int)
    return


@app.cell
def _(df_tudd_filt):
    df_tudd_filt.dtypes
    return


@app.cell
def _(df_tudd_filter):
    df_tudd_filter.isna()
    return


@app.cell
def _(df_mimic):
    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt

    df_mimic
    missing_per_row = df_mimic.isna().sum(axis=1)

    ax = sns.histplot(
        missing_per_row, 
        bins=np.arange(missing_per_row.max() + 2) - 0.5,
    )
    ax.set(xlabel="Number of missing features")
    plt.tight_layout()
    plt.axvline(x=15.5, color='red', linestyle=':', linewidth=2)
    plt.savefig("./plots/custom/feature_missingness_mimic.pdf")
    plt.show()

    return np, plt, sns


@app.cell
def _(df_tudd, np, plt, sns):
    missing_per_row_tudd = df_tudd.isna().sum(axis=1)

    ax2 = sns.histplot(missing_per_row_tudd, bins=np.arange(missing_per_row_tudd.max() + 2) - 0.5)
    ax2.set(xlabel="Number of missing features")
    plt.tight_layout()
    plt.axvline(x=15.5, color='red', linestyle=':', linewidth=2)
    plt.savefig("./plots/custom/feature_missingness_tudd.pdf")
    plt.show()
    return


@app.cell
def _(df_mimic_read, np, plt, sns):
    missing_per_row_read = df_mimic_read.isna().sum(axis=1)

    ax_r = sns.histplot(
        missing_per_row_read, 
        bins=np.arange(missing_per_row_read.max() + 2) - 0.5,
    )
    ax_r.set(xlabel="Number of missing features")
    plt.tight_layout()
    plt.axvline(x=15.5, color='red', linestyle=':', linewidth=2)
    plt.savefig("./plots/custom/feature_missingness_read_mimic.pdf")
    plt.show()
    return


@app.cell
def _(df_tudd_read, np, plt, sns):
    missing_per_row_read_tudd = df_tudd_read.isna().sum(axis=1)

    ax2_r = sns.histplot(missing_per_row_read_tudd, bins=np.arange(missing_per_row_read_tudd.max() + 2) - 0.5)
    ax2_r.set(xlabel="Number of missing features")
    plt.tight_layout()
    plt.axvline(x=15.5, color='red', linestyle=':', linewidth=2)
    plt.savefig("./plots/custom/feature_missingness_read_tudd.pdf")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
