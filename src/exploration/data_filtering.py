import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import os
    import sys
    from pathlib import Path

    import numpy as np
    import pandas as pd

    sys.path.append("/var/home/keks/projects/tab")

    from src.config import config
    from src.utils.dataset_utils import standard_preprocessing

    return config, pd, standard_preprocessing


@app.cell
def _(config, pd):
    df_tudd = pd.read_csv(config.dir_data / "extracted" / "tudd_mean_100_full.csv")

    df_mimic = pd.read_csv(config.dir_data / "extracted" / "mimic4_mean_100_full.csv")

    df_tudd_read = pd.read_csv(config.dir_data / "extracted" / "tudd_readmission.csv")

    df_mimic_read = pd.read_csv(
        config.dir_data / "extracted" / "mimic4_readmission.csv"
    )
    return df_tudd, df_tudd_read


@app.function
def filter_many_missing(df, threshold_row=0.5, threshold_col=0.5):
    print("shape before missing filter", df.shape)

    df_features = df.drop(columns=["mortality", "LOS"], inplace=False)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    row_null = df.isnull().sum(axis=1)
    df = df[row_null < (threshold_row * len(df.columns))]

    df = df.loc[:, df.isnull().mean() < 0.5]

    print("shape after missing filter", df.shape)
    return df


@app.cell
def _(df_tudd):
    df_tudd.isnull().mean() < 0.5
    return


@app.cell
def _(config, df_tudd, pd, standard_preprocessing):
    df_tudd_filter = standard_preprocessing(df_tudd, readmission=False)

    df_tudd_filt = pd.read_csv(config.dir_data / "filtered" / "tudd_mean_100_full.csv")

    print("original:\t\t", len(df_tudd))
    print("filter paper:\t", len(df_tudd_filt))
    print("my filter:\t\t", len(df_tudd_filter))

    print(df_tudd_filt.shape)
    print(df_tudd_filter.shape)
    return df_tudd_filt, df_tudd_filter


@app.cell
def _(config, df_tudd_read, pd, standard_preprocessing):
    df_tudd_read_filter = standard_preprocessing(df_tudd_read, readmission=False)

    df_tudd_read_filt = pd.read_csv(
        config.dir_data / "filtered" / "tudd_readmission.csv"
    )

    print("original: \t\t", len(df_tudd_read))
    print("filter paper: \t", len(df_tudd_read_filt))
    print("my filter: \t\t", len(df_tudd_read_filter))

    print(df_tudd_read_filt.shape)
    print(df_tudd_read_filter.shape)
    return


@app.cell
def _(df_tudd_filt, df_tudd_filter):
    missing_cols = df_tudd_filt.columns[
        ~df_tudd_filt.columns.isin(df_tudd_filter.columns)
    ]
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
def _(df_tudd_filt):
    df_tudd_filt["Age"] = min(df_tudd_filt["Age"], 90)
    df_tudd_filt["Age"]
    return


if __name__ == "__main__":
    app.run()
