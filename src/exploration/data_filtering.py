import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import sys
    import os
    from pathlib import Path
    import pandas as pd
    import numpy as np

    sys.path.append("/var/home/keks/projects/tab")

    from src.config import config
    from src.utils.datset_utils import standart_preprocessing

    return config, pd, standart_preprocessing


@app.cell
def _(config, pd):
    df_tudd = pd.read_csv(config.dir_data / "extracted" / "tudd_mean_100_full.csv")

    df_mimic = pd.read_csv(config.dir_data / "extracted" / "mimic4_mean_100_full.csv")

    df_tudd_read = pd.read_csv(config.dir_data / "extracted" / "tudd_readmission.csv")

    df_mimic_read = pd.read_csv(config.dir_data / "extracted" / "mimic4_readmission.csv")
    return (df_tudd,)


@app.function
def filter_many_missing(df, threshold_row=0.5, threshold_col=0.5): 
    print("shape before missing filter", df.shape)

    df_features = df.drop(columns=["mortality", "LOS"], inplace=False)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    row_null = df.isnull().sum(axis=1)
    df = df[row_null < (threshold_row*len(df.columns))]
        
    df = df.loc[:, df.isnull().mean() < 0.5]
    
    print("shape after missing filter", df.shape)
    return df


@app.cell
def _(df_tudd):
    df_tudd.isnull().mean() < 0.5
    return


@app.cell
def _(df_tudd, standart_preprocessing):
    df_tudd_filter = standart_preprocessing(df_tudd)
    df_tudd_filter = filter_many_missing(df_tudd_filter)

    df_tudd_filter
    return (df_tudd_filter,)


@app.cell
def _(config, df_tudd, df_tudd_filter, pd):
    df_tudd_filt = pd.read_csv(config.dir_data / "filtered" / "tudd_mean_100_full.csv")

    print(len(df_tudd))
    print(len(df_tudd_filt))
    print(len(df_tudd_filter))
    return


if __name__ == "__main__":
    app.run()
