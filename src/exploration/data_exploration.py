import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")


@app.cell
def _():
    import sys

    sys.path.append("/var/home/keks/projects/tab")

    import dtale
    import pandas as pd

    from src.config import config

    df_tudd = pd.read_csv(config.dir_data / "extracted" / "tudd_mean_100_full.csv")
    df_mimic = pd.read_csv(config.dir_data / "extracted" / "mimic4_mean_100_full.csv")

    return df_mimic, df_tudd, dtale


@app.cell
def _(df_mimic, dtale):
    dtale.show(df_mimic)


@app.cell
def _(df_tudd, dtale):
    dtale.show(df_tudd)


@app.cell
def _():
    # we make a script that turns filtered -> extracted with more rigorous rules and filterings
    return


@app.cell
def _():
    # convert mimic to metric system

    # height measures in inch
    #

    return


if __name__ == "__main__":
    app.run()
