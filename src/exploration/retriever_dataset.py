import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path
    from time import perf_counter

    import marimo as mo
    import pandas as pd

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.classes.dataset import Dataset
    from src.schemas.dataset_schemas import CustomRetrieverConfig, DataSplitConfig
    from src.utils.config_io import load_pipeline_config

    return (
        CustomRetrieverConfig,
        DataSplitConfig,
        Dataset,
        load_pipeline_config,
        mo,
        pd,
        perf_counter,
        project_root,
    )


@app.cell
def _(mo):
    mo.md("""
    # Retriever dataset

    Build and inspect a retriever dataset without training or evaluating a model.
    Edit the sizes and seed below; use the two selectors to compare retrieval strategies and distances.
    """)
    return


@app.cell
def _():
    CONFIG_PATH = "configs/pipeline/retriever.yaml"
    RETRIEVER_TRAIN_SIZE = 800
    TEST_ON = (("tudd", 100),)
    SAMPLE_SEED = 1337
    DIVERSITY_POOL_MULTIPLIER = 2.0
    DIVERSITY_CLUSTERS = 32
    return (
        CONFIG_PATH,
        DIVERSITY_CLUSTERS,
        DIVERSITY_POOL_MULTIPLIER,
        RETRIEVER_TRAIN_SIZE,
        SAMPLE_SEED,
        TEST_ON,
    )


@app.cell
def _(mo):
    selection_strategy = mo.ui.dropdown(
        options=["random", "knn", "knn-diverse"],
        value="knn",
        label="Selection strategy",
    )
    distance_metric = mo.ui.dropdown(
        options=["euclidean", "manhattan"],
        value="euclidean",
        label="KNN distance",
    )
    mo.hstack([selection_strategy, distance_metric], justify="start")
    return distance_metric, selection_strategy


@app.cell
def _(
    CONFIG_PATH,
    CustomRetrieverConfig,
    DIVERSITY_CLUSTERS,
    DIVERSITY_POOL_MULTIPLIER,
    DataSplitConfig,
    RETRIEVER_TRAIN_SIZE,
    TEST_ON,
    distance_metric,
    load_pipeline_config,
    project_root,
    selection_strategy,
):
    _pipeline_config = load_pipeline_config(project_root / CONFIG_PATH)
    retriever_config = CustomRetrieverConfig(
        train_size=RETRIEVER_TRAIN_SIZE,
        test_on=tuple(DataSplitConfig(dataset=origin, fraction=size) for origin, size in TEST_ON),
        selection_strategy=selection_strategy.value,
        distance_metric=distance_metric.value,
        diversity_pool_multiplier=DIVERSITY_POOL_MULTIPLIER,
        diversity_clusters=DIVERSITY_CLUSTERS,
    )
    original_dataset_config = _pipeline_config.dataset.model_copy(update={"custom_retriever": None})
    dataset_config = _pipeline_config.dataset.model_copy(update={"custom_retriever": retriever_config})
    return dataset_config, original_dataset_config, retriever_config


@app.cell
def _(
    Dataset,
    SAMPLE_SEED,
    dataset_config,
    original_dataset_config,
    perf_counter,
):
    _original_started_at = perf_counter()
    _original_dataset = Dataset(original_dataset_config, sample_seed=SAMPLE_SEED)
    original_data = _original_dataset.get_dataset()
    original_build_seconds = perf_counter() - _original_started_at

    _retriever_started_at = perf_counter()
    _retriever_dataset = Dataset(dataset_config, sample_seed=SAMPLE_SEED)
    data = _retriever_dataset.get_dataset()
    retriever_build_seconds = perf_counter() - _retriever_started_at

    if data.test_retriever is None:
        raise RuntimeError("Retriever dataset was not created")
    return data, original_build_seconds, original_data, retriever_build_seconds


@app.cell
def _(
    data,
    original_build_seconds,
    original_data,
    pd,
    retriever_build_seconds,
    retriever_config,
):
    def _summarize_part(name, part):
        target_counts = part.y.value_counts(dropna=False).sort_index()
        target_summary = (
            target_counts.to_dict()
            if len(target_counts) <= 10
            else {
                "mean": float(part.y.mean()),
                "std": float(part.y.std()),
            }
        )
        return {
            "part": name,
            "rows": len(part.X),
            "features": part.X.shape[1],
            "missing_fraction": float(part.X.isna().mean().mean()),
            "target": target_summary,
        }

    def _feature_stats(frame, prefix):
        stats = frame.describe().T[["mean", "std", "50%"]].rename(columns={"50%": "median"})
        stats["missing_fraction"] = frame.isna().mean()
        return stats.add_prefix(f"{prefix}_")

    settings_table = pd.DataFrame(
        [
            {
                "strategy": retriever_config.selection_strategy,
                "distance": retriever_config.distance_metric,
                "train_size": retriever_config.train_size,
                "test_on": [(split.dataset, split.fraction) for split in retriever_config.test_on],
                "diversity_pool_multiplier": retriever_config.diversity_pool_multiplier,
                "diversity_clusters": retriever_config.diversity_clusters,
                "original_build_seconds": original_build_seconds,
                "retriever_build_seconds": retriever_build_seconds,
            }
        ]
    )
    parts_table = pd.DataFrame(
        [
            _summarize_part("original_train_pool", original_data.train_data),
            _summarize_part("retrieved_train", data.train_data),
            _summarize_part("retriever_test", data.test_retriever),
        ]
    )

    feature_comparison = (
        _feature_stats(original_data.train_data.X, "pool")
        .join(_feature_stats(data.train_data.X, "selected"))
        .join(_feature_stats(data.test_retriever.X, "test"))
        .reset_index(names="feature")
    )
    return feature_comparison, parts_table, settings_table


@app.cell
def _(feature_comparison, mo, parts_table, settings_table):
    mo.vstack(
        [
            mo.md("## Result"),
            settings_table,
            parts_table,
            mo.md("## Original pool vs. retrieved training vs. target-test features"),
            feature_comparison,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
