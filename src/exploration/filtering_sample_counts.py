"""Print the pipeline's sample flow without training models or replacing data files.

Run from the repository root:
    uv run python -m src.exploration.filtering_sample_counts -c baseline
    uv run python -m src.exploration.filtering_sample_counts -c baseline --target hours_to_readmit_72

Always rebuild filtering from extracted data in memory (even if cached filtered
CSVs exist). Raw-to-extracted processing is not implemented in DataCleaner, so
this report starts at extracted. Only aggregate counts are printed.
"""

import argparse
from contextlib import ExitStack, contextmanager
from functools import wraps
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.classes.data_registry import TARGET_DEFINITIONS
from src.classes.dataset import Dataset
from src.config import config
from src.utils import dataset_utils
from src.utils.config_io import load_pipeline_config


def print_stage(label: str, df: pd.DataFrame, before: int | None = None) -> None:
    removed = "" if before is None else f"  removed={before - len(df):>7,}"
    print(f"  {label:<42} samples={len(df):>7,}  columns={df.shape[1]:>3}{removed}")


def observe_stage(function, label):
    """Observe real preprocessing calls, rather than duplicating their rules."""

    @wraps(function)
    def observed(df, *args, **kwargs):
        before = len(df)
        result = function(df, *args, **kwargs)
        frame = result[0] if isinstance(result, tuple) else result
        print_stage(label, frame, before)
        if isinstance(result, tuple):
            print(f"    Values replaced with missing: {sum(result[1].values()):,} (not rows removed)")
        return result

    return observed


@contextmanager
def observe_filtering():
    """Temporary observers keep standard_preprocessing's execution order intact."""
    stages = {
        "_convert_units": "Unit conversion (TUDD urea -> BUN)",
        "remove_impossible_values": "Impossible values -> missing",
        "remove_unused_columns": "Retain configured columns",
        "_filter_reasonable_los": "LOS range filter",
        "_filter_childs": "Minimum age filter",
        "_clip_max_age": "Clip maximum age (no row removal)",
        "_clean_dtypes": "Encode Sex / filtered result",
    }
    original_convert = dataset_utils._convert_units
    original_missing = dataset_utils._filter_many_missing

    def convert(df, *args, **kwargs):
        # standard_preprocessing replaces infinities immediately before this call.
        print_stage("Infinite values -> missing", df, len(df))
        return observe_stage(original_convert, stages["_convert_units"])(df, *args, **kwargs)

    def missing(df, readmission, feature_cols, threshold_row=0.5):
        before = len(df)
        if readmission:
            survivors = df.loc[df["mortality"] == 0]
            print_stage("Readmission: retain survivors", survivors, before)
            before = len(survivors)
        result = original_missing(df, readmission, feature_cols, threshold_row)
        print_stage(f"Feature missingness <= {threshold_row:.0%}", result, before)
        return result

    with ExitStack() as stack:
        for name, label in stages.items():
            function = getattr(dataset_utils, name)
            observer = convert if name == "_convert_units" else observe_stage(function, label)
            stack.enter_context(patch.object(dataset_utils, name, observer))
        stack.enter_context(patch.object(dataset_utils, "_filter_many_missing", missing))
        yield


class FilteringExploration(Dataset):
    """Use normal Dataset splitting, with fresh filtering kept entirely in memory."""

    def _load_data(self):
        dfs = {}
        for data_file in self._task.data_files.values():
            print(f"\n{data_file.data_origin.upper()} / {data_file.dataset_kind}")
            df = pd.read_csv(config.dir_data / "extracted" / data_file.file_name)
            print_stage("Extracted input", df)
            with observe_filtering():
                # Same arguments as DataCleaner.preprocess_extracted_to_filtered.
                df = dataset_utils.standard_preprocessing(
                    df,
                    data_file.data_origin,
                    data_file.dataset_kind == "readmission",
                    self.config.data_cleaner.missing_threshold_row,
                    self.config.data_cleaner.outlier_limits_path,
                )
            # Reproduce DataCleaner's CSV write + Dataset's read (including index
            # reset and dtype inference), without touching the filtered cache.
            df = pd.read_csv(StringIO(df.to_csv(index=False)))
            before = len(df)
            df = self._runtime_preprocessing(df)
            print_stage("Runtime cleanup / age clipping", df, before)
            dfs[data_file.data_origin] = df
        return dfs

    def _split_data(self, dfs):
        origins = iter(dfs)
        original_split = self._split_single_df

        def split(df):
            origin = next(origins)
            result = original_split(df)
            X_train, X_test, _, _ = result
            print(f"\n{origin.upper()} / target={self.config.target}")
            print_stage("Target features (target columns removed)", self._task.features_from(df))
            print_stage("Available training pool", X_train)
            print_stage("Held-out test", X_test)
            selected = 0
            for contribution in self.config.train_on:
                if contribution.dataset == origin:
                    fraction = contribution.fraction
                    selected = int(fraction * len(X_train)) if isinstance(fraction, float) else fraction
            print(f"  Configured training contribution: {selected:,} samples")
            return result

        # The real implementation still handles stratification, feature alignment,
        # source sampling, concatenation, shuffling, and optional retrieval.
        with patch.object(self, "_split_single_df", split):
            return super()._split_data(dfs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "-c", "--config", default="baseline", help="Pipeline config name or YAML path (default: baseline)"
    )
    parser.add_argument(
        "--target", choices=list(TARGET_DEFINITIONS), help="Explore another target using the same settings"
    )
    args = parser.parse_args()
    path = Path(args.config)
    if path.suffix not in {".yaml", ".yml"}:
        path = config.dir_pipelines / f"{args.config}.yaml"
    pipeline_config = load_pipeline_config(path)
    dataset_config = pipeline_config.dataset
    if args.target:
        # Revalidate target-dependent settings rather than bypassing validators.
        dataset_config = type(dataset_config).model_validate({**dataset_config.model_dump(), "target": args.target})

    print(f"Config: {path}\nTarget: {dataset_config.target}")
    print("Flow: extracted -> freshly filtered -> runtime cleanup -> split -> sample")
    print("Cached filtered files are neither read nor overwritten. No model training or MLflow runs.")
    print(
        f"Split seed: {dataset_config.random_state}; sample seed: {pipeline_config.random_states.training_sample_seed}"
    )
    dataset = FilteringExploration(dataset_config, sample_seed=pipeline_config.random_states.training_sample_seed)
    bundle = dataset.get_dataset()
    print("\nFINAL DATASET (aligned feature columns)")
    print_stage("Training (after sampling / retrieval)", bundle.train_data.X)
    print_stage("MIMIC held-out test", bundle.test_mimic.X)
    print_stage("TUDD held-out test", bundle.test_tudd.X)
    if bundle.test_retriever is not None:
        print_stage("Retriever test cohort", bundle.test_retriever.X)
    print("Model-specific imputation/scaling happens later and is outside this filtering report.")


if __name__ == "__main__":
    main()
