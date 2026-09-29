"""Reusable data and rendering utilities for reproducible figures."""

from src.plotting.utils.aggregation import AggregatedEvaluation, aggregate_evaluation_runs
from src.plotting.utils.artifacts import (
    MissingExperimentError,
    MissingFullTrainingRunError,
    PlotArtifacts,
    TargetMismatchError,
    load_plot_artifacts,
    select_single_source_full_data_run_ids,
)
from src.plotting.utils.augmentation import (
    FixedLocalAugmentation,
    FixedLocalBudgetView,
    FullExternalAugmentation,
    prepare_fixed_local_augmentation,
    prepare_full_external_augmentation,
)
from src.plotting.utils.composition import (
    CompositionBudgetView,
    CompositionEvaluation,
    prepare_composition_evaluation,
)
from src.plotting.utils.grouped import GroupedEvaluation, aggregate_runs_by_setting
from src.plotting.utils.pairwise import PairwiseSummary, load_pairwise_inputs, prepare_pairwise_summary
from src.plotting.utils.ranking import RankSummary, prepare_rank_summary
from src.plotting.utils.retrieval import (
    BatchIdentity,
    RetrievalBatchView,
    RetrievalBudgetEvaluation,
    RetrieverDesign,
    prepare_retrieval_budget_evaluation,
    read_batch_identity,
    read_retriever_design,
)
from src.plotting.utils.runs import (
    IncompleteExperimentError,
    TrainOnDesign,
    TrainOnEntry,
    read_train_on_design,
    read_training_sample_seed,
)
from src.plotting.utils.sample_size import (
    FullDataBenchmark,
    SampleSizeEvaluation,
    XGBoostDifferenceEvaluation,
    full_training_count,
    prepare_full_data_benchmark,
    prepare_sample_size_evaluation,
    prepare_xgboost_difference_evaluation,
)
from src.plotting.utils.source_contrast import (
    PairingEvidence,
    SourceContrastEvaluation,
    prepare_source_contrast,
)
from src.plotting.utils.transfer import TransferSummary, prepare_transfer_summary

__all__ = [
    "AggregatedEvaluation",
    "BatchIdentity",
    "CompositionBudgetView",
    "CompositionEvaluation",
    "FixedLocalAugmentation",
    "FixedLocalBudgetView",
    "FullDataBenchmark",
    "FullExternalAugmentation",
    "GroupedEvaluation",
    "IncompleteExperimentError",
    "MissingExperimentError",
    "MissingFullTrainingRunError",
    "PairingEvidence",
    "PairwiseSummary",
    "PlotArtifacts",
    "RankSummary",
    "RetrievalBatchView",
    "RetrievalBudgetEvaluation",
    "RetrieverDesign",
    "SampleSizeEvaluation",
    "SourceContrastEvaluation",
    "TargetMismatchError",
    "TrainOnDesign",
    "TrainOnEntry",
    "TransferSummary",
    "XGBoostDifferenceEvaluation",
    "aggregate_evaluation_runs",
    "aggregate_runs_by_setting",
    "full_training_count",
    "load_pairwise_inputs",
    "load_plot_artifacts",
    "prepare_composition_evaluation",
    "prepare_fixed_local_augmentation",
    "prepare_full_data_benchmark",
    "prepare_full_external_augmentation",
    "prepare_pairwise_summary",
    "prepare_rank_summary",
    "prepare_retrieval_budget_evaluation",
    "prepare_sample_size_evaluation",
    "prepare_source_contrast",
    "prepare_transfer_summary",
    "prepare_xgboost_difference_evaluation",
    "read_batch_identity",
    "read_retriever_design",
    "read_train_on_design",
    "read_training_sample_seed",
    "select_single_source_full_data_run_ids",
]
