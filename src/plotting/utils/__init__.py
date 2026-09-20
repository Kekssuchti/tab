"""Reusable data and rendering utilities for reproducible figures."""

from src.plotting.utils.aggregation import AggregatedEvaluation, aggregate_evaluation_runs
from src.plotting.utils.artifacts import PlotArtifacts, load_plot_artifacts, select_full_training_run_ids
from src.plotting.utils.grouped import GroupedEvaluation, aggregate_runs_by_setting
from src.plotting.utils.pairwise import PairwiseSummary, load_pairwise_inputs, prepare_pairwise_summary
from src.plotting.utils.ranking import RankSummary, prepare_rank_summary
from src.plotting.utils.sample_size import SampleSizeEvaluation, prepare_sample_size_evaluation
from src.plotting.utils.transfer import TransferSummary, prepare_transfer_summary

__all__ = [
    "AggregatedEvaluation",
    "GroupedEvaluation",
    "PairwiseSummary",
    "PlotArtifacts",
    "RankSummary",
    "SampleSizeEvaluation",
    "TransferSummary",
    "aggregate_evaluation_runs",
    "aggregate_runs_by_setting",
    "load_pairwise_inputs",
    "load_plot_artifacts",
    "prepare_pairwise_summary",
    "prepare_rank_summary",
    "prepare_sample_size_evaluation",
    "prepare_transfer_summary",
    "select_full_training_run_ids",
]
