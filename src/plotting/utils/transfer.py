"""Derive cross-dataset transfer contrasts from aggregated evaluations."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.plotting.utils.aggregation import AggregatedEvaluation
from src.schemas.training_schemas import LOWER_IS_BETTER_SCORING


@dataclass(frozen=True)
class TransferSummary:
    """Absolute performance and two positive generalizability losses."""

    performance: pd.DataFrame
    degradation: pd.DataFrame
    relative_loss: pd.DataFrame
    aggregated: AggregatedEvaluation
    external_dataset: str

    @property
    def model_metadata(self) -> pd.DataFrame:
        return self.aggregated.model_metadata

    @property
    def model_instances(self) -> tuple[str, ...]:
        return self.aggregated.model_instances

    @property
    def metrics(self) -> tuple[str, ...]:
        return self.aggregated.metrics

    @property
    def trained_on(self) -> str:
        return self.aggregated.trained_on

    @property
    def target(self) -> str:
        return self.aggregated.target

    @property
    def run_ids(self) -> tuple[str, ...]:
        return self.aggregated.run_ids

    @property
    def run_count(self) -> int:
        return self.aggregated.run_count

    @property
    def bootstrap_count(self) -> int:
        return self.aggregated.bootstrap_count

    @property
    def ci_level(self) -> float:
        return self.aggregated.ci_level


def prepare_transfer_summary(aggregated: AggregatedEvaluation) -> TransferSummary:
    """Calculate positive degradation and loss to the best external model.

    Degradation is oriented so positive values are worse transfer. Relative loss
    is calculated against the model with the best aggregated external point
    estimate for each metric, making that reference model exactly zero.
    """
    external_datasets = [dataset for dataset in aggregated.datasets if dataset != aggregated.trained_on]
    if aggregated.trained_on not in aggregated.datasets or len(external_datasets) != 1:
        raise ValueError(
            "Transfer summaries require exactly one in-domain and one external test dataset; "
            f"trained_on={aggregated.trained_on!r}, datasets={list(aggregated.datasets)}"
        )
    external_dataset = external_datasets[0]

    performance = aggregated.performance.copy()
    indexed_points = performance.set_index(["model_instance", "dataset", "metric"])["estimate"]
    names_by_instance = aggregated.model_metadata.set_index("model_instance")["model_name"]
    alpha = (1.0 - aggregated.ci_level) / 2.0
    degradation_rows: list[dict[str, object]] = []
    relative_rows: list[dict[str, object]] = []

    for metric in aggregated.metrics:
        internal_scores = aggregated.scores(aggregated.trained_on, metric)
        external_scores = aggregated.scores(external_dataset, metric)
        external_points = performance.loc[
            performance["dataset"].eq(external_dataset) & performance["metric"].eq(metric)
        ].set_index("model_instance")["estimate"]
        lower_is_better = metric in LOWER_IS_BETTER_SCORING
        best_value = external_points.min() if lower_is_better else external_points.max()
        best_instance = next(
            instance for instance in aggregated.model_instances if external_points.loc[instance] == best_value
        )

        for instance in aggregated.model_instances:
            internal_point = float(indexed_points.loc[(instance, aggregated.trained_on, metric)])
            external_point = float(indexed_points.loc[(instance, external_dataset, metric)])
            if lower_is_better:
                degradation_estimate = external_point - internal_point
                degradation_draws = external_scores[instance] - internal_scores[instance]
                relative_estimate = external_point - float(best_value)
                relative_draws = external_scores[instance] - external_scores[best_instance]
            else:
                degradation_estimate = internal_point - external_point
                degradation_draws = internal_scores[instance] - external_scores[instance]
                relative_estimate = float(best_value) - external_point
                relative_draws = external_scores[best_instance] - external_scores[instance]

            degradation_lower, degradation_upper = degradation_draws.quantile([alpha, 1.0 - alpha])
            relative_lower, relative_upper = relative_draws.quantile([alpha, 1.0 - alpha])
            common = {
                "model_name": str(names_by_instance.loc[instance]),
                "model_instance": instance,
                "metric": metric,
            }
            degradation_rows.append(
                {
                    **common,
                    "estimate": degradation_estimate,
                    "lower": float(degradation_lower),
                    "upper": float(degradation_upper),
                }
            )
            relative_rows.append(
                {
                    **common,
                    "estimate": relative_estimate,
                    "lower": float(relative_lower),
                    "upper": float(relative_upper),
                    "reference_model_instance": best_instance,
                }
            )

    return TransferSummary(
        performance=performance,
        degradation=pd.DataFrame(degradation_rows),
        relative_loss=pd.DataFrame(relative_rows),
        aggregated=aggregated,
        external_dataset=external_dataset,
    )
