"""Derive cross-dataset transfer contrasts from aggregated evaluations."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.plotting.utils.aggregation import AggregatedEvaluation
from src.schemas.training_schemas import scoring_is_lower_better


@dataclass(frozen=True)
class TransferSummary:
    """Absolute performance and signed model-specific and comparative changes."""

    performance: pd.DataFrame
    delta_spec: pd.DataFrame
    delta_comp: pd.DataFrame
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
    def ci_level(self) -> float | None:
        return self.aggregated.ci_level


def prepare_transfer_summary(aggregated: AggregatedEvaluation) -> TransferSummary:
    """Calculate signed model-specific and comparative generalizability changes.

    ``delta_spec`` is target minus source performance, oriented so negative
    values consistently indicate worse target-cohort performance. ``delta_comp``
    compares each target-cohort result with the best transferred model for that
    metric; the reference is exactly zero and worse models are negative.
    """
    external_datasets = [dataset for dataset in aggregated.datasets if dataset != aggregated.trained_on]
    if aggregated.trained_on not in aggregated.datasets or len(external_datasets) != 1:
        raise ValueError(
            "Transfer summaries require exactly one in-domain and one external test dataset; "
            f"trained_on={aggregated.trained_on!r}, datasets={list(aggregated.datasets)}"
        )
    if aggregated.ci_level is None:
        raise ValueError("Transfer summaries report interval endpoints, so they need a ci_level")
    external_dataset = external_datasets[0]

    performance = aggregated.performance.copy()
    indexed_points = performance.set_index(["model_instance", "dataset", "metric"])["estimate"]
    names_by_instance = aggregated.model_metadata.set_index("model_instance")["model_name"]
    alpha = (1.0 - aggregated.ci_level) / 2.0
    delta_spec_rows: list[dict[str, object]] = []
    delta_comp_rows: list[dict[str, object]] = []

    for metric in aggregated.metrics:
        internal_scores = aggregated.scores(aggregated.trained_on, metric)
        external_scores = aggregated.scores(external_dataset, metric)
        external_points = performance.loc[
            performance["dataset"].eq(external_dataset) & performance["metric"].eq(metric)
        ].set_index("model_instance")["estimate"]
        lower_is_better = scoring_is_lower_better(metric)
        best_value = external_points.min() if lower_is_better else external_points.max()
        best_instance = next(
            instance for instance in aggregated.model_instances if external_points.loc[instance] == best_value
        )

        for instance in aggregated.model_instances:
            internal_point = float(indexed_points.loc[(instance, aggregated.trained_on, metric)])
            external_point = float(indexed_points.loc[(instance, external_dataset, metric)])
            if lower_is_better:
                delta_spec_estimate = internal_point - external_point
                delta_spec_draws = internal_scores[instance] - external_scores[instance]
                delta_comp_estimate = float(best_value) - external_point
                delta_comp_draws = external_scores[best_instance] - external_scores[instance]
            else:
                delta_spec_estimate = external_point - internal_point
                delta_spec_draws = external_scores[instance] - internal_scores[instance]
                delta_comp_estimate = external_point - float(best_value)
                delta_comp_draws = external_scores[instance] - external_scores[best_instance]

            delta_spec_lower, delta_spec_upper = delta_spec_draws.quantile([alpha, 1.0 - alpha])
            delta_comp_lower, delta_comp_upper = delta_comp_draws.quantile([alpha, 1.0 - alpha])
            common = {
                "model_name": str(names_by_instance.loc[instance]),
                "model_instance": instance,
                "metric": metric,
            }
            delta_spec_rows.append(
                {
                    **common,
                    "estimate": delta_spec_estimate,
                    "lower": float(delta_spec_lower),
                    "upper": float(delta_spec_upper),
                }
            )
            delta_comp_rows.append(
                {
                    **common,
                    "estimate": delta_comp_estimate,
                    "lower": float(delta_comp_lower),
                    "upper": float(delta_comp_upper),
                    "reference_model_instance": best_instance,
                }
            )

    return TransferSummary(
        performance=performance,
        delta_spec=pd.DataFrame(delta_spec_rows),
        delta_comp=pd.DataFrame(delta_comp_rows),
        aggregated=aggregated,
        external_dataset=external_dataset,
    )
