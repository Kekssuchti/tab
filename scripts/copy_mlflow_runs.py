"""Copy MLflow runs to another experiment without deleting the originals.

Examples:
    # One run (its parents and children are included when needed)
    uv run python scripts/copy_mlflow_runs.py DESTINATION_EXPERIMENT --run-id RUN_ID
    uv run python scripts/copy_mlflow_runs.py test \
        --run-id 4fd292d65cf540bbbe67d2e1967b989b

    # Several runs from the same source experiment
    uv run python scripts/copy_mlflow_runs.py test --run-id RUN_1 RUN_2 RUN_3

    # Every active run in an experiment
    uv run python scripts/copy_mlflow_runs.py test --all source_experiment

Copies get new run IDs; nested links and this project's embedded run-ID references
are rewritten in the destination copy. Re-running the same command creates duplicates.

The mlflow-export-import package needs an HTTP(S) tracking URI. For this repository,
start MLflow with:

    uv run mlflow server --backend-store-uri sqlite:///mlflow.db \
        --default-artifact-root ./mlartifacts --serve-artifacts \
        --host 127.0.0.1 --port 5000

If no defaults change (like store uri) the command below suffices too:
    uv run mlflow server
"""

from __future__ import annotations

import argparse
import os
from collections import deque
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from mlflow.entities import Experiment, Run
from mlflow.exceptions import MlflowException
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
from mlflow_export_import.experiment import export_experiment, import_experiment

from mlflow import MlflowClient

DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"


def _experiment(client: MlflowClient, name_or_id: str) -> Experiment:
    experiment = client.get_experiment_by_name(name_or_id)
    if experiment is None:
        try:
            experiment = client.get_experiment(name_or_id)
        except Exception as error:
            raise ValueError(f"MLflow experiment not found: {name_or_id!r}") from error
    return experiment


def _children(client: MlflowClient, run: Run) -> list[Run]:
    filter_string = f"tags.`{MLFLOW_PARENT_RUN_ID}` = '{run.info.run_id}'"
    children: list[Run] = []
    page_token: str | None = None
    while True:
        page = client.search_runs(
            experiment_ids=[run.info.experiment_id],
            filter_string=filter_string,
            max_results=1000,
            page_token=page_token,
        )
        children.extend(page)
        page_token = page.token
        if not page_token:
            return children


def _expanded_run_ids(client: MlflowClient, requested_ids: list[str]) -> tuple[Experiment, list[str]]:
    """Include ancestors and descendants so nested-run links remain valid."""
    requested = [client.get_run(run_id) for run_id in dict.fromkeys(requested_ids)]
    experiment_ids = {run.info.experiment_id for run in requested}
    if len(experiment_ids) != 1:
        raise ValueError("All selected runs must belong to the same source experiment.")

    source = client.get_experiment(experiment_ids.pop())
    expanded: dict[str, Run] = {}

    for run in requested:
        ancestors: list[Run] = []
        current = run
        visited = {current.info.run_id}
        while parent_id := current.data.tags.get(MLFLOW_PARENT_RUN_ID):
            if parent_id in visited:
                raise ValueError(f"Nested-run cycle detected at {parent_id}.")
            parent = client.get_run(parent_id)
            if parent.info.experiment_id != source.experiment_id:
                raise ValueError(f"Parent run {parent_id} is outside source experiment {source.name!r}.")
            ancestors.append(parent)
            visited.add(parent_id)
            current = parent

        for ancestor in reversed(ancestors):
            expanded.setdefault(ancestor.info.run_id, ancestor)
        expanded.setdefault(run.info.run_id, run)

        queue = deque([run])
        while queue:
            parent = queue.popleft()
            for child in _children(client, parent):
                if child.info.run_id not in expanded:
                    expanded[child.info.run_id] = child
                    queue.append(child)

    return source, list(expanded)


def _rewrite_copied_references(
    client: MlflowClient,
    run_id_map: dict[str, str],
    destination_experiment: str,
) -> None:
    """Update this project's copied parent IDs while retaining source-lineage tags."""
    with TemporaryDirectory(prefix="mlflow-run-rewrite-") as directory:
        for destination_id in run_id_map.values():
            run = client.get_run(destination_id)
            source_parent_id = run.data.tags.get("pipeline_mlflow_run_id")
            if source_parent_id in run_id_map:
                client.set_tag(destination_id, "pipeline_mlflow_run_id", run_id_map[source_parent_id])

            run_directory = Path(directory) / destination_id
            run_directory.mkdir()
            try:
                path = Path(client.download_artifacts(destination_id, "prediction_metrics.csv", str(run_directory)))
            except MlflowException as error:
                if error.error_code == "RESOURCE_DOES_NOT_EXIST":
                    continue
                raise

            frame = pd.read_csv(path)
            for column in ("pipeline_mlflow_run_id", "model_mlflow_run_id"):
                if column in frame:
                    frame[column] = frame[column].replace(run_id_map)
            if "experiment_name" in frame:
                frame["experiment_name"] = destination_experiment
            frame.to_csv(path, index=False)
            client.log_artifact(destination_id, str(path))


def copy_runs(
    destination_experiment: str,
    *,
    run_ids: list[str] | None = None,
    source_experiment: str | None = None,
    tracking_uri: str = DEFAULT_TRACKING_URI,
) -> dict[str, str]:
    """Copy selected runs, or all runs, and return source-to-destination run IDs."""
    if tracking_uri.startswith(("sqlite:", "file:")):
        raise ValueError(
            "mlflow-export-import requires an HTTP(S) tracking URI, not a direct database/file URI. "
            "Start the MLflow server shown in this script's docstring and use http://127.0.0.1:5000."
        )

    client = MlflowClient(tracking_uri=tracking_uri)
    if run_ids:
        source, selected_run_ids = _expanded_run_ids(client, run_ids)
    elif source_experiment:
        source = _experiment(client, source_experiment)
        selected_run_ids = None
    else:
        raise ValueError("Provide --run-id or --all.")

    destination = client.get_experiment_by_name(destination_experiment)
    if destination is not None and destination.experiment_id == source.experiment_id:
        raise ValueError("Source and destination experiments must be different.")

    with TemporaryDirectory(prefix="mlflow-run-copy-") as directory:
        exported, failed = export_experiment.export_experiment(
            experiment_id_or_name=source.experiment_id,
            output_dir=directory,
            run_ids=selected_run_ids,
            mlflow_client=client,
        )
        if failed:
            raise RuntimeError(f"Export failed for {failed} run(s); nothing was imported.")
        if exported == 0:
            raise RuntimeError("No runs were exported; nothing was imported.")

        imported = import_experiment.import_experiment(
            experiment_name=destination_experiment,
            input_dir=directory,
            import_source_tags=True,
            mlflow_client=client,
        )

    run_id_map = {source_id: info.run_id for source_id, info in imported.items()}
    _rewrite_copied_references(client, run_id_map, destination_experiment)
    return run_id_map


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy MLflow runs to another experiment; never delete the source runs."
    )
    parser.add_argument("destination", help="Destination experiment name (created if missing).")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--run-id",
        nargs="+",
        metavar="RUN_ID",
        help="One or more run IDs from the same source experiment.",
    )
    selection.add_argument(
        "--all",
        metavar="EXPERIMENT",
        help="Copy every active run from this source experiment name or ID.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI),
        help="HTTP(S) MLflow tracking URI (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    copied = copy_runs(
        args.destination,
        run_ids=args.run_id,
        source_experiment=args.all_from,
        tracking_uri=args.tracking_uri,
    )
    print(f"Copied {len(copied)} run(s) to experiment {args.destination!r}:")
    for source_id, destination_id in copied.items():
        print(f"  {source_id} -> {destination_id}")
    print("Source runs were not modified or deleted.")


if __name__ == "__main__":
    main()
