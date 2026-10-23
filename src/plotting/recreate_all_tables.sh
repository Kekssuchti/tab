#!/usr/bin/env bash
# Recreate descriptive and full-sample baseline LaTeX tables.
# Optional arguments are passed only to the descriptive generator, e.g. --kind normal.
# PLOT_METRIC optionally selects one baseline metric; all models are always included.
set -euo pipefail
cd "$(dirname "$0")/../.."

steps=(
    src.plotting.descriptive_tables
)
for step in "${steps[@]}"; do
    echo "=== $step"
    uv run python -m "$step" "$@"
done

metrics=(auroc auprc)
if [[ -n "${PLOT_METRIC:-}" ]]; then metrics=("$PLOT_METRIC"); fi
for metric in "${metrics[@]}"; do
    echo "=== src.plotting.baseline_tables [$metric]"
    PLOT_METRIC="$metric" uv run python -m src.plotting.baseline_tables --ci both
done
