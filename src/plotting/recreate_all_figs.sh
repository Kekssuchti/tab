#!/usr/bin/env bash
# Rebuild every figure (and print the LaTeX tables) from the recorded MLflow runs.
#
#   src/plotting/recreate_all_figs.sh              # every step
#   src/plotting/recreate_all_figs.sh pairwise     # only steps whose module name matches

set -uo pipefail

cd "$(dirname "$0")/../.." || exit 1

steps=(
    src.plotting.baseline_transfer
    src.plotting.sample_size
    src.plotting.ablation_estimators
    src.plotting.feature_distributions
    src.plotting.pairwise_wins
    src.plotting.pairwise_tables
)

filter="${1:-}"
failed=()
ran=0

for step in "${steps[@]}"; do
    if [[ -n "$filter" && "$step" != *"$filter"* ]]; then
        continue
    fi
    ran=$((ran + 1))
    echo
    echo "=== $step"
    uv run python -m "$step" || failed+=("$step")
done

echo
if ((${#failed[@]})); then
    echo "failed: ${failed[*]}"
    exit 1
fi
if ((ran == 0)); then
    echo "no step matched '${filter}'"
    exit 1
fi
echo "rebuilt ${ran} step(s)"
