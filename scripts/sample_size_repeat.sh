#!/usr/bin/env bash
set -euo pipefail

# This assumes running a baseline task.
# This assumes the config is done and ready.
# Reruns this baseline x times (FIRST ARGUMENT).
#
# Between runs only the SEED environment variable changes (the one that
# src/config.py reads into config.seed). The dataset random_state in the
# config YAML is left untouched, so the train/test split and CV folds stay
# identical and the models only see minimal, controlled variation.

cd "$(dirname "$0")/.."

BASELINE_CONFIG_NAME=tudd_small
DEFAULT_BASE_SEED=1338

REPEATS="${1:-}"
BASE_SEED="${2:-${SEED:-$DEFAULT_BASE_SEED}}"

if [[ ! "$REPEATS" =~ ^[0-9]+$ ]] || (( REPEATS < 1 )); then
    echo "Usage: $0 <repeat_count> [base_seed]" >&2
    echo "  repeat_count: how many times to rerun the baseline (>= 1)" >&2
    echo "  base_seed: SEED used for the first run; incremented by 1 per run" >&2
    echo "             (default: \$SEED if set, else $DEFAULT_BASE_SEED)" >&2
    exit 1
fi

for ((i = 0; i < REPEATS; i++)); do
    seed=$((BASE_SEED + i))
    echo "=== Baseline run $((i + 1))/${REPEATS} (SEED=${seed}) ==="
    SEED="${seed}" uv run python -m src.run_pipeline -c "${BASELINE_CONFIG_NAME} --suite"
done
