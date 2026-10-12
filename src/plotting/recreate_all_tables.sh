#!/usr/bin/env bash
# Recreate local LaTeX tables. Add future table modules to this list.
# Optional arguments are passed to the descriptive generator, e.g. --kind normal.
set -euo pipefail
cd "$(dirname "$0")/../.."

steps=(
    src.plotting.descriptive_tables
)
for step in "${steps[@]}"; do
    echo "=== $step"
    uv run python -m "$step" "$@"
done
