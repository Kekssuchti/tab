#!/usr/bin/env bash
# Rebuild every figure (and print the LaTeX tables) from the recorded MLflow runs.

set -uo pipefail

cd "$(dirname "$0")/../.." || exit 1

steps=(
    src.plotting.baseline_transfer
    src.plotting.training_source_contrast
    src.plotting.sample_size
    src.plotting.training_composition
    src.plotting.augmentation
    src.plotting.ablation_estimators
#    src.plotting.feature_distributions
    src.plotting.pairwise_wins
    src.plotting.pairwise_tables
    src.plotting.retriever_comparison
    src.plotting.retrieval_budget
)

filter="${1:-}"
jobs="${JOBS:-3}"
mem_floor_mb="${MEM_FLOOR_MB:-8192}"

selected=()
for step in "${steps[@]}"; do
    if [[ -n "$filter" && "$step" != *"$filter"* ]]; then
        continue
    fi
    selected+=("$step")
done

if ((${#selected[@]} == 0)); then
    echo "no step matched '${filter}'"
    exit 1
fi

failed=()
logs=""
declare -A running=()

cleanup() {
    [[ -n "$logs" ]] && rm -rf "$logs"
}
interrupted() {
    for pid in "${!running[@]}"; do
        kill "$pid" 2>/dev/null
    done
    exit 130
}
trap cleanup EXIT
trap interrupted INT TERM

# Memory available right now, in MiB, or empty when /proc/meminfo is unusable.
mem_available_mb() {
    awk '/^MemAvailable:/ {print int($2 / 1024); exit}' /proc/meminfo 2>/dev/null
}

# A running step is never interrupted for memory; the floor only decides whether
# another step may be started next to it.
room_for_another_step() {
    ((mem_floor_mb == 0)) && return 0
    local available
    available="$(mem_available_mb)"
    [[ -z "$available" ]] && return 0
    ((available >= mem_floor_mb))
}

if ((jobs == 1 || ${#selected[@]} == 1)); then
    for step in "${selected[@]}"; do
        echo
        echo "=== $step"
        uv run python -m "$step" || failed+=("$step")
    done
else
    logs="$(mktemp -d)"

    # Without this every step would start its own full-width BLAS/OpenMP thread
    # pool and the concurrent jobs would fight over the same CPUs.
    cores="$(nproc 2>/dev/null || echo "$jobs")"
    threads=$((cores / jobs))
    ((threads < 1)) && threads=1
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$threads}"
    export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$threads}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$threads}"

    echo "running ${#selected[@]} step(s), up to ${jobs} at a time (memory floor ${mem_floor_mb} MiB)"

    next=0
    while ((next < ${#selected[@]} || ${#running[@]} > 0)); do
        while ((next < ${#selected[@]} && ${#running[@]} < jobs)); do
            if ((${#running[@]} > 0)) && ! room_for_another_step; then
                echo "waiting for memory: $(mem_available_mb) MiB available, floor is ${mem_floor_mb} MiB"
                break
            fi
            step="${selected[next]}"
            stem="${logs}/${step//./_}"
            ( uv run python -m "$step" >"${stem}.log" 2>&1; echo $? >"${stem}.rc" ) &
            running[$!]="$step"
            next=$((next + 1))
        done

        # Nothing running means the queue is empty; the outer loop then ends.
        ((${#running[@]} == 0)) && break

        wait -n -p done_pid || true
        step="${running[$done_pid]:-}"
        unset "running[$done_pid]"
        [[ -z "$step" ]] && continue

        stem="${logs}/${step//./_}"
        rc="$(cat "${stem}.rc" 2>/dev/null || echo 1)"
        echo
        echo "=== $step"
        cat "${stem}.log"
        if ((rc != 0)); then
            echo "--- $step exited with status ${rc}"
            failed+=("$step")
        fi
    done
fi

echo
if ((${#failed[@]})); then
    echo "failed: ${failed[*]}"
    exit 1
fi
echo "rebuilt ${#selected[@]} step(s)"
