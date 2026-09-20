# Plotting structure

Current figure scripts:

- `baseline_transfer.py` — full-data performance and generalizability
- `sample_size.py` — AUROC/AUPRC progression over training sample size
- `ablation_estimators.py` — estimator-count performance and runtime
- `pairwise_wins.py` — cross-cohort win shares, paired differences, and rank summaries
- `pairwise_tables.py` — the pairwise and average-rank LaTeX tables
- `feature_distributions.py` — filtered-cohort feature distributions

Shared mechanics live in `utils/`:

- `artifacts.py` — MLflow artifact loading and full-training-run selection
- `aggregation.py` — strict repeated-run validation and bootstrap averaging
- `transfer.py` — positive transfer degradation and relative external loss
- `grouped.py` — repeated-run aggregation by an explicit experimental setting
- `sample_size.py` — sample-count inference and repeated-run preparation
- `pairwise.py` — win shares, paired differences, and the split-encoded LaTeX matrices
- `ranking.py` — average ranks per block, Friedman tests, and Nemenyi post-hoc results
- `settings.py` — assignment of pipeline runs to experimental settings
- `rendering.py` — model styles, forest panels, and interval-aware axis limits
- `distributions.py` — low-level feature-distribution drawing

`defaults.py` SOURCE OF TRUTH, colors styles etc.
`scientific_figstyle.py` (from other skill extracted but wrapped in defaults.py `set_plot_style()`)
The Nemenyi diagrams themselves come from `scikit-posthocs`.
## Regeneration

`recreate_all_figs.sh` runs every figure script below and reports any step that
failed; pass a substring to run only the matching steps (`recreate_all_figs.sh
pairwise`). The interpretability figures come from the marimo notebook instead and
are regenerated interactively.

```bash
src/plotting/recreate_all_figs.sh

uv run python -m src.plotting.baseline_transfer
uv run python -m src.plotting.sample_size
uv run python -m src.plotting.ablation_estimators
uv run python -m src.plotting.pairwise_wins
uv run python -m src.plotting.pairwise_tables
uv run python -m src.plotting.feature_distributions
```
