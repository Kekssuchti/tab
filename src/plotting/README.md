# Plotting structure

Current figure scripts:

- `baseline_transfer.py` — full-data performance and generalizability
- `ablation_estimators.py` — estimator-count performance and runtime
- `feature_distributions.py` — filtered-cohort feature distributions

Shared mechanics live in `utils/`:

- `artifacts.py` — MLflow artifact loading and full-training-run selection
- `aggregation.py` — strict repeated-run validation and bootstrap averaging
- `transfer.py` — positive transfer degradation and relative external loss
- `grouped.py` — repeated-run aggregation by an explicit experimental setting
- `rendering.py` — model styles, forest panels, and interval-aware axis limits
- `distributions.py` — low-level feature-distribution drawing

`defaults.py` SOURCE OF TRUTH, colors styles etc.
`scientific_figstyle.py` (from other skill extracted but wrapped in defaults.py `set_plot_style()`)
## Regeneration

```bash
uv run python -m src.plotting.baseline_transfer
uv run python -m src.plotting.ablation_estimators
uv run python -m src.plotting.feature_distributions
```
