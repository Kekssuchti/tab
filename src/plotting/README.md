# Plotting structure

Current figure scripts:

- `baseline_transfer.py` — full-data performance and generalizability
- `training_source_contrast.py` — same-target local-versus-external full-data cost
- `sample_size.py` — AUROC/AUPRC progression over training sample size, plus
  same-size XGBoost differences
- `training_composition.py` — fixed-budget MIMIC/TUDD composition curves
- `augmentation.py` — local-only versus external-augmented training curves
- `ablation_estimators.py` — estimator-count performance and runtime
- `pairwise_wins.py` — cross-cohort win shares, paired differences, and rank summaries
- `pairwise_tables.py` — the pairwise and average-rank LaTeX tables
- `feature_distributions.py` — filtered-cohort feature distributions
- `retriever_comparison.py` — paired retrieval-strategy effects per budget/batch
- `retrieval_budget.py` — retrieval performance over the selected budget with the
  matched unrestricted candidate-pool reference

`experiments.py` is the single place that says which MLflow experiment backs
which figure; `defaults.py` is the source of truth for colors and styles.

Shared mechanics live in `utils/`:

- `artifacts.py` — MLflow artifact loading, full-training-run selection, and the
  declared-target check
- `aggregation.py` — strict repeated-run validation and bootstrap averaging
- `transfer.py` — positive transfer degradation and relative external loss
- `grouped.py` — repeated-run aggregation by an explicit experimental setting
- `sample_size.py` — sample-count inference and repeated-run preparation
- `pairwise.py` — win shares, paired differences, and the split-encoded LaTeX matrices
- `ranking.py` — average ranks per block, Friedman tests, and Nemenyi post-hoc results
- `settings.py` — assignment of pipeline runs to experimental settings
- `rendering.py` — model styles, forest panels, and interval-aware axis limits
- `runs.py` — authoritative training-source design, realized count, and
  bootstrap alignment of selected runs
- `composition.py` / `augmentation.py` — fixed-budget composition and
  augmentation preparation
- `retrieval.py` — retriever design, target-batch identity, and budget curves
- `distributions.py` — low-level feature-distribution drawing

`defaults.py` SOURCE OF TRUTH, colors styles etc.
`scientific_figstyle.py` (from other skill extracted but wrapped in defaults.py `set_plot_style()`)
The Nemenyi diagrams themselves come from `scikit-posthocs`.
## Regeneration

`recreate_all_figs.sh` runs every figure script below and reports any step that
failed; pass a substring to run only the matching steps (`recreate_all_figs.sh
pairwise`). It stays a flat list of families, because the prediction tasks a
family covers are declared in Python, not in the shell.

```bash
src/plotting/recreate_all_figs.sh

uv run python -m src.plotting.baseline_transfer
uv run python -m src.plotting.training_source_contrast
uv run python -m src.plotting.sample_size
uv run python -m src.plotting.training_composition
uv run python -m src.plotting.augmentation
uv run python -m src.plotting.ablation_estimators
uv run python -m src.plotting.pairwise_wins
uv run python -m src.plotting.pairwise_tables
uv run python -m src.plotting.feature_distributions
uv run python -m src.plotting.retriever_comparison
uv run python -m src.plotting.retrieval_budget
```

### Adding a prediction task

`experiments.py` maps each figure family to the experiments it reads:

```python
"baseline": (
    FigureTask("mortality", "sample_size_mimic_mortality", ("roc_auc", "prc_auc")),
    FigureTask("LOS", "sample_size_mimic_LOS", ("mae", "rmse")),
),
```

- `baseline`, `sample_size`, `pairwise_wins`, and `pairwise_tables` are
  task-scoped: they loop over every declared task and write into
  `plots/<family>/<target>/`, so one entry per task is all a new experiment
  needs, and `src/plotting/recreate_all_figs.sh` needs no change at all.
  Metrics and the percent scale come from the declared task, so a regression
  task is not plotted on a 0-100 axis.
- A task whose experiment does not exist yet is skipped with a warning instead
  of failing the run, so declaring an upcoming task in advance is safe.
- An experiment that contains a different target than declared is an error, not
  a warning: it is the only way a figure could silently show the wrong task.
- `ablation_estimators.py`, `retriever_comparison.py`, and
  `feature_distributions.py` are investigation figures: one experiment, one
  question, and an output path that does not contain the prediction task. They
  keep naming their own experiment, and their data comes from MLflow, the
  filtered CSVs, and the marimo notebook respectively.

`--target mortality` narrows a task-scoped script to selected tasks, and
`--run-id` still pins explicit pipeline runs (one target at a time).
