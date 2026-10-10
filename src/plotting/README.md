# Plotting structure

Current figure scripts:

- `baseline_transfer.py` — full-data performance and generalizability
- `training_source_contrast.py` — same-target local-versus-external full-data contrast
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
- `transfer.py` — signed model-specific and comparative generalizability changes
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

The steps are independent, so the script runs up to three of them at a time by
default. `JOBS` raises or lowers that ceiling and `JOBS=1` restores plain
sequential execution with streaming output; parallel steps buffer their console
output and print it as one block when they finish, so every printed caption still
sits next to the step that produced it.

The ceiling is about CPU only; the real limit is memory. The heavy families
(`sample_size`, `augmentation`, `training_composition`, `retrieval_budget`) hold
several GB each, and a run is added only while `MemAvailable` is at least
`MEM_FLOOR_MB` (default 4096), so a loaded desktop degrades the run towards
sequential instead of letting the kernel OOM-kill a step. `MEM_FLOOR_MB=0`
disables that guard.

```bash
JOBS=4 src/plotting/recreate_all_figs.sh
```

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

### Output paths

Every figure path reads `plots/<experiment>/<target>/<source>_<what>.pdf`:

- `<experiment>` is the experiment family the figure reports on (`baseline`,
  `sample_size`, `composition`, `augmentation`, `retrieval`), not the kind of
  chart. The full-data cross-cohort win shares, the reciprocal source contrast
  and the paired forest therefore live under `baseline/`, while the rank and
  rank-movement diagrams belong to the `sample_size/` experiment they are
  computed from.
- `<target>` is the prediction task.
- `<source>` is the training source, the `external_to_local` augmentation
  direction, the `source_to_target` retrieval direction, or the source pair of
  a reciprocal contrast. A design with one training pool, such as fixed-budget
  composition, starts the file name with `<what>` instead.
- `<what>` names the figure and the design settings it fixes, for example
  `generalizability`, `roc_auc_ranks`, `roc_auc_budget-1600`, or
  `xgboost_difference`.

Estimator-count ablations and feature distributions are investigation figures and
keep their own flat directories.

### Baseline uncertainty versions

Set `VISUAL.plot_type` in `baseline_transfer.py` to select:

- `"current"`: mean across runs with the existing 95% percentile
  interval from aligned, run-averaged cohort-bootstrap draws.
- `"bootstrap"`: a small hollow point and its own bootstrap CI for each run,
  plus a larger filled mean point without a whisker.
- `"min_max"` (default): one filled mean point and one whisker spanning the
  minimum and maximum run estimates. Individual runs are not drawn. This range
  is not a CI.

The setting applies to both performance and generalizability figures. Contrasts
are calculated within each run; the comparative reference is the strongest
model by mean external performance and stays fixed across repeats. Run count
comes from the selected logical repeats, rather than assuming exactly three.
`show_ci=False` hides whiskers in every version; individual run points remain
visible only in the `"bootstrap"` version.

CLI overrides and comparison rendering:

```bash
uv run python -m src.plotting.baseline_transfer --plot-type bootstrap
uv run python -m src.plotting.baseline_transfer --plot-type min_max
uv run python -m src.plotting.baseline_transfer --plot-type all --output-dir plots/baseline/versions
```

The current version retains the original filenames. Alternatives append
`_bootstrap` or `_min_max`; each version is exported as PDF and SVG. The
comparison outputs under `plots/baseline/versions/<target>/` preserve existing
baseline PDFs. Captions printed by the script describe each version's markers
and interval interpretation.

### Figure geometry

Every height is stated per grid row, as a fraction of the figure width:
`figure_grid(nrows, ncols, width, row_height, overhead)` multiplies it by the
row count, and `figure(row_height=...)` is the one-row case. A figure therefore
grows and shrinks with its rows instead of stretching the panels it keeps, so
dropping a metric from `VISUAL.metrics` shortens the figure by exactly one row
height. Never pass a total height for a grid: that is what silently doubled the
panel height of every one-metric figure. `overhead` is the only absolute height,
in inches, for what belongs to no single row (a band of panel labels).

### Where a figure's metrics come from

The registry orchestrates inputs only: which experiment, which target, which
training source, which evaluation center. Which metrics a figure plots is a
plot-level decision in the figure's own `VISUAL` (or `TABLES`) settings, next to
its colors, sizes, and axis labels, so registering a new input never changes what
an existing figure shows. `score_scale` is derived from `metrics` through
`defaults.panel_scale`, so the two cannot disagree; a tuple mixing percentage and
original-unit metrics is rejected instead of being plotted on one axis.

### Registering an experiment

A figure reads one MLflow experiment name per declared input. The name is
registered in the `_EXPERIMENT_NAMES` mapping in `experiments.py` with a
four-part key:

```python
(family, target, training_source, evaluation_center): "mlflow_experiment_name"
```

- `training_source` trains the model. For augmentation and retrieval it is the
  **external** source / candidate pool, not the center being predicted.
- `evaluation_center` is the center a directional figure evaluates: the local
  target center for augmentation, the target batch center for retrieval. Figures
  that evaluate both centers (single-source sweeps, composition) use `None`.
- A key that matches no declared task is simply unused, so the figure keeps
  printing `(not registered)` and skips. That is the usual reason a finished
  run produces no plot: compare the printed family/direction with the key.

One registration example per planned figure:

```python
_EXPERIMENT_NAMES = {
    # (family, target, training_source, evaluation_center): "mlflow_experiment_name"

    # F1/F2 (absolute, generalizability) and F4/F5 (learning curves, XGBoost delta):
    # one single-source sweep per center, evaluated on both centers.
    (SINGLE_SOURCE, "mortality", "mimic", None): "sample_size_mimic_mortality",
    (SINGLE_SOURCE, "mortality", "tudd", None): "sample_size_tudd_mortality",
    # F3 additionally needs the reciprocal center of the same target: it is plotted
    # from the two single-source names above, so it needs no registration of its own.
    # F6 fixed-budget composition: one experiment per target, no single source.
    (COMPOSITION, "mortality", None, None): "composition_mortality",
    # F7 complete external pool of training_source plus a growing local sample at
    # evaluation_center.
    (AUGMENTATION_FULL_EXTERNAL, "mortality", "tudd", "mimic"): "mixed_sample_size_mimic_mortality",
    (AUGMENTATION_FULL_EXTERNAL, "mortality", "mimic", "tudd"): "mixed_sample_size_tudd_mortality",
    # F9 paired strategy effects for one candidate-source/target-batch direction.
    (RETRIEVAL, "mortality", "tudd", "tudd"): "retriever_tudd_mortality",
    # F10 budget curves, plus the matched unrestricted reference in its own experiment.
    (RETRIEVAL, "mortality", "tudd", "tudd"): "retrieval_budget_tudd_mortality",
    (RETRIEVAL_UNRESTRICTED, "mortality", "tudd", "tudd"): "retrieval_unrestricted_tudd_mortality",
}
```

What the runs of an experiment have to record:

| Figure | runs |
| --- | --- |
| F1/F2 | one run per center with `train_on: [{"dataset": "<center>", "fraction": 1.0}]` |
| F3 | the same, for both centers and one target |
| F4/F5 | that sweep, one run per absolute count `fraction: <int>`, plus the `1.0` run |
| F6 | `train_on` as integer counts of both sources summing to one fixed total per budget |
| F7 | `[{"dataset": "<external>", "fraction": 1.0}, {"dataset": "<local>", "fraction": <int>}]` per local size, optionally the external-only run and the both-full run |
| F9 | `custom_retriever` runs whose strategies share one budget, batch center, batch size, and batch seed |
| F10 | the same at several budgets, plus a separately registered unrestricted candidate-pool run per direction |

The local-only and external-only reference curves of F7 are taken from the
single-source sweeps already registered for that center, matched cell by cell on
`training_sample_seed` and realized count; unmatched repeats are reported and
left out. All inputs of one figure must share one model roster — models missing
from any input are reported and dropped, and a pinned `DATA.models` roster is an
error if an input does not cover it.

F7 also reuses a missing `1.0 + 1.0` endpoint from the registered reciprocal
mixed-sample sweep for the same seeds, keeping existing endpoints. To plot the
completed LOS7 seed:

```bash
uv run python -m src.plotting.augmentation --target LOS7 --seed 1337
```

Then rebuild that figure:

```bash
uv run python -m src.plotting.augmentation   # F7
uv run python -m src.plotting.retrieval_budget   # F10
```

A registered experiment that does not measure the design yet is skipped with a
warning; a registered experiment whose runs contradict each other (different
targets, unpaired batches, mismatched counts) fails loudly instead of drawing a
misleading figure.

### Adding a prediction task

`experiments.py` maps each figure family to the experiments it reads:

```python
"baseline": (
    FigureTask("mortality", "sample_size_mimic_mortality", ("roc_auc", "prc_auc")),
    FigureTask("LOS", "sample_size_mimic_LOS", ("mae", "rmse")),
),
```

- `baseline`, `sample_size`, `pairwise_wins`, and `pairwise_tables` are
  task-scoped: they loop over every declared task and write under
  `plots/<experiment>/<target>/` (`pairwise_wins` splits its full-data views into
  `baseline/` and its rank views into `sample_size/`), so one entry per task is
  all a new experiment needs, and `src/plotting/recreate_all_figs.sh` needs no
  change at all.
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
