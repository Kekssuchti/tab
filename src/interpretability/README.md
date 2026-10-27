# Mortality interpretability experiment

Plain Python, no notebook or YAML configuration. EBM/XGBoost use the existing
registry-based HPO; TabPFN3.5 stays fixed-parameter. The old interpretability
experiment is not used. No trained models are saved or logged to MLflow.

## Run

From the repository root, edit the `SETTINGS` instance and the `ebm_kwargs`,
`xgboost_kwargs`, and `tabpfn_kwargs` dictionaries near the top of
[compute.py](<src/interpretability/compute.py>). For EBM/XGBoost, these
dictionaries supply extra kwargs; parameters sampled from the registry take
precedence for searched keys (including XGBoost aliases). `ebm_hpo` and
`xgboost_hpo` expose the trial counts and named search spaces. TabPFN uses its
kwargs without tuning. The experiment fixes EBM interactions to zero. Defaults
remain EBM, XGBoost, and **TabPFN3.5**; `--models tabpfn-3.5-fast` selects the
registry's **v3.5-fast** variant instead. Both TabPFN variants share `tabpfn_kwargs`
and the same marginal OddSHAP method, with separate result directories.

```bash
# Inspect cohort counts, sampling, and resume status. Does not fit any model.
uv run python -m src.interpretability.compute --dry-run

# Compute missing results. This can be expensive; fill the kwargs first.
uv run python -m src.interpretability.compute

# Optional: complete the experiment in smaller invocations.
uv run python -m src.interpretability.compute --models ebm xgboost
uv run python -m src.interpretability.compute --models tabpfn-3.5 --seeds 1337
uv run python -m src.interpretability.compute --models tabpfn-3.5-fast --seeds 1337

# Plot existing results only. Never fits models or computes SHAP.
uv run python -m src.interpretability.plot
uv run python -m src.interpretability.plot --no-bands --features Age Sex
# Compare baselines with the fast variant (or include both TabPFN variants).
uv run python -m src.interpretability.plot --models ebm xgboost tabpfn-3.5-fast
```

Both entry points accept `--results PATH`; the default is
`run_results/interpretability/mortality/`. This directory is **gitignored**:
raw feature values, predictions, explanation arrays, and scatter figures are
private clinical artifacts. Do not commit or publish them without review.

## Design

- Fixed core mortality train/test split (seed 1337, 80/20), aligned original
  features, and existing clinical filtering. No new split definition is introduced.
- Three repeats (1337–1339): stratified training selection **without replacement**,
  10,000 rows from each source. All models receive identical selected patients and
  row order within a source/repeat. An excessive requested count raises an error;
  it is not silently reduced or sampled with replacement.
- Tune EBM/XGBoost separately for each source/repeat, using **only that repeat's
  selected 10,000 training rows**. Match the baseline experiment: Optuna TPE,
  five shuffled stratified CV folds, mean CV AUROC, and existing stale-study
  stopping. EBM uses registry `default` (up to 50 trials, 5 startup); XGBoost uses
  registry `good` (up to 100 trials, 10 startup). The registry owns the ranges;
  they are not copied into this experiment. CV uses the repeat seed; the Optuna
  sampler seed stays 1337. Each fold model is released immediately in `finally`.
- Refit the selected baseline parameters once on all selected training rows.
  TabPFN3.5 skips HPO and fits once with its supplied kwargs. For each final model,
  evaluate and explain **both** held-out cohorts, save results, then release its
  RAM/VRAM in `finally`.
- Full-test AUROC/AUPRC (and the core evaluator's other point metrics), without
  bootstrap computation. Held-out sources remain separate and never enter HPO.
- SHAP on 100 fixed, stratified explanation rows per held-out cohort; all features.
  The background is fixed across models/repeats, drawn without replacement from
  the full training pools: `background_tudd = 16`, `background_mimic = 16`.
  EUH's internal source name is `tudd`. Background, explanation-patient, and
  explainer seeds stay fixed; training selection/model/CV seeds follow the repeat.
- Native missing-value handling; no external scaling, imputation, encoding, or
  additional missingness features. Sex remains a single original feature.

## Explanation semantics

Every model explains **binary mortality natural-log odds**, with positive values
pushing toward higher predicted mortality relative to that model's background
mean. The reference patients are shared; numerical base values are model-specific.

The empirical marginal game averages predictions over whole background rows
for masked coordinates. It is not conditional SHAP, and mixed feature combinations
need not represent clinically realistic patients. These are predictive, not causal,
associations.

- **EBM:** exact main-effect marginal SHAP, computed by centering each additive
  log-odds term on the shared background. Native uncentered terms are not reused.
- **XGBoost:** interventional TreeSHAP in raw log-odds space, retaining the complete
  supplied background (no implicit 100-row truncation). SHAP receives the fitted
  booster directly: XGBoost 3.4 defaults the sklearn wrapper's categorical-support
  flag to true even for numeric inputs, which SHAP 0.52 otherwise rejects. This
  does not change the model or switch to tree-path-dependent SHAP; actual
  categorical features remain unsupported by this explanation method.
- **TabPFN3.5 / TabPFN3.5-fast:** marginal OddSHAP on `log(p_mortality) - log(p_survival)` from final
  class probabilities, **not** the extension's automatic single-class logit.
  `shap_budget = 512` is the coalition budget **per explained patient**. Computation
  grows with explanation patients × coalitions × background patients × ensemble
  size. Timings and additivity residuals are recorded; a small residual does not
  establish attribution convergence. Zero probabilities fail explicitly rather
  than silently clipping/changing the explained function.

## Saved results and reruns

Each completed fitted model produces these files:

```text
run_results/interpretability/mortality/
  <training_source>/<model>/seed_<seed>.npz
  <training_source>/<model>/seed_<seed>.json
  <training_source>/<model>/seed_<seed>.importance.csv
  figures/dependence_<feature>_<name_hash>.pdf
```

The NPZ is the completion marker, written atomically after both explanation
cohorts finish. It contains feature names, selected training/background row IDs,
background values, explanation row IDs/feature values, SHAP/base/output arrays,
mean absolute SHAP importance, full-test labels/probabilities/row IDs, and JSON
metadata. JSON sidecars expose parameters, seeds, settings, metrics, timings,
package versions, and data/code fingerprints without loading arrays. For tuned
models, `base_params` records the input kwargs, `params` the selected final kwargs,
and `tuning` the resolved search configuration, best CV score, candidates, and
per-fold scores/metrics/timings. No Optuna study or fitted fold estimator is stored.
Nonfinite parameter defaults such as XGBoost's `missing=np.nan` use explicit JSON
tags (`{"__float__": "nan"}` means `float("nan")`). Importance CSV rows contain
`cohort`, `feature`, and `mean_abs_shap`; importance describes the fixed
**explanation sample**, not the complete test cohort. Additional stability analysis
is deliberately deferred.

Completed compatible models are skipped. An interrupted/failed fitted-model unit
restarts on rerun; there is no patient-level or per-cohort checkpointing. A model
failure is recorded in its private JSON sidecar and other models continue; the
command exits nonzero if any model failed. Plotting uses the seeds actually saved,
not every seed planned in `repeat_seeds`. One saved repetition per model/source is
sufficient. Each saved seed must cover all selected models and both training sources;
incomplete comparisons still raise an error instead of silently dropping results.

Changed compute settings/data/parameters are not silently reused. Choose another
`--results PATH`, or pass `--recompute` to replace selected units deliberately.
A forced rerun first moves the previous completed unit to `.previous` backup files,
so a failed replacement cannot be plotted as if the new run succeeded.
Changing one model's kwargs/HPO settings need not invalidate the other models.
HPO configuration and tuning code enter baseline cache fingerprints; adding HPO
does not invalidate otherwise compatible fixed TabPFN results. Plotting allows
selected hyperparameters to differ across repeats/sources while checking that
their input kwargs and HPO recipes agree. Plot settings are not part of computation
fingerprints. Changing the explanation method or patients requires refitting
because models are intentionally not persisted.

## Figures

Edit `PLOT_SETTINGS` in [plot.py](<src/interpretability/plot.py>) for histogram bin
count, EBM bands, feature selection, and visual styling. Rows are training source
(EUH, MIMIC-IV); columns are explanation cohort (EUH, MIMIC-IV). Output is **PDF only**.

- **EBM is a black line without markers**, connecting saved centered main-effect
  values in feature-value order, averaged across repeats. There is no binning or
  smoothing. Both explanation cohorts supply the observed values, so the same
  fitted EBM has the same line in both columns. Native EBM bin boundaries are not
  saved: this is a sampled main-effect line, not a reconstruction of its exact
  step function. No tails are extrapolated beyond the pooled observed range.
- **XGBoost is orange scatter; TabPFN3.5 is blue; TabPFN3.5-fast is cyan.** All use
  the same small dots, with no model-specific marker shapes or fitted/binned lines.
  Each dot is one patient's **mean SHAP across repeats**. The scatter's vertical
  spread is patient-level attribution variation, not an uncertainty interval.
- Optional EBM bands show **min–max across repeat main effects**, not confidence
  intervals or within-bin patient variability. With one saved repeat, the legend
  says **single repeat** and bands are omitted automatically.
- Count strips show cohort-specific observed-value support with common pooled
  histogram edges. Separate strips show missing-value contributions on the same
  SHAP scale (a black segment for EBM, patient dots for XGBoost/TabPFN).
  Sex is displayed discretely, with separate EBM segments rather than a line
  connecting categories.
- Axes are shared across the four panels. Units are not guessed from feature names
  or clinical plausibility. All-missing/constant features remain visible.

Rebinning, toggling bands, selecting features, or changing style only reads saved
results. It never reruns the expensive model/explanation computation.
