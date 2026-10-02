# cUPMI User Guide

This guide covers the full workflow: building the inputs, evaluating cUPMI
honestly, choosing settings, and fitting a final model. For the CSV and JSON
helper formats, see [input_contracts.md](input_contracts.md).

## 1. What cUPMI does

You have several base models ("streams"), each producing class probabilities
for the same samples. A level-1 combiner learns from the stacked
log-probabilities. With small cohorts that combiner overfits easily.

cUPMI regularizes it. Inside each training fold, it fits one Gaussian per class
in the log-probability meta-space, sharing one covariance matrix across
classes. It then appends `rho * n` synthetic rows, split evenly across classes,
and trains the combiner on real plus synthetic rows. `rho` is chosen by inner
cross-validation on the training fold only. Evaluation rows are never augmented.

## 2. Workflow

### 2.1 Build out-of-fold probability streams

The meta-features must be **out-of-fold** (OOF): each sample's probabilities
come from a base model that did not train on that sample. In-fold probabilities
are overconfident, which inflates both the stack and cUPMI.

```python
import numpy as np
from sklearn.model_selection import StratifiedKFold

folds = np.empty(len(y), dtype=int)
for k, (_, test_idx) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(X, y)):
    folds[test_idx] = k

def oof_proba(model, X, y, folds):
    proba = np.zeros((len(y), len(np.unique(y))))
    for k in np.unique(folds):
        train, test = folds != k, folds == k
        proba[test] = model.fit(X[train], y[train]).predict_proba(X[test])
    return proba

streams = [oof_proba(model_a, X_a, y, folds), oof_proba(model_b, X_b, y, folds)]
```

Rules:

- Every stream has shape `(n_samples, n_classes)`, with rows in the same sample
  order and columns in the same class order (sorted labels).
- Use the **same outer folds** for the base models and for the combiner
  evaluation below. Otherwise information leaks through the base models.
- If several samples come from one patient (or site), build folds by group, for
  example with `StratifiedGroupKFold`.

### 2.2 Evaluate over several seeds

```python
from cupmi import evaluate_over_seeds

sweep = evaluate_over_seeds(
    streams, y, folds,
    seeds=5,                           # seeds 0..4; or a list such as [11, 23, 42]
    metrics=["roc_auc_ovr", "qwk"],    # what to report
    scoring="roc_auc_ovr",             # what selects rho inside each fold
    estimator="rf",                    # "rf", "xgb", "lr", or any sklearn classifier
)
print(sweep.summary())                 # one row per metric
print(sweep.table)                     # one row per (seed, metric)
print(sweep.selected_rhos)             # how often each rho was chosen
```

For each seed and each outer fold, this trains the plain stack and the cUPMI
stack on the same rows and scores both on the held-out fold. The `delta` columns
are `cUPMI - stack`.

**Report several seeds, not one.** A seed controls the synthetic draws, the
inner-CV split used to pick `rho`, and the combiner's own randomness. On small
cohorts the delta can move between seeds by about as much as the effect itself.
`evaluate_precomputed_streams` runs a single seed and is mainly a building block.

### 2.3 Fit the final model

```python
from cupmi import CUPMICombiner, stack_log_proba

U = stack_log_proba(streams)           # OOF streams for all training samples
clf = CUPMICombiner(estimator="rf", seed=0).fit(U, y)

# New samples: probabilities from base models refit on all training data.
U_new = stack_log_proba([model_a_full.predict_proba(X_a_new),
                         model_b_full.predict_proba(X_b_new)])
proba_new = clf.predict_proba(U_new)
print(clf.rho_, clf.rho_scores_)       # selected rho and its inner-CV scores
```

This is standard stacking. The combiner learns from OOF probabilities, and at
prediction time the base models are refit on all training data.

## 3. Choosing settings

| Setting | Default | Guidance |
| --- | --- | --- |
| `estimator` | `"rf"` | Augmentation helps most with flexible combiners (gradient boosting). With a strongly regularized combiner (shallow random forest, logistic regression) expect smaller effects. |
| `rhos` | `(0, 1, 2, 3, 4)` | Keep `0` in the grid, so the selector can choose no augmentation. |
| `inner_cv` | `3` | Lowered automatically to the smallest class count. If that is below 2, every `rho` scores 0 and the smallest `rho` in the grid is used. |
| `scoring` | `"roc_auc_ovr"` | Selects `rho`. Options: `"roc_auc_ovr"`, `"qwk"` (ordinal labels), `"neg_log_loss"`, `"accuracy"`, or a callable `f(y_true, proba, classes)`. Ideally it matches the metric you report. |
| `covariance` | `"total"` | `"total"` is the paper's estimator. Its spread includes between-class scatter, so it is wide. `"within_lw"` (shrunk within-class) is a tighter alternative and was the most consistent in our first study. It is not yet the default. |
| `center` | `"class_mean"` | `"per_point"` jitters real rows instead (`bandwidth` sets the scale). Useful as a comparison arm. |
| `ridge` | `1e-4` | Added to the covariance diagonal for numerical stability. |
| `eps` | `1e-6` | Probabilities are clipped to `[eps, 1]` before taking logs. |

To compare settings, run `evaluate_over_seeds` once per setting with the **same
folds and seeds**, then compare the `delta` columns seed by seed.

## 4. Reading the results

`sweep.summary()` columns:

- `stack_mean`, `cupmi_mean`: mean score over seeds.
- `delta_mean`, `delta_sd`: mean and standard deviation of `cUPMI - stack` over seeds.
- `delta_ci_low`, `delta_ci_high`: t-interval of the mean delta over seeds
  (95% by default; `summary(confidence=0.9)` changes it).
- `n_positive`: number of seeds where cUPMI beat the stack.

The interval shows how stable the result is across seeds **on this dataset**. It
is not a confidence interval over patients or over new datasets. For that, you
need resampling of samples (e.g. a bootstrap of the outer-fold predictions) or
several datasets.

`sweep.selected_rhos` counts the chosen `rho` over seeds × outer folds. If it is
mostly `0`, the selector found no benefit from augmentation for this setup.

## 5. Reproducibility

- A fixed seed gives identical results within one Python environment.
- With a full covariance (`"total"`, `"within"`, `"within_lw"`, `"within_oas"`),
  synthetic rows come from NumPy's `Generator.multivariate_normal`. Its output
  for a given seed depends on the NumPy build and its linear-algebra library
  (e.g. Apple Accelerate vs OpenBLAS). Across environments, the distribution is
  the same but individual draws differ. This is another reason to report several
  seeds. `"diagonal"` draws are the same across environments.
- Pass the combiner as a name (`"rf"`) or with `random_state=None` so each seed
  also reseeds it. An estimator with a fixed `random_state` keeps that value for
  every seed, so only the augmentation and inner CV vary.
- Record the package version, NumPy and scikit-learn versions, folds, and seeds
  alongside reported numbers.

## 6. Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `ValueError: X contains NaN or infinite values` | A stream contains NaN or inf. Probabilities are clipped at `eps`, so zeros are fine; NaN is not. |
| `probabilities must have rows and at least two classes` | A binary model returned only one column. Pass both columns `[1 - p, p]`. |
| `shape mismatch` inside the evaluator | A training fold is missing a class. Use stratified folds. |
| `FutureWarning: covariance='pooled' is deprecated` | Use `covariance="total"`; results are identical. |
| cUPMI is worse than the stack | Possible, especially with a regularized combiner or a large cohort. Check `selected_rhos` and the delta across seeds before drawing conclusions. |

## 7. API summary

| Object | Purpose |
| --- | --- |
| `stack_log_proba(streams, eps=1e-6)` | Clip, log and concatenate aligned probability streams. |
| `class_conditional_gaussian_augment(X, y, rho, *, seed, covariance, ridge, center, bandwidth, return_info)` | The sampler on its own. Returns real rows followed by synthetic rows. |
| `CUPMICombiner(estimator, *, rhos, inner_cv, scoring, seed, covariance, ridge, center, bandwidth)` | scikit-learn classifier: selects `rho`, then fits on augmented data. Fitted attributes: `rho_`, `rho_scores_`, `estimator_`, `classes_`. |
| `evaluate_precomputed_streams(streams, y, folds, *, seed, ...)` | Plain stack vs cUPMI on fixed outer folds, for one seed. Returns `EvaluationResult`. |
| `evaluate_over_seeds(streams, y, folds, *, seeds=5, metrics, scoring, **kwargs)` | The same over several seeds. Returns `SeedSweepResult` with `.table`, `.summary()` and `.selected_rhos`. |
| `make_default_estimator(kind, *, seed, n_classes)` | The documented `"rf"`, `"xgb"` and `"lr"` combiners. `"xgb"` needs `pip install "cupmi[xgboost]"`. |
| `cupmi.metrics.macro_auc_ovr`, `cupmi.metrics.quadratic_weighted_kappa` | Metrics with the `f(y_true, proba, classes)` signature. |
| `cupmi.io.read_probability_csv`, `cupmi.io.read_split_json` | Optional file helpers; see [input_contracts.md](input_contracts.md). |
