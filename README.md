# cUPMI

[User guide](docs/guide.md) · [Changelog](CHANGELOG.md) · [Roadmap](ROADMAP.md)

cUPMI is a small Python package for **class-conditional Gaussian augmentation of
stacking meta-features**. It is intended for multi-stream classification systems
where several base models produce class probabilities and a level-1 combiner
learns from their stacked log-probabilities.

The core idea is simple: fit one Gaussian per class in log-probability
meta-space, share one covariance matrix across classes for stability, append synthetic
meta-feature rows to the combiner's training fold, and keep the evaluation fold
untouched.

This repository is deliberately self-contained: it ships only source code,
documentation, tests, and synthetic examples.

## Install

```bash
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
pytest
```

## Quickstart

Inputs: aligned **out-of-fold** probability streams from your base models (each
`(n_samples, n_classes)`), labels `y`, and the outer fold of each sample.

```python
from cupmi import evaluate_over_seeds

sweep = evaluate_over_seeds(
    [stream_a_proba, stream_b_proba, stream_c_proba], y, folds,
    seeds=5, metrics=["roc_auc_ovr", "qwk"], estimator="rf",
)
print(sweep.summary())   # plain stack vs cUPMI: mean delta, SD and interval over seeds
```

Fit a final combiner on all training samples:

```python
from cupmi import CUPMICombiner, stack_log_proba

U = stack_log_proba([stream_a_proba, stream_b_proba, stream_c_proba])
clf = CUPMICombiner(estimator="rf", seed=0).fit(U, y)
proba = clf.predict_proba(U_new)   # U_new: stacked log-probabilities of new samples
```

The [user guide](docs/guide.md) covers building out-of-fold streams, avoiding
leakage, choosing settings, reading results, and reproducibility.

For a complete synthetic demonstration:

```bash
python examples/synthetic_demo.py
```

## What the package provides

- `class_conditional_gaussian_augment`: the pure NumPy cUPMI sampler.
- `CUPMICombiner`: a scikit-learn-compatible wrapper that selects the synthesis
  ratio by inner cross-validation and fits the wrapped estimator on augmented
  meta-features.
- `stack_log_proba`: converts multiple base-model probability streams into the
  log-probability meta-feature matrix used by the combiner.
- `evaluate_over_seeds`: plain stack vs cUPMI on fixed outer folds, repeated over
  seeds, with a per-metric summary. Use this to report results.
- `evaluate_precomputed_streams`: the same comparison for a single seed.

## Method defaults

The default sampler matches the conservative version used in the paper:

- log-probability meta-features, clipped at `1e-6`;
- one Gaussian per class;
- shared **total** covariance of all training rows (`covariance="total"`; this
  includes between-class scatter, so it is wider than the LDA pooled within-class
  covariance);
- ridge regularization `1e-4` on the covariance diagonal;
- balanced synthetic generation across classes;
- synthesis ratio `rho` selected inside the training fold.

`covariance="pooled"`, the v0.1 name for the same estimator, still works but emits a
`FutureWarning`.

## Augmentation options

`class_conditional_gaussian_augment`, `CUPMICombiner`,
`evaluate_precomputed_streams` and `evaluate_over_seeds` share these knobs:

| Option | Values | Meaning |
| --- | --- | --- |
| `covariance` | `"total"` (default), `"within"`, `"within_lw"`, `"within_oas"`, `"diagonal"` | Shared covariance: all rows; pooled within-class (LDA); within-class with Ledoit-Wolf or OAS shrinkage; featurewise variances only |
| `center` | `"class_mean"` (default), `"per_point"` | Sample around the class mean (cUPMI), or around a random real row of the class (Gaussian-noise jitter) |
| `bandwidth` | float, default `0.5` | Noise scale `h` for `center="per_point"`: synthetic row = real row + `N(0, h^2 * Sigma)` |

cUPMI and noise jitter are two corners of the same family:

```python
from cupmi import CUPMICombiner

cupmi = CUPMICombiner(estimator="xgb", covariance="total")              # paper
shrunk = CUPMICombiner(estimator="xgb", covariance="within_lw")         # shrunk within-class
jitter = CUPMICombiner(estimator="xgb", covariance="diagonal",
                       center="per_point", bandwidth=0.5)               # noise jitter
```

On the IPMN 3-class task (XGB combiner, 5 seeds), `within_lw` gave the most
consistent QWK gain over the un-augmented stack; see [ROADMAP.md](ROADMAP.md). The
default stays `"total"` until this is confirmed on public benchmarks.

Draws from a full (non-diagonal) covariance use NumPy's
`Generator.multivariate_normal`, whose samples for a fixed seed can differ between
NumPy/LAPACK builds. Results are reproducible within one environment; across
environments only the distribution is. Report results over several seeds
(`evaluate_over_seeds`); see [Reproducibility](docs/guide.md#5-reproducibility).

## Optional file formats

The main API works directly with NumPy arrays, so no file format is required.
For file-based workflows, see [docs/input_contracts.md](docs/input_contracts.md)
for the optional CSV and split-JSON helper formats.
