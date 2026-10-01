# cUPMI

See the [development roadmap](ROADMAP.md).

cUPMI is a small Python package for **class-conditional Gaussian augmentation of
stacking meta-features**. It is intended for multi-stream classification systems
where several base models produce class probabilities and a level-1 combiner
learns from their stacked log-probabilities.

The core idea is simple: fit one Gaussian per class in log-probability
meta-space, use a shared pooled covariance matrix for stability, append synthetic
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

```python
from sklearn.ensemble import RandomForestClassifier
from cupmi import CUPMICombiner, stack_log_proba

# Each probability stream has shape (n_samples, n_classes).
U = stack_log_proba([stream_a_proba, stream_b_proba, stream_c_proba])

clf = CUPMICombiner(
    estimator=RandomForestClassifier(n_estimators=200, max_depth=4),
    rhos=(0.0, 1.0, 2.0, 3.0, 4.0),
    inner_cv=3,
    scoring="roc_auc_ovr",
    seed=0,
)
clf.fit(U_train, y_train)
proba = clf.predict_proba(U_test)
```

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
- `evaluate_precomputed_streams`: a fold-locked evaluator for precomputed
  probability streams.

## Method defaults

The default sampler matches the conservative version used in the paper:

- log-probability meta-features, clipped at `1e-6`;
- one Gaussian per class;
- shared pooled covariance across all training rows;
- ridge regularization `1e-4` on the covariance diagonal;
- balanced synthetic generation across classes;
- synthesis ratio `rho` selected inside the training fold.

## Optional file formats

The main API works directly with NumPy arrays, so no file format is required.
For file-based workflows, see [docs/input_contracts.md](docs/input_contracts.md)
for the optional CSV and split-JSON helper formats.
