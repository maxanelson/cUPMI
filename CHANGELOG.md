# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `center="per_point"` with `bandwidth` in `class_conditional_gaussian_augment`,
  `CUPMICombiner` and `evaluate_precomputed_streams`: Gaussian-noise jitter of real
  rows, `x_i + N(0, h^2 * Sigma)`.
- Covariance modes `"total"`, `"within"` (pooled within-class, as in LDA),
  `"within_lw"` (Ledoit-Wolf shrinkage) and `"within_oas"` (OAS shrinkage).
- `AugmentationInfo.center` and `AugmentationInfo.bandwidth`.
- `evaluate_over_seeds` and `SeedSweepResult`: repeat the fold-locked comparison
  over several seeds and report a per-metric table, a summary (mean, SD,
  t-interval and positive-seed count of the delta) and the selected `rho` counts.
- `scoring="qwk"` (quadratic-weighted kappa) for `CUPMICombiner` and the evaluators.
- `docs/guide.md`: user guide covering workflow, leakage rules, settings, result
  interpretation, reproducibility and troubleshooting.
- GitHub Actions workflow running `pytest` on Python 3.10-3.13.
- `ROADMAP.md`.

### Changed

- The default covariance is now spelled `"total"`. Output for a fixed seed is
  unchanged from v0.1.0 in the same environment.
- `evaluate_precomputed_streams` now forwards `covariance` and `ridge` to
  `CUPMICombiner`; before, it silently used the defaults.
- ruff `target-version` set to `py310` to match `requires-python`.
- `examples/synthetic_demo.py` reports results over 5 seeds instead of one.
- README Quickstart fixed (it fitted on an undefined `U_train`) and now leads with
  multi-seed evaluation.

### Deprecated

- `covariance="pooled"`: it always meant the total covariance of all rows, not the
  pooled within-class covariance. Use `"total"` (same behavior) or `"within"`.

## [0.1.0] - 2026-08-20

- Initial public release.
