# cUPMI Roadmap

## Vision

cUPMI is a small, well-tested library for **Gaussian augmentation in stacking meta-space**: synthesizing
rows in the space of base-model log-probabilities to regularize the level-1 combiner. The goal is to turn
the paper's single recipe (class means + shared covariance) into a family of orthogonal, documented knobs,
and to establish with public benchmarks when, and why, this kind of augmentation helps.

## Current state (v0.1.0)

- Public API (`src/cupmi/__init__.py`): `class_conditional_gaussian_augment`, `CUPMICombiner`,
  `make_default_estimator`, `stack_log_proba`, `to_log_proba`, `evaluate_precomputed_streams`,
  `EvaluationResult`.
- `sampler.py`: one Gaussian per class (class mean), one shared covariance, `covariance in {"pooled","diagonal"}`,
  `ridge=1e-4`, balanced allocation `per_class = int(rho * n) // n_classes`, synthetic rows appended, no sample weights.
- `combiner.py`: `CUPMICombiner` selects `rho` by inner `StratifiedKFold` over `rhos`, then refits on augmented data.
- `meta_features.py`: log-prob clipping at `eps=1e-6`; `align_predict_proba`.
- `stacking.py`: `evaluate_precomputed_streams` (fold-locked stack vs cUPMI). Note: it does not forward `covariance` / `ridge`
  to `CUPMICombiner` (always uses defaults).
- `metrics.py`: `macro_auc_ovr`, `quadratic_weighted_kappa`. `io.py`: `read_probability_csv`, `read_split_json` (read-only; no split writers or leakage checks).
- Tests: 8 passing (`tests/test_sampler.py`, `test_combiner.py`, `test_stacking.py`, `test_io.py`). Package is not installed by
  default; `PYTHONPATH=src pytest` works.
- Gaps: no CI (`.github/` absent), no CHANGELOG, no docs beyond `docs/input_contracts.md`, no benchmarks, `pyproject.toml` ruff
  `target-version = "py39"` vs `requires-python >= 3.10`, authors field is generic.
- Known naming issue: `_pooled_covariance` (`sampler.py`) calls `np.cov` on **all** rows, i.e. the **total** covariance
  (within-class + between-class scatter), not the pooled within-class covariance used in LDA. This is paper-faithful (the paper code did the
  same) but the name `"pooled"` (and the README's "shared pooled covariance") is misleading. See v0.2.

## Non-goals

- SMOTE, random oversampling, and other non-Gaussian resamplers will **not** be maintained in the package. For comparison they live only
  in `benchmarks/`, using `imbalanced-learn` as an optional extra (`benchmarks` extra), never a core dependency.
- No domain-specific feature extraction, imaging, or radiomics code. The package starts from probability streams.
- No new base-model zoo or AutoML. The combiner stays "any estimator with `fit` / `predict_proba`".

## Milestones

Order is priority order. Each milestone is releasable on its own; v0.1 default behavior must stay reproducible throughout.

### v0.2: Generalized Gaussian augmenter

**Goal:** one sampler with orthogonal knobs; cUPMI and noise-jitter become two settings of the same function.

- [ ] Add `center` option in `sampler.py`: `"class_mean"` (current) | `"per_point"` with bandwidth `h` (sample a real row of the class, add `N(0, h^2 * Sigma)`).
  Small `h` = noise jitter of real rows; large `h` approximates cUPMI.
- [ ] Add `covariance` options, factored out of `_pooled_covariance` into a `_estimate_covariance(X, y, mode, ...)`:
  `"total"` | `"pooled_within"` | `"per_class"` | `"rda"` (blend `lambda` between pooled-within and per-class) |
  `"ledoit_wolf"` / `"oas"` (via `sklearn.covariance`) | `"diagonal"`.
- [ ] Add `allocation` option: `"balanced"` (current) | `"proportional"` | `"inverse_frequency"`; keep `AugmentationInfo` reporting per-class counts
  (replace scalar `n_synthetic_per_class` with a dict, keeping the old field for compatibility).
- [ ] Return a `sample_weight` for synthetic rows (separate from `rho`); `CUPMICombiner` passes it to `estimator.fit` when supported, with a `synthetic_weight` parameter.
- [ ] Backward compatibility: `covariance="pooled"` stays accepted as a deprecated alias of `"total"` and emits `FutureWarning`;
  the default call (`center="class_mean"`, `covariance="total"`, `allocation="balanced"`, weight 1) must give bit-identical output to v0.1 for a fixed seed.
  Update README "Method defaults" to say "total covariance".
- [ ] Expose the new knobs on `CUPMICombiner` and forward them in `evaluate_precomputed_streams` (fixing the dropped `covariance` / `ridge`).
- [ ] Optionally let `CUPMICombiner` search over a small grid of `(rho, h)` or covariance mode in the inner CV.

**Acceptance:**
- Regression test in `tests/test_sampler.py`: v0.1 default output equals a stored reference array (generated from the v0.1 code, tiny synthetic data).
- Unit tests: `per_point` with `h -> 0` returns near-copies of real rows; `pooled_within` equals LDA pooled covariance on a toy example; each allocation mode yields the expected counts.
- Experiment: on the synthetic demo plus one real stream set, a grid over `center x covariance` reproduces the paper finding (cUPMI approx. noise-jitter) and maps the family (see Open questions).

**Item: total vs pooled-within vs shrinkage covariance**
- Experiment: same folds, seeds and combiner, vary only `covariance in {total, pooled_within, ledoit_wolf, oas, diagonal}` at `center="class_mean"`; report delta vs un-augmented stack.
- Result (IPMN, 2026-10-01; UPMI-2026 `covariance_study/`): Ledoit-Wolf-shrunk within-class covariance was the most consistent arm for XGB: ΔQWK +0.028 (S8) and +0.026 (fused S16) vs. +0.022 / +0.005 for the paper's total covariance, positive in 5/5 seeds. No effect on RF. Rescaling within-class to total trace did not reproduce the gain, so shrinkage (estimation noise), not spread, looks like the lever. **Candidate default for v0.2: `covariance="within_lw"`, pending benchmarks.**
- At rho=2, 45–92% of synthetic rows contain a log-probability > 0 (invalid). Raises the priority of v0.3.

### v0.3: Sampling space (log-prob vs logit vs ILR)

**Goal:** synthetic rows that map back to valid per-stream probability simplices.

- [ ] New `src/cupmi/spaces.py` with transforms and inverses: `"logprob"` (current), `"logit"` (binary streams), `"ilr"` (isometric log-ratio, per stream, with a fixed Helmert basis); `stack_log_proba` gains a `space=` argument (or a sibling `stack_meta_features`).
- [ ] `class_conditional_gaussian_augment(..., space=...)`: sample in the chosen space, optionally return back-mapped probabilities.
- [ ] Diagnostic `invalid_row_rate(X_syn, n_streams, n_classes)` in a new `diagnostics.py`: fraction of synthetic rows where any stream's `exp(logp)` does not sum to 1 within tolerance or has entries > 1; expose in `AugmentationInfo`.
- [ ] Decide and document the clipping/renormalization policy under `logprob` (default stays "no projection" for v0.1 parity).

**Acceptance:**
- Round-trip tests: `inverse(forward(P)) == P` to numerical tolerance for each space; ILR synthetic rows always yield valid simplices (invalid rate 0).
- Experiment: on 2-3 stream sets, report invalid-row rate under `logprob` vs the delta from each space; keep `logprob` the default unless another space wins consistently.

### v0.4: Leakage-safe helpers

**Goal:** make the correct, leakage-free workflow the easy one.

- [ ] `src/cupmi/oof.py`: `build_oof_streams(base_models, X_by_stream, y, groups=None, cv=StratifiedGroupKFold(...))` returning aligned OOF probability streams (patient/center groups).
- [ ] `src/cupmi/splits.py`: `make_splits(y, groups, n_splits, seed)` (stratified + grouped), `save_split_json` / `load_split_json` (round-trip with `io.read_split_json` format), `check_no_leakage(folds, groups)` raising on any group in both train and test.
- [ ] `CUPMIAugmenter` (sklearn-compatible, fit-time-only): augments only inside `fit`, passes through at `predict`; works inside `Pipeline` and `cross_validate`. Likely via a sampler-style wrapper (`imblearn`-like semantics, implemented natively with no dependency).
- [ ] Reuse `CUPMIAugmenter` inside `CUPMICombiner` / `evaluate_precomputed_streams` so there is one code path.
- [ ] Group-aware inner CV in `CUPMICombiner._select_rho` (accept `groups`/splitter, not just `inner_cv: int`).

**Acceptance:**
- Tests: `check_no_leakage` catches an injected overlap; `cross_validate(Pipeline([augmenter, estimator]))` never scores synthetic rows (assert row counts at predict); JSON splits round-trip.
- Example in `examples/` using OOF streams + grouped splits end to end on synthetic data.

### v0.5: Diagnostics & evaluation

**Goal:** make results and failures inspectable and statistically honest.

- [ ] `diagnostics.py`: `plot_pca_real_vs_synthetic(X, y, X_syn, y_syn)` (matplotlib, `examples` extra), coverage metrics, `mmd(X_real, X_syn, kernel="rbf")`.
- [ ] `evaluation.py`: `paired_bootstrap_delta(scores_aug, scores_stack, n_boot, seed)` returning delta and CI, resampling seeds/folds paired; wrapper over repeated seeds of `evaluate_precomputed_streams`.
- [ ] `metrics.py`: add `neg_log_loss`-style and QWK scorers usable as `CUPMICombiner(scoring=...)` (QWK as a callable matching `(y, proba, classes)`).
- [ ] Report multi-seed delta tables (mean, CI, selected rho distribution) from `EvaluationResult`s.

**Acceptance:**
- Bootstrap CI covers 0 on a null synthetic case (augmentation has no effect) in a seeded test and excludes 0 on a constructed positive case.
- PCA/MMD figures produced by `examples/synthetic_demo.py`.

### v0.6: Ordinal-aware sampling

**Goal:** a QWK-oriented niche: use class order in the meta-space.

- [ ] `ordinal=True` mode in sampler: interpolate/sample between adjacent class Gaussians (mixing coefficient `t`), emit soft labels (e.g. `1-t`, `t` over the two classes) plus `sample_weight`; needs a soft-label-capable estimator or label-splitting (duplicate row with weights).
- [ ] Combiner support for soft-label augmentation (via weighted duplicate rows to stay sklearn-compatible).
- [ ] QWK as a first-class scoring option in `CUPMICombiner` and `evaluate_precomputed_streams`.

**Acceptance:** on ordinal benchmarks (v0.8 datasets), QWK delta vs the standard Gaussian augmenter, paired bootstrap CI reported; ship only if it beats the plain augmenter, otherwise keep as `experimental`.

### v0.7: Mixture per class (investigation)

**Goal:** decide, with evidence, whether a per-class mixture belongs in the package.

- [ ] Re-run the prior internal comparison (`gmmcmp_bic` vs `gmmcmp_single`) with the v0.2 knobs as baselines, on the v0.8 benchmark datasets.
- [ ] If justified: `mixture="gmm"` with BIC-selected components per class (`sklearn.mixture.GaussianMixture`), sampling from the fitted mixture.

**Acceptance:** include only if it beats single-Gaussian + best covariance on a majority of benchmark settings with non-overlapping paired-bootstrap CIs; otherwise document the negative result and drop.

### v0.8: Benchmarks (highest external credibility; start early, in parallel with v0.2)

**Goal:** characterize when Gaussian meta-space augmentation helps, on public data.

- [ ] `benchmarks/` directory (not packaged): 5-10 public multiclass and ordinal datasets (e.g. OpenML), loader with pinned dataset IDs, no data committed.
- [ ] Multiple base learners producing OOF streams (via `build_oof_streams`): e.g. LR, RF, XGB, kNN, naive Bayes; several stream counts.
- [ ] Combiners: LR, RF, XGB (`make_default_estimator`).
- [ ] Arms: un-augmented stack, cUPMI default, noise-jitter (`center="per_point"`), covariance variants; optional comparison-only arms SMOTE/oversample via `imbalanced-learn` (`benchmarks` extra).
- [ ] Factors to vary: combiner capacity, meta-set size (subsampled training rows), number of streams.
- [ ] Output: tidy CSV of seed-level deltas + paired-bootstrap summary; script entry points runnable from a clean checkout.

**Acceptance:** one command regenerates the headline table/figure; results state which regimes show a positive delta, with CIs.

## Open research questions

- What about Gaussian meta-space augmentation matters? Reviewer-response evidence (3-class QWK delta vs un-augmented stack, XGB on radiomics-S8, 5 seeds):
  cUPMI +0.022, Gaussian-noise jitter of real rows +0.024, oversample +0.007, SMOTE -0.010. On fused 16-stream XGB: noise +0.019 vs cUPMI +0.005.
  On the RF flagship, nothing helped. cUPMI and noise-jitter appear to be corners of one family (v0.2 `center` x `covariance`).
- Is the gain from class-mean centering, the covariance shape, the amount of synthetic mass (`rho`), or just smoothing the combiner's decision function?
- Total vs pooled-within vs shrinkage covariance. First IPMN result favours shrunk within-class covariance (see v0.2); needs confirmation on the benchmark suite.
- Does the effect depend on combiner capacity (XGB helps, RF does not) and on meta-set size / number of streams (16-stream fused favors jitter)?
- How often does log-prob sampling leave the simplex, and does fixing that (ILR) matter for performance?
- Is `rho` selection by inner CV (3 folds, `scoring="roc_auc_ovr"`) well matched to the outer metric (QWK)?
- Should synthetic rows be down-weighted (`sample_weight`) instead of choosing a small `rho`?

## Cross-cutting

- [ ] **CI:** `.github/workflows/ci.yml` running `ruff check` and `pytest` on Python 3.10-3.12 (install with `pip install -e ".[dev]"`).
- [ ] **Install parity:** document/test the installed path (not only `PYTHONPATH=src pytest`); fix ruff `target-version` to `py310`.
- [ ] **Versioning:** SemVer; `CHANGELOG.md` (Keep a Changelog format); single-source `__version__` between `pyproject.toml` and `src/cupmi/__init__.py`; deprecations last >= 1 minor release.
- [ ] **PyPI release:** build with `python -m build`, TestPyPI dry run, then PyPI via trusted publishing on tag; run `docs/publication_checklist.md` scans first.
- [ ] **Docs:** API reference (generated from docstrings, e.g. mkdocs or Sphinx) plus a method write-up (`docs/method.md`: sampler, covariance modes, spaces, leakage rules); keep `docs/input_contracts.md` in sync.
- [ ] **Type hints:** complete annotations, `py.typed` marker, mypy/pyright check in CI; tidy `Scoring` and `CovarianceMode` literals.
- [ ] **Tests:** keep the v0.1 reference-output regression test; add property tests (counts, determinism by seed); target coverage reported in CI.
- [ ] **Packaging metadata:** real author/maintainer, project URLs, new extras (`benchmarks`: imbalanced-learn, openml; `examples`: matplotlib).
