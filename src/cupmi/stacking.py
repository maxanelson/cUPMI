# SPDX-License-Identifier: Apache-2.0
"""Fold-locked evaluation helpers for precomputed probability streams."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .combiner import CUPMICombiner, _clone_with_seed, _score_probabilities
from .meta_features import align_predict_proba, stack_log_proba


@dataclass
class EvaluationResult:
    """Result returned by ``evaluate_precomputed_streams``."""

    stack_score: float
    cupmi_score: float
    delta: float
    selected_rhos: tuple[float, ...]
    classes: tuple
    stack_proba: np.ndarray
    cupmi_proba: np.ndarray


def evaluate_precomputed_streams(
    probability_streams: Sequence[np.ndarray],
    y: np.ndarray,
    folds: np.ndarray,
    *,
    estimator=None,
    rhos: Sequence[float] = (0.0, 1.0, 2.0, 3.0, 4.0),
    inner_cv: int = 3,
    scoring="roc_auc_ovr",
    seed: int | None = None,
    eps: float = 1e-6,
    covariance: str = "total",
    ridge: float = 1e-4,
    center: str = "class_mean",
    bandwidth: float = 0.5,
) -> EvaluationResult:
    """Evaluate plain stacking and cUPMI on fixed outer folds.

    Parameters
    ----------
    probability_streams:
        Aligned ``(n_samples, n_classes)`` probability arrays. These can be
        out-of-fold predictions from base models.
    y:
        Labels aligned to the probability streams.
    folds:
        Integer fold assignment per row. The function trains only on rows whose
        fold differs from the scoring fold.
    estimator:
        Level-1 estimator for both the plain stack and the cUPMI stack.
    covariance, ridge, center, bandwidth:
        Passed to ``CUPMICombiner``; see ``class_conditional_gaussian_augment``.
    """

    y = np.asarray(y)
    folds = np.asarray(folds)
    if y.ndim != 1 or folds.ndim != 1 or len(y) != len(folds):
        raise ValueError("y and folds must be aligned 1D arrays.")
    X = stack_log_proba(probability_streams, eps=eps)
    if X.shape[0] != len(y):
        raise ValueError("probability_streams and y have different row counts.")

    classes = np.unique(y)
    stack_proba = np.zeros((len(y), len(classes)), dtype=float)
    cupmi_proba = np.zeros_like(stack_proba)
    selected_rhos = []

    for fold in np.unique(folds):
        test = folds == fold
        train = ~test
        if not np.any(test) or not np.any(train):
            continue

        stacker = _clone_with_seed(estimator, seed, len(classes))
        stacker.fit(X[train], y[train])
        stack_proba[test] = align_predict_proba(stacker, X[test], classes)

        cupmi = CUPMICombiner(
            estimator=estimator,
            rhos=rhos,
            inner_cv=inner_cv,
            scoring=scoring,
            seed=seed,
            covariance=covariance,
            ridge=ridge,
            center=center,
            bandwidth=bandwidth,
        )
        cupmi.fit(X[train], y[train])
        cupmi_proba[test] = cupmi.predict_proba(X[test])
        selected_rhos.append(float(cupmi.rho_))

    stack_score = _score_probabilities(y, stack_proba, classes, scoring)
    cupmi_score = _score_probabilities(y, cupmi_proba, classes, scoring)
    return EvaluationResult(
        stack_score=stack_score,
        cupmi_score=cupmi_score,
        delta=cupmi_score - stack_score,
        selected_rhos=tuple(selected_rhos),
        classes=tuple(classes.tolist()),
        stack_proba=stack_proba,
        cupmi_proba=cupmi_proba,
    )


@dataclass
class SeedSweepResult:
    """Result returned by ``evaluate_over_seeds``.

    ``table`` has one row per (seed, metric) with the plain-stack score, the
    cUPMI score and their difference. ``summary()`` aggregates it over seeds.
    """

    seeds: tuple[int, ...]
    metrics: tuple[str, ...]
    results: list[EvaluationResult] = field(repr=False)
    table: pd.DataFrame = field(repr=False)

    def summary(self, confidence: float = 0.95) -> pd.DataFrame:
        """Per-metric mean scores and the distribution of ``delta`` over seeds.

        The interval is a t-interval over seeds. It describes how much the
        result moves with the random seed (augmentation draws, inner CV and the
        combiner's own randomness) on this one dataset. It is not a confidence
        interval over patients or datasets.
        """

        rows = []
        for metric, group in self.table.groupby("metric", sort=False):
            deltas = group["delta"].to_numpy(dtype=float)
            n = len(deltas)
            sd = float(deltas.std(ddof=1)) if n > 1 else float("nan")
            if n > 1:
                half = stats.t.ppf(0.5 + confidence / 2, df=n - 1) * sd / np.sqrt(n)
            else:
                half = float("nan")
            rows.append(
                {
                    "metric": metric,
                    "n_seeds": n,
                    "stack_mean": float(group["stack"].mean()),
                    "cupmi_mean": float(group["cupmi"].mean()),
                    "delta_mean": float(deltas.mean()),
                    "delta_sd": sd,
                    "delta_ci_low": float(deltas.mean() - half),
                    "delta_ci_high": float(deltas.mean() + half),
                    "n_positive": int((deltas > 0).sum()),
                }
            )
        return pd.DataFrame(rows).set_index("metric")

    @property
    def selected_rhos(self) -> pd.Series:
        """How often each ``rho`` was selected, over all seeds and outer folds."""

        rhos = [rho for result in self.results for rho in result.selected_rhos]
        return pd.Series(rhos, name="rho").value_counts().sort_index()


def _metric_name(metric) -> str:
    return metric if isinstance(metric, str) else getattr(metric, "__name__", repr(metric))


def evaluate_over_seeds(
    probability_streams: Sequence[np.ndarray],
    y: np.ndarray,
    folds: np.ndarray,
    *,
    seeds: int | Sequence[int] = 5,
    metrics: Sequence[Any] | None = None,
    scoring="roc_auc_ovr",
    **kwargs,
) -> SeedSweepResult:
    """Run ``evaluate_precomputed_streams`` once per seed and collect the deltas.

    A single seed is one draw of the synthetic rows, the inner-CV split and the
    combiner's randomness; the cUPMI-vs-stack delta can move by about as much as
    the effect itself between seeds. Report the summary over several seeds.

    Parameters
    ----------
    probability_streams, y, folds:
        As in ``evaluate_precomputed_streams``. The outer folds stay fixed across
        seeds.
    seeds:
        Number of seeds (``5`` means ``0..4``) or an explicit list.
    metrics:
        Metrics reported in ``table`` and ``summary()``. Each is a scoring name
        accepted by ``CUPMICombiner`` (``"roc_auc_ovr"``, ``"qwk"``,
        ``"neg_log_loss"``, ``"accuracy"``) or a callable
        ``f(y_true, proba, classes) -> float``. Defaults to ``[scoring]``.
    scoring:
        Metric used to select ``rho`` inside each training fold.
    **kwargs:
        Passed to ``evaluate_precomputed_streams``: ``estimator``, ``rhos``,
        ``inner_cv``, ``eps``, ``covariance``, ``ridge``, ``center``,
        ``bandwidth``. Pass ``estimator`` as a name (``"rf"``, ``"xgb"``,
        ``"lr"``) or with ``random_state=None`` so the combiner is reseeded too;
        an estimator with a fixed ``random_state`` keeps it for every seed.
    """

    if "seed" in kwargs:
        raise TypeError("Use seeds=..., not seed=..., with evaluate_over_seeds.")
    seed_list = tuple(range(seeds)) if isinstance(seeds, (int, np.integer)) else tuple(seeds)
    if not seed_list:
        raise ValueError("seeds must contain at least one seed.")
    metric_list = list(metrics) if metrics is not None else [scoring]
    names = tuple(_metric_name(metric) for metric in metric_list)
    y = np.asarray(y)

    results: list[EvaluationResult] = []
    rows = []
    for seed in seed_list:
        result = evaluate_precomputed_streams(
            probability_streams, y, folds, seed=int(seed), scoring=scoring, **kwargs
        )
        results.append(result)
        classes = np.asarray(result.classes)
        for metric, name in zip(metric_list, names):
            stack = _score_probabilities(y, result.stack_proba, classes, metric)
            cupmi = _score_probabilities(y, result.cupmi_proba, classes, metric)
            rows.append(
                {"seed": int(seed), "metric": name, "stack": stack, "cupmi": cupmi,
                 "delta": cupmi - stack}
            )

    return SeedSweepResult(
        seeds=tuple(int(seed) for seed in seed_list),
        metrics=names,
        results=results,
        table=pd.DataFrame(rows),
    )
