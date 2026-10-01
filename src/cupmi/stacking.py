# SPDX-License-Identifier: Apache-2.0
"""Fold-locked evaluation helpers for precomputed probability streams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

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
    covariance: str = "pooled",
    ridge: float = 1e-4,
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
    covariance, ridge:
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
