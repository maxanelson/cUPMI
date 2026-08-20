# SPDX-License-Identifier: Apache-2.0
"""Small metric helpers used by the examples and evaluation harness."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import cohen_kappa_score, roc_auc_score


def macro_auc_ovr(y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray | None = None) -> float:
    """Macro one-vs-rest AUC for binary or multiclass probabilities."""

    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    if classes is None:
        classes = np.unique(y_true)
    classes = np.asarray(classes)
    if len(classes) == 2:
        return float(roc_auc_score(y_true, proba[:, 1]))
    return float(
        roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=classes)
    )


def quadratic_weighted_kappa(y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray | None = None) -> float:
    """Quadratic-weighted kappa after argmax decoding."""

    y_true = np.asarray(y_true)
    if classes is None:
        classes = np.unique(y_true)
    classes = np.asarray(classes)
    pred = classes[np.asarray(proba).argmax(axis=1)]
    return float(cohen_kappa_score(y_true, pred, weights="quadratic"))
