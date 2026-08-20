# SPDX-License-Identifier: Apache-2.0
"""Utilities for stacking probability streams into log-probability meta-features."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

EPS = 1e-6


def to_log_proba(probabilities: np.ndarray, *, eps: float = EPS) -> np.ndarray:
    """Convert class probabilities to clipped log probabilities."""

    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a 2D array.")
    if probabilities.shape[0] == 0 or probabilities.shape[1] < 2:
        raise ValueError("probabilities must have rows and at least two classes.")
    if eps <= 0 or eps >= 1:
        raise ValueError("eps must be in (0, 1).")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("probabilities contain NaN or infinite values.")
    return np.log(np.clip(probabilities, eps, 1.0))


def stack_log_proba(probability_streams: Sequence[np.ndarray], *, eps: float = EPS) -> np.ndarray:
    """Stack multiple probability streams into one log-probability meta-matrix.

    Each item in ``probability_streams`` must be an ``(n_samples, n_classes)``
    array aligned to the same sample order.
    """

    if not probability_streams:
        raise ValueError("At least one probability stream is required.")
    log_streams = [to_log_proba(stream, eps=eps) for stream in probability_streams]
    n_rows = {stream.shape[0] for stream in log_streams}
    if len(n_rows) != 1:
        raise ValueError("All probability streams must have the same number of rows.")
    return np.hstack(log_streams)


def align_predict_proba(
    model,
    X: np.ndarray,
    classes: np.ndarray,
) -> np.ndarray:
    """Return ``predict_proba`` columns aligned to a fixed class order."""

    if not hasattr(model, "predict_proba"):
        raise TypeError("The estimator must implement predict_proba.")
    predicted = model.predict_proba(X)
    out = np.zeros((len(X), len(classes)), dtype=float)
    for source_col, cls in enumerate(model.classes_):
        target = np.flatnonzero(classes == cls)
        if len(target):
            out[:, target[0]] = predicted[:, source_col]
    return out
