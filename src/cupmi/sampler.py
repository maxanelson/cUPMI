# SPDX-License-Identifier: Apache-2.0
"""Class-conditional Gaussian augmentation for stacking meta-features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

CovarianceMode = Literal["pooled", "diagonal"]


@dataclass(frozen=True)
class AugmentationInfo:
    """Metadata describing one augmentation call."""

    rho: float
    n_real: int
    n_synthetic_per_class: int
    classes: tuple
    covariance: str
    ridge: float


def _validate_arrays(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if X.ndim != 2:
        raise ValueError("X must be a 2D meta-feature matrix.")
    if y.ndim != 1:
        raise ValueError("y must be a 1D label array.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of rows.")
    if X.shape[0] == 0:
        raise ValueError("X and y must be non-empty.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains NaN or infinite values.")
    return X, y


def _pooled_covariance(X: np.ndarray, ridge: float, covariance: CovarianceMode) -> np.ndarray:
    if ridge < 0:
        raise ValueError("ridge must be non-negative.")
    cov = np.cov(X, rowvar=False)
    cov = np.asarray(cov, dtype=float)
    if cov.ndim == 0:
        cov = cov.reshape(1, 1)
    if covariance == "diagonal":
        cov = np.diag(np.diag(cov))
    elif covariance != "pooled":
        raise ValueError("covariance must be 'pooled' or 'diagonal'.")
    return cov + ridge * np.eye(X.shape[1])


def class_conditional_gaussian_augment(
    X: np.ndarray,
    y: np.ndarray,
    rho: float,
    *,
    seed: int | None = None,
    covariance: CovarianceMode = "pooled",
    ridge: float = 1e-4,
    return_info: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, AugmentationInfo]:
    """Append class-balanced Gaussian synthetic rows to a meta-feature matrix.

    Parameters
    ----------
    X:
        Two-dimensional stacking meta-feature matrix. In the cUPMI paper this is
        the concatenation of per-stream log class probabilities.
    y:
        Class labels for the rows of ``X``.
    rho:
        Synthetic-to-real ratio. ``rho=2`` requests roughly ``2 * len(y)``
        synthetic rows, distributed evenly across observed classes. ``rho=0``
        returns copies of the original arrays.
    seed:
        Random seed for reproducible synthesis.
    covariance:
        ``"pooled"`` uses one shared covariance matrix estimated from all rows.
        ``"diagonal"`` keeps only the featurewise variances.
    ridge:
        Non-negative diagonal regularizer added to the covariance matrix.
    return_info:
        If true, also return an ``AugmentationInfo`` record.

    Returns
    -------
    X_aug, y_aug:
        Original rows followed by synthetic rows.
    """

    X, y = _validate_arrays(X, y)
    rho = float(rho)
    if rho < 0:
        raise ValueError("rho must be non-negative.")

    classes = tuple(np.unique(y).tolist())
    n_classes = len(classes)
    per_class = int(rho * len(y)) // max(n_classes, 1)
    info = AugmentationInfo(
        rho=rho,
        n_real=len(y),
        n_synthetic_per_class=per_class,
        classes=classes,
        covariance=covariance,
        ridge=ridge,
    )
    if per_class <= 0:
        if return_info:
            return X.copy(), y.copy(), info
        return X.copy(), y.copy()

    cov = _pooled_covariance(X, ridge=ridge, covariance=covariance)
    rng = np.random.default_rng(seed)
    xs: list[np.ndarray] = [X]
    ys: list[np.ndarray] = [y]

    for cls in classes:
        Xc = X[y == cls]
        if Xc.size == 0:
            continue
        mean = Xc.mean(axis=0)
        synthetic = rng.multivariate_normal(mean, cov, size=per_class, check_valid="ignore")
        xs.append(synthetic)
        ys.append(np.full(per_class, cls, dtype=y.dtype))

    X_aug = np.vstack(xs)
    y_aug = np.concatenate(ys)
    if return_info:
        return X_aug, y_aug, info
    return X_aug, y_aug
