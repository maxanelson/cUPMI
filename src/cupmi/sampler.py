# SPDX-License-Identifier: Apache-2.0
"""Class-conditional Gaussian augmentation for stacking meta-features."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.covariance import OAS, LedoitWolf

CovarianceMode = Literal["total", "within", "within_lw", "within_oas", "diagonal", "pooled"]
CenterMode = Literal["class_mean", "per_point"]

_COVARIANCE_MODES = ("total", "within", "within_lw", "within_oas", "diagonal")
_CENTER_MODES = ("class_mean", "per_point")


@dataclass(frozen=True)
class AugmentationInfo:
    """Metadata describing one augmentation call."""

    rho: float
    n_real: int
    n_synthetic_per_class: int
    classes: tuple
    covariance: str
    ridge: float
    center: str = "class_mean"
    bandwidth: float | None = None


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


def _resolve_covariance(covariance: str) -> str:
    if covariance == "pooled":
        warnings.warn(
            "covariance='pooled' is deprecated and will be removed in a future release; "
            "it has always meant the total covariance of all rows, so use covariance='total'. "
            "For the pooled within-class (LDA) covariance use covariance='within'.",
            FutureWarning,
            stacklevel=3,
        )
        return "total"
    if covariance not in _COVARIANCE_MODES:
        raise ValueError(f"covariance must be one of {_COVARIANCE_MODES}.")
    return covariance


def _within_class_residuals(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.vstack([X[y == cls] - X[y == cls].mean(axis=0) for cls in np.unique(y)])


def _estimate_covariance(X: np.ndarray, y: np.ndarray, covariance: str, ridge: float) -> np.ndarray:
    """Shared covariance used by every class, plus ``ridge`` on the diagonal.

    ``total`` and ``diagonal`` ignore labels (``np.cov`` over all rows, as in the
    paper). The ``within*`` modes estimate the pooled within-class covariance from
    class-centred residuals, optionally with Ledoit-Wolf or OAS shrinkage.
    """

    if ridge < 0:
        raise ValueError("ridge must be non-negative.")
    if covariance in ("total", "diagonal"):
        cov = np.atleast_2d(np.cov(X, rowvar=False))
        if covariance == "diagonal":
            cov = np.diag(np.diag(cov))
    else:
        n_classes = len(np.unique(y))
        if X.shape[0] <= n_classes:
            raise ValueError("within-class covariance needs more rows than classes.")
        residuals = _within_class_residuals(X, y)
        if covariance == "within":
            cov = residuals.T @ residuals / (X.shape[0] - n_classes)
        elif covariance == "within_lw":
            cov = LedoitWolf(assume_centered=True).fit(residuals).covariance_
        else:
            cov = OAS(assume_centered=True).fit(residuals).covariance_
    return np.asarray(cov, dtype=float) + ridge * np.eye(X.shape[1])


def class_conditional_gaussian_augment(
    X: np.ndarray,
    y: np.ndarray,
    rho: float,
    *,
    seed: int | None = None,
    covariance: CovarianceMode = "total",
    ridge: float = 1e-4,
    center: CenterMode = "class_mean",
    bandwidth: float = 0.5,
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
        Shared covariance ``Sigma`` used for every class.

        - ``"total"``: covariance of all rows, ignoring labels (within-class plus
          between-class scatter). This is the paper's estimator and the default.
        - ``"within"``: pooled within-class covariance, as in LDA.
        - ``"within_lw"`` / ``"within_oas"``: pooled within-class covariance with
          Ledoit-Wolf / OAS shrinkage.
        - ``"diagonal"``: featurewise variances of all rows only.
        - ``"pooled"``: deprecated alias of ``"total"``.
    ridge:
        Non-negative diagonal regularizer added to the covariance matrix.
    center:
        Where synthetic rows are centred.

        - ``"class_mean"``: draw from ``N(class mean, Sigma)`` (cUPMI).
        - ``"per_point"``: pick a real row of the class uniformly at random and
          add ``N(0, bandwidth**2 * Sigma)`` (Gaussian-noise jitter).
    bandwidth:
        Noise scale ``h`` for ``center="per_point"``; ignored otherwise. With
        ``covariance="diagonal"`` and ``bandwidth=0.5`` this is the jitter
        baseline from the paper's reviewer response.
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
    covariance = _resolve_covariance(covariance)
    if center not in _CENTER_MODES:
        raise ValueError(f"center must be one of {_CENTER_MODES}.")
    bandwidth = float(bandwidth)
    if center == "per_point" and bandwidth < 0:
        raise ValueError("bandwidth must be non-negative.")

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
        center=center,
        bandwidth=bandwidth if center == "per_point" else None,
    )
    if per_class <= 0:
        if return_info:
            return X.copy(), y.copy(), info
        return X.copy(), y.copy()

    cov = _estimate_covariance(X, y, covariance=covariance, ridge=ridge)
    rng = np.random.default_rng(seed)
    xs: list[np.ndarray] = [X]
    ys: list[np.ndarray] = [y]

    for cls in classes:
        Xc = X[y == cls]
        if Xc.size == 0:
            continue
        if center == "class_mean":
            synthetic = rng.multivariate_normal(
                Xc.mean(axis=0), cov, size=per_class, check_valid="ignore"
            )
        else:
            anchors = Xc[rng.integers(0, len(Xc), size=per_class)]
            noise = rng.multivariate_normal(
                np.zeros(X.shape[1]), bandwidth**2 * cov, size=per_class, check_valid="ignore"
            )
            synthetic = anchors + noise
        xs.append(synthetic)
        ys.append(np.full(per_class, cls, dtype=y.dtype))

    X_aug = np.vstack(xs)
    y_aug = np.concatenate(ys)
    if return_info:
        return X_aug, y_aug, info
    return X_aug, y_aug
