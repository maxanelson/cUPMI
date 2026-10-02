# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from sklearn.covariance import OAS, LedoitWolf

from cupmi import class_conditional_gaussian_augment
from cupmi.sampler import _estimate_covariance


def test_rho_zero_returns_original_values():
    X = np.arange(12, dtype=float).reshape(6, 2)
    y = np.array([0, 0, 1, 1, 2, 2])

    X_aug, y_aug = class_conditional_gaussian_augment(X, y, rho=0, seed=0)

    assert np.array_equal(X_aug, X)
    assert np.array_equal(y_aug, y)
    assert X_aug is not X
    assert y_aug is not y


def test_balanced_synthetic_counts_and_info():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 4))
    y = np.repeat([0, 1, 2], 20)

    X_aug, y_aug, info = class_conditional_gaussian_augment(
        X, y, rho=1.0, seed=3, return_info=True
    )

    assert X_aug.shape == (120, 4)
    assert y_aug.shape == (120,)
    assert info.n_synthetic_per_class == 20
    assert np.bincount(y_aug).tolist() == [40, 40, 40]


def test_sampler_is_deterministic_by_seed():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(45, 3))
    y = np.repeat([0, 1, 2], 15)

    X1, y1 = class_conditional_gaussian_augment(X, y, rho=2.0, seed=11)
    X2, y2 = class_conditional_gaussian_augment(X, y, rho=2.0, seed=11)

    assert np.allclose(X1, X2)
    assert np.array_equal(y1, y2)


def _v0_1_reference(X, y, rho, seed, diagonal=False, ridge=1e-4):
    """The v0.1.0 sampler, verbatim in behavior, for same-environment parity checks."""
    classes = np.unique(y)
    per_class = int(rho * len(y)) // len(classes)
    cov = np.atleast_2d(np.cov(X, rowvar=False))
    if diagonal:
        cov = np.diag(np.diag(cov))
    cov = cov + ridge * np.eye(X.shape[1])
    rng = np.random.default_rng(seed)
    xs, ys = [X], [y]
    for cls in classes:
        mean = X[y == cls].mean(axis=0)
        xs.append(rng.multivariate_normal(mean, cov, size=per_class, check_valid="ignore"))
        ys.append(np.full(per_class, cls, dtype=y.dtype))
    return np.vstack(xs), np.concatenate(ys)


def _toy(seed=0, n_per_class=20, n_features=4):
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1, 2], n_per_class)
    X = rng.normal(size=(len(y), n_features)) + 3.0 * y[:, None]
    return X, y


def test_default_matches_v0_1_output():
    X, y = _toy(4)
    for covariance, diagonal in [("total", False), ("diagonal", True)]:
        X_new, y_new = class_conditional_gaussian_augment(
            X, y, rho=1.5, seed=7, covariance=covariance
        )
        X_ref, y_ref = _v0_1_reference(X, y, rho=1.5, seed=7, diagonal=diagonal)
        assert np.array_equal(X_new, X_ref)
        assert np.array_equal(y_new, y_ref)

    X_def, _ = class_conditional_gaussian_augment(X, y, rho=1.5, seed=7)
    assert np.array_equal(X_def, _v0_1_reference(X, y, rho=1.5, seed=7)[0])


def test_pooled_is_deprecated_alias_of_total():
    X, y = _toy(5)
    with pytest.warns(FutureWarning, match="covariance='total'"):
        X_pooled, _, info = class_conditional_gaussian_augment(
            X, y, rho=1.0, seed=3, covariance="pooled", return_info=True
        )
    X_total, _ = class_conditional_gaussian_augment(X, y, rho=1.0, seed=3, covariance="total")
    assert np.array_equal(X_pooled, X_total)
    assert info.covariance == "total"


def test_within_covariance_is_lda_pooled_covariance():
    X, y = _toy(6, n_per_class=15)
    n, k = len(y), 3
    expected = sum((np.sum(y == c) - 1) * np.cov(X[y == c], rowvar=False) for c in range(k))
    expected = expected / (n - k)

    within = _estimate_covariance(X, y, "within", ridge=0.0)
    total = _estimate_covariance(X, y, "total", ridge=0.0)

    assert np.allclose(within, expected)
    # Classes are shifted by 3 per step, so between-class scatter inflates the total.
    assert np.trace(total) > 2 * np.trace(within)


def test_shrunk_within_covariances_match_sklearn_on_residuals():
    X, y = _toy(7)
    residuals = np.vstack([X[y == c] - X[y == c].mean(axis=0) for c in range(3)])
    lw = LedoitWolf(assume_centered=True).fit(residuals).covariance_
    oas = OAS(assume_centered=True).fit(residuals).covariance_
    assert np.allclose(_estimate_covariance(X, y, "within_lw", ridge=1e-4), lw + 1e-4 * np.eye(4))
    assert np.allclose(_estimate_covariance(X, y, "within_oas", ridge=0.0), oas)


def test_per_point_small_bandwidth_returns_near_copies_of_same_class_rows():
    X, y = _toy(8)
    X_aug, y_aug, info = class_conditional_gaussian_augment(
        X, y, rho=1.0, seed=1, center="per_point", bandwidth=1e-6, return_info=True
    )
    synthetic, labels = X_aug[len(y):], y_aug[len(y):]
    for row, cls in zip(synthetic, labels):
        nearest = np.abs(X - row).max(axis=1).argmin()
        assert np.abs(X[nearest] - row).max() < 1e-4
        assert y[nearest] == cls
    assert info.center == "per_point"
    assert info.bandwidth == 1e-6


def test_per_point_noise_scale_follows_bandwidth():
    # Every row of a class is the same point, so each synthetic row's anchor is known.
    y = np.repeat([0, 1], 50)
    X = np.where(y[:, None] == 0, [0.0, 0.0], [4.0, -2.0])
    X_aug, y_aug = class_conditional_gaussian_augment(
        X, y, rho=40.0, seed=2, covariance="diagonal", center="per_point", bandwidth=0.5
    )
    synthetic = X_aug[len(y):][y_aug[len(y):] == 0]
    expected_std = 0.5 * np.sqrt(np.diag(np.cov(X, rowvar=False)) + 1e-4)
    assert np.allclose(synthetic.mean(axis=0), 0.0, atol=0.05)
    assert np.allclose(synthetic.std(axis=0), expected_std, rtol=0.05)


@pytest.mark.parametrize(
    "kwargs",
    [{"covariance": "per_class"}, {"center": "kde"}, {"center": "per_point", "bandwidth": -1.0}],
)
def test_invalid_options_raise(kwargs):
    X, y = _toy(10)
    with pytest.raises(ValueError):
        class_conditional_gaussian_augment(X, y, rho=1.0, seed=0, **kwargs)
