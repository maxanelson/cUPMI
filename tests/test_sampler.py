# SPDX-License-Identifier: Apache-2.0

import numpy as np

from cupmi import class_conditional_gaussian_augment


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
