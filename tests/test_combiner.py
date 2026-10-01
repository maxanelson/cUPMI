# SPDX-License-Identifier: Apache-2.0

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from cupmi import CUPMICombiner


def test_combiner_fit_predict_proba_contract():
    X, y = make_classification(
        n_samples=180,
        n_features=8,
        n_informative=6,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=4,
    )

    clf = CUPMICombiner(
        estimator=RandomForestClassifier(n_estimators=20, max_depth=3, random_state=0),
        rhos=(0.0, 0.5, 1.0),
        inner_cv=3,
        scoring="neg_log_loss",
        seed=0,
    )
    clf.fit(X, y)
    proba = clf.predict_proba(X[:7])

    assert clf.rho_ in {0.0, 0.5, 1.0}
    assert set(clf.rho_scores_) == {0.0, 0.5, 1.0}
    assert proba.shape == (7, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert set(clf.predict(X[:7])).issubset(set(np.unique(y)))


def test_combiner_selects_zero_when_inner_cv_disabled():
    X, y = make_classification(
        n_samples=80,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        n_classes=2,
        random_state=1,
    )

    clf = CUPMICombiner(estimator="lr", rhos=(0.0, 2.0), inner_cv=1, seed=0)
    clf.fit(X, y)

    assert clf.rho_ == 0.0


def test_combiner_supports_within_lw_noise_jitter():
    X, y = make_classification(
        n_samples=120,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=2,
    )

    clf = CUPMICombiner(
        estimator="lr",
        rhos=(0.0, 1.0),
        seed=0,
        covariance="within_lw",
        center="per_point",
        bandwidth=0.5,
    )
    clf.fit(X, y)

    assert clf.get_params()["center"] == "per_point"
    assert clf.predict_proba(X[:5]).shape == (5, 3)
