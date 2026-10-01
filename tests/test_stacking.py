# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from cupmi import evaluate_precomputed_streams


def _prob_stream(y, seed, confidence=0.72):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    n_classes = len(np.unique(y))
    proba = np.full((len(y), n_classes), (1.0 - confidence) / (n_classes - 1))
    proba[np.arange(len(y)), y] = confidence
    proba += rng.normal(0, 0.035, size=proba.shape)
    proba = np.clip(proba, 1e-4, None)
    proba /= proba.sum(axis=1, keepdims=True)
    return proba


def test_evaluate_precomputed_streams_shapes_and_scores():
    y = np.tile([0, 1, 2], 50)
    folds = np.arange(len(y)) % 5
    streams = [_prob_stream(y, 0), _prob_stream(y, 1, confidence=0.65)]

    result = evaluate_precomputed_streams(
        streams,
        y,
        folds,
        estimator=RandomForestClassifier(n_estimators=20, max_depth=3, random_state=0),
        rhos=(0.0, 1.0),
        inner_cv=3,
        scoring="accuracy",
        seed=0,
    )

    assert result.stack_proba.shape == (150, 3)
    assert result.cupmi_proba.shape == (150, 3)
    assert len(result.selected_rhos) == 5
    assert 0.0 <= result.stack_score <= 1.0
    assert 0.0 <= result.cupmi_score <= 1.0
    assert result.delta == result.cupmi_score - result.stack_score


def test_evaluate_precomputed_streams_forwards_covariance():
    y = np.tile([0, 1, 2], 30)
    folds = np.arange(len(y)) % 3
    streams = [_prob_stream(y, 0), _prob_stream(y, 1)]
    with pytest.raises(ValueError, match="covariance"):
        evaluate_precomputed_streams(
            streams,
            y,
            folds,
            estimator=RandomForestClassifier(n_estimators=5, random_state=0),
            rhos=(1.0,),
            seed=0,
            covariance="not-a-mode",
        )
