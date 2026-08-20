# SPDX-License-Identifier: Apache-2.0
"""Run cUPMI on synthetic probability streams."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cupmi import evaluate_precomputed_streams  # noqa: E402
from cupmi.metrics import quadratic_weighted_kappa  # noqa: E402
from cupmi.meta_features import align_predict_proba  # noqa: E402


def _stream_oof_proba(X: np.ndarray, y: np.ndarray, folds: np.ndarray, seed: int) -> np.ndarray:
    classes = np.unique(y)
    proba = np.zeros((len(y), len(classes)), dtype=float)
    for fold in np.unique(folds):
        test = folds == fold
        train = ~test
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, C=0.8, class_weight="balanced"),
        )
        model.fit(X[train], y[train])
        proba[test] = align_predict_proba(model, X[test], classes)
    return proba


def make_synthetic_streams(seed: int = 7):
    rng = np.random.default_rng(seed)
    X, y = make_classification(
        n_samples=360,
        n_features=20,
        n_informative=10,
        n_redundant=3,
        n_classes=3,
        n_clusters_per_class=1,
        weights=[0.25, 0.5, 0.25],
        class_sep=1.15,
        flip_y=0.08,
        random_state=seed,
    )
    folds = np.empty(len(y), dtype=int)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (_, test_idx) in enumerate(splitter.split(X, y)):
        folds[test_idx] = fold

    streams = []
    for stream_idx in range(3):
        cols = rng.choice(X.shape[1], size=10, replace=False)
        noisy_view = X[:, cols] + rng.normal(0.0, 0.25 + 0.08 * stream_idx, size=(len(y), 10))
        streams.append(_stream_oof_proba(noisy_view, y, folds, seed + stream_idx))
    return streams, y, folds


def main() -> None:
    streams, y, folds = make_synthetic_streams()
    stacker = RandomForestClassifier(
        n_estimators=80,
        max_depth=5,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=0,
        n_jobs=1,
    )
    result = evaluate_precomputed_streams(
        streams,
        y,
        folds,
        estimator=stacker,
        rhos=(0.0, 1.0, 2.0),
        inner_cv=3,
        scoring="roc_auc_ovr",
        seed=0,
    )

    stack_qwk = quadratic_weighted_kappa(y, result.stack_proba, np.asarray(result.classes))
    cupmi_qwk = quadratic_weighted_kappa(y, result.cupmi_proba, np.asarray(result.classes))
    print("Synthetic cUPMI demo")
    print(f"Macro AUC stack: {result.stack_score:.3f}")
    print(f"Macro AUC cUPMI: {result.cupmi_score:.3f}")
    print(f"Delta AUC:       {result.delta:+.3f}")
    print(f"QWK stack:       {stack_qwk:.3f}")
    print(f"QWK cUPMI:       {cupmi_qwk:.3f}")
    print(f"Selected rhos:   {list(result.selected_rhos)}")
    print("Note: cUPMI regularizes the combiner; it is not expected to win on every draw.")


if __name__ == "__main__":
    main()
