# SPDX-License-Identifier: Apache-2.0
"""Scikit-learn-compatible cUPMI combiner."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from .meta_features import align_predict_proba
from .sampler import class_conditional_gaussian_augment

Scoring = str | Callable[[np.ndarray, np.ndarray, np.ndarray], float]


def make_default_estimator(kind: str = "rf", *, seed: int | None = None, n_classes: int | None = None):
    """Create one of the documented level-1 estimators."""

    if kind == "lr":
        return LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=4,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        )
    if kind == "xgb":
        from xgboost import XGBClassifier

        params: dict[str, Any] = {
            "n_estimators": 200,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "mlogloss" if n_classes != 2 else "logloss",
            "random_state": seed,
            "n_jobs": 1,
            "verbosity": 0,
        }
        if n_classes == 2:
            params["objective"] = "binary:logistic"
        else:
            params["objective"] = "multi:softprob"
            params["num_class"] = n_classes
        return XGBClassifier(**params)
    raise ValueError("kind must be 'lr', 'rf', or 'xgb'.")


def _clone_with_seed(estimator, seed: int | None, n_classes: int):
    if estimator is None:
        return make_default_estimator("rf", seed=seed, n_classes=n_classes)
    if isinstance(estimator, str):
        return make_default_estimator(estimator, seed=seed, n_classes=n_classes)
    fresh = clone(estimator)
    params = fresh.get_params(deep=False)
    if seed is not None and "random_state" in params and params["random_state"] is None:
        fresh.set_params(random_state=seed)
    return fresh


def _score_probabilities(
    y_true: np.ndarray,
    proba: np.ndarray,
    classes: np.ndarray,
    scoring: Scoring,
) -> float:
    if callable(scoring):
        return float(scoring(y_true, proba, classes))
    if scoring == "roc_auc_ovr":
        if len(classes) == 2:
            return float(roc_auc_score(y_true, proba[:, 1]))
        return float(
            roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=classes)
        )
    if scoring == "neg_log_loss":
        clipped = np.clip(proba, 1e-12, 1.0)
        clipped /= clipped.sum(axis=1, keepdims=True)
        return float(-log_loss(y_true, clipped, labels=classes))
    if scoring == "accuracy":
        pred = classes[proba.argmax(axis=1)]
        return float(accuracy_score(y_true, pred))
    raise ValueError("scoring must be 'roc_auc_ovr', 'neg_log_loss', 'accuracy', or a callable.")


class CUPMICombiner(BaseEstimator, ClassifierMixin):
    """Level-1 combiner with inner-CV cUPMI augmentation.

    The wrapped estimator must implement ``fit`` and ``predict_proba``. ``rho``
    is selected only on the training data supplied to ``fit``.
    """

    def __init__(
        self,
        estimator=None,
        *,
        rhos: Sequence[float] = (0.0, 1.0, 2.0, 3.0, 4.0),
        inner_cv: int = 3,
        scoring: Scoring = "roc_auc_ovr",
        seed: int | None = None,
        covariance: str = "pooled",
        ridge: float = 1e-4,
    ):
        self.estimator = estimator
        self.rhos = rhos
        self.inner_cv = inner_cv
        self.scoring = scoring
        self.seed = seed
        self.covariance = covariance
        self.ridge = ridge

    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=float)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.rhos_ = tuple(float(rho) for rho in self.rhos)
        self.rho_scores_ = self._select_rho(X, y)
        self.rho_ = max(self.rho_scores_, key=lambda rho: (self.rho_scores_[rho], -rho))

        X_aug, y_aug = class_conditional_gaussian_augment(
            X,
            y,
            self.rho_,
            seed=self.seed,
            covariance=self.covariance,
            ridge=self.ridge,
        )
        self.estimator_ = _clone_with_seed(self.estimator, self.seed, len(self.classes_))
        self.estimator_.fit(X_aug, y_aug)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "estimator_")
        X = check_array(X, dtype=float)
        return align_predict_proba(self.estimator_, X, self.classes_)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def score(self, X, y):
        proba = self.predict_proba(X)
        return _score_probabilities(np.asarray(y), proba, self.classes_, self.scoring)

    def _select_rho(self, X: np.ndarray, y: np.ndarray) -> dict[float, float]:
        if not self.rhos_:
            raise ValueError("rhos must contain at least one value.")
        if int(self.inner_cv) < 2:
            return {rho: 0.0 for rho in self.rhos_}

        _, counts = np.unique(y, return_counts=True)
        n_splits = min(int(self.inner_cv), int(counts.min()))
        if n_splits < 2:
            return {rho: 0.0 for rho in self.rhos_}

        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        scores: dict[float, float] = {}
        for rho in self.rhos_:
            fold_scores = []
            for train_idx, valid_idx in splitter.split(X, y):
                X_aug, y_aug = class_conditional_gaussian_augment(
                    X[train_idx],
                    y[train_idx],
                    rho,
                    seed=self.seed,
                    covariance=self.covariance,
                    ridge=self.ridge,
                )
                model = _clone_with_seed(self.estimator, self.seed, len(self.classes_))
                model.fit(X_aug, y_aug)
                proba = align_predict_proba(model, X[valid_idx], self.classes_)
                fold_scores.append(
                    _score_probabilities(y[valid_idx], proba, self.classes_, self.scoring)
                )
            scores[rho] = float(np.mean(fold_scores))
        return scores
