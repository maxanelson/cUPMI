# SPDX-License-Identifier: Apache-2.0
"""Public API for cUPMI."""

from .combiner import CUPMICombiner, make_default_estimator
from .meta_features import stack_log_proba, to_log_proba
from .sampler import class_conditional_gaussian_augment
from .stacking import EvaluationResult, evaluate_precomputed_streams

__all__ = [
    "CUPMICombiner",
    "EvaluationResult",
    "class_conditional_gaussian_augment",
    "evaluate_precomputed_streams",
    "make_default_estimator",
    "stack_log_proba",
    "to_log_proba",
]

__version__ = "0.1.0"
