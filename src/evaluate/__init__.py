"""
Evaluation module for assessing simplification quality.

This module provides functionality to evaluate simplified text quality
by computing semantic similarity metrics between original and simplified texts.
"""

from .config import EvaluatorConfig
from .evaluator import SimplificationEvaluator
from .resimplify import ResimplifyPolicy

__all__ = [
    "EvaluatorConfig",
    "SimplificationEvaluator",
    "ResimplifyPolicy",
]
