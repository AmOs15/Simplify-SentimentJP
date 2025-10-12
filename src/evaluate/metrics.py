"""
Additional metrics for simplification quality evaluation.
"""

from typing import Dict, List, Tuple

import numpy as np
from bert_score import score as bert_score_func
from sacrebleu.metrics import CHRF
from tqdm import tqdm

from .config import EvaluatorConfig


class MetricsComputer:
    """Computes various metrics for simplification quality evaluation."""

    def __init__(self, config: EvaluatorConfig):
        """Initialize the metrics computer.

        Args:
            config: Evaluator configuration
        """
        self.config = config

    def compute_bertscore(
        self,
        original_texts: List[str],
        simplified_texts: List[str],
        show_progress: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Compute BERTScore for pairs of original and simplified texts.

        Args:
            original_texts: List of original texts (references)
            simplified_texts: List of simplified texts (candidates)
            show_progress: Whether to show progress bar

        Returns:
            Dictionary containing 'precision', 'recall', 'f1' arrays
        """
        if not self.config.use_bertscore:
            raise ValueError("BERTScore is disabled in configuration")

        if len(original_texts) != len(simplified_texts):
            raise ValueError(
                f"Number of original texts ({len(original_texts)}) must match "
                f"number of simplified texts ({len(simplified_texts)})"
            )

        if show_progress:
            print(f"Computing BERTScore for {len(original_texts)} pairs...")

        # Compute BERTScore
        # Note: bert_score expects candidates (simplified) and references (original)
        P, R, F1 = bert_score_func(
            cands=simplified_texts,
            refs=original_texts,
            model_type=self.config.bertscore_model_name,
            lang="ja",
            verbose=show_progress,
            device=self.config.device if self.config.device != "auto" else None,
            batch_size=self.config.batch_size,
        )

        # Convert to numpy arrays
        precision = P.numpy()
        recall = R.numpy()
        f1 = F1.numpy()

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def compute_chrf(
        self,
        original_texts: List[str],
        simplified_texts: List[str],
        show_progress: bool = True,
    ) -> np.ndarray:
        """Compute chrF score for pairs of original and simplified texts.

        Args:
            original_texts: List of original texts (references)
            simplified_texts: List of simplified texts (hypotheses)
            show_progress: Whether to show progress bar

        Returns:
            Array of chrF scores
        """
        if not self.config.use_chrf:
            raise ValueError("chrF is disabled in configuration")

        if len(original_texts) != len(simplified_texts):
            raise ValueError(
                f"Number of original texts ({len(original_texts)}) must match "
                f"number of simplified texts ({len(simplified_texts)})"
            )

        if show_progress:
            print(f"Computing chrF for {len(original_texts)} pairs...")

        chrf = CHRF()
        scores = []

        iterator = (
            tqdm(zip(original_texts, simplified_texts), total=len(original_texts))
            if show_progress
            else zip(original_texts, simplified_texts)
        )

        for orig, simp in iterator:
            # chrF expects a list of references and a hypothesis
            score = chrf.sentence_score(hypothesis=simp, references=[orig])
            scores.append(score.score / 100.0)  # Normalize to [0, 1]

        return np.array(scores)

    def compute_all_metrics(
        self,
        original_texts: List[str],
        simplified_texts: List[str],
        show_progress: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Compute all enabled metrics.

        Args:
            original_texts: List of original texts
            simplified_texts: List of simplified texts
            show_progress: Whether to show progress bar

        Returns:
            Dictionary containing all computed metrics
        """
        metrics = {}

        # Compute BERTScore if enabled
        if self.config.use_bertscore:
            bertscore_results = self.compute_bertscore(
                original_texts, simplified_texts, show_progress=show_progress
            )
            metrics["bertscore_precision"] = bertscore_results["precision"]
            metrics["bertscore_recall"] = bertscore_results["recall"]
            metrics["bertscore_f1"] = bertscore_results["f1"]

        # Compute chrF if enabled
        if self.config.use_chrf:
            chrf_scores = self.compute_chrf(
                original_texts, simplified_texts, show_progress=show_progress
            )
            metrics["chrf"] = chrf_scores

        return metrics
