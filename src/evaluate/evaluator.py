"""
Main evaluator class that orchestrates simplification quality evaluation.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .config import EvaluatorConfig
from .embeddings import EmbeddingComputer
from .metrics import MetricsComputer
from .plotting import Plotter
from .resimplify import ResimplifyPolicy


class SimplificationEvaluator:
    """Main evaluator for simplification quality assessment."""

    def __init__(self, config: Optional[EvaluatorConfig] = None):
        """Initialize the evaluator.

        Args:
            config: Evaluator configuration (uses default if None)
        """
        self.config = config or EvaluatorConfig()

        # Initialize components
        self.embedding_computer = EmbeddingComputer(self.config)
        self.metrics_computer = MetricsComputer(self.config)
        self.plotter = Plotter(self.config)
        self.resimplify_policy = ResimplifyPolicy(self.config)

        # Set random seed for reproducibility
        np.random.seed(self.config.seed)

    def evaluate(
        self,
        original_texts: List[str],
        simplified_texts: List[str],
        ids: Optional[List[any]] = None,
        use_cache: bool = True,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Evaluate simplification quality for text pairs.

        Args:
            original_texts: List of original texts
            simplified_texts: List of simplified texts
            ids: Optional list of IDs for each text pair
            use_cache: Whether to use cached embeddings
            show_progress: Whether to show progress bars

        Returns:
            DataFrame containing evaluation results
        """
        if len(original_texts) != len(simplified_texts):
            raise ValueError(
                f"Number of original texts ({len(original_texts)}) must match "
                f"number of simplified texts ({len(simplified_texts)})"
            )

        print(f"\nEvaluating {len(original_texts)} text pairs...")
        print("=" * 60)

        # Create IDs if not provided
        if ids is None:
            ids = list(range(len(original_texts)))

        # Initialize results dictionary
        results = {
            "id": ids,
            "original_text": original_texts,
            "simplified_text": simplified_texts,
        }

        # Compute cosine similarity (always computed)
        print("\n1. Computing sentence embeddings and cosine similarity...")
        cosine_similarities = self.embedding_computer.compute_similarity_for_pairs(
            original_texts=original_texts,
            simplified_texts=simplified_texts,
            use_cache=use_cache,
            show_progress=show_progress,
        )
        results["cosine_similarity"] = cosine_similarities

        # Compute additional metrics
        additional_metrics = self.metrics_computer.compute_all_metrics(
            original_texts=original_texts,
            simplified_texts=simplified_texts,
            show_progress=show_progress,
        )

        # Add additional metrics to results
        for metric_name, metric_values in additional_metrics.items():
            results[metric_name] = metric_values

        # Create DataFrame
        df = pd.DataFrame(results)
        df.set_index("id", inplace=True)

        print("\n" + "=" * 60)
        print("Evaluation completed!")

        return df

    def compute_statistics(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Compute summary statistics for all metrics.

        Args:
            df: DataFrame containing evaluation results

        Returns:
            Dictionary of statistics for each metric
        """
        statistics = {}

        metric_columns = [
            "cosine_similarity",
            "bertscore_f1",
            "bertscore_precision",
            "bertscore_recall",
            "chrf",
        ]

        for col in metric_columns:
            if col in df.columns:
                values = df[col].values
                statistics[col] = {
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "q25": float(np.percentile(values, 25)),
                    "q75": float(np.percentile(values, 75)),
                    "skewness": float(stats.skew(values)),
                    "kurtosis": float(stats.kurtosis(values)),
                }

        return statistics

    def save_results(
        self,
        df: pd.DataFrame,
        prefix: str = "evaluation",
    ):
        """Save evaluation results to files.

        Args:
            df: DataFrame containing evaluation results
            prefix: Prefix for output filenames
        """
        # Save DataFrame to CSV
        csv_path = self.config.output_dir / f"{prefix}_results.csv"
        df.to_csv(csv_path, index=True)
        print(f"\nSaved results to: {csv_path}")

        # Compute and save statistics
        statistics = self.compute_statistics(df)
        stats_path = self.config.output_dir / f"{prefix}_statistics.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(statistics, f, indent=2, ensure_ascii=False)
        print(f"Saved statistics to: {stats_path}")

        # Print statistics summary
        print("\nStatistics Summary:")
        print("-" * 60)
        for metric_name, metric_stats in statistics.items():
            print(
                f"{metric_name}: "
                f"mean={metric_stats['mean']:.3f}, "
                f"median={metric_stats['median']:.3f}, "
                f"std={metric_stats['std']:.3f}"
            )

    def generate_visualizations(self, df: pd.DataFrame):
        """Generate all visualizations for evaluation results.

        Args:
            df: DataFrame containing evaluation results
        """
        self.plotter.plot_all_metrics(df)

    def generate_quality_report(
        self,
        df: pd.DataFrame,
        prefix: str = "evaluation",
    ):
        """Generate a comprehensive quality report.

        Args:
            df: DataFrame containing evaluation results
            prefix: Prefix for output filenames
        """
        # Print quality report
        self.resimplify_policy.print_quality_report(df)

        # Export low quality pairs
        candidates_path = self.config.output_dir / f"{prefix}_resimplify_candidates.csv"
        self.resimplify_policy.export_resimplification_candidates(
            df=df,
            output_path=str(candidates_path),
            use_or_logic=True,
        )

        # Get quality summary
        quality_summary = self.resimplify_policy.get_quality_summary(df)

        # Save quality summary
        summary_path = self.config.output_dir / f"{prefix}_quality_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(quality_summary, f, indent=2, ensure_ascii=False)
        print(f"Saved quality summary to: {summary_path}")

    def run_full_evaluation(
        self,
        original_texts: List[str],
        simplified_texts: List[str],
        ids: Optional[List[any]] = None,
        prefix: str = "evaluation",
        use_cache: bool = True,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Run complete evaluation pipeline.

        Args:
            original_texts: List of original texts
            simplified_texts: List of simplified texts
            ids: Optional list of IDs for each text pair
            prefix: Prefix for output filenames
            use_cache: Whether to use cached embeddings
            show_progress: Whether to show progress bars

        Returns:
            DataFrame containing evaluation results
        """
        # Evaluate
        df = self.evaluate(
            original_texts=original_texts,
            simplified_texts=simplified_texts,
            ids=ids,
            use_cache=use_cache,
            show_progress=show_progress,
        )

        # Save results
        self.save_results(df, prefix=prefix)

        # Generate visualizations
        self.generate_visualizations(df)

        # Generate quality report
        self.generate_quality_report(df, prefix=prefix)

        return df
