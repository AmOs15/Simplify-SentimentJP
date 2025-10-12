"""
Policy for determining which text pairs need re-simplification.
"""

from typing import Dict, List, Optional, Tuple

import pandas as pd

from .config import EvaluatorConfig


class ResimplifyPolicy:
    """Determines which text pairs should be re-simplified based on quality metrics."""

    def __init__(self, config: EvaluatorConfig):
        """Initialize the resimplify policy.

        Args:
            config: Evaluator configuration
        """
        self.config = config

    def identify_low_quality_pairs(
        self,
        df: pd.DataFrame,
        use_or_logic: bool = True,
    ) -> pd.DataFrame:
        """Identify text pairs that need re-simplification.

        Args:
            df: DataFrame containing evaluation results
            use_or_logic: If True, flag pairs that fail ANY threshold (OR logic).
                         If False, flag pairs that fail ALL thresholds (AND logic).

        Returns:
            DataFrame containing only the low-quality pairs
        """
        # Initialize mask (all False)
        if use_or_logic:
            mask = pd.Series([False] * len(df), index=df.index)
        else:
            mask = pd.Series([True] * len(df), index=df.index)

        # Check cosine similarity threshold
        if "cosine_similarity" in df.columns:
            cosine_mask = df["cosine_similarity"] < self.config.cosine_threshold
            if use_or_logic:
                mask = mask | cosine_mask
            else:
                mask = mask & cosine_mask

        # Check BERTScore F1 threshold
        if "bertscore_f1" in df.columns:
            bertscore_mask = df["bertscore_f1"] < self.config.bertscore_threshold
            if use_or_logic:
                mask = mask | bertscore_mask
            else:
                mask = mask & bertscore_mask

        # Filter DataFrame
        low_quality_df = df[mask].copy()

        return low_quality_df

    def get_quality_summary(self, df: pd.DataFrame) -> Dict[str, any]:
        """Get summary statistics about quality.

        Args:
            df: DataFrame containing evaluation results

        Returns:
            Dictionary containing quality summary statistics
        """
        summary = {
            "total_pairs": len(df),
            "low_quality_pairs": 0,
            "low_quality_percentage": 0.0,
            "failing_metrics": {},
        }

        # Count failures for each metric
        if "cosine_similarity" in df.columns:
            cosine_failures = (df["cosine_similarity"] < self.config.cosine_threshold).sum()
            summary["failing_metrics"]["cosine_similarity"] = {
                "count": int(cosine_failures),
                "percentage": float(cosine_failures / len(df) * 100),
                "threshold": self.config.cosine_threshold,
            }

        if "bertscore_f1" in df.columns:
            bertscore_failures = (df["bertscore_f1"] < self.config.bertscore_threshold).sum()
            summary["failing_metrics"]["bertscore_f1"] = {
                "count": int(bertscore_failures),
                "percentage": float(bertscore_failures / len(df) * 100),
                "threshold": self.config.bertscore_threshold,
            }

        # Get low quality pairs (using OR logic by default)
        low_quality_df = self.identify_low_quality_pairs(df, use_or_logic=True)
        summary["low_quality_pairs"] = len(low_quality_df)
        summary["low_quality_percentage"] = len(low_quality_df) / len(df) * 100

        return summary

    def create_resimplification_list(
        self,
        df: pd.DataFrame,
        use_or_logic: bool = True,
        max_samples: Optional[int] = None,
        sort_by: Optional[str] = None,
    ) -> List[Dict[str, any]]:
        """Create a list of text pairs that need re-simplification.

        Args:
            df: DataFrame containing evaluation results
            use_or_logic: If True, flag pairs that fail ANY threshold
            max_samples: Maximum number of samples to return (None for all)
            sort_by: Column name to sort by (lower is worse)

        Returns:
            List of dictionaries containing text pairs and their scores
        """
        # Identify low quality pairs
        low_quality_df = self.identify_low_quality_pairs(df, use_or_logic=use_or_logic)

        # Sort if requested
        if sort_by and sort_by in low_quality_df.columns:
            low_quality_df = low_quality_df.sort_values(by=sort_by, ascending=True)

        # Limit number of samples if requested
        if max_samples is not None:
            low_quality_df = low_quality_df.head(max_samples)

        # Convert to list of dictionaries
        resimplification_list = []
        for idx, row in low_quality_df.iterrows():
            entry = {
                "index": idx,
                "original_text": row.get("original_text", ""),
                "simplified_text": row.get("simplified_text", ""),
                "scores": {},
            }

            # Add all score columns
            score_columns = [
                "cosine_similarity",
                "bertscore_f1",
                "bertscore_precision",
                "bertscore_recall",
                "chrf",
            ]
            for col in score_columns:
                if col in row:
                    entry["scores"][col] = float(row[col])

            resimplification_list.append(entry)

        return resimplification_list

    def print_quality_report(self, df: pd.DataFrame):
        """Print a detailed quality report.

        Args:
            df: DataFrame containing evaluation results
        """
        summary = self.get_quality_summary(df)

        print("\n" + "=" * 60)
        print("SIMPLIFICATION QUALITY REPORT")
        print("=" * 60)
        print(f"\nTotal text pairs evaluated: {summary['total_pairs']}")
        print(
            f"Low quality pairs (need re-simplification): {summary['low_quality_pairs']} "
            f"({summary['low_quality_percentage']:.2f}%)"
        )
        print("\nFailures by metric:")
        print("-" * 60)

        for metric_name, metric_info in summary["failing_metrics"].items():
            print(
                f"  {metric_name}:"
                f" {metric_info['count']} failures ({metric_info['percentage']:.2f}%)"
                f" [threshold: {metric_info['threshold']:.2f}]"
            )

        print("=" * 60 + "\n")

    def export_resimplification_candidates(
        self,
        df: pd.DataFrame,
        output_path: str,
        use_or_logic: bool = True,
        max_samples: Optional[int] = None,
    ):
        """Export low quality pairs to a file for re-simplification.

        Args:
            df: DataFrame containing evaluation results
            output_path: Path to save the candidates
            use_or_logic: If True, flag pairs that fail ANY threshold
            max_samples: Maximum number of samples to export (None for all)
        """
        low_quality_df = self.identify_low_quality_pairs(df, use_or_logic=use_or_logic)

        # Sort by cosine similarity (worst first)
        if "cosine_similarity" in low_quality_df.columns:
            low_quality_df = low_quality_df.sort_values(by="cosine_similarity", ascending=True)

        # Limit if requested
        if max_samples is not None:
            low_quality_df = low_quality_df.head(max_samples)

        # Save to CSV
        low_quality_df.to_csv(output_path, index=True)
        print(f"Exported {len(low_quality_df)} candidates to: {output_path}")
