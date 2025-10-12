"""
Visualization functions for evaluation results.
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import EvaluatorConfig


class Plotter:
    """Creates visualizations for evaluation results."""

    def __init__(self, config: EvaluatorConfig):
        """Initialize the plotter.

        Args:
            config: Evaluator configuration
        """
        self.config = config

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "DejaVu Sans"]
        plt.rcParams["figure.figsize"] = (10, 6)
        plt.rcParams["figure.dpi"] = 100

    def plot_histogram(
        self,
        values: np.ndarray,
        title: str,
        xlabel: str,
        output_filename: str,
        bins: int = 50,
        threshold: Optional[float] = None,
    ):
        """Plot histogram for a metric.

        Args:
            values: Array of metric values
            title: Plot title
            xlabel: X-axis label
            output_filename: Output filename (without directory)
            bins: Number of bins
            threshold: Optional threshold line to draw
        """
        plt.figure(figsize=(10, 6))

        # Plot histogram
        plt.hist(values, bins=bins, alpha=0.7, color="steelblue", edgecolor="black")

        # Add threshold line if provided
        if threshold is not None:
            plt.axvline(
                x=threshold,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Threshold: {threshold:.2f}",
            )
            plt.legend()

        # Add statistics
        mean_val = np.mean(values)
        median_val = np.median(values)
        plt.axvline(x=mean_val, color="green", linestyle="--", linewidth=1.5, alpha=0.7)
        plt.axvline(x=median_val, color="orange", linestyle="--", linewidth=1.5, alpha=0.7)

        # Labels and title
        plt.xlabel(xlabel)
        plt.ylabel("Frequency")
        plt.title(f"{title}\n(Mean: {mean_val:.3f}, Median: {median_val:.3f})")
        plt.grid(True, alpha=0.3)

        # Save
        output_path = self.config.plot_dir / output_filename
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved histogram: {output_path}")

    def plot_boxplot(
        self,
        data_dict: Dict[str, np.ndarray],
        title: str,
        ylabel: str,
        output_filename: str,
    ):
        """Plot boxplot for multiple metrics.

        Args:
            data_dict: Dictionary mapping metric names to value arrays
            title: Plot title
            ylabel: Y-axis label
            output_filename: Output filename (without directory)
        """
        plt.figure(figsize=(max(10, len(data_dict) * 2), 6))

        # Prepare data for boxplot
        data = list(data_dict.values())
        labels = list(data_dict.keys())

        # Create boxplot
        bp = plt.boxplot(data, labels=labels, patch_artist=True)

        # Customize colors
        for patch in bp["boxes"]:
            patch.set_facecolor("lightblue")

        # Labels and title
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks(rotation=45, ha="right")
        plt.grid(True, alpha=0.3, axis="y")

        # Save
        output_path = self.config.plot_dir / output_filename
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved boxplot: {output_path}")

    def plot_scatter(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray,
        xlabel: str,
        ylabel: str,
        title: str,
        output_filename: str,
        x_threshold: Optional[float] = None,
        y_threshold: Optional[float] = None,
    ):
        """Plot scatter plot for two metrics.

        Args:
            x_values: X-axis values
            y_values: Y-axis values
            xlabel: X-axis label
            ylabel: Y-axis label
            title: Plot title
            output_filename: Output filename (without directory)
            x_threshold: Optional threshold for x-axis
            y_threshold: Optional threshold for y-axis
        """
        plt.figure(figsize=(10, 8))

        # Create scatter plot
        plt.scatter(x_values, y_values, alpha=0.5, s=20, color="steelblue")

        # Add threshold lines if provided
        if x_threshold is not None:
            plt.axvline(
                x=x_threshold,
                color="red",
                linestyle="--",
                linewidth=2,
                alpha=0.7,
                label=f"{xlabel} Threshold: {x_threshold:.2f}",
            )
        if y_threshold is not None:
            plt.axhline(
                y=y_threshold,
                color="orange",
                linestyle="--",
                linewidth=2,
                alpha=0.7,
                label=f"{ylabel} Threshold: {y_threshold:.2f}",
            )

        # Add correlation coefficient
        correlation = np.corrcoef(x_values, y_values)[0, 1]

        # Labels and title
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f"{title}\n(Correlation: {correlation:.3f})")
        plt.grid(True, alpha=0.3)

        if x_threshold is not None or y_threshold is not None:
            plt.legend()

        # Save
        output_path = self.config.plot_dir / output_filename
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved scatter plot: {output_path}")

    def plot_all_metrics(self, df: pd.DataFrame):
        """Create all standard plots for evaluation results.

        Args:
            df: DataFrame containing evaluation results with metrics as columns
        """
        print("Creating visualizations...")

        # Histogram for each metric
        if "cosine_similarity" in df.columns:
            self.plot_histogram(
                df["cosine_similarity"].values,
                title="Cosine Similarity Distribution",
                xlabel="Cosine Similarity",
                output_filename="hist_cosine_similarity.png",
                threshold=self.config.cosine_threshold,
            )

        if "bertscore_f1" in df.columns:
            self.plot_histogram(
                df["bertscore_f1"].values,
                title="BERTScore F1 Distribution",
                xlabel="BERTScore F1",
                output_filename="hist_bertscore_f1.png",
                threshold=self.config.bertscore_threshold,
            )

        if "chrf" in df.columns:
            self.plot_histogram(
                df["chrf"].values,
                title="chrF Score Distribution",
                xlabel="chrF Score",
                output_filename="hist_chrf.png",
            )

        # Boxplot for all metrics
        metrics_for_boxplot = {}
        for col in ["cosine_similarity", "bertscore_f1", "bertscore_precision", "bertscore_recall", "chrf"]:
            if col in df.columns:
                metrics_for_boxplot[col] = df[col].values

        if metrics_for_boxplot:
            self.plot_boxplot(
                metrics_for_boxplot,
                title="Comparison of All Metrics",
                ylabel="Score",
                output_filename="boxplot_all_metrics.png",
            )

        # Scatter plots for metric pairs
        if "cosine_similarity" in df.columns and "bertscore_f1" in df.columns:
            self.plot_scatter(
                df["cosine_similarity"].values,
                df["bertscore_f1"].values,
                xlabel="Cosine Similarity",
                ylabel="BERTScore F1",
                title="Cosine Similarity vs BERTScore F1",
                output_filename="scatter_cosine_vs_bertscore.png",
                x_threshold=self.config.cosine_threshold,
                y_threshold=self.config.bertscore_threshold,
            )

        if "cosine_similarity" in df.columns and "chrf" in df.columns:
            self.plot_scatter(
                df["cosine_similarity"].values,
                df["chrf"].values,
                xlabel="Cosine Similarity",
                ylabel="chrF Score",
                title="Cosine Similarity vs chrF",
                output_filename="scatter_cosine_vs_chrf.png",
                x_threshold=self.config.cosine_threshold,
            )

        if "bertscore_f1" in df.columns and "chrf" in df.columns:
            self.plot_scatter(
                df["bertscore_f1"].values,
                df["chrf"].values,
                xlabel="BERTScore F1",
                ylabel="chrF Score",
                title="BERTScore F1 vs chrF",
                output_filename="scatter_bertscore_vs_chrf.png",
                y_threshold=self.config.bertscore_threshold,
            )

        print("All visualizations created successfully!")
