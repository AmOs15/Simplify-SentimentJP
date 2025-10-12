"""
Configuration for simplification evaluation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class EvaluatorConfig:
    """Configuration for simplification evaluator.

    Attributes:
        device: Device to use for computation ('cuda', 'mps', 'cpu', or 'auto')
        batch_size: Batch size for embedding and metric computation
        embedding_model_name: Model name for sentence embeddings (e.g., 'intfloat/multilingual-e5-large')
        bertscore_model_name: Model name for BERTScore computation (e.g., 'cl-tohoku/bert-base-japanese-v3')
        use_bertscore: Whether to compute BERTScore
        use_chrf: Whether to compute chrF score
        cache_dir: Directory to cache embeddings
        output_dir: Directory to save evaluation results
        plot_dir: Directory to save plots
        seed: Random seed for reproducibility
        cosine_threshold: Threshold for cosine similarity to flag low-quality pairs
        bertscore_threshold: Threshold for BERTScore F1 to flag low-quality pairs
        precision: Precision for embeddings ('float32' or 'float16')
    """

    # Device configuration
    device: str = "auto"
    batch_size: int = 32

    # Model configuration
    embedding_model_name: str = "intfloat/multilingual-e5-large"
    bertscore_model_name: str = "cl-tohoku/bert-base-japanese-v3"

    # Metrics to compute
    use_bertscore: bool = True
    use_chrf: bool = True

    # Directory configuration
    cache_dir: Path = field(default_factory=lambda: Path("cache/embeddings"))
    output_dir: Path = field(default_factory=lambda: Path("eval_results"))
    plot_dir: Path = field(default_factory=lambda: Path("eval_results/plots"))

    # Reproducibility
    seed: int = 42

    # Thresholds for quality control
    cosine_threshold: float = 0.7
    bertscore_threshold: float = 0.8

    # Precision
    precision: str = "float32"  # or "float16"

    def __post_init__(self):
        """Ensure directories are Path objects and create them if needed."""
        self.cache_dir = Path(self.cache_dir)
        self.output_dir = Path(self.output_dir)
        self.plot_dir = Path(self.plot_dir)

        # Create directories if they don't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)

        # Validate precision
        if self.precision not in ["float32", "float16"]:
            raise ValueError(f"Invalid precision: {self.precision}. Must be 'float32' or 'float16'")

        # Validate device
        if self.device not in ["auto", "cuda", "mps", "cpu"]:
            raise ValueError(f"Invalid device: {self.device}. Must be 'auto', 'cuda', 'mps', or 'cpu'")
