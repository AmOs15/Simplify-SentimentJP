"""
Evaluation script for simplification quality assessment.

This script evaluates the quality of simplified texts by computing various
similarity metrics between original and simplified text pairs.

Usage:
    # Evaluate all splits (train, valid, test)
    python src/evaluate_simplification.py

    # Evaluate specific split
    python src/evaluate_simplification.py --split train

    # Evaluate with custom configuration
    python src/evaluate_simplification.py --split train --batch_size 16 --device cuda

    # Evaluate with custom thresholds
    python src/evaluate_simplification.py --cosine_threshold 0.75 --bertscore_threshold 0.85

    # Disable specific metrics
    python src/evaluate_simplification.py --no-bertscore --no-chrf
"""

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd

from evaluate import EvaluatorConfig, SimplificationEvaluator


def load_data(split: str, data_dir: Path = Path("data")) -> tuple:
    """Load original and simplified data for a given split.

    Args:
        split: Data split ('train', 'valid', or 'test')
        data_dir: Directory containing data files

    Returns:
        Tuple of (ids, original_texts, simplified_texts)
    """
    # Load original data
    orig_path = data_dir / f"wrime_subset_{split}.pkl"
    # orig_path = data_dir / f"wrime_{split}.pkl"
    if not orig_path.exists():
        raise FileNotFoundError(f"Original data not found: {orig_path}")

    orig_df = pd.read_pickle(orig_path)

    # Load simplified data
    simp_path = data_dir / f"wrime_simp_subset_{split}.pkl"
    if not simp_path.exists():
        raise FileNotFoundError(f"Simplified data not found: {simp_path}")

    simp_df = pd.read_pickle(simp_path)

    # Verify data integrity
    if len(orig_df) != len(simp_df):
        raise ValueError(
            f"Mismatch in data size: original={len(orig_df)}, simplified={len(simp_df)}"
        )

    # Extract texts
    ids = orig_df.index.tolist()
    original_texts = orig_df["text"].tolist()
    simplified_texts = simp_df["text"].tolist()

    return ids, original_texts, simplified_texts


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate simplification quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Data options
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "valid", "test", "all"],
        help="Data split to evaluate (default: all)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing data files (default: data)",
    )

    # Device and performance options
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use (default: auto)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for processing (default: 32)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="float32",
        choices=["float32", "float16"],
        help="Precision for embeddings (default: float32)",
    )

    # Model options
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="intfloat/multilingual-e5-large",
        help="Model for sentence embeddings (default: intfloat/multilingual-e5-large)",
    )
    parser.add_argument(
        "--bertscore_model",
        type=str,
        default="cl-tohoku/bert-base-japanese-v3",
        help="Model for BERTScore (default: cl-tohoku/bert-base-japanese-v3)",
    )

    # Metrics options
    parser.add_argument(
        "--no-bertscore",
        action="store_true",
        help="Disable BERTScore computation",
    )
    parser.add_argument(
        "--no-chrf",
        action="store_true",
        help="Disable chrF computation",
    )

    # Threshold options
    parser.add_argument(
        "--cosine_threshold",
        type=float,
        default=0.7,
        help="Threshold for cosine similarity (default: 0.7)",
    )
    parser.add_argument(
        "--bertscore_threshold",
        type=float,
        default=0.8,
        help="Threshold for BERTScore F1 (default: 0.8)",
    )

    # Output options
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_results",
        help="Directory to save results (default: eval_results)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Prefix for output files (default: split name)",
    )

    # Cache options
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable embedding cache",
    )

    # Other options
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress bars",
    )

    args = parser.parse_args()

    # Create evaluator configuration
    config = EvaluatorConfig(
        device=args.device,
        batch_size=args.batch_size,
        embedding_model_name=args.embedding_model,
        bertscore_model_name=args.bertscore_model,
        use_bertscore=not args.no_bertscore,
        use_chrf=not args.no_chrf,
        output_dir=Path(args.output_dir),
        seed=args.seed,
        cosine_threshold=args.cosine_threshold,
        bertscore_threshold=args.bertscore_threshold,
        precision=args.precision,
    )

    # Create evaluator
    evaluator = SimplificationEvaluator(config)

    # Determine splits to evaluate
    if args.split == "all":
        splits = ["train", "valid", "test"]
    else:
        splits = [args.split]

    # Evaluate each split
    data_dir = Path(args.data_dir)
    for split in splits:
        print(f"\n{'=' * 60}")
        print(f"Evaluating {split} split")
        print(f"{'=' * 60}")

        try:
            # Load data
            ids, original_texts, simplified_texts = load_data(split, data_dir)

            # Determine prefix
            prefix = args.prefix if args.prefix else split

            # Run evaluation
            df = evaluator.run_full_evaluation(
                original_texts=original_texts,
                simplified_texts=simplified_texts,
                ids=ids,
                prefix=prefix,
                use_cache=not args.no_cache,
                show_progress=not args.quiet,
            )

            print(f"\nCompleted evaluation for {split} split!")
            print(f"Results saved to: {config.output_dir}")

        except FileNotFoundError as e:
            print(f"Error: {e}")
            print(f"Skipping {split} split...")
            continue
        except Exception as e:
            print(f"Error evaluating {split} split: {e}")
            import traceback

            traceback.print_exc()
            continue

    print(f"\n{'=' * 60}")
    print("All evaluations completed!")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
