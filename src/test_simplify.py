"""
LLMによる平易化の検証用スクリプト

train, valid, testの各セットから、各クラス(neg, neu, pos)をバランスよく
10件ずつサンプリングして平易化を行い、結果を保存します。
モデルやプロンプトを変更して検証できます。
"""

from pathlib import Path
from typing import Optional
import argparse
import json

import pandas as pd

from load_wrime import WRIMELoader
from simplify_wrime import WRIMESimplifier


class TestSimplifier(WRIMESimplifier):
    """検証用の平易化クラス（カスタムプロンプト対応）"""

    def __init__(
        self,
        data_dir: str = "data",
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "auto",
        batch_size: int = 1,
        verbose: bool = True,
        custom_prompt_template: Optional[str] = None,
    ):
        """
        Args:
            custom_prompt_template: カスタムプロンプトテンプレート
                                   {text}が元の文章に置き換えられます
        """
        super().__init__(data_dir, model_name, device, batch_size, verbose)
        self.custom_prompt_template = custom_prompt_template

    def _create_simplification_prompt(self, text: str) -> str:
        """カスタムプロンプトがあればそれを使用"""
        if self.custom_prompt_template:
            return self.custom_prompt_template.format(text=text)
        return super()._create_simplification_prompt(text)


def sample_balanced_data(df: pd.DataFrame, n_samples: int = 10) -> pd.DataFrame:
    """
    各クラスからバランスよくサンプリング

    Args:
        df: 元のDataFrame
        n_samples: サンプリングする総数

    Returns:
        サンプリングされたDataFrame
    """
    # クラスごとのサンプル数を計算
    classes = df["label"].unique()
    n_classes = len(classes)
    samples_per_class = n_samples // n_classes
    remaining = n_samples % n_classes

    sampled_dfs = []
    for i, label in enumerate(sorted(classes)):
        class_df = df[df["label"] == label]
        # 余りを最初のクラスに配分
        n = samples_per_class + (1 if i < remaining else 0)
        # クラスのサンプル数が要求より少ない場合は全て取得
        n = min(n, len(class_df))
        sampled = class_df.sample(n=n, random_state=42)
        sampled_dfs.append(sampled)

    return pd.concat(sampled_dfs, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(
        description="LLM平易化の検証用スクリプト"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="使用するLLMモデル名",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10,
        help="各セット（train/valid/test）からサンプリングする数",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/test_simplification",
        help="結果の保存先ディレクトリ",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default=None,
        help="カスタムプロンプトテンプレートファイル（オプション）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="使用するデバイス",
    )

    args = parser.parse_args()

    # 出力ディレクトリ作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # カスタムプロンプトの読み込み（オプション）
    custom_prompt = None
    if args.prompt_file:
        prompt_path = Path(args.prompt_file)
        if prompt_path.exists():
            custom_prompt = prompt_path.read_text(encoding="utf-8")
            print(f"Loaded custom prompt from: {args.prompt_file}")
        else:
            print(f"Warning: Prompt file not found: {args.prompt_file}")

    # データセットの読み込み
    print("Loading WRIME dataset...")
    wrime_loader = WRIMELoader()
    train_df, valid_df, test_df = wrime_loader.load_or_download()

    # 各セットからバランスよくサンプリング
    print(f"\nSampling {args.n_samples} samples per split...")
    train_sample = sample_balanced_data(train_df, args.n_samples)
    valid_sample = sample_balanced_data(valid_df, args.n_samples)
    test_sample = sample_balanced_data(test_df, args.n_samples)

    # サンプル数とクラス分布を表示
    for name, df in [("train", train_sample), ("valid", valid_sample), ("test", test_sample)]:
        print(f"\n{name.upper()} samples:")
        print(f"  Total: {len(df)}")
        print(f"  Class distribution:")
        for label, count in df["label"].value_counts().sort_index().items():
            print(f"    {label}: {count}")

    # 平易化処理
    print(f"\nInitializing simplifier with model: {args.model}")
    simplifier = TestSimplifier(
        model_name=args.model,
        device=args.device,
        verbose=True,
        custom_prompt_template=custom_prompt,
    )

    # 各セットを平易化
    results = {}
    for split_name, df in [
        ("train", train_sample),
        ("valid", valid_sample),
        ("test", test_sample),
    ]:
        print(f"\n{'='*60}")
        print(f"Simplifying {split_name} samples...")
        print(f"{'='*60}")

        df_result = df.copy()
        df_result["text_orig"] = df_result["text"]
        simplified_texts = []

        for idx, (_, row) in enumerate(df.iterrows(), 1):
            print(f"\n[{idx}/{len(df)}] Processing...")
            print(f"Original: {row['text'][:100]}...")

            simplified = simplifier._simplify_text(row["text"])
            simplified_texts.append(simplified)

            if simplified:
                print(f"Simplified: {simplified[:100]}...")
            else:
                print("Warning: Simplification failed")

        df_result["text_simplified"] = simplified_texts
        results[split_name] = df_result

    # 結果を保存
    print(f"\n{'='*60}")
    print("Saving results...")
    print(f"{'='*60}")

    for split_name, df_result in results.items():
        # CSV形式で保存
        csv_path = output_dir / f"{split_name}_samples.csv"
        df_result.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"Saved: {csv_path}")

        # Pickle形式でも保存（日本語の扱いが安全）
        pkl_path = output_dir / f"{split_name}_samples.pkl"
        df_result.to_pickle(pkl_path)
        print(f"Saved: {pkl_path}")

    # メタデータを保存
    metadata = {
        "model": args.model,
        "n_samples": args.n_samples,
        "device": args.device,
        "custom_prompt_used": args.prompt_file is not None,
        "prompt_file": args.prompt_file,
        "splits": {
            split_name: {
                "total_samples": len(df),
                "class_distribution": df["label"].value_counts().to_dict(),
                "failed_count": df["text_simplified"].isna().sum(),
            }
            for split_name, df in results.items()
        },
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Saved: {metadata_path}")

    print(f"\n{'='*60}")
    print("Completed!")
    print(f"{'='*60}")
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
