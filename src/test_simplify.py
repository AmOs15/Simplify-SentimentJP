"""
LLMによる平易化の検証用スクリプト

train, valid, testの各セットから、各クラス(neg, neu, pos)をバランスよく
10件ずつサンプリングして平易化を行い、結果を保存します。
"""

from pathlib import Path
from typing import Optional
import argparse
import json

import pandas as pd

from load_wrime import WRIMELoader
from simplify_wrime import WRIMESimplifier
from utils import detect_device

# ============================================================
# グローバル設定
# ============================================================
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # 使用するLLMモデル
OUTPUT_DIR = "outputs/test_simplification"  # 結果の保存先


class TestSimplifier(WRIMESimplifier):
    """検証用の平易化クラス（カスタムプロンプト対応）"""

    # プロンプトファイルのパス（クラス定数）
    PROMPT_FILE = "test_simplification/custom_prompt_example.txt"

    def __init__(
        self,
        data_dir: str = "data",
        model_name: str = MODEL_NAME,
        device: str = "auto",
        batch_size: int = 1,
        verbose: bool = True,
    ):
        """
        Args:
            data_dir: データ保存先ディレクトリ
            model_name: 使用するLLMのモデル名
            device: デバイス指定
            batch_size: バッチサイズ
            verbose: 詳細なログ出力
        """
        super().__init__(data_dir, model_name, device, batch_size, verbose)
        # カスタムプロンプトテンプレートを読み込み
        self.custom_prompt_template = self._load_custom_prompt()

    def _load_custom_prompt(self) -> str:
        """
        カスタムプロンプトテンプレートをファイルから読み込む

        Returns:
            プロンプトテンプレート文字列
        """
        prompt_path = Path(self.PROMPT_FILE)
        if not prompt_path.exists():
            if self.verbose:
                print(f"Warning: Prompt file not found: {prompt_path}, using default prompt")
            return None

        with open(prompt_path, 'r', encoding='utf-8') as f:
            content = f.read()

        if self.verbose:
            print(f"Loaded custom prompt from: {prompt_path}")

        return content

    def _create_simplification_prompt(self, text: str) -> str:
        """カスタムプロンプトがあればそれを使用"""
        if self.custom_prompt_template:
            # プロンプトテンプレートの{text}を実際のテキストに置換
            if self.custom_prompt_template.startswith('prompt = f"""'):
                # テンプレートをコードとして評価
                local_vars = {'text': text}
                exec(self.custom_prompt_template, {}, local_vars)
                return local_vars['prompt']
            else:
                # シンプルなテンプレート文字列の場合
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
        description="WRIMEコーパスの平易化検証用スクリプト"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10,
        help="各セット（train/valid/test）からサンプリングする数（デフォルト: 10）",
    )

    args = parser.parse_args()

    # デバイスを自動検出
    device = detect_device()
    print(f"Using device: {device}")

    # 出力ディレクトリ作成
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    print(f"\nInitializing simplifier with model: {MODEL_NAME}")
    simplifier = TestSimplifier(
        model_name=MODEL_NAME,
        device=device,
        verbose=True,
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
        "model": MODEL_NAME,
        "n_samples": args.n_samples,
        "device": device,
        "custom_prompt_file": TestSimplifier.PROMPT_FILE,
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
