"""
LLMによる平易化の検証用スクリプト

train, valid, testの各セットから、各クラス(neg, neu, pos)をバランスよく
10件ずつサンプリングして平易化を行い、結果を保存します。
"""

from pathlib import Path
import argparse
import json

import pandas as pd

from load_wrime import WRIMELoader
from simplify_wrime import WRIMESimplifier

# ============================================================
# グローバル設定
# ============================================================
MODEL_NAME = "tokyotech-llm/Swallow-7b-instruct-v0.1"  # 使用するLLMモデル
OUTPUT_DIR = "outputs/test_simplification"  # 結果の保存先

# プロンプト設定
SYSTEM_PROMPT = """あなたは誠実で優秀な日本人のアシスタントです。
# 命令書

あなたは、文章を平易化する専門家です。
与えられた【元の文章】を、日本に住み始めたばかりの外国人を対象読者として、以下の#ルールと#出力形式に厳密に従って、やさしい日本語の文章に書き換えてください。

# ルール
- **語彙**: 難しい専門用語や漢語を避け、小学校中学年までで習うような、具体的で身近な言葉を選びます。
- **文法**:
    - 一つの文は短く、簡潔にします。一つの文には一つの情報だけを入れます。
    - 尊敬語や謙譲語は使わず、「です・ます」調の丁寧語に統一します。
    - 受け身の文（例：「〜される」）ではなく、誰が何をするのかが明確な能動態の文（例：「〜が〜する」）を使います。
    - 二重否定や曖昧な表現は使わず、直接的で分かりやすい表現にします。
- **構成**: 情報の構造を分かりやすく整理します。
- **情報**: 元の文章が持つ最も重要な意味、意図、ニュアンスは必ず維持してください。

# 出力形式
- 必ず書き換えた「やさしい日本語の文章」の本文のみを出力してください。これは必須要件です。
- 挨拶、導入、追加の説明や解説、言い訳など、本文以外の情報は一切含めてはいけません。禁止です。

# 例
Input
当市役所にて住民票の写しの交付申請をされる際は、本人確認書類の提示が義務付けられておりますので、ご持参くださいますようお願い申し上げます。

Output
市役所で「住民票の写し」を申し込むときは、本人確認の書類を見せてください。運転免許証や在留カードなどを、忘れずに持ってきてください。
"""

USER_PROMPT_TEMPLATE = """{text}
"""


class TestSimplifier(WRIMESimplifier):
    """検証用の平易化クラス"""

    def _create_simplification_prompt(self, text: str) -> list:
        """
        グローバル変数で定義されたプロンプトを使用してmessages形式で返す

        Args:
            text: 平易化したい文章

        Returns:
            messages形式のリスト
        """
        user_content = USER_PROMPT_TEMPLATE.format(text=text)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]
        return messages


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
        device="auto",
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
        "n_samples": int(args.n_samples),
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt_template": USER_PROMPT_TEMPLATE,
        "splits": {},
    }

    for split_name, df in results.items():
        total_samples = int(len(df))

        raw_dist = df["label"].value_counts().to_dict()
        class_distribution = {str(k): int(v) for k, v in raw_dist.items()}

        failed_count = int(df["text_simplified"].isna().sum())

        metadata["splits"][split_name] = {
            "total_samples": total_samples,
            "class_distribution": class_distribution,
            "failed_count": failed_count,
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
