"""
OpenAI APIを使用した平易化の検証用スクリプト

train, valid, testの各セットから、各クラス(neg, neu, pos)をバランスよく
10件ずつサンプリングして平易化を行い、結果を保存します。
"""

from pathlib import Path
import argparse
import json
import os
import time
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

from load_wrime import WRIMELoader

# API プロバイダーのインポート（遅延インポートで対応）
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

try:
    from google import genai
except ImportError:
    genai = None

# .envファイルを読み込み
load_dotenv()

# ============================================================
# グローバル設定
# ============================================================
DEFAULT_PROVIDER = "openai"
DEFAULT_MODEL_NAME = "gpt-4o-mini"  # 使用するOpenAI APIモデル
OUTPUT_DIR = "outputs/test_simplification_api"  # 結果の保存先

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
"""

USER_PROMPT_TEMPLATE = """
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

以下は平易化して欲しい文章です。
Input
{text}

Output
"""

# 簡潔版（実際に使用）
USER_PROMPT_TEMPLATE = """{text}
"""


class APISimplifier:
    """複数のAPI（OpenAI、Anthropic、Google）を使用した平易化クラス"""

    SUPPORTED_PROVIDERS = ["openai", "anthropic", "gemini"]

    def __init__(
        self,
        provider: str = DEFAULT_PROVIDER,
        model_name: str = DEFAULT_MODEL_NAME,
        verbose: bool = True,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Args:
            provider: APIプロバイダー ("openai", "anthropic", "gemini")
            model_name: 使用するAPIモデル名
            verbose: 詳細なログ出力
            max_retries: API呼び出しの最大リトライ回数
            retry_delay: リトライ時の待機時間（秒）
        """
        self.provider = provider.lower()
        self.model_name = model_name
        self.verbose = verbose
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.client = None

        # プロバイダーの検証
        if self.provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported provider: {self.provider}. "
                f"Supported providers: {', '.join(self.SUPPORTED_PROVIDERS)}"
            )

        # プロバイダーごとにクライアントを初期化
        self._initialize_client()

        if self.verbose:
            print(f"Initialized APISimplifier with provider: {self.provider}, model: {self.model_name}")

    def _initialize_client(self):
        """プロバイダーに応じてAPIクライアントを初期化"""
        if self.provider == "openai":
            if OpenAI is None:
                raise ImportError("openai package is not installed. Install it with: pip install openai")

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY environment variable not found. "
                    "Please set it in .env file or as an environment variable."
                )
            self.client = OpenAI(api_key=api_key)

        elif self.provider == "anthropic":
            if Anthropic is None:
                raise ImportError("anthropic package is not installed. Install it with: pip install anthropic")

            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY environment variable not found. "
                    "Please set it in .env file or as an environment variable."
                )
            self.client = Anthropic(api_key=api_key)

        elif self.provider == "gemini":
            if genai is None:
                raise ImportError("google-generativeai package is not installed. Install it with: pip install google-generativeai")

            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError(
                    "GOOGLE_API_KEY environment variable not found. "
                    "Please set it in .env file or as an environment variable."
                )
            self.client = genai.Client(api_key=api_key)
            # genai.configure(api_key=api_key)
            # self.client = genai.GenerativeModel(self.model_name)

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

    def simplify_text(self, text: str) -> Optional[str]:
        """
        選択されたAPIを使用して文章を平易化

        Args:
            text: 平易化したい文章

        Returns:
            平易化された文章（失敗時はNone）
        """
        for attempt in range(self.max_retries):
            try:
                if self.provider == "openai":
                    simplified = self._simplify_with_openai(text)
                elif self.provider == "anthropic":
                    simplified = self._simplify_with_anthropic(text)
                elif self.provider == "gemini":
                    simplified = self._simplify_with_gemini(text)
                else:
                    raise ValueError(f"Unknown provider: {self.provider}")

                # 空文字列や元の文章と同じ場合はリトライ
                if simplified and simplified != text:
                    return simplified

                if self.verbose:
                    print(f"Warning: Empty or identical result at attempt {attempt + 1}/{self.max_retries}")

            except Exception as e:
                if self.verbose:
                    print(f"Error at attempt {attempt + 1}/{self.max_retries}: {e}")

                # リトライ前に待機
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)

        # リトライ上限に達した場合はNoneを返す
        return None

    def _simplify_with_openai(self, text: str) -> str:
        """OpenAI APIを使用して平易化"""
        messages = self._create_simplification_prompt(text)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.7,
            max_tokens=512,
        )

        return response.choices[0].message.content.strip()

    def _simplify_with_anthropic(self, text: str) -> str:
        """Anthropic (Claude) APIを使用して平易化"""
        messages = self._create_simplification_prompt(text)

        # システムプロンプトを分離
        system_content = None
        user_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                user_messages.append(msg)

        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=512,
            temperature=0.7,
            system=system_content if system_content else "あなたは誠実で優秀な日本人のアシスタントです。",
            messages=user_messages,
        )

        return response.content[0].text.strip()

    def _simplify_with_gemini(self, text: str) -> str:
        """Google Gemini APIを使用して平易化"""
        messages = self._create_simplification_prompt(text)

        # Gemini用にプロンプトを結合
        full_prompt = ""
        for msg in messages:
            if msg["role"] == "system":
                full_prompt += f"{msg['content']}\n\n"
            elif msg["role"] == "user":
                full_prompt += msg["content"]

        # response = self.client.generate_content(
        #     full_prompt,
        #     generation_config={
        #         "temperature": 0.7,
        #         "max_output_tokens": 512,
        #     }
        # )
        print("debug")
        print(full_prompt)
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=full_prompt,
            # generation_config={
            #     "temperature": 0.7,
            #     "max_output_tokens": 512,
            # }
        )
        print(response.text)

        return response.text.strip()


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
        description="複数のLLM API（OpenAI/Anthropic/Google）を使用したWRIMEコーパスの平易化検証用スクリプト"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10,
        help="各セット（train/valid/test）からサンプリングする数（デフォルト: 10）",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=DEFAULT_PROVIDER,
        choices=["openai", "anthropic", "gemini"],
        help="使用するAPIプロバイダー（デフォルト: openai）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"使用するAPIモデル（デフォルト: {DEFAULT_MODEL_NAME}）",
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
    print(f"\nInitializing API simplifier with provider: {args.provider}, model: {args.model}")
    simplifier = APISimplifier(
        provider=args.provider,
        model_name=args.model,
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

            simplified = simplifier.simplify_text(row["text"])
            simplified_texts.append(simplified)

            if simplified:
                print(f"Simplified: {simplified[:100]}...")
            else:
                print("Warning: Simplification failed")

            # API レート制限対策として少し待機
            time.sleep(10)

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
        "provider": args.provider,
        "model": args.model,
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
