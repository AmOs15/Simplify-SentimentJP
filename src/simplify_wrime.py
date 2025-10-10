"""
WRIME v2データセットの平易化処理

このモジュールは以下の機能を提供します：
1. WRIMEデータセットの読み込み
2. ローカルLLMを使用した文章の平易化（意味・極性保持）
3. 平易化データセットの保存とロード
4. 統計情報の表示
"""

from pathlib import Path
from typing import Dict, Tuple, Optional
import time
from datetime import datetime, timedelta
import re

import pandas as pd

from load_wrime import WRIMELoader
from utils import detect_device, is_fp16_available


SYSTEM_PROMPT = """あなたは日本語文章の平易化の専門家です。
次の制約を必ず守ってください。

# 目的
- 文化庁や自治体の「やさしい日本語」指針に基づき、初心者にも読みやすい文に書き換える。
- 元の重要な意味・意図・感情の極性（ポジ/ネガ）は必ず保持する。

# 書き換え原則（抜粋）
- 語彙：難語・専門用語・漢語を避け、具体的で身近な語を選ぶ。
- 文法：一文は短く、能動態で、二重否定や曖昧表現を避ける。「です・ます」に統一。
- 構成：情報を整理し、要点を前に出す。冗長な言い回しを省く。

# 出力仕様（厳守）
- 出力は必ず 1 行。
- 先頭に `output: ` を付け、その後に平易化文のみを書く。
- 複数文でも途中で改行しない（句点「。」で区切るが改行は入れない）。
- 説明・前置き・引用符・記号・装飾・箇条書き・余談は一切入れない。
- 文末は日本語の句点「。」で終える。
- 上記を守れない場合は、例外的に `output: [SKIP]` のみを出力する。
"""


USER_PROMPT_TEMPLATE = """
# 入力
{text}

# 作業
上の「入力」を、上記の書き換え原則に沿って平易化してください。

# 出力
下の行に、仕様どおり 1 行で出力してください。

output:
"""



class WRIMESimplifier:
    """WRIME v2データセットの平易化を行うクラス"""

    def __init__(
        self,
        data_dir: str = "data",
        model_name: str = "tokyotech-llm/Swallow-7b-instruct-v0.1",
        device: str = "auto",
        batch_size: int = 1,
        verbose: bool = False,
    ):
        """
        Args:
            data_dir: データ保存先ディレクトリ
            model_name: 使用するLLMのモデル名
            device: デバイス指定 ("auto", "cuda", "mps", "cpu")
            batch_size: バッチサイズ
            verbose: 詳細なログ出力（Falseで速度重視）
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.verbose = verbose

        # 平易化データセットの保存先
        self.simp_train_file = self.data_dir / "wrime_simp_train.pkl"
        self.simp_valid_file = self.data_dir / "wrime_simp_valid.pkl"
        self.simp_test_file = self.data_dir / "wrime_simp_test.pkl"

        # モデルとトークナイザー（遅延初期化）
        self.model = None
        self.tokenizer = None

    def _initialize_model(self):
        """LLMモデルとトークナイザーの初期化"""
        if self.model is not None:
            return

        if self.verbose:
            print(f"\nInitializing LLM model: {self.model_name}")

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
        except ImportError:
            raise ImportError(
                "transformers and torch are required. "
                "Install them with: pip install transformers torch"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

                # デバイス設定
        if self.device == "auto":
            device_map = detect_device()
        elif self.device == "gpu" or self.device == "cuda":
            device_map = "cuda"
        elif self.device == "mps":
            device_map = "mps"
        elif self.device == "cpu":
            device_map = "cpu"
        else:
            device_map = None

        # モデルの読み込み
        # CUDA, MPSが利用可能ならfloat16、CPUならfloat32
        use_fp16 = is_fp16_available()
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=device_map,
            torch_dtype=torch.float16 if use_fp16 else torch.float32,
        )

        if self.verbose:
            print(f"Model loaded on device: {self.model.device}")

    def _create_simplification_prompt(self, text: str) -> list:
        """
        文化庁「やさしい日本語」書換え原則に基づくプロンプトを作成

        Args:
            text: 平易化したい文章

        Returns:
            messages形式のリスト（Swallowモデル用）
        """
        system_prompt = SYSTEM_PROMPT
        user_content = USER_PROMPT_TEMPLATE.format(text=text)  # ★埋め込み

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        return messages
    
    def _normalize_one_line(self, s: str) -> str:
        # 改行や重複空白をスペース1個に
        s = re.sub(r"\s*\n+\s*", " ", s)
        s = re.sub(r"\s{2,}", " ", s)
        return s.strip()

    def _extract_output(self, generated_text: str) -> Optional[str]:
        """
        Chat生成テキストから `output:` の値だけを安全に抽出し、1行に正規化して返す
        """
        # apply_chat_templateで生成されたプロンプト部分を除去
        # Swallowモデルの場合、[/INST]の後の出力を抽出
        tail = generated_text.split("[/INST]", 1)[-1] if "[/INST]" in generated_text else generated_text

        # 最初の output: を見つける
        m = re.search(r"output\s*:\s*(.+)", tail, flags=re.IGNORECASE | re.DOTALL)
        if not m:
            return None

        # 次の役割ラベルや空行の手前までを切り出す
        candidate = m.group(1)
        candidate = re.split(
            r"\n{2,}|(?:^|\n)\s*(?:system|user|assistant)\s*:",
            candidate, maxsplit=1, flags=re.IGNORECASE
        )[0]

        # コードブロック等の終端があれば手前まで
        candidate = re.split(r"```|~~~", candidate, maxsplit=1)[0]

        # 1行化 & 余計な引用符・括弧を除去
        candidate = self._normalize_one_line(candidate)
        candidate = candidate.strip(" \"'“”「」『』（）()")

        return candidate or None


    def _simplify_text(self, text: str, max_retries: int = 3) -> Optional[str]:
        """
        LLMを使用して文章を平易化

        Args:
            text: 平易化したい文章
            max_retries: 失敗時のリトライ回数

        Returns:
            平易化された文章（失敗時はNone）
        """
        if self.model is None:
            self._initialize_model()

        messages = self._create_simplification_prompt(text)

        for attempt in range(max_retries):
            try:
                # apply_chat_templateを使用してプロンプトを生成
                inputs = self.tokenizer.apply_chat_template(
                    messages,
                    return_tensors="pt",
                    add_generation_prompt=True
                )

                # モデルのデバイスに入力を移動
                model_device = next(self.model.parameters()).device
                inputs = inputs.to(model_device)

                # 生成
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=256,
                    temperature=0.3,   # ★書式安定のため低温度
                    do_sample=False,   # ★決定的生成でフォーマット遵守を優先
                    top_p=1.0,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                simplified = self._extract_output(generated_text)
                

                # 仕様：常に1行（途中改行なし）に正規化
                if simplified:
                    simplified = self._normalize_one_line(simplified)

                if simplified and simplified != text:
                    print(simplified)
                    return simplified

            except Exception:
                if self.verbose:
                    print(f"Error at attempt {attempt + 1}/{max_retries}")
                time.sleep(1)

        return None

    def simplify_dataset(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        test_df: pd.DataFrame,
        force_regenerate: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        データセット全体を平易化

        Args:
            train_df, valid_df, test_df: 元のDataFrame
            force_regenerate: Trueの場合、既存ファイルがあっても再生成

        Returns:
            (train_simp_df, valid_simp_df, test_simp_df)のタプル
        """
        # 既存ファイルがある場合はロード
        if not force_regenerate and all([
            self.simp_train_file.exists(),
            self.simp_valid_file.exists(),
            self.simp_test_file.exists(),
        ]):
            if self.verbose:
                print("Loading simplified dataset from local files...")
            train_simp = pd.read_pickle(self.simp_train_file)
            valid_simp = pd.read_pickle(self.simp_valid_file)
            test_simp = pd.read_pickle(self.simp_test_file)
            return train_simp, valid_simp, test_simp

        # モデルの初期化
        self._initialize_model()

        # 各splitを平易化
        results = {}
        for split_name, df in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
            total_samples = len(df)
            print(f"\nSimplifying {split_name} set ({total_samples} samples)...", flush=True)

            start_time = time.time()
            simplified_texts = []
            failed_count = 0

            for idx, (_, row) in enumerate(df.iterrows(), 1):
                simplified = self._simplify_text(row['text'])
                simplified_texts.append(simplified)

                if simplified is None:
                    failed_count += 1

                # 進捗表示（10サンプルごと、または最後）
                if idx % 10 == 0 or idx == total_samples:
                    elapsed = time.time() - start_time
                    avg_time_per_sample = elapsed / idx
                    remaining_samples = total_samples - idx
                    eta_seconds = avg_time_per_sample * remaining_samples
                    eta = datetime.now() + timedelta(seconds=eta_seconds)

                    print(f"  [{idx}/{total_samples}] "
                          f"Elapsed: {elapsed/60:.1f}min, "
                          f"ETA: {eta.strftime('%H:%M:%S')}", flush=True)

            # 新しいDataFrameを作成
            df_simp = df.copy()
            df_simp['text_orig'] = df_simp['text']
            df_simp['text'] = simplified_texts

            elapsed_total = time.time() - start_time
            print(f"  Completed in {elapsed_total/60:.1f} minutes", flush=True)
            if failed_count > 0:
                print(f"  Warning: {failed_count} samples failed", flush=True)

            results[split_name] = df_simp

        train_simp = results["train"]
        valid_simp = results["valid"]
        test_simp = results["test"]

        # 保存
        train_simp.to_pickle(self.simp_train_file)
        valid_simp.to_pickle(self.simp_valid_file)
        test_simp.to_pickle(self.simp_test_file)

        return train_simp, valid_simp, test_simp

    def get_statistics(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Dict:
        """
        平易化データセットの統計情報を取得

        Args:
            train_df, valid_df, test_df: 各splitのDataFrame

        Returns:
            統計情報の辞書
        """
        stats = {}

        for name, df in [('train', train_df), ('valid', valid_df), ('test', test_df)]:
            # サンプル数
            stats[f'{name}_size'] = len(df)

            # クラス分布
            class_dist = df['label'].value_counts().to_dict()
            stats[f'{name}_class_dist'] = class_dist

            # 文章長統計（平易化後）
            df['text_length'] = df['text'].str.len()
            stats[f'{name}_text_length_mean'] = df['text_length'].mean()
            stats[f'{name}_text_length_std'] = df['text_length'].std()

            # 元の文章との長さの比較
            if 'text_orig' in df.columns:
                df['text_orig_length'] = df['text_orig'].str.len()
                stats[f'{name}_orig_length_mean'] = df['text_orig_length'].mean()
                stats[f'{name}_length_ratio'] = df['text_length'].mean() / df['text_orig_length'].mean()

        return stats

    def print_statistics(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ):
        """
        平易化データセットの統計情報を表示

        Args:
            train_df, valid_df, test_df: 各splitのDataFrame
        """
        stats = self.get_statistics(train_df, valid_df, test_df)

        print("\n" + "="*60)
        print("Simplified WRIME Dataset Statistics")
        print("="*60)

        for split in ['train', 'valid', 'test']:
            print(f"\n[{split.upper()} SET]")
            print(f"  Total samples: {stats[f'{split}_size']}")
            print(f"  Class distribution:")
            for label, count in stats[f'{split}_class_dist'].items():
                percentage = count / stats[f'{split}_size'] * 100
                print(f"    {label}: {count:5d} ({percentage:5.2f}%)")
            print(f"  Text length (simplified):")
            print(f"    Mean: {stats[f'{split}_text_length_mean']:.2f}")
            print(f"    Std:  {stats[f'{split}_text_length_std']:.2f}")

            if f'{split}_orig_length_mean' in stats:
                print(f"  Text length (original):")
                print(f"    Mean: {stats[f'{split}_orig_length_mean']:.2f}")
                print(f"  Length ratio (simplified/original):")
                print(f"    {stats[f'{split}_length_ratio']:.2%}")

        print("\n" + "="*60)


def main():
    """メイン処理：データセットの平易化と統計情報の表示"""
    # 元のデータセットを読み込み
    wrime_loader = WRIMELoader()
    train_df, valid_df, test_df = wrime_loader.load_or_download()

    # 平易化処理
    simplifier = WRIMESimplifier(
        model_name="tokyotech-llm/Swallow-7b-instruct-v0.1",
        device="auto",
        verbose=True,  # 詳細ログを表示する場合はTrue
    )

    train_simp, valid_simp, test_simp = simplifier.simplify_dataset(
        train_df, valid_df, test_df
    )

    # 統計情報を表示
    simplifier.print_statistics(train_simp, valid_simp, test_simp)

    # サンプルデータを表示
    print("\n[Sample simplified data from train set]")
    for idx, row in train_simp.head(3).iterrows():
        print(f"\n--- Sample {idx} ---")
        print(f"Original: {row['text_orig']}")
        print(f"Simplified: {row['text']}")
        print(f"Label: {row['label']}")


if __name__ == "__main__":
    main()
