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

import pandas as pd

from load_wrime import WRIMELoader
from utils import detect_device, is_fp16_available


class WRIMESimplifier:
    """WRIME v2データセットの平易化を行うクラス"""

    def __init__(
        self,
        data_dir: str = "data",
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
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
            device_map = "cuda"
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

    def _create_simplification_prompt(self, text: str) -> str:
        """
        文化庁「やさしい日本語」書換え原則に基づくプロンプトを作成

        Args:
            text: 平易化したい文章

        Returns:
            プロンプト文字列
        """
        prompt = f"""あなたは日本語の文章を平易化する専門家です。以下の文章を「やさしい日本語」に書き換えてください。

【やさしい日本語の書換え原則】
1. 難しい言葉を簡単な言葉に置き換える
2. 長い文を短く分割する
3. 漢字を減らし、ひらがなを増やす（ただし読みやすさを保つ）
4. 二重否定を避ける
5. 受動態を能動態にする
6. 曖昧な表現を具体的にする

【重要な制約】
- 元の文章の意味を必ず保持してください
- 元の文章の感情（ポジティブ・ネガティブ・中立）を必ず保持してください
- 書き換え後の文章のみを出力してください（説明や追加情報は不要です）

【元の文章】
{text}

【やさしい日本語に書き換えた文章】
"""
        return prompt

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

        prompt = self._create_simplification_prompt(text)

        for attempt in range(max_retries):
            try:
                # トークナイズ
                inputs = self.tokenizer(prompt, return_tensors="pt")

                # モデルのデバイスに入力を移動
                model_device = next(self.model.parameters()).device
                inputs = inputs.to(model_device)

                # 生成
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

                # デコード
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                # プロンプト部分を除去して出力のみを抽出
                simplified = generated_text.split("【やさしい日本語に書き換えた文章】")[-1].strip()

                # 空文字列や元の文章と同じ場合はリトライ
                if simplified and simplified != text:
                    return simplified

            except Exception:
                if self.verbose:
                    print(f"Error at attempt {attempt + 1}/{max_retries}")
                time.sleep(1)

        # リトライ上限に達した場合はNoneを返す
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
        model_name="Qwen/Qwen2.5-7B-Instruct",
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
