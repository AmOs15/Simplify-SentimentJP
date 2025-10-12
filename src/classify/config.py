"""
設定ファイル

学習・評価に関する設定をまとめて管理
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class TrainingConfig:
    """学習に関する設定"""

    # モデル設定
    model_name: str = "cl-tohoku/bert-base-japanese-v3"  # Tohoku University BERT
    num_labels: int = 3  # neg, neu, pos

    # データ設定
    data_dir: str = "data"
    condition: str = "orig"  # "orig", "simp", "concat", "noise-norm"
    use_subset: bool = True
    max_length: int = 512  # トークン最大長

    # 学習設定
    batch_size: int = 16
    num_epochs: int = 10
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1

    # 最適化設定
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_metric: str = "macro_f1"  # "macro_f1" or "loss"

    # 再現性
    seed: int = 42
    seeds: List[int] = field(default_factory=lambda: [42, 43, 44, 45, 46])  # 複数シード実験用

    # ログ・保存設定
    output_dir: str = "outputs"
    log_interval: int = 10  # 学習ログの表示間隔（ステップ数）
    eval_interval: int = 100  # 評価の実行間隔（ステップ数）
    save_best_model: bool = True

    # デバイス設定
    device: str = "auto"  # "auto", "cuda", "mps", "cpu" (autoはcuda→mps→cpuの順で検出)

    # クラスウェイト（クラス不均衡対応）
    use_class_weights: bool = True

    # ラベルマッピング
    label2id: dict = field(default_factory=lambda: {"neg": 0, "neu": 1, "pos": 2})
    id2label: dict = field(default_factory=lambda: {0: "neg", 1: "neu", 2: "pos"})

    def __post_init__(self):
        """初期化後の処理"""
        # conditionに応じてデータファイル名を設定
        if self.condition == "orig":
            if self.use_subset:
                self.train_file = f"{self.data_dir}/wrime_subset_train.pkl"
                self.valid_file = f"{self.data_dir}/wrime_subset_valid.pkl"
                self.test_file = f"{self.data_dir}/wrime_subset_test.pkl"
            else:
                self.train_file = f"{self.data_dir}/wrime_train.pkl"
                self.valid_file = f"{self.data_dir}/wrime_valid.pkl"
                self.test_file = f"{self.data_dir}/wrime_test.pkl"
        elif self.condition == "simp":
            if self.use_subset:
                self.train_file = f"{self.data_dir}/wrime_simp_subset_train.pkl"
                self.valid_file = f"{self.data_dir}/wrime_simp_subset_valid.pkl"
                self.test_file = f"{self.data_dir}/wrime_simp_subset_test.pkl"
            else:
                self.train_file = f"{self.data_dir}/wrime_simp_train.pkl"
                self.valid_file = f"{self.data_dir}/wrime_simp_valid.pkl"
                self.test_file = f"{self.data_dir}/wrime_simp_test.pkl"
        elif self.condition == "concat":
            # Concatの場合は別途処理が必要
            if self.use_subset:
                self.train_file = f"{self.data_dir}/wrime_subset_train.pkl"
                self.valid_file = f"{self.data_dir}/wrime_subset_valid.pkl"
                self.test_file = f"{self.data_dir}/wrime_subset_test.pkl"
                self.train_file_simp = f"{self.data_dir}/wrime_simp_subset_train.pkl"
                self.valid_file_simp = f"{self.data_dir}/wrime_simp_subset_valid.pkl"
                self.test_file_simp = f"{self.data_dir}/wrime_simp_subset_test.pkl"
            else:
                self.train_file = f"{self.data_dir}/wrime_train.pkl"
                self.valid_file = f"{self.data_dir}/wrime_valid.pkl"
                self.test_file = f"{self.data_dir}/wrime_test.pkl"
                self.train_file_simp = f"{self.data_dir}/wrime_simp_train.pkl"
                self.valid_file_simp = f"{self.data_dir}/wrime_simp_valid.pkl"
                self.test_file_simp = f"{self.data_dir}/wrime_simp_test.pkl"
        elif self.condition == "noise-norm":
            # Noise-normの場合は別途実装が必要
            raise NotImplementedError("noise-norm condition is not implemented yet")
        else:
            raise ValueError(f"Invalid condition: {self.condition}")

        # 出力ディレクトリをconditionごとに分ける
        self.output_dir = f"{self.output_dir}/{self.condition}"


def get_default_config() -> TrainingConfig:
    """デフォルト設定を取得"""
    return TrainingConfig()
