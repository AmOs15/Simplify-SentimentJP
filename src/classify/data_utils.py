"""
データ処理ユーティリティ

データセット、データローダーの作成など
"""

from typing import Dict, Tuple, Optional
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from classify.config import TrainingConfig


class SentimentDataset(Dataset):
    """
    感情分類用のデータセットクラス

    text, labelを保持し、トークナイズを行う
    """

    def __init__(
        self,
        texts: list,
        labels: list,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        label2id: dict = None,
    ):
        """
        Args:
            texts: テキストのリスト
            labels: ラベルのリスト
            tokenizer: トークナイザー
            max_length: トークン最大長
            label2id: ラベル→IDのマッピング辞書
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = label2id or {"neg": 0, "neu": 1, "pos": 2}

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # トークナイズ
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # ラベルをIDに変換
        label_id = self.label2id[label]

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding.get("token_type_ids", torch.zeros_like(encoding["input_ids"])).squeeze(0),
            "labels": torch.tensor(label_id, dtype=torch.long),
        }


def load_data_from_config(config: TrainingConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    設定に基づいてデータを読み込む

    Args:
        config: TrainingConfig

    Returns:
        (train_df, valid_df, test_df)のタプル
    """
    print(f"Loading data for condition: {config.condition}")

    if config.condition in ["orig", "simp"]:
        train_df = pd.read_pickle(config.train_file)
        valid_df = pd.read_pickle(config.valid_file)
        test_df = pd.read_pickle(config.test_file)

    elif config.condition == "concat":
        # 原文と平易化版を両方読み込んで結合
        train_orig = pd.read_pickle(config.train_file)
        valid_orig = pd.read_pickle(config.valid_file)
        test_orig = pd.read_pickle(config.test_file)

        train_simp = pd.read_pickle(config.train_file_simp)
        valid_simp = pd.read_pickle(config.valid_file_simp)
        test_simp = pd.read_pickle(config.test_file_simp)

        # 平易化版のテキストを使用（text_origがあればそちらを使う）
        if 'text_orig' in train_simp.columns:
            # text_simpを残してorigと結合
            train_simp_copy = train_simp.copy()
            train_simp_copy['text'] = train_simp_copy['text']  # 平易化テキスト
            train_df = pd.concat([train_orig, train_simp_copy], ignore_index=True)
        else:
            train_df = pd.concat([train_orig, train_simp], ignore_index=True)

        if 'text_orig' in valid_simp.columns:
            valid_simp_copy = valid_simp.copy()
            valid_simp_copy['text'] = valid_simp_copy['text']
            valid_df = pd.concat([valid_orig, valid_simp_copy], ignore_index=True)
        else:
            valid_df = pd.concat([valid_orig, valid_simp], ignore_index=True)

        if 'text_orig' in test_simp.columns:
            test_simp_copy = test_simp.copy()
            test_simp_copy['text'] = test_simp_copy['text']
            test_df = pd.concat([test_orig, test_simp_copy], ignore_index=True)
        else:
            test_df = pd.concat([test_orig, test_simp], ignore_index=True)

    else:
        raise ValueError(f"Invalid condition: {config.condition}")

    print(f"Loaded: train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}")

    return train_df, valid_df, test_df


def create_dataloaders(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    config: TrainingConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    DataLoaderを作成

    Args:
        train_df, valid_df, test_df: 各splitのDataFrame
        tokenizer: トークナイザー
        config: TrainingConfig

    Returns:
        (train_loader, valid_loader, test_loader)のタプル
    """
    # Datasetを作成
    train_dataset = SentimentDataset(
        texts=train_df["text"].tolist(),
        labels=train_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=config.max_length,
        label2id=config.label2id,
    )

    valid_dataset = SentimentDataset(
        texts=valid_df["text"].tolist(),
        labels=valid_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=config.max_length,
        label2id=config.label2id,
    )

    test_dataset = SentimentDataset(
        texts=test_df["text"].tolist(),
        labels=test_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=config.max_length,
        label2id=config.label2id,
    )

    # DataLoaderを作成
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # macOSでのエラーを防ぐため0に設定
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, valid_loader, test_loader


def compute_class_weights(train_df: pd.DataFrame, label2id: dict) -> torch.Tensor:
    """
    クラス重みを計算

    クラス不均衡に対応するため、各クラスの逆頻度を重みとする

    Args:
        train_df: 学習データのDataFrame
        label2id: ラベル→IDのマッピング辞書

    Returns:
        クラス重みのテンソル [num_labels]
    """
    label_counts = train_df["label"].value_counts()
    total_count = len(train_df)

    # 各クラスの重みを計算（逆頻度）
    weights = []
    for label in sorted(label2id.keys(), key=lambda x: label2id[x]):
        count = label_counts.get(label, 1)  # ゼロ除算を防ぐ
        weight = total_count / (len(label2id) * count)
        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float32)


def print_data_statistics(train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    データセットの統計情報を表示

    Args:
        train_df, valid_df, test_df: 各splitのDataFrame
    """
    print("\n" + "="*60)
    print("Dataset Statistics")
    print("="*60)

    for split_name, df in [("TRAIN", train_df), ("VALID", valid_df), ("TEST", test_df)]:
        print(f"\n[{split_name} SET]")
        print(f"  Total samples: {len(df)}")
        print(f"  Class distribution:")

        label_counts = df["label"].value_counts()
        for label in ["neg", "neu", "pos"]:
            count = label_counts.get(label, 0)
            percentage = count / len(df) * 100
            print(f"    {label}: {count:5d} ({percentage:5.2f}%)")

    print("\n" + "="*60)
