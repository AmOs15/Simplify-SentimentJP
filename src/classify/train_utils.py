"""
学習・評価のユーティリティ関数

学習ループ、評価、メトリクス計算など
"""

import os
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from tqdm import tqdm

from classify.config import TrainingConfig
from utils import detect_device


def set_seed(seed: int):
    """
    再現性のためのシード設定

    Args:
        seed: シード値
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # 再現性を最大化（速度は若干低下）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(config: TrainingConfig) -> torch.device:
    """
    デバイスを取得（CUDA, MPS, CPUの順で優先）

    Args:
        config: TrainingConfig

    Returns:
        torch.device
    """
    if config.device == "auto":
        device_str = detect_device()
        device = torch.device(device_str)
    else:
        device = torch.device(config.device)

    return device


def setup_optimizer_and_scheduler(
    model: nn.Module,
    train_loader: DataLoader,
    config: TrainingConfig,
) -> Tuple[AdamW, object]:
    """
    オプティマイザーとスケジューラーをセットアップ

    Args:
        model: モデル
        train_loader: 学習データローダー
        config: TrainingConfig

    Returns:
        (optimizer, scheduler)のタプル
    """
    # オプティマイザー
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # スケジューラー
    num_training_steps = len(train_loader) * config.num_epochs // config.gradient_accumulation_steps
    num_warmup_steps = int(num_training_steps * config.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    return optimizer, scheduler


def compute_metrics(predictions: np.ndarray, labels: np.ndarray, id2label: dict = None) -> Dict:
    """
    評価メトリクスを計算

    Args:
        predictions: 予測ラベル
        labels: 正解ラベル
        id2label: ID→ラベルのマッピング辞書

    Returns:
        メトリクスの辞書
    """
    id2label = id2label or {0: "neg", 1: "neu", 2: "pos"}

    # 全体のメトリクス
    accuracy = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average="macro")
    macro_precision = precision_score(labels, predictions, average="macro", zero_division=0)
    macro_recall = recall_score(labels, predictions, average="macro", zero_division=0)

    # クラスごとのF1
    per_class_f1 = f1_score(labels, predictions, average=None, zero_division=0)
    per_class_precision = precision_score(labels, predictions, average=None, zero_division=0)
    per_class_recall = recall_score(labels, predictions, average=None, zero_division=0)

    # 混同行列
    cm = confusion_matrix(labels, predictions)

    metrics = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
    }

    # クラスごとのメトリクスを追加
    for idx, label_name in id2label.items():
        metrics[f"{label_name}_f1"] = per_class_f1[idx]
        metrics[f"{label_name}_precision"] = per_class_precision[idx]
        metrics[f"{label_name}_recall"] = per_class_recall[idx]

    metrics["confusion_matrix"] = cm.tolist()

    return metrics


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: AdamW,
    scheduler: object,
    device: torch.device,
    config: TrainingConfig,
    epoch: int,
) -> Dict:
    """
    1エポック分の学習を実行

    Args:
        model: モデル
        train_loader: 学習データローダー
        optimizer: オプティマイザー
        scheduler: スケジューラー
        device: デバイス
        config: TrainingConfig
        epoch: エポック数

    Returns:
        学習結果の辞書（loss, accuracyなど）
    """
    model.train()

    total_loss = 0
    all_predictions = []
    all_labels = []

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.num_epochs}")

    for step, batch in enumerate(progress_bar):
        # バッチをデバイスに移動
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )

        loss = outputs["loss"]
        logits = outputs["logits"]

        # Gradient accumulation
        loss = loss / config.gradient_accumulation_steps
        loss.backward()

        if (step + 1) % config.gradient_accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # メトリクス計算用
        total_loss += loss.item() * config.gradient_accumulation_steps
        predictions = torch.argmax(logits, dim=-1)
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # 進捗表示
        if (step + 1) % config.log_interval == 0:
            avg_loss = total_loss / (step + 1)
            progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})

    # エポック全体のメトリクス
    avg_loss = total_loss / len(train_loader)
    metrics = compute_metrics(np.array(all_predictions), np.array(all_labels), config.id2label)
    metrics["loss"] = avg_loss

    return metrics


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    config: TrainingConfig,
    desc: str = "Evaluating",
) -> Dict:
    """
    評価を実行

    Args:
        model: モデル
        data_loader: 評価データローダー
        device: デバイス
        config: TrainingConfig
        desc: 進捗バーの説明文

    Returns:
        評価結果の辞書（loss, accuracy, f1など）
    """
    model.eval()

    total_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=desc):
            # バッチをデバイスに移動
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )

            loss = outputs["loss"]
            logits = outputs["logits"]

            # メトリクス計算用
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 評価メトリクス
    avg_loss = total_loss / len(data_loader)
    metrics = compute_metrics(np.array(all_predictions), np.array(all_labels), config.id2label)
    metrics["loss"] = avg_loss

    return metrics


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    config: TrainingConfig,
    output_dir: Optional[str] = None,
) -> Dict:
    """
    モデルの学習を実行

    Args:
        model: モデル
        train_loader: 学習データローダー
        valid_loader: 検証データローダー
        config: TrainingConfig
        output_dir: モデル保存先ディレクトリ

    Returns:
        学習履歴の辞書
    """
    device = get_device(config)
    model.to(device)

    # オプティマイザーとスケジューラーのセットアップ
    optimizer, scheduler = setup_optimizer_and_scheduler(model, train_loader, config)

    # 学習履歴
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "train_macro_f1": [],
        "valid_loss": [],
        "valid_accuracy": [],
        "valid_macro_f1": [],
    }

    # Early stopping用
    best_metric = -float("inf") if config.early_stopping_metric != "loss" else float("inf")
    best_epoch = 0
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"Training on device: {device}")
    print(f"{'='*60}\n")

    for epoch in range(config.num_epochs):
        # 学習
        train_metrics = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            config=config,
            epoch=epoch,
        )

        # 検証
        valid_metrics = evaluate(
            model=model,
            data_loader=valid_loader,
            device=device,
            config=config,
            desc=f"Validation (Epoch {epoch + 1})",
        )

        # 履歴に追加
        history["train_loss"].append(train_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["train_macro_f1"].append(train_metrics["macro_f1"])
        history["valid_loss"].append(valid_metrics["loss"])
        history["valid_accuracy"].append(valid_metrics["accuracy"])
        history["valid_macro_f1"].append(valid_metrics["macro_f1"])

        # 結果表示
        print(f"\n[Epoch {epoch + 1}/{config.num_epochs}]")
        print(f"  Train Loss: {train_metrics['loss']:.4f}, "
              f"Acc: {train_metrics['accuracy']:.4f}, "
              f"Macro-F1: {train_metrics['macro_f1']:.4f}")
        print(f"  Valid Loss: {valid_metrics['loss']:.4f}, "
              f"Acc: {valid_metrics['accuracy']:.4f}, "
              f"Macro-F1: {valid_metrics['macro_f1']:.4f}")

        # Early stopping判定
        current_metric = valid_metrics[config.early_stopping_metric]

        if config.early_stopping_metric == "loss":
            is_better = current_metric < best_metric
        else:
            is_better = current_metric > best_metric

        if is_better:
            best_metric = current_metric
            best_epoch = epoch
            patience_counter = 0

            # ベストモデルを保存
            if config.save_best_model and output_dir:
                print(f"  Saving best model (epoch {epoch + 1})...")
                save_path = Path(output_dir) / "best_model"
                save_path.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(str(save_path))

        else:
            patience_counter += 1
            print(f"  Early stopping counter: {patience_counter}/{config.early_stopping_patience}")

            if patience_counter >= config.early_stopping_patience:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                print(f"Best {config.early_stopping_metric}: {best_metric:.4f} at epoch {best_epoch + 1}")
                break

    return history


def save_results(
    history: Dict,
    test_metrics: Dict,
    config: TrainingConfig,
    output_dir: str,
):
    """
    学習結果を保存

    Args:
        history: 学習履歴
        test_metrics: テストセットの評価結果
        config: TrainingConfig
        output_dir: 保存先ディレクトリ
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 学習履歴を保存
    history_path = output_path / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # テスト結果を保存
    test_results_path = output_path / "test_results.json"
    with open(test_results_path, "w") as f:
        json.dump(test_metrics, f, indent=2)

    # 混同行列を別ファイルに保存
    cm_path = output_path / "confusion_matrix.txt"
    with open(cm_path, "w") as f:
        f.write("Confusion Matrix:\n")
        f.write(str(np.array(test_metrics["confusion_matrix"])))

    print(f"\nResults saved to {output_dir}")


def print_test_results(test_metrics: Dict, id2label: dict = None):
    """
    テスト結果を表示

    Args:
        test_metrics: テスト結果のメトリクス
        id2label: ID→ラベルのマッピング辞書
    """
    id2label = id2label or {0: "neg", 1: "neu", 2: "pos"}

    print("\n" + "="*60)
    print("Test Results")
    print("="*60)
    print(f"  Accuracy:        {test_metrics['accuracy']:.4f}")
    print(f"  Macro-F1:        {test_metrics['macro_f1']:.4f}")
    print(f"  Macro-Precision: {test_metrics['macro_precision']:.4f}")
    print(f"  Macro-Recall:    {test_metrics['macro_recall']:.4f}")
    print("\nPer-class Metrics:")

    for idx, label_name in id2label.items():
        print(f"  [{label_name}]")
        print(f"    F1:        {test_metrics[f'{label_name}_f1']:.4f}")
        print(f"    Precision: {test_metrics[f'{label_name}_precision']:.4f}")
        print(f"    Recall:    {test_metrics[f'{label_name}_recall']:.4f}")

    print("\nConfusion Matrix:")
    cm = np.array(test_metrics["confusion_matrix"])
    print(f"              Predicted")
    print(f"              neg   neu   pos")
    for i, label_name in enumerate(["neg", "neu", "pos"]):
        print(f"  Actual {label_name}  {cm[i, 0]:4d}  {cm[i, 1]:4d}  {cm[i, 2]:4d}")

    print("="*60)
