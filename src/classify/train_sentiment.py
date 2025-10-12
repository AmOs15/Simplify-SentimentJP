"""
感情分類モデルの学習・評価

WRIME v2データセットでBERTベースの感情分類モデルを学習
条件: orig, simp, concat, noise-norm
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from classify.config import TrainingConfig
from classify.models import BERTSentimentClassifier, WeightedBERTSentimentClassifier
from classify.data_utils import (
    load_data_from_config,
    create_dataloaders,
    compute_class_weights,
    print_data_statistics,
)
from classify.train_utils import (
    set_seed,
    train_model,
    evaluate,
    get_device,
    save_results,
    print_test_results,
)


def main():
    """メイン処理"""
    # コマンドライン引数のパース
    parser = argparse.ArgumentParser(description="Train sentiment classification model")
    parser.add_argument(
        "--condition",
        type=str,
        default="orig",
        choices=["orig", "simp", "concat", "noise-norm"],
        help="Data condition to use",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="cl-tohoku/bert-base-japanese-v3",
        help="Pretrained model name",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory",
    )
    parser.add_argument(
        "--use_class_weights",
        action="store_true",
        help="Use class weights for imbalanced data",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )

    args = parser.parse_args()

    # 設定の作成
    config = TrainingConfig(
        model_name=args.model_name,
        condition=args.condition,
        # ToDo: コマンドライン引数からの読み取り
        use_subset=True,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        output_dir=args.output_dir,
        use_class_weights=args.use_class_weights,
        max_length=args.max_length,
    )

    print("\n" + "="*60)
    print("Sentiment Classification Training")
    print("="*60)
    print(f"Condition: {config.condition}")
    print(f"Model: {config.model_name}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Seed: {config.seed}")
    print(f"Use class weights: {config.use_class_weights}")
    print("="*60 + "\n")

    # シード設定
    set_seed(config.seed)

    # データの読み込み
    train_df, valid_df, test_df = load_data_from_config(config)

    # 統計情報の表示
    print_data_statistics(train_df, valid_df, test_df)

    # トークナイザーの読み込み
    print(f"\nLoading tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # DataLoaderの作成
    print("Creating dataloaders...")
    train_loader, valid_loader, test_loader = create_dataloaders(
        train_df, valid_df, test_df, tokenizer, config
    )

    # クラス重みの計算
    class_weights = None
    if config.use_class_weights:
        class_weights = compute_class_weights(train_df, config.label2id)
        print(f"\nClass weights: {class_weights.tolist()}")

    # モデルの作成
    print(f"\nInitializing model: {config.model_name}")
    if config.use_class_weights:
        model = WeightedBERTSentimentClassifier(
            model_name=config.model_name,
            num_labels=config.num_labels,
            class_weights=class_weights,
        )
    else:
        model = BERTSentimentClassifier(
            model_name=config.model_name,
            num_labels=config.num_labels,
        )

    # 学習の実行
    print("\nStarting training...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        config=config,
        output_dir=config.output_dir,
    )

    # テストセットでの評価
    print("\nEvaluating on test set...")
    device = get_device(config)

    # ベストモデルが保存されている場合は読み込む
    best_model_path = Path(config.output_dir) / "best_model"
    if best_model_path.exists() and config.save_best_model:
        print(f"Loading best model from {best_model_path}")
        model = BERTSentimentClassifier.from_pretrained(str(best_model_path))
        model.to(device)

    test_metrics = evaluate(
        model=model,
        data_loader=test_loader,
        device=device,
        config=config,
        desc="Testing",
    )

    # 結果の表示
    print_test_results(test_metrics, config.id2label)

    # 結果の保存
    save_results(history, test_metrics, config, config.output_dir)

    print(f"\nTraining completed!")
    print(f"Results saved to: {config.output_dir}")


if __name__ == "__main__":
    main()
