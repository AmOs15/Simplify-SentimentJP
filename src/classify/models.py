"""
感情分類モデルの定義

Tohoku BERTベースの3クラス分類モデル
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class BERTSentimentClassifier(nn.Module):
    """
    BERTベースの感情分類モデル

    CLS表現 → 全結合層 → 3クラス出力
    """

    def __init__(
        self,
        model_name: str = "cl-tohoku/bert-base-japanese-v3",
        num_labels: int = 3,
        dropout_rate: float = 0.1,
    ):
        """
        Args:
            model_name: 使用するBERTモデルの名前
            num_labels: クラス数
            dropout_rate: ドロップアウト率
        """
        super().__init__()

        # BERTモデルの読み込み
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)

        # 分類ヘッド
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

        self.num_labels = num_labels

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor = None,
        labels: torch.Tensor = None,
    ):
        """
        順伝播

        Args:
            input_ids: 入力トークンID [batch_size, seq_len]
            attention_mask: アテンションマスク [batch_size, seq_len]
            token_type_ids: トークンタイプID [batch_size, seq_len]
            labels: ラベル [batch_size] (optional)

        Returns:
            loss, logits, hidden_statesを含む辞書
        """
        # BERT forward
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # CLS表現を取得
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]

        # 分類
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)  # [batch_size, num_labels]

        # 損失計算
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": cls_output,
        }

    def save_pretrained(self, save_directory: str):
        """
        モデルを保存

        Args:
            save_directory: 保存先ディレクトリ
        """
        import os

        os.makedirs(save_directory, exist_ok=True)

        # BERTモデルを保存
        self.bert.save_pretrained(save_directory)

        # 分類ヘッドを保存
        classifier_path = os.path.join(save_directory, "classifier.pt")
        torch.save(
            {
                "classifier": self.classifier.state_dict(),
                "dropout_rate": self.dropout.p,
                "num_labels": self.num_labels,
            },
            classifier_path,
        )

    @classmethod
    def from_pretrained(cls, model_directory: str):
        """
        保存したモデルを読み込み

        Args:
            model_directory: モデルが保存されているディレクトリ

        Returns:
            読み込んだモデル
        """
        import os

        # 分類ヘッドの情報を読み込み
        classifier_path = os.path.join(model_directory, "classifier.pt")
        checkpoint = torch.load(classifier_path, map_location="cpu")

        # モデルを初期化
        model = cls(
            model_name=model_directory,
            num_labels=checkpoint["num_labels"],
            dropout_rate=checkpoint["dropout_rate"],
        )

        # 分類ヘッドの重みを復元
        model.classifier.load_state_dict(checkpoint["classifier"])

        return model


class WeightedBERTSentimentClassifier(BERTSentimentClassifier):
    """
    クラス重み付き損失を使用するBERT分類モデル

    クラス不均衡に対応するため、クラスごとに重みを設定
    """

    def __init__(
        self,
        model_name: str = "cl-tohoku/bert-base-japanese-v3",
        num_labels: int = 3,
        dropout_rate: float = 0.1,
        class_weights: torch.Tensor = None,
    ):
        """
        Args:
            model_name: 使用するBERTモデルの名前
            num_labels: クラス数
            dropout_rate: ドロップアウト率
            class_weights: クラスごとの重み [num_labels]
        """
        super().__init__(model_name, num_labels, dropout_rate)

        self.class_weights = class_weights

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor = None,
        labels: torch.Tensor = None,
    ):
        """
        順伝播（重み付き損失版）

        Args:
            input_ids: 入力トークンID [batch_size, seq_len]
            attention_mask: アテンションマスク [batch_size, seq_len]
            token_type_ids: トークンタイプID [batch_size, seq_len]
            labels: ラベル [batch_size] (optional)

        Returns:
            loss, logits, hidden_statesを含む辞書
        """
        # BERT forward
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # CLS表現を取得
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]

        # 分類
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)  # [batch_size, num_labels]

        # 損失計算（重み付き）
        loss = None
        if labels is not None:
            # クラス重みをモデルと同じデバイスに移動
            weights = self.class_weights
            if weights is not None:
                weights = weights.to(logits.device)

            loss_fct = nn.CrossEntropyLoss(weight=weights)
            loss = loss_fct(logits, labels)

        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": cls_output,
        }
