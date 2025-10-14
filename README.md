# Simplify-SentimentJP
# 読みやすさ強化による日本語感情分類：平易化利用の検討

## 背景と目的
本研究は、**日本語文章の感情（極性）分類**タスクにおいて、**平易化（simplification）** を用いることで分類性能を改善できるかを検証することを目的とする。
主仮説は以下の通り：
> 平易化により冗長・曖昧・ノイズ要素が除去され、モデル学習が容易になり、三値（ネガティブ／中立／ポジティブ）分類において平易化文のみを使ったモデルが、原文のみを使ったモデルより優れた性能を示すかもしれない。

さらに、**原文＋平易化文併用（Concat）** などの条件を設け、それぞれの利点・欠点を比較・分析する。

## プロジェクト構成

### 実装済み機能

#### データパイプライン
- **データ読み込み** (`src/load_wrime.py`): WRIME v2データセットのダウンロード、3クラス変換、著者単位での層化分割
- **サブセット作成** (`src/select_wrime_subset.py`): 高速実験用に30%サブセットを抽出（クラスバランス維持）
- **データ確認ツール** (`src/view_data.py`): pklファイルの内容確認

#### 平易化パイプライン
- **ローカルLLM平易化** (`src/simplify_wrime.py`): Swallow-7b-instruct-v0.1を使用した平易化
- **API平易化検証** (`test_simplification/test_simplify_api.py`): OpenAI/Anthropic/Google APIによる小規模検証

#### 分類モデル (`src/classify/`)
- **モデル定義** (`models.py`): BERTベース3クラス分類器（クラス重み付き版含む）
- **データ処理** (`data_utils.py`): データロード、トークナイズ、DataLoader作成
- **学習・評価** (`train_utils.py`): 学習ループ、評価メトリクス（Macro-F1、Accuracy等）
- **設定管理** (`config.py`): 実験条件（orig/simp/concat）の設定
- **学習スクリプト** (`train_sentiment.py`): CLIから実行可能なメインスクリプト

### データファイル

| データファイル | 説明 | 用途 |
|---|---|---|
| `wrime_{train,valid,test}.pkl` | 完全版オリジナルデータ | ベースライン学習 |
| `wrime_subset_{train,valid,test}.pkl` | 30%サブセット（オリジナル） | 高速実験・検証 |
| `wrime_simp_{train,valid,test}.pkl` | 完全版平易化データ | 本実験 |
| `wrime_simp_subset_{train,valid,test}.pkl` | 30%サブセット（平易化） | 高速実験・検証 |

## セットアップ

### 環境構築
```bash
# 依存関係のインストール
poetry install
```

### 必要な環境変数（API使用時）
`.env`ファイルを作成し、以下を設定：
```bash
OPENAI_API_KEY=your_openai_api_key      # OpenAI使用時
ANTHROPIC_API_KEY=your_anthropic_key     # Claude使用時
GOOGLE_API_KEY=your_google_api_key       # Gemini使用時
```

## 使用方法

### 1. データセットの準備

#### 完全版データセットのダウンロード
```bash
poetry run python src/load_wrime.py
```
- WRIME v2を自動ダウンロード
- 3クラス変換（neg/neu/pos）
- 著者単位での層化分割（train/valid/test）
- 出力: `data/wrime_{train,valid,test}.pkl`

#### サブセット（30%）の作成
```bash
poetry run python src/select_wrime_subset.py
```
- 高速実験用に30%サブセット作成
- クラスバランス維持
- 出力: `data/wrime_subset_{train,valid,test}.pkl`

#### データ内容の確認
```bash
# 完全版の確認
poetry run python src/view_data.py --file data/wrime_train.pkl --n 5

# サブセットの確認
poetry run python src/view_data.py --file data/wrime_subset_train.pkl --n 5
```

### 2. 平易化の実行

#### ローカルLLMによる平易化（完全版）
```bash
poetry run python src/simplify_wrime.py
```
- モデル: Swallow-7b-instruct-v0.1
- 処理時間: 数時間〜（データサイズによる）
- 出力: `data/wrime_simp_{train,valid,test}.pkl`

#### API平易化の検証（小規模サンプル）
```bash
# OpenAI (ChatGPT)
poetry run python test_simplification/test_simplify_api.py \
  --provider openai \
  --model gpt-4o-mini \
  --n_samples 10

# Anthropic (Claude)
poetry run python test_simplification/test_simplify_api.py \
  --provider anthropic \
  --model claude-3-5-sonnet-20241022 \
  --n_samples 10

# Google (Gemini)
poetry run python test_simplification/test_simplify_api.py \
  --provider gemini \
  --model gemini-1.5-flash \
  --n_samples 10
```
- 各セット（train/valid/test）から指定件数をサンプリング
- 出力: `outputs/test_simplification_api/`

### 3. 感情分類モデルの学習

#### 基本的な学習
```bash
# 原文で学習（ベースライン）
poetry run python src/classify/train_sentiment.py --condition orig --seed 42

# 平易化文で学習
poetry run python src/classify/train_sentiment.py --condition simp --seed 42

# 原文+平易化文で学習（データ拡張）
poetry run python src/classify/train_sentiment.py --condition concat --seed 42
```

#### パラメータのカスタマイズ
```bash
poetry run python src/classify/train_sentiment.py \
  --condition orig \
  --model_name cl-tohoku/bert-base-japanese-v3 \
  --batch_size 16 \
  --num_epochs 10 \
  --learning_rate 2e-5 \
  --seed 42 \
  --use_class_weights \
  --max_length 512
```

**利用可能なパラメータ:**
- `--condition`: データ条件 (orig/simp/concat)
- `--model_name`: 事前学習モデル名
- `--batch_size`: バッチサイズ
- `--num_epochs`: エポック数
- `--learning_rate`: 学習率
- `--seed`: 乱数シード
- `--use_class_weights`: クラス重み付け（不均衡対応）
- `--max_length`: 最大系列長
- `--output_dir`: 出力ディレクトリ

## データとラベル変換
- 使用データ：**WRIME v2**
- 元の極性ラベル（−2, −1, 0, +1, +2）を次のように三値分類用に変換：
  - neg: {−2, −1}
  - neu: {0}
  - pos: {+1, +2}
- 極性参照視点：通常は "reader（読者視点）" を基準とする
- データ分割：三クラス比をできるだけ維持。

## 実験条件（モデル入力パターン）

| 条件名 | 入力データ | 意図・目的 | 実装状況 |
|---|---|---|---|
| Orig | 原文そのまま | ベースライン | ✅ 実装済み |
| Simp | 平易化文のみ | 平易化の効果を直接評価 | ✅ 実装済み |
| Concat | 原文 + 平易化文（同ラベル） | データ拡張・情報併用の効果を検討 | ✅ 実装済み |
| Noise-norm | 表記ゆれ正規化等の軽処理 | 平易化・正規化効果の切り分けを可能にする比較条件 | 🚧 未実装 |

### 平易化の品質管理
- LLMに**意味保持**と**極性保持**を明示的に指示
- 文化庁「やさしい日本語」書換え原則に基づくプロンプト設計
- **品質評価** (計画中):
  1. **BERTScore** による意味保持チェック
  2. **Cosine類似度** による意味的一致性評価
  3. 原文と平易化文での極性不一致サンプルの検出

## モデル設定と学習
- 基盤モデル：**東北大学版日本語 BERT**（例：`cl-tohoku/bert-base-japanese-v3`）
- 分類ヘッド：CLS 表現 → 全結合 → 3 クラス出力
- 損失関数：加重クロスエントロピー（クラス不均衡対応）
- 学習設定：学習率、バッチサイズ、エポック数、早期終了、複数乱数シードを用いた平均化
- 再現性：乱数シードを複数設定

## 評価指標と分析方法
- **主評価指標**：Macro-F1
- **補助指標**：Accuracy、クラス別 F1、混同行列
- **意味保持評価** (計画中)：BERTScore
- **極性不一致率** (計画中)：原文と平易化文でのラベル一致率
- **統計的検定** (計画中)：McNemar 検定またはブートストラップ法によるモデル間差の有意性確認
- **層別分析** (計画中)：投稿長、語彙難易度、表記ノイズ量などの属性別性能比較
- **失敗例抽出** (計画中)：Simp で性能が落ちたケースを収集して、改善指針を得る

## 想定される知見・示唆
1. 平易化（Simp）は冗長表現・曖昧表現・ノイズ要素が多い文で有効に働く可能性。
2. ただし、皮肉・含意・程度副詞・否定表現などを簡約中に失うと性能低下を招く可能性。
3. Concat 条件は、品質管理が十分なら最も性能向上が得られるが、ノイズ混入で逆効果となりうる。
4. Noise-norm 条件を導入することで、簡約の効果が表記正規化不足ではなく内容の変化によるものかを検証できる。

## 実装の進行状況
詳細は `STEP.md` と `CLAUDE.md` を参照してください。

### 完了したタスク
- ✅ データセット読み込みと3クラス変換
- ✅ サブセット作成機能
- ✅ ローカルLLMによる平易化
- ✅ API平易化検証スクリプト
- ✅ BERT分類モデルの実装
- ✅ 学習・評価パイプライン
- ✅ 複数条件対応（orig/simp/concat）

### 今後のタスク
- 🚧 Noise-norm条件の実装
- 🚧 平易化品質評価パイプライン
- 🚧 複数シードでの実験と統計分析
- 🚧 層別分析・失敗例解析
- 🚧 結果の可視化とレポート生成

## ライセンス
本プロジェクトで使用するデータセット（WRIME v2）のライセンスに従ってください。

## 参考文献
- WRIME v2: [GitHub Repository](https://github.com/ids-cv/wrime)
- 文化庁「やさしい日本語」: [文化庁ウェブサイト](https://www.bunka.go.jp/seisaku/kokugo_nihongo/kyoiku/nihongo_curriculum/)
