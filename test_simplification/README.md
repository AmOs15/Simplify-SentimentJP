# 平易化検証用スクリプトの使い方
**要修正**
古いバージョン

## 概要

`src/test_simplify.py`は、LLMによる平易化を小規模サンプルで検証するためのスクリプトです。
train/valid/testの各セットから、各クラス（neg/neu/pos）をバランスよくサンプリングして平易化を行います。

## 基本的な使い方

### デフォルト設定での実行

```bash
cd src
python test_simplify.py
```

- モデル: `Qwen/Qwen2.5-7B-Instruct`
- サンプル数: 各セット10件ずつ
- 出力先: `outputs/test_simplification/`

### モデルを変更する

```bash
python test_simplify.py --model "rinna/japanese-gpt-neox-3.6b"
```

### サンプル数を変更する

```bash
python test_simplify.py --n_samples 20
```

各セットから20件ずつサンプリングします。

### カスタムプロンプトを使用する

```bash
python test_simplify.py --prompt_file ../outputs/test_simplification/custom_prompt_v2.txt
```

プロンプトファイルには、`{text}`というプレースホルダーを含めてください。

### デバイスを指定する

```bash
# M1/M2 MacでMPSを使用
python test_simplify.py --device mps

# CUDAを使用
python test_simplify.py --device cuda

# CPUを使用
python test_simplify.py --device cpu
```

## 出力ファイル

### CSV/Pickleファイル
- `train_samples.csv` / `train_samples.pkl`
- `valid_samples.csv` / `valid_samples.pkl`
- `test_samples.csv` / `test_samples.pkl`

各ファイルには以下のカラムが含まれます：
- `text_orig`: 元の文章
- `text_simplified`: 平易化された文章
- `label`: クラスラベル（neg/neu/pos）
- その他のWRIMEデータセットの元カラム

### メタデータ
- `metadata.json`: 実行設定とクラス分布、失敗数などの情報

## プロンプト例

### デフォルトプロンプト
デフォルトでは、文化庁「やさしい日本語」書換え原則に基づくプロンプトを使用します。
（`src/simplify_wrime.py`の`_create_simplification_prompt`メソッドを参照）

### カスタムプロンプトの例

`outputs/test_simplification/custom_prompt_v2.txt`として保存:

```
あなたは日本語の文章を平易化する専門家です。
以下の文章を小学5年生でも理解できる簡単な日本語に書き換えてください。

【重要な制約】
- 元の文章の意味を必ず保ってください
- 元の文章の感情（ポジティブ・ネガティブ・中立）を必ず保ってください
- 難しい言葉は簡単な言葉に置き換えてください
- 長い文は短く分けてください
- 書き換えた文章のみを出力してください

【元の文章】
{text}

【簡単な日本語に書き換えた文章】
```

## 検証の流れ

1. 異なるモデルで実行して結果を比較
   ```bash
   python test_simplify.py --model "Qwen/Qwen2.5-7B-Instruct" --output_dir ../outputs/test_qwen
   python test_simplify.py --model "rinna/japanese-gpt-neox-3.6b" --output_dir ../outputs/test_rinna
   ```

2. 異なるプロンプトで実行して結果を比較
   ```bash
   python test_simplify.py --prompt_file ../outputs/prompt_v1.txt --output_dir ../outputs/test_prompt_v1
   python test_simplify.py --prompt_file ../outputs/prompt_v2.txt --output_dir ../outputs/test_prompt_v2
   ```

3. CSVファイルを開いて、元の文章と平易化された文章を人手で確認

4. 良い結果が得られたら、そのモデル・プロンプトを使って全データセットを平易化
   （`src/simplify_wrime.py`の`WRIMESimplifier`クラスを修正）
