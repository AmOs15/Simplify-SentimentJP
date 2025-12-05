# 実験結果

## 実験の概要

本実験では、テキストの平易化が感情分類性能に与える影響を検証しました。WRIME v2データセットを用い、3クラス感情分類（negative / neutral / positive）を以下の3条件で比較しました：

- **Baseline (OriginalData)**: 元のテキストのみで学習
- **SimplificationData**: LLMで生成した簡略化テキストのみで学習
- **ConcatData**: 元のテキストと簡略化テキストを連結したデータで学習（データ拡張）

## 結果の要約

実験の結果、**元のテキストを使用したBaseline（Macro-F1: 0.719）が最も高い性能**を示しました。簡略化テキストのみを使用した場合（Macro-F1: 0.675）は性能が低下し、連結データ（Macro-F1: 0.697）は中間的な性能となりました。

### 主な知見

1. **簡略化による性能低下**: SimplificationDataでは全体的な性能が約4ポイント低下（Accuracy: 0.720 → 0.681）
2. **neutral クラスへの影響が顕著**: 特にneutralクラスのF1スコアが大きく低下（0.599 → 0.528）
3. **BERTScoreによる品質評価**: 簡略化テキストと元のテキストの意味的類似度は平均0.840（中央値0.846）と高く、簡略化自体の品質は保たれている
4. **データ拡張の効果は限定的**: ConcatDataは性能向上に寄与しなかった

### 示唆

テキスト簡略化は意味の保持には成功しているものの、感情分類に重要な微妙なニュアンス（皮肉、含意、程度副詞など）が失われている可能性が示唆されます。特にneutralクラスのような曖昧な感情の識別において、元の表現の複雑さが重要な手がかりとなっていると考えられます。

---

## Sentiment classification (neg / neu / pos) – Evaluation results

### Overall
| Experiment | Eval N | Accuracy | Macro-F1 | Macro-P | Macro-R | Loss |
|---|---:|---:|---:|---:|---:|---:|
| Baseline (OriginalData) | 1,860 | 0.720 | 0.719 | 0.718 | 0.720 | 0.640 |
| SimplificationData | 1,860 | 0.681 | 0.675 | 0.676 | 0.681 | 0.849 |
| ConcatData | 3,720 | 0.695 | 0.697 | 0.707 | 0.695 | 0.761 |

### Per-class (F1 with Precision/Recall)
| Experiment | neg F1 (P/R) | neu F1 (P/R) | pos F1 (P/R) |
|---|---:|---:|---:|
| Baseline (OriginalData) | 0.772 (0.750 / 0.797) | 0.599 (0.616 / 0.582) | 0.785 (0.789 / 0.782) |
| SimplificationData | 0.739 (0.682 / 0.808) | 0.528 (0.585 / 0.481) | 0.757 (0.760 / 0.755) |
| ConcatData | 0.757 (0.726 / 0.791) | 0.590 (0.563 / 0.619) | 0.744 (0.832 / 0.673) |

### BERTScore (OriginalData vs SimplificationData)
| Metric | Value |
|---|---:|
| count | 7050 |
| mean | 0.8396 |
| std | 0.0699 |
| min | 0.4862 |
| 25% | 0.7978 |
| median | 0.8460 |
| 75% | 0.8900 |
| max | 1.0000 |

<details>
<summary>Confusion matrices（rows=true, cols=pred / order: neg, neu, pos）</summary>

**Baseline (OriginalData)**
| true\pred | neg | neu | pos |
|---|---:|---:|---:|
| neg | 494 | 109 | 17 |
| neu | 146 | 361 | 113 |
| pos | 19 | 116 | 485 |

**SimplificationData**
| true\pred | neg | neu | pos |
|---|---:|---:|---:|
| neg | 501 | 94 | 25 |
| neu | 199 | 298 | 123 |
| pos | 35 | 117 | 468 |

**ConcatData**
| true\pred | neg | neu | pos |
|---|---:|---:|---:|
| neg | 981 | 240 | 19 |
| neu | 322 | 768 | 150 |
| pos | 48 | 357 | 835 |

</details>
