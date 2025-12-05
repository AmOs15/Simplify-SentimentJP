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
