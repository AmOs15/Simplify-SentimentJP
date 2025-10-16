#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compute_bertscore.py
.pkl 内のレコードから 'text' と 'text_orig' を取り出し、
各ペアの BERTScore (P, R, F1) を計算して DataFrame に追加、
F1 のヒストグラムを表示・保存し、基本統計量を表示・保存します。

想定される pkl の形式:
 - pandas.DataFrame を to_pickle したもの
 - list of dicts を pickle.dump したもの
 - dict を pickle.dump したもの
"""

import sys
import os
import pickle
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# BERTScore
from bert_score import score

def load_possible_pickle(path: str) -> pd.DataFrame:
    """
    .pkl を安全に読み込み、DataFrame を返す。
    いくつかの可能な構造に対応。
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{path} does not exist")

    # まず pandas の read_pickle を試す
    try:
        df = pd.read_pickle(path)
        if isinstance(df, pd.DataFrame):
            return df.reset_index(drop=True)
    except Exception:
        pass

    # フォールバック: 標準の pickle.load
    with open(path, "rb") as f:
        obj = pickle.load(f)

    # list of dicts -> DataFrame
    if isinstance(obj, list):
        try:
            df = pd.DataFrame(obj)
            return df.reset_index(drop=True)
        except Exception:
            raise ValueError("Loaded object is a list but cannot be converted to DataFrame")

    # dict -> DataFrame (single record) or dict of lists
    if isinstance(obj, dict):
        # dict of lists?
        lengths = [len(v) for v in obj.values() if hasattr(v, "__len__")]
        if len(lengths) > 0 and all(l == lengths[0] for l in lengths):
            return pd.DataFrame(obj)
        else:
            # single record
            return pd.DataFrame([obj])

    raise ValueError(f"Unsupported pickle content: {type(obj)}")

def ensure_text_columns(df: pd.DataFrame):
    """
    text と text_orig カラムを探す・整形する。
    少し名前が違うケースにも対応。
    """
    candidates = df.columns.tolist()
    # common variants
    if "text" not in df.columns:
        # try to infer
        for c in ["sentence", "text_pred", "hyp", "text_candidate"]:
            if c in df.columns:
                df = df.rename(columns={c: "text"})
                break
    if "text_orig" not in df.columns:
        for c in ["text_orig", "text_ref", "reference", "ref", "text_origin", "text_original"]:
            if c in df.columns:
                df = df.rename(columns={c: "text_orig"})
                break

    if "text" not in df.columns or "text_orig" not in df.columns:
        raise KeyError("Required columns 'text' and 'text_orig' not found in DataFrame. "
                       f"Available columns: {df.columns.tolist()}")

    # ensure strings
    df["text"] = df["text"].astype(str).fillna("")
    df["text_orig"] = df["text_orig"].astype(str).fillna("")
    return df

# --- 置き換える関数 ---
def compute_bertscore_on_pairs(df: pd.DataFrame, model_type: str, lang: str, device: str, batch_size: int, num_layers: int = None):
    cand = df["text"].tolist()
    ref  = df["text_orig"].tolist()

    # try/except で model2layers に載っていないケースをキャッチしてフォールバック
    try:
        if num_layers is None:
            # 普通は num_layers を渡さずに library の既知リストを使う
            P, R, F1 = score(cand,
                             ref,
                             model_type=model_type,
                             lang=lang,
                             verbose=True,
                             batch_size=batch_size,
                             device=device)
        else:
            P, R, F1 = score(cand,
                             ref,
                             model_type=model_type,
                             lang=lang,
                             num_layers=num_layers,
                             verbose=True,
                             batch_size=batch_size,
                             device=device)
    except KeyError as e:
        # model_type が内部 map にない場合はここに来ることがある（例: tohoku-nlp/bert-base-japanese-v3）
        # よくある BERT-base の場合は 12 層を使うのが安全
        fallback_layers = 12 if num_layers is None else num_layers
        print(f"Warning: model_type '{model_type}' not found in bert-score known list. Retrying with num_layers={fallback_layers}. (Original error: {e})")
        P, R, F1 = score(cand,
                         ref,
                         model_type=model_type,
                         lang=lang,
                         num_layers=fallback_layers,
                         verbose=True,
                         batch_size=batch_size,
                         device=device)

    # convert to numpy floats
    try:
        p_scores = P.cpu().numpy() if hasattr(P, "cpu") else np.array(P)
        r_scores = R.cpu().numpy() if hasattr(R, "cpu") else np.array(R)
        f1_scores = F1.cpu().numpy() if hasattr(F1, "cpu") else np.array(F1)
    except Exception:
        p_scores = np.array([float(x) for x in P])
        r_scores = np.array([float(x) for x in R])
        f1_scores = np.array([float(x) for x in F1])

    df = df.copy()
    df["bertscore_p"] = p_scores
    df["bertscore_r"] = r_scores
    df["bertscore_f1"] = f1_scores
    return df

def plot_and_stats(df: pd.DataFrame, out_prefix: str):
    col = "bertscore_f1"
    arr = df[col].dropna().astype(float).values

    # basic stats
    stats = {
        "count": int(len(arr)),
        "mean": float(np.mean(arr)) if len(arr)>0 else None,
        "std": float(np.std(arr, ddof=1)) if len(arr)>1 else None,
        "min": float(np.min(arr)) if len(arr)>0 else None,
        "25%": float(np.percentile(arr, 25)) if len(arr)>0 else None,
        "50% (median)": float(np.median(arr)) if len(arr)>0 else None,
        "75%": float(np.percentile(arr, 75)) if len(arr)>0 else None,
        "max": float(np.max(arr)) if len(arr)>0 else None,
    }

    # print stats
    print("\n=== BERTScore F1 basic statistics ===")
    for k,v in stats.items():
        print(f"{k:12s}: {v}")
    print("====================================\n")

    # histogram
    plt.figure(figsize=(8,5))
    plt.hist(arr, bins=30)
    plt.xlabel("BERTScore F1")
    plt.ylabel("count")
    plt.title("Histogram of BERTScore F1")
    hist_path = f"{out_prefix}_bertscore_hist.png"
    plt.tight_layout()
    plt.savefig(hist_path)
    print(f"Histogram saved to: {hist_path}")
    plt.show()

    # save stats to csv
    stats_df = pd.DataFrame.from_dict(stats, orient="index", columns=["value"])
    stats_df.to_csv(f"{out_prefix}_bertscore_stats.csv")
    print(f"Stats saved to: {out_prefix}_bertscore_stats.csv")

    return stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pkl_path", help=".pkl file containing records (DataFrame or list/dict)")
    parser.add_argument("--num_layers", type=int, default=12, help="(optional) number of representation layers to use. If omitted, bert-score tries to pick a default; if model_type not known, specify e.g. 12 for BERT-base.")
    parser.add_argument("--model", default="tohoku-nlp/bert-base-japanese-v3",
                        help="Hugging Face model for BERTScore (default: Japanese Tohoku BERT)")
    parser.add_argument("--lang", default="ja", help="language code for bert-score (default: 'ja')")
    parser.add_argument("--device", default="cuda" if (("CUDA_VISIBLE_DEVICES" in os.environ) or False) else "cpu",
                        help="device to use, e.g. 'cuda' or 'cpu'")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--out_prefix", default="bertscore_output")
    args = parser.parse_args()

    df = load_possible_pickle(args.pkl_path)
    df = ensure_text_columns(df)

    print(f"Loaded {len(df)} records. Computing BERTScore using model={args.model} on device={args.device} ...")

    df_out = compute_bertscore_on_pairs(df, model_type=args.model, lang=args.lang, device=args.device, batch_size=args.batch_size)

    # save augmented dataframe
    out_pkl = f"{args.out_prefix}_with_bertscore.pkl"
    df_out.to_pickle(out_pkl)
    print(f"Augmented DataFrame saved to: {out_pkl}")

    # csv for easy viewing
    out_csv = f"{args.out_prefix}_with_bertscore.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"CSV saved to: {out_csv}")

    # histogram + stats
    stats = plot_and_stats(df_out, args.out_prefix)

    # print top-k lowest/highest for quick check
    print("\nTop 5 lowest F1:")
    print(df_out[["id", "text", "text_orig", "bertscore_f1"]].sort_values("bertscore_f1").head(5).to_string(index=False))
    print("\nTop 5 highest F1:")
    print(df_out[["id", "text", "text_orig", "bertscore_f1"]].sort_values("bertscore_f1", ascending=False).head(5).to_string(index=False))

if __name__ == "__main__":
    main()
