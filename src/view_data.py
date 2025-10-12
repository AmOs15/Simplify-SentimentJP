"""
pklファイルの内容を確認するためのツール

Usage:
    python src/view_data.py --file data/wrime_train.pkl --n 5
    python src/view_data.py -f data/wrime_simp_train.pkl -n 10
"""

import argparse
from pathlib import Path
import pandas as pd


def view_pkl_data(file_path: str, n: int = 5):
    """
    pklファイルの上位n件を表示

    Args:
        file_path: pklファイルのパス
        n: 表示する件数
    """
    path = Path(file_path)

    # ファイルの存在確認
    if not path.exists():
        print(f"Error: File not found: {file_path}")
        return

    # pklファイルを読み込み
    print(f"Loading: {file_path}")
    try:
        df = pd.read_pickle(path)
    except Exception as e:
        print(f"Error loading pkl file: {e}")
        return

    # データの基本情報を表示
    print(f"\n{'='*80}")
    print(f"File: {path.name}")
    print(f"{'='*80}")
    print(f"Total records: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # カラムのデータ型を表示
    print(f"\nColumn types:")
    for col, dtype in df.dtypes.items():
        print(f"  {col}: {dtype}")

    # ラベル分布を表示（labelカラムがあれば）
    if 'label' in df.columns:
        print(f"\nLabel distribution:")
        label_counts = df['label'].value_counts()
        for label, count in label_counts.items():
            percentage = count / len(df) * 100
            print(f"  {label}: {count:5d} ({percentage:5.2f}%)")

    # 上位n件を表示
    print(f"\n{'='*80}")
    print(f"Top {n} records:")
    print(f"{'='*80}")

    # pd.optionsを設定して全文表示
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', 100)
    pd.set_option('display.width', None)

    top_n = df.head(n)

    # 各レコードを見やすく表示
    for idx, row in top_n.iterrows():
        print(f"\n--- Record {idx} ---")
        for col in df.columns:
            value = row[col]
            # 長いテキストの場合は改行して表示
            if isinstance(value, str) and len(value) > 100:
                print(f"{col}:")
                print(f"  {value}")
            else:
                print(f"{col}: {value}")

    print(f"\n{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description='pklファイルの上位n件を表示するツール',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/view_data.py --file data/wrime_train.pkl --n 5
  python src/view_data.py -f data/wrime_simp_train.pkl -n 10
  python src/view_data.py -f data/wrime_subset_test.pkl -n 3
        """
    )

    parser.add_argument(
        '-f', '--file',
        type=str,
        required=True,
        help='pklファイルのパス（例: data/wrime_train.pkl）'
    )

    parser.add_argument(
        '-n', '--num',
        type=int,
        default=5,
        help='表示する件数（デフォルト: 5）'
    )

    args = parser.parse_args()

    # データを表示
    view_pkl_data(args.file, args.num)


if __name__ == "__main__":
    main()
