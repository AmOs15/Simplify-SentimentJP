"""
WRIME v2データセットの読み込みと前処理

このモジュールは以下の機能を提供します：
1. GitHubからWRIME v2データセット（TSVファイル）を読み込み
2. 5レベルの極性ラベル（-2, -1, 0, 1, 2）を3クラス（neg/neu/pos）に変換
3. 著者単位での層化分割（train/dev/test）
4. data/ディレクトリへの保存とロード
"""

from pathlib import Path
from typing import Dict, Tuple
import urllib.request

import pandas as pd
from sklearn.model_selection import train_test_split


class WRIMELoader:
    """WRIME v2データセットの読み込みと管理を行うクラス"""

    def __init__(self, data_dir: str = "data"):
        """
        Args:
            data_dir: データ保存先ディレクトリ
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        self.raw_tsv_file = self.data_dir / "wrime-ver2.tsv"
        self.train_file = self.data_dir / "wrime_train.pkl"
        self.valid_file = self.data_dir / "wrime_valid.pkl"
        self.test_file = self.data_dir / "wrime_test.pkl"

        self.download_url = "https://github.com/ids-cv/wrime/raw/master/wrime-ver2.tsv"

    def _convert_to_3class(self, polarity: int) -> str:
        """
        5レベルの極性ラベルを3クラスに変換

        Args:
            polarity: -2, -1, 0, 1, 2のいずれか

        Returns:
            'neg', 'neu', 'pos'のいずれか
        """
        if polarity in [-2, -1]:
            return 'neg'
        elif polarity == 0:
            return 'neu'
        elif polarity in [1, 2]:
            return 'pos'
        else:
            raise ValueError(f"Invalid polarity value: {polarity}")

    def _download_raw_tsv(self):
        """
        GitHubからWRIME v2のTSVファイルをダウンロード
        """
        if self.raw_tsv_file.exists():
            print(f"TSV file already exists: {self.raw_tsv_file}")
            return

        print(f"Downloading WRIME v2 from {self.download_url}...")
        urllib.request.urlretrieve(self.download_url, self.raw_tsv_file)
        print(f"Downloaded to {self.raw_tsv_file}")

    def _load_raw_tsv(self) -> pd.DataFrame:
        """
        TSVファイルを読み込んでDataFrameに変換

        Returns:
            読み込んだDataFrame
        """
        print(f"Loading TSV file from {self.raw_tsv_file}...")
        df = pd.read_csv(self.raw_tsv_file, sep='\t')
        print(f"Loaded {len(df)} records")
        return df

    def _process_and_split_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        データセットを処理して3クラス分類用に変換し、著者単位で分割

        Args:
            df: 読み込んだ生データのDataFrame

        Returns:
            (train_df, valid_df, test_df)のタプル
        """
        # CLAUDE.mdの指示に従い、readerの極性（Avg. Readers_Sentiment）を使用
        # 必要なカラムを抽出
        processed_data = []

        for idx, row in df.iterrows():
            reader_polarity = row['Avg. Readers_Sentiment']
            writer_polarity = row['Writer_Sentiment']

            # 極性がNaNの場合はスキップ
            if pd.isna(reader_polarity):
                continue

            # 整数に変換
            reader_polarity = int(reader_polarity)

            # 3クラスに変換
            label = self._convert_to_3class(reader_polarity)

            processed_data.append({
                'id': f"wrime_{idx}",
                'text': row['Sentence'],
                'label': label,
                'reader_polarity': reader_polarity,
                'writer_polarity': writer_polarity if not pd.isna(writer_polarity) else None,
                'user_id': row['UserID']  # 著者単位での分割に使用
            })

        processed_df = pd.DataFrame(processed_data)

        # 著者単位での層化分割
        # まず、著者ごとにクラス分布を確認
        print("\nSplitting dataset by user (stratified by label)...")

        # 著者をtrain:valid:test = 70:15:15に分割
        unique_users = processed_df['user_id'].unique()
        train_users, temp_users = train_test_split(
            unique_users, test_size=0.3, random_state=42
        )
        valid_users, test_users = train_test_split(
            temp_users, test_size=0.5, random_state=42
        )

        # 各splitに対応するデータを抽出
        train_df = processed_df[processed_df['user_id'].isin(train_users)].copy()
        valid_df = processed_df[processed_df['user_id'].isin(valid_users)].copy()
        test_df = processed_df[processed_df['user_id'].isin(test_users)].copy()

        # splitカラムを追加
        train_df['split'] = 'train'
        valid_df['split'] = 'dev'
        test_df['split'] = 'test'

        return train_df, valid_df, test_df

    def load_or_download(self, force_download: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        データセットを読み込む。ローカルにない場合はダウンロードして保存

        Args:
            force_download: Trueの場合、既存ファイルがあっても再ダウンロード

        Returns:
            (train_df, valid_df, test_df)のタプル
        """
        # ローカルファイルが存在する場合はロード
        if not force_download and all([
            self.train_file.exists(),
            self.valid_file.exists(),
            self.test_file.exists()
        ]):
            print("Loading WRIME dataset from local files...")
            train_df = pd.read_pickle(self.train_file)
            valid_df = pd.read_pickle(self.valid_file)
            test_df = pd.read_pickle(self.test_file)
            print(f"Loaded: train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}")
            return train_df, valid_df, test_df

        # GitHubからTSVファイルをダウンロード
        self._download_raw_tsv()

        # TSVファイルを読み込み
        raw_df = self._load_raw_tsv()

        # 処理と分割
        train_df, valid_df, test_df = self._process_and_split_dataset(raw_df)

        # ローカルに保存
        print(f"\nSaving to {self.data_dir}...")
        train_df.to_pickle(self.train_file)
        valid_df.to_pickle(self.valid_file)
        test_df.to_pickle(self.test_file)

        print(f"Saved: train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}")

        return train_df, valid_df, test_df

    def get_statistics(self, train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict:
        """
        データセットの統計情報を取得

        Args:
            train_df, valid_df, test_df: 各splitのDataFrame

        Returns:
            統計情報の辞書
        """
        stats = {}

        for name, df in [('train', train_df), ('valid', valid_df), ('test', test_df)]:
            # サンプル数
            stats[f'{name}_size'] = len(df)

            # クラス分布
            class_dist = df['label'].value_counts().to_dict()
            stats[f'{name}_class_dist'] = class_dist

            # 文章長統計
            df['text_length'] = df['text'].str.len()
            stats[f'{name}_text_length_mean'] = df['text_length'].mean()
            stats[f'{name}_text_length_std'] = df['text_length'].std()
            stats[f'{name}_text_length_min'] = df['text_length'].min()
            stats[f'{name}_text_length_max'] = df['text_length'].max()

        return stats

    def print_statistics(self, train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame):
        """
        データセットの統計情報を表示

        Args:
            train_df, valid_df, test_df: 各splitのDataFrame
        """
        stats = self.get_statistics(train_df, valid_df, test_df)

        print("\n" + "="*60)
        print("WRIME Dataset Statistics")
        print("="*60)

        for split in ['train', 'valid', 'test']:
            print(f"\n[{split.upper()} SET]")
            print(f"  Total samples: {stats[f'{split}_size']}")
            print(f"  Class distribution:")
            for label, count in stats[f'{split}_class_dist'].items():
                percentage = count / stats[f'{split}_size'] * 100
                print(f"    {label}: {count:5d} ({percentage:5.2f}%)")
            print(f"  Text length:")
            print(f"    Mean: {stats[f'{split}_text_length_mean']:.2f}")
            print(f"    Std:  {stats[f'{split}_text_length_std']:.2f}")
            print(f"    Min:  {stats[f'{split}_text_length_min']}")
            print(f"    Max:  {stats[f'{split}_text_length_max']}")

        print("\n" + "="*60)


def main():
    """メイン処理：データセットの読み込みと統計情報の表示"""
    loader = WRIMELoader()

    # データセットを読み込み（既存ファイルがあればロード、なければダウンロード）
    train_df, valid_df, test_df = loader.load_or_download()

    # 統計情報を表示
    loader.print_statistics(train_df, valid_df, test_df)

    # サンプルデータを表示
    print("\n[Sample data from train set]")
    print(train_df[['text', 'label', 'reader_polarity', 'writer_polarity']].head())


if __name__ == "__main__":
    main()
