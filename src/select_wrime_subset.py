"""
WRIMEデータセットからサブセット（約10,500件）を抽出

このモジュールは以下の機能を提供します：
1. 既存のWRIMEデータセット（train/valid/test）から30%をサンプリング
2. 各split内でクラス（neg/neu/pos）バランスを維持
3. random seedを固定して再現性を確保
4. data/ディレクトリへの保存とロード
"""

from pathlib import Path
from typing import Tuple, Dict, List
import pandas as pd
from load_wrime import WRIMELoader


class WRIMESubsetSelector:
    """WRIMEデータセットからサブセットを抽出するクラス"""

    def __init__(self, data_dir: str = "data", sample_ratio: float = 0.3, random_seed: int = 42):
        """
        Args:
            data_dir: データ保存先ディレクトリ
            sample_ratio: 各splitからのサンプリング割合（デフォルト: 0.3 = 30%）
            random_seed: 再現性のための乱数シード
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        self.sample_ratio = sample_ratio
        self.random_seed = random_seed

        # 出力ファイルパス
        self.subset_train_file = self.data_dir / "wrime_subset_train.pkl"
        self.subset_valid_file = self.data_dir / "wrime_subset_valid.pkl"
        self.subset_test_file = self.data_dir / "wrime_subset_test.pkl"

        # 既存のWRIMELoaderを使用
        self.loader = WRIMELoader(data_dir=data_dir)

        self.labels: List[str] = ['neg', 'neu', 'pos']

    def _assign_counts_per_label(
        self,
        target_total: int,
        avail_counts: Dict[str, int],
        labels: List[str]
    ) -> Dict[str, int]:
        """
        各split内で target_total を labels にできるだけ均等に割り当てる。
        - 割り切れない分は Largest Remainder 的に +1 を配分（labels の固定順で安定化）
        - 各ラベルの母数上限（avail_counts）を超えない
        """
        n_labels = len(labels)
        base = target_total // n_labels
        remainder = target_total % n_labels

        # まず均等割り
        assigned = {lbl: base for lbl in labels}

        # 余りをラベル順で +1 ずつ（順序固定で決定的）
        for i in range(remainder):
            assigned[labels[i]] += 1

        # 上限超過の調整（不足が出た分は他ラベルへ回す）
        # 1) 上限超過を削って不足バケツに溜める
        deficit = 0
        for lbl in labels:
            cap = avail_counts.get(lbl, 0)
            if assigned[lbl] > cap:
                over = assigned[lbl] - cap
                assigned[lbl] = cap
                deficit += over

        # 2) 余りがあるラベルに配る（headroom大：cap - assigned の大きい順）
        if deficit > 0:
            headrooms = sorted(
                [(lbl, avail_counts.get(lbl, 0) - assigned[lbl]) for lbl in labels],
                key=lambda x: (-x[1], labels.index(x[0]))  # headroom大→同率はラベル順
            )
            for lbl, room in headrooms:
                if deficit == 0:
                    break
                if room <= 0:
                    continue
                take = min(room, deficit)
                assigned[lbl] += take
                deficit -= take

        total_assigned = sum(assigned.values())
        if total_assigned != target_total:
            # ここに来るのは「総母数がそもそも足りない」場合など。
            print(f"[WARN] Assigned {total_assigned} != target {target_total}. "
                  f"Check avail_counts or target_total feasibility.")

        return assigned
    
    def _stratified_sample_fixed_total(
        self,
        df: pd.DataFrame,
        split: str,
        target_total: int
    ) -> pd.DataFrame:
        """
        各split内で target_total 件を厳密に確保しつつ、ラベルはできるだけ均等にサンプリング。
        """
        # 各ラベルの母数
        avail = {lbl: int((df['label'] == lbl).sum()) for lbl in self.labels}

        # ラベル配分を決定（均等割り＋Largest Remainder＋上限考慮）
        assigned = self._assign_counts_per_label(target_total, avail, self.labels)

        # 実サンプリング
        sampled_dfs = []
        for lbl in self.labels:
            n = assigned[lbl]
            if n <= 0:
                continue
            class_df = df[df['label'] == lbl]
            seed = self.random_seed
            picked = class_df.sample(n=n, random_state=seed, replace=False)
            sampled_dfs.append(picked)

        # 結合してシャッフル（順序安定のため seed は base）
        result_df = pd.concat(sampled_dfs, ignore_index=True)
        result_df = result_df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)

        # 最終チェック
        assert len(result_df) == sum(assigned.values()), "Assigned合計と抽出数が不一致です"
        return result_df

    def create_subset(self, force_recreate: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        サブセットを作成

        Args:
            force_recreate: Trueの場合、既存ファイルがあっても再作成

        Returns:
            (subset_train_df, subset_valid_df, subset_test_df)のタプル
        """
        # ローカルファイルが存在する場合はロード
        if not force_recreate and all([
            self.subset_train_file.exists(),
            self.subset_valid_file.exists(),
            self.subset_test_file.exists()
        ]):
            print("Loading WRIME subset from local files...")
            subset_train_df = pd.read_pickle(self.subset_train_file)
            subset_valid_df = pd.read_pickle(self.subset_valid_file)
            subset_test_df = pd.read_pickle(self.subset_test_file)
            print(f"Loaded: train={len(subset_train_df)}, valid={len(subset_valid_df)}, test={len(subset_test_df)}")
            return subset_train_df, subset_valid_df, subset_test_df

        # 元データをロード
        print("Loading original WRIME dataset...")
        train_df, valid_df, test_df = self.loader.load_or_download()

        print(f"\nOriginal sizes: train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}")
        print(f"Sampling ratio: {self.sample_ratio:.1%}")
        print(f"Random seed: {self.random_seed}")

        
        # 各splitのターゲット件数（整数で厳密に固定）
        target_train = int(round(len(train_df) * self.sample_ratio))
        target_valid = int(round(len(valid_df) * self.sample_ratio))
        target_test  = int(round(len(test_df)  * self.sample_ratio))

        # 各splitで抽出（均等配分＋上限考慮）
        print("\nSampling with fixed totals per split...")
        subset_train_df = self._stratified_sample_fixed_total(train_df, split="train", target_total=target_train)
        subset_valid_df = self._stratified_sample_fixed_total(valid_df, split="valid", target_total=target_valid)
        subset_test_df  = self._stratified_sample_fixed_total(test_df,  split="test",  target_total=target_test)

        print(f"Subset sizes: train={len(subset_train_df)}, valid={len(subset_valid_df)}, test={len(subset_test_df)}")
        print(f"Total subset size: {len(subset_train_df) + len(subset_valid_df) + len(subset_test_df)}")

        # 保存
        print(f"\nSaving subset to {self.data_dir}...")
        subset_train_df.to_pickle(self.subset_train_file)
        subset_valid_df.to_pickle(self.subset_valid_file)
        subset_test_df.to_pickle(self.subset_test_file)
        print("Saved successfully!")

        return subset_train_df, subset_valid_df, subset_test_df

    def get_statistics(self, train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict:
        """
        データセットの統計情報を取得

        Args:
            train_df, valid_df, test_df: 各splitのDataFrame

        Returns:
            統計情報の辞書
        """
        return self.loader.get_statistics(train_df, valid_df, test_df)

    def print_statistics(self, train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame):
        """
        データセットの統計情報を表示

        Args:
            train_df, valid_df, test_df: 各splitのDataFrame
        """
        stats = self.get_statistics(train_df, valid_df, test_df)

        print("\n" + "="*60)
        print("WRIME Subset Dataset Statistics")
        print("="*60)

        total_size = 0
        for split in ['train', 'valid', 'test']:
            print(f"\n[{split.upper()} SET]")
            split_size = stats[f'{split}_size']
            total_size += split_size
            print(f"  Total samples: {split_size}")
            print(f"  Class distribution:")
            for label in ['neg', 'neu', 'pos']:
                count = stats[f'{split}_class_dist'].get(label, 0)
                percentage = count / split_size * 100 if split_size > 0 else 0
                print(f"    {label}: {count:5d} ({percentage:5.2f}%)")
            print(f"  Text length:")
            print(f"    Mean: {stats[f'{split}_text_length_mean']:.2f}")
            print(f"    Std:  {stats[f'{split}_text_length_std']:.2f}")
            print(f"    Min:  {stats[f'{split}_text_length_min']}")
            print(f"    Max:  {stats[f'{split}_text_length_max']}")

        print(f"\n[TOTAL]")
        print(f"  Total samples: {total_size}")
        print("\n" + "="*60)


def main():
    """メイン処理：サブセットの作成と統計情報の表示"""
    selector = WRIMESubsetSelector(
        data_dir="data",
        sample_ratio=0.3,
        random_seed=42
    )

    # サブセットを作成（既存ファイルがあればロード、なければ作成）
    subset_train_df, subset_valid_df, subset_test_df = selector.create_subset()

    # 統計情報を表示
    selector.print_statistics(subset_train_df, subset_valid_df, subset_test_df)

    # サンプルデータを表示
    print("\n[Sample data from subset train set]")
    print(subset_train_df[['text', 'label', 'reader_polarity', 'writer_polarity']].head())


if __name__ == "__main__":
    main()
