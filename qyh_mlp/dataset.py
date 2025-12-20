import pandas as pd
from torch.utils.data import Dataset

class LOBWindowDataset(Dataset):
    def __init__(self, csv_path, window_size=100):
        """
        csv_path: str or List[str]
        window_size: int
        """
        self.window_size = window_size

        # 统一成 list
        if isinstance(csv_path, str):
            csv_path = [csv_path]
        self.csv_paths = csv_path

        self.dfs = []               # 每个 csv 一个 df
        self.lengths = []           # 每个 csv 能产生的 window 数
        self.cum_lengths = []       # 累积长度，用于全局 index 映射

        total = 0
        for path in self.csv_paths:
            df = pd.read_csv(path)

            # 校验 label
            assert "label_5" in df.columns, f"{path} 中缺少 label_5"

            n = len(df) - window_size + 1
            if n <= 0:
                raise ValueError(f"{path} tick 数不足 window_size={window_size}")

            self.dfs.append(df)
            self.lengths.append(n)

            total += n
            self.cum_lengths.append(total)

        self.total_length = total

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        """
        返回：
        - window_df: DataFrame (window_size 行)
        - label: int（window 最后一个 tick 的 label_5）
        """
        if idx < 0 or idx >= self.total_length:
            raise IndexError

        # ===== 找 idx 属于哪个 csv =====
        csv_id = 0
        while idx >= self.cum_lengths[csv_id]:
            csv_id += 1

        # 该 csv 内的起始位置
        prev_cum = 0 if csv_id == 0 else self.cum_lengths[csv_id - 1]
        local_idx = idx - prev_cum

        df = self.dfs[csv_id]

        window_df = df.iloc[
            local_idx : local_idx + self.window_size
        ].copy()

        label = int(window_df.iloc[-1]["label_5"])

        return window_df, label
