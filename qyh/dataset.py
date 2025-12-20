import pandas as pd
from torch.utils.data import Dataset

class LOBWindowDataset(Dataset):
    def __init__(self, csv_path, window_size=100):
        self.df = pd.read_csv(csv_path)
        self.window_size = window_size

        # 假设 label 在 csv 中，比如叫 label
        assert "label_5" in self.df.columns
        assert "label_10" in self.df.columns
        assert "label_20" in self.df.columns
        assert "label_40" in self.df.columns
        assert "label_60" in self.df.columns

        self.length = len(self.df) - window_size + 1
        assert self.length > 0, "CSV tick 数量不足一个 window"

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        返回：
        - 一个 DataFrame（window）
        - 一个 label（最后一个 tick 的 label）
        """
        window_df = self.df.iloc[idx:idx + self.window_size].copy()
        label = int(window_df.iloc[-1]["label_5"])
        return window_df, label