import os
import random

def split_csv_files(
    data_dir,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    # 读取所有 csv
    csv_files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".csv")
    ]

    csv_files.sort()  # 保证稳定
    random.seed(seed)
    random.shuffle(csv_files)

    n = len(csv_files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_files = csv_files[:n_train]
    val_files = csv_files[n_train:n_train + n_val]
    test_files = csv_files[n_train + n_val:]

    return train_files, val_files, test_files

data_dir = "data/data_raw"  # ← 放很多 csv 的目录

train_csvs, val_csvs, test_csvs = split_csv_files(data_dir)

print("test_csvs: ", test_csvs)

print(f"Train CSVs: {len(train_csvs)}")
print(f"Val CSVs:   {len(val_csvs)}")
print(f"Test CSVs:  {len(test_csvs)}")