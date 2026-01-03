import os
import glob
import pandas as pd
import numpy as np

def estimate_price_scale_from_folder(
    folder_path: str,
    mid_col: str = 'midprice'
):
    min_abs_diff = np.inf
    min_info = None

    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

    if len(csv_files) == 0:
        raise ValueError(f"No csv files found in {folder_path}")

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)

        if mid_col not in df.columns:
            raise ValueError(f"{csv_path} has no column '{mid_col}'")

        mid = df[mid_col].to_numpy()

        # 一阶差分
        diffs = np.diff(mid)

        for i, diff in enumerate(diffs):
            if diff < 1e-12:
                continue

            abs_diff = abs(diff)

            if abs_diff < min_abs_diff:
                min_abs_diff = abs_diff
                min_info = {
                    'file': os.path.basename(csv_path),
                    'index_prev': i,
                    'index_curr': i + 1,
                    'midprice_prev': mid[i],
                    'midprice_curr': mid[i + 1],
                    'diff': diff
                }

    if min_info is None:
        raise ValueError("All midprice diffs are zero")

    original_price_scale = 0.1 / min_abs_diff

    return {
        'min_abs_midprice_diff': min_abs_diff,
        'original_price_scale': original_price_scale,
        'source': min_info
    }


# ===== 使用 =====
symbols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

results = {}

for sym in symbols:
    folder = f"data/data_sym{sym}"

    result = estimate_price_scale_from_folder(
        folder_path=folder,
        mid_col='n_midprice'
    )

    results[f"sym{sym}"] = result

    print(f"\n=== sym{sym} ===")
    print("最小非零 midprice 差分:", result['min_abs_midprice_diff'])
    print("股价还原比例:", result['original_price_scale'])
    print("来源:", result['source'])