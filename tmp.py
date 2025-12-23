# # import os
# # import pandas as pd

# # data_dir = "/hdd/yyh/src/quant/data/data_raw"

# # for fname in os.listdir(data_dir):
# #     if not fname.endswith(".csv"):
# #         continue

# #     file_path = os.path.join(data_dir, fname)

# #     try:
# #         df = pd.read_csv(file_path)

# #         # 缺失值总数
# #         nan_count = df.isna().sum().sum()
# #         import numpy as np

# #         inf_count = np.isinf(df.select_dtypes(include='number')).sum().sum()

# #         if nan_count > 0 or inf_count > 0:
# #             print(f"{fname}: NaN={nan_count}, Inf={inf_count}")


# #     except Exception as e:
# #         print(f"{fname}: 读取失败，错误信息 -> {e}")

# import os
# import numpy as np
# import pandas as pd

# data_dir = "/hdd/yyh/src/quant/data/data_raw"

# all_data = []

# # 读取并汇总所有 csv
# for fname in os.listdir(data_dir):
#     if fname.endswith(".csv"):
#         path = os.path.join(data_dir, fname)
#         df = pd.read_csv(path)
#         # 强制转为数值，非数值变 NaN
#         df = df.apply(pd.to_numeric, errors="coerce")
#         all_data.append(df.to_numpy())

# # 拼接为 (N_total, D)
# data = np.vstack(all_data)[:, :-5]

# # 统计量（按列）
# median = np.nanmedian(data, axis=0)
# mean = np.nanmean(data, axis=0)
# std = np.nanstd(data, axis=0)

# # 保存
# np.savez(
#     "scaler.npz",
#     median=median,
#     mean=mean,
#     std=std
# )

# print("Saved scalar.npz")
# print(f"Data shape: {data.shape}")
# print(f"Median: {median}")
# print(f"Mean: {mean}")
# print(f"Std: {std}")

import os
import pandas as pd

root = "/hdd/yyh/src/quant/data/data_raw"
limit_files = []

# 涨跌停阈值（涨跌幅表示）
UP_LIMIT = 0.099
DOWN_LIMIT = -0.099

for dirpath, _, filenames in os.walk(root):
    for fn in filenames:
        if not fn.lower().endswith(".csv"):
            continue
        path = os.path.join(dirpath, fn)
        try:
            df = pd.read_csv(path)
        except Exception:
            continue

        # 只基于“涨跌幅表示”的字段判断
        price_cols = [c for c in df.columns if c == "n_close"]

        hit = False
        for col in price_cols:
            if ((df[col] >= UP_LIMIT) | (df[col] <= DOWN_LIMIT)).any():
                hit = True
                break

        if hit:
            limit_files.append(path)

print(len(limit_files))
print(limit_files)