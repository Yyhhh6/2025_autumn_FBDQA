import os
import pandas as pd
import matplotlib.pyplot as plt

# ====== 配置 ======
dates = ["date12_am", "date20_am", "date30_am", "date40_am", "date55_am", "date64_am", "date78_am"]
col_name = "n_midprice"
save_dir = "mmpc/plots"
os.makedirs(save_dir, exist_ok=True)
# ==================

for date in dates:
    csv_path = f"data/data_sym0_train/snapshot_sym0_{date}.csv"
    save_name = f"mid_price_features_{date}.png"

    # 读取数据
    df = pd.read_csv(csv_path)
    assert col_name in df.columns, f"{col_name} not found in csv columns"

    # 原始 mid_price
    mid = df[col_name] + 1
    ticks = df.index

    # # 移动平均
    # ma5  = mid.rolling(5).mean()
    # ma10 = mid.rolling(10).mean()
    # ma20 = mid.rolling(20).mean()
    # ma40 = mid.rolling(40).mean()

    # ====== 画图 ======
    plt.figure(figsize=(14, 6))

    # 主价格 & 均线
    plt.plot(ticks, mid, label="mid_price", linewidth=1.2)
    # plt.plot(ticks, ma5,  label="MA5",  linestyle="--")
    # plt.plot(ticks, ma10, label="MA10", linestyle="--")
    # plt.plot(ticks, ma20, label="MA20", linestyle="--")
    # plt.plot(ticks, ma40, label="MA40", linestyle="--")

    # # 一阶
    # diff1 = ma40.diff(1) * 1000

    # # 差分（通常量级不同，用较细线）
    # plt.plot(ticks, diff1, label="Δ1 (1st diff)", alpha=0.6)

    plt.xlabel("Tick")
    plt.ylabel("Value")
    plt.title(f"Mid Price & Derived Features ({date})")
    plt.legend(ncol=3)
    plt.grid(True)

    # 保存
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved figure to: {save_path}")

    plt.show()
