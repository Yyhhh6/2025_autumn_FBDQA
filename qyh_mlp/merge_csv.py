import pandas as pd
import os

def merge_csv(csv_list, output_path):
    """
    合并多个 CSV 文件为一个 CSV 文件
    :param csv_list: List[str], csv 文件路径列表
    :param output_path: str, 输出合并后的 CSV 文件路径
    """
    dfs = []
    for csv_file in csv_list:
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            dfs.append(df)
        else:
            print(f"Warning: {csv_file} 不存在，跳过。")

    if not dfs:
        print("没有可用的 CSV 文件，退出。")
        return

    # 纵向拼接
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df.to_csv(output_path, index=False)
    print(f"合并完成，保存为: {output_path}")


if __name__ == "__main__":
    # 示例
    # csv_files = [
    #     "data/data_raw/snapshot_sym0_date0_am.csv",
    #     "data/data_raw/snapshot_sym0_date0_pm.csv",
    #     "data/data_raw/snapshot_sym0_date1_am.csv",
    #     "data/data_raw/snapshot_sym0_date1_pm.csv",
    #     "data/data_raw/snapshot_sym0_date2_am.csv",
    #     "data/data_raw/snapshot_sym0_date2_pm.csv",
    #     "data/data_raw/snapshot_sym0_date3_am.csv",
    #     "data/data_raw/snapshot_sym0_date3_pm.csv",
    #     "data/data_raw/snapshot_sym0_date4_am.csv",
    #     "data/data_raw/snapshot_sym0_date4_pm.csv",
    #     "data/data_raw/snapshot_sym0_date5_am.csv",
    #     "data/data_raw/snapshot_sym0_date5_pm.csv",
    #     "data/data_raw/snapshot_sym0_date6_am.csv",
    #     "data/data_raw/snapshot_sym0_date6_pm.csv",
    #     "data/data_raw/snapshot_sym0_date7_am.csv",
    #     "data/data_raw/snapshot_sym0_date7_pm.csv",
    #     "data/data_raw/snapshot_sym0_date8_am.csv",
    #     "data/data_raw/snapshot_sym0_date8_pm.csv",
    #     "data/data_raw/snapshot_sym0_date9_am.csv",
    #     "data/data_raw/snapshot_sym0_date9_pm.csv",
    #     "data/data_raw/snapshot_sym0_date10_am.csv",
    #     "data/data_raw/snapshot_sym0_date10_pm.csv",
    #     "data/data_raw/snapshot_sym0_date11_am.csv",
    #     "data/data_raw/snapshot_sym0_date11_pm.csv",
    #     "data/data_raw/snapshot_sym0_date12_am.csv",
    #     "data/data_raw/snapshot_sym0_date12_pm.csv",
    # ]
    # output_csv = "data/data_try/train.csv"
    csv_files = [
        "data/data_raw/snapshot_sym0_date13_am.csv",
        "data/data_raw/snapshot_sym0_date13_pm.csv",
        "data/data_raw/snapshot_sym0_date14_am.csv",
        "data/data_raw/snapshot_sym0_date14_pm.csv",
        "data/data_raw/snapshot_sym0_date15_am.csv",
        "data/data_raw/snapshot_sym0_date15_pm.csv",
    ]
    output_csv = "data/data_try/eval.csv"
    merge_csv(csv_files, output_csv)
