from .model import XGBModel
from .Predictor import preprocess
from .data_process import *
import os
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss, mean_squared_log_error
from tqdm import tqdm

from collections import defaultdict

# 去掉涨跌停的文件
# EXCLUDE_FILES = ['./data/data_raw/snapshot_sym1_date33_pm.csv', './data/data_raw/snapshot_sym7_date42_am.csv', './data/data_raw/snapshot_sym1_date25_am.csv', './data/data_raw/snapshot_sym6_date32_pm.csv', './data/data_raw/snapshot_sym4_date33_pm.csv', './data/data_raw/snapshot_sym2_date59_pm.csv', './data/data_raw/snapshot_sym1_date26_pm.csv', './data/data_raw/snapshot_sym2_date59_am.csv', './data/data_raw/snapshot_sym2_date57_pm.csv', './data/data_raw/snapshot_sym4_date34_am.csv', './data/data_raw/snapshot_sym5_date38_am.csv', './data/data_raw/snapshot_sym0_date64_pm.csv', './data/data_raw/snapshot_sym1_date33_am.csv', './data/data_raw/snapshot_sym1_date34_pm.csv', './data/data_raw/snapshot_sym0_date63_pm.csv', './data/data_raw/snapshot_sym0_date71_pm.csv', './data/data_raw/snapshot_sym7_date10_pm.csv', './data/data_raw/snapshot_sym4_date32_pm.csv', './data/data_raw/snapshot_sym6_date42_pm.csv', './data/data_raw/snapshot_sym4_date33_am.csv', './data/data_raw/snapshot_sym7_date42_pm.csv', './data/data_raw/snapshot_sym0_date63_am.csv', './data/data_raw/snapshot_sym2_date42_pm.csv', './data/data_raw/snapshot_sym4_date34_pm.csv', './data/data_raw/snapshot_sym1_date25_pm.csv', './data/data_raw/snapshot_sym5_date23_pm.csv', './data/data_raw/snapshot_sym6_date33_pm.csv', './data/data_raw/snapshot_sym4_date31_pm.csv', './data/data_raw/snapshot_sym7_date10_am.csv', './data/data_raw/snapshot_sym0_date64_pm.csv', './data/data_raw/snapshot_sym0_date71_pm.csv', './data/data_raw/snapshot_sym0_date63_am.csv',]
EXCLUDE_FILES = []

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1 
SEED = 42

# N_list = [5, 10, 20, 40, 60]
N_list = [20]
alpha_map = {5: 0.0005, 10: 0.0005, 20: 0.001, 40: 0.001, 60: 0.001}

def split_csv_files(
    data_dir,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=SEED
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    csv_files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".csv") and 
        os.path.join(data_dir, f) not in EXCLUDE_FILES
    ]

    csv_files.sort() 
    random.seed(seed)
    random.shuffle(csv_files)

    n = len(csv_files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_files = csv_files[:n_train]
    val_files = csv_files[n_train:n_train + n_val]
    test_files = csv_files[n_train + n_val:]
    print(f"Total CSV files: {n}, Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

    return train_files, val_files, test_files

def extract_feature(files_dir, N):
    csv_files = files_dir
    def process_file(file, N):
        if os.path.exists(file):
            df = pd.read_csv(file)#[:-N]
            if df.empty:
                raise ValueError(f"File {file} is empty.")
            df = df.reset_index(drop=True)
            df, labels = preprocess(df, N)
            df = df.squeeze(axis=0)   # (1, T, D) -> (T, D)
            # 去除前100个tick
            labels = labels[99:]
            df_list = df[99:]
        else:
            print("file: ", file)
            raise FileNotFoundError(f"File {file} not found.")
        return df_list, labels

    data = []
    labels_list = []

    for file in tqdm(csv_files, total=len(csv_files), desc="Extracting features"):
        df, labels = process_file(file, N)
        data.append(df)
        labels_list.append(labels)

    data = np.concatenate(data, axis=0)
    labels_list = np.concatenate(labels_list, axis=0)

    return data, labels_list

def extract_feature_test(files_dir, N):
    csv_files = files_dir
    def process_file(file, N):
        if os.path.exists(file):
            df = pd.read_csv(file)#[:-N]
            n_midprice = df['n_midprice'].values
            if df.empty:
                raise ValueError(f"File {file} is empty.")
            df = df.reset_index(drop=True)
            df, labels = preprocess(df, N)
            df = df.squeeze(axis=0)
            labels = labels[99:]
            df_list = df[99:]
            n_midprice = n_midprice[99:]
        else:
            print("file: ", file)
            raise FileNotFoundError(f"File {file} not found.")
        return df_list, labels, n_midprice

    data = []
    labels_list = []
    midprice_list = []

    for file in tqdm(csv_files, total=len(csv_files), desc="Extracting features"):
        df, labels, n_midprice = process_file(file, N)
        data.append(df)
        labels_list.append(labels)
        midprice_list.append(n_midprice)
    data = np.concatenate(data, axis=0)
    labels_list = np.concatenate(labels_list, axis=0)
    midprice_list = np.concatenate(midprice_list, axis=0)

    return data, labels_list, midprice_list

def group_files_by_sym(file_list):
    """
    使用字符串切分方法按 sym 分类文件
    返回 defaultdict(list)
    """
    sym_dict = defaultdict(list)
    for f in file_list:
        # 假设文件名中有 snapshot_symX_，提取 symX
        basename = os.path.basename(f)
        # print("basename: ", basename)
        sym = basename.split("snapshot_")[1].split("_")[0]  # e.g., sym0, sym1
        sym_dict[sym].append(f)
    return sym_dict

def test_by_sym_function(test_files, N, models):
    sym_files_dict = group_files_by_sym(test_files)
    overall_metrics = []

    for sym, sym_files in sym_files_dict.items():
        model = models.get(sym)
        # if not model_path:
        #     print(f"No model found for sym{sym}, skipping.")
        #     continue
        print(f"Evaluating sym{sym} with {len(sym_files)} files using model_{sym}")
        test_data, test_labels, n_midprice = extract_feature_test(sym_files, N)
        # model = XGBModel(model_path)
        y_pred = model.predict(test_data)

        target_confidences = [0.7, 0.725, 0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.9]
        _all_y = {}
        _all_labels = {}
        _all_pnl = {}
        for target_confidence in target_confidences:
            _all_y[target_confidence] = []
            _all_labels[target_confidence] = []
            _all_pnl[target_confidence] = []

            confidence = np.max(y_pred, axis=1)
            signal = np.argmax(y_pred, axis=1)
            signal[confidence < target_confidence] = 1  # Hold

            # 拼接所有 sym
            _all_y[target_confidence].append(signal)
            _all_labels[target_confidence].append(test_labels)

            y = signal
            index_recall = test_labels != 1
            recall = sum(y[index_recall] == test_labels[index_recall]) / sum(index_recall)
            index_precision = y != 1
            precision = sum(y[index_precision] == test_labels[index_precision]) / sum(index_precision)
            beta = 0.5
            f05 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)

            pnl = []
            for i, s in enumerate(signal):
                if i + N >= len(n_midprice):
                    continue
                if s == 2:
                    pnl.append(n_midprice[i+N] - n_midprice[i])
                elif s == 0:
                    pnl.append(n_midprice[i] - n_midprice[i+N])
            _all_pnl[target_confidence].append(np.array(pnl))

            pnl = np.array(pnl)
            total_pnl = pnl.sum()
            avg_pnl = pnl.mean()
            trade_pnl = pnl[pnl != 0]
            win_rate = (trade_pnl > 0).mean() if len(trade_pnl) > 0 else 0
            num_trades = len(trade_pnl)

            final_score = f05 * (avg_pnl - 0.0006) * (avg_pnl - 0.0006) * 10000 * 10000
            if avg_pnl - 0.0006 < 0:
                final_score = -final_score

            overall_metrics.append({
                "sym": sym,
                "target_confidence": target_confidence,
                "Precision": precision,
                "Recall": recall,
                "F0.5": f05,
                "Total_PNL": total_pnl,
                "Avg_PNL": avg_pnl,
                "Win_Rate": win_rate,
                "Num_Trades": num_trades,
                "Final_Score": final_score, 
            })

    for target_confidence in target_confidences:
        all_y = _all_y[target_confidence]
        all_labels = _all_labels[target_confidence]
        all_pnl = _all_pnl[target_confidence]

        # 合并所有 sym 数据
        all_y = np.concatenate(all_y)
        all_labels = np.concatenate(all_labels)
        all_pnl = np.concatenate(all_pnl)

        # 总体指标计算
        index_recall = all_labels != 1
        recall = sum(all_y[index_recall] == all_labels[index_recall]) / sum(index_recall)
        index_precision = all_y != 1
        precision = sum(all_y[index_precision] == all_labels[index_precision]) / sum(index_precision)
        beta = 0.5
        f05 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)

        total_pnl = all_pnl.sum()
        avg_pnl = all_pnl.mean()
        trade_pnl = all_pnl[all_pnl != 0]
        win_rate = (trade_pnl > 0).mean() if len(trade_pnl) > 0 else 0
        num_trades = len(trade_pnl)

        final_score = f05 * (avg_pnl - 0.0006) * (avg_pnl - 0.0006) * 10000 * 10000
        if avg_pnl - 0.0006 < 0:
            final_score = -final_score

        overall_metrics.append({
            "sym": "Overall",
            "target_confidence": target_confidence,
            "Precision": precision,
            "Recall": recall,
            "F0.5": f05,
            "Total_PNL": total_pnl,
            "Avg_PNL": avg_pnl,
            "Win_Rate": win_rate,
            "Num_Trades": num_trades,
            "Final_Score": final_score, 
        })

    # ------------------ 输出指标 ------------------
    df_metrics = pd.DataFrame(overall_metrics)
    print("Per-sym metrics:\n", df_metrics)

    # 保存到 CSV 文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"./results/per_sym_metrics_{timestamp}.csv"
    df_metrics.to_csv(output_file, index=False)  # 不保存行索引
    print(f"Per-sym metrics saved to {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train and test XGBModel")
    parser.add_argument("--num_boost_round", type=int, default=8000, help="Number of boosting rounds")
    parser.add_argument("--weight1", type=float, default=1.0, help="Weight 1 for custom loss")
    parser.add_argument("--weight2", type=float, default=0.5, help="Weight 2 for custom loss")
    parser.add_argument("--weight3", type=float, default=1.0, help="Weight 3 for custom loss")

    parser.add_argument("--max_depth", type=int, default=3, help="Maximum depth of trees")
    parser.add_argument("--subsample", type=float, default=0.5, help="Subsample ratio of training instances")
    parser.add_argument("--colsample_bytree", type=float, default=0.48, help="Subsample ratio of columns per tree")
    parser.add_argument("--min_child_weight", type=int, default=18, help="Minimum sum of instance weight in a child")
    parser.add_argument("--gamma", type=float, default=4.3, help="Minimum loss reduction to make a split")
    
    parser.add_argument("--file_dir", type=str, default="./data/data_sym_test", help="file_dir")
    parser.add_argument("--save_path", type=str, default="./models_sym/", help="save_path")

    args = parser.parse_args()

    # 打印参数
    print("===== Training Parameters =====")
    print(f"num_boost_round: {args.num_boost_round}")
    print(f"weight1: {args.weight1}")
    print(f"weight2: {args.weight2}")
    print(f"weight3: {args.weight3}")
    print(f"max_depth: {args.max_depth}")
    print(f"subsample: {args.subsample}")
    print(f"colsample_bytree: {args.colsample_bytree}")
    print(f"min_child_weight: {args.min_child_weight}")
    print(f"gamma: {args.gamma}")
    print(f"file_dir: {args.file_dir}")
    print(f"save_path: {args.save_path}")
    print("===============================")

    for N in N_list:
        # 划分训练集、验证集、测试集
        train_files, val_files, test_files = split_csv_files(data_dir=args.file_dir, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, test_ratio=1-TRAIN_RATIO-VAL_RATIO, seed=SEED)

        # 提取训练集、验证集、测试集的特征
        train_by_sym = group_files_by_sym(train_files)
        val_by_sym   = group_files_by_sym(val_files)

        models = {}

        print("train_by_sym.keys(): ", train_by_sym.keys())
        for sym in train_by_sym.keys():
            print(f"\n===== Training model for {sym} =====")

            train_data, train_labels = extract_feature(train_by_sym[sym], N)
            val_data, val_labels     = extract_feature(val_by_sym.get(sym, []), N)

            model = XGBModel()
            model.train(
                train_data,
                train_labels,
                val_data if len(val_data) > 0 else None,
                val_labels if len(val_labels) > 0 else None,
                num_boost_round=args.num_boost_round,
                early_stopping_rounds=150,
                N=N,
                weight1=args.weight1,
                weight2=args.weight2,
                weight3=args.weight3,
                max_depth=args.max_depth,
                subsample=args.subsample,
                colsample_bytree=args.colsample_bytree,
                min_child_weight=args.min_child_weight,
                gamma=args.gamma,
                save_path=args.save_path,
                sym=sym,
            )

            models[sym] = model

        print("*"*50)
        print("Finish Traing, Starting Testing...")
        print("*"*50)

        test_by_sym_function(test_files, N=N, models=models)
    
    print("\n\n\n")