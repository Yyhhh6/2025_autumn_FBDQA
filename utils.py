import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss, mean_squared_log_error, precision_score, recall_score
import tqdm, sys, os, gc, argparse, warnings
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
from pathlib import Path

def calc_factor_correlation(
    datas: pd.DataFrame,
    factor_list: list,
    save_path: str = "factor_corr_heatmap.png",
    stock_id_col: str = "sym"
):
    """
    Tick级别横截面因子相关性计算 + 热力图绘制
    
    Parameters
    ----------
    datas : pd.DataFrame
        必须包含 ['date', 'time', stock_id_col] 和 factor_list 中的因子列。
        每个 date-time 下应有多个股票（横截面）。
    factor_list : list
        需要分析相关性的因子列名列表。
    save_path : str
        保存热力图路径。
    stock_id_col : str
        股票 ID（或任何样本标识）列名，默认为 'code'。
    
    Returns
    -------
    pd.DataFrame
        平均横截面相关系数矩阵
    """

    # ---- Step 1：数据检查 ----
    required_cols = {"date", "time", stock_id_col} | set(factor_list)
    missing = required_cols - set(datas.columns)
    if missing:
        raise ValueError(f"数据缺少必要列: {missing}")

    # ---- Step 2：分组计算相关系数（date × time 横截面）----
    corr_sum = None
    valid_count = 0

    # 按 tick 分组（每个 tick 必须含多股票才有横截面）
    for (date, t), group in datas.groupby(["date", "time"]):

        # 至少两个股票才能计算相关性
        if group[stock_id_col].nunique() < 2:
            continue

        X = group[factor_list]

        # 若某个 tick 所有样本某因子值一致，corr 会为 NaN，跳过
        if X.shape[0] > 1:
            corr = X.corr()

            # 若全为 NaN 说明这一组不可用（所有因子无波动）
            if corr.isna().all().all():
                continue

            if corr_sum is None:
                corr_sum = corr.copy()
            else:
                corr_sum += corr

            valid_count += 1

    if corr_sum is None:
        raise ValueError("未找到任何有效横截面用于计算相关性，请检查数据。")

    corr_mean = corr_sum / valid_count

    # ---- Step 3：绘制热力图 ----
    plt.figure(figsize=(22, 18))

    sns.heatmap(
        corr_mean,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        linewidths=0.05,
        linecolor="white",
        annot_kws={"size": 9, "weight": "bold"},
        center=0
    )

    plt.title("Tick-level Factor Correlation (Cross-sectional)", fontsize=20, fontweight="bold")
    plt.tight_layout()

    # ---- Step 4：保存图片 ----
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor="white")

    plt.close()

    print(f"相关性热力图已保存至：{os.path.abspath(save_path)}")
    print(f"有效横截面数量：{valid_count}")

    return corr_mean


def lowdin_orthogonal(data:pd.DataFrame, col:list)->pd.DataFrame:

    data_ = data.copy() # 创建副本不影响原数据
    F = np.asmatrix(data_[col])  # 除去行业指标,将数据框转化为矩阵
    M = F.T @ F # 等价于 (F.shape[0] - 1) * np.cov(F.T)
    a,U = np.linalg.eig(M)  # a为特征值，U为特征向量
    D_inv = np.linalg.inv(np.diag(a))
    S = U @ np.sqrt(D_inv) @ U.T
    data_[col] = data_[col].dot(S)
    
    return data_

def cv_model(clf, train_x, train_y, test_x, clf_name, seed):
    folds = 5
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.zeros([train_x.shape[0], 3])
    # test_predict = np.zeros([test_x.shape[0], 3])
    f1_score_list = []
    precision_list = []
    recall_list = []
    
    for i, (train_index, valid_index) in enumerate(kf.split(train_x, train_y)):
        print('************************************ {} ************************************'.format(str(i+1)))
        trn_x, trn_y, val_x, val_y = train_x.iloc[train_index], train_y[train_index], train_x.iloc[valid_index], train_y[valid_index]
       
        if clf_name == "cat":
            params = {'learning_rate': 0.2, 'depth': 6, 'bootstrap_type':'Bernoulli','random_seed':seed,
                      'od_type': 'Iter', 'od_wait': 100, 'allow_writing_files': False,
                      'loss_function': 'MultiClass'}        
            model = model(iterations=1000, **params)
            model.fit(trn_x, trn_y, eval_set=(val_x, val_y),
                      metric_period=1000,
                      use_best_model=True, 
                      cat_features=[],
                      verbose=1)
            
            val_pred  = model.predict_proba(val_x)
            # test_pred = model.predict_proba(test_x)
        
        oof[valid_index] = val_pred
        # test_predict += test_pred / kf.n_splits
        
        F1_score = f1_score(val_y, np.argmax(val_pred, axis=1), average='macro')
        # precision 和 recall
        precision = precision_score(val_y, np.argmax(val_pred, axis=1), average='macro')
        recall = recall_score(val_y, np.argmax(val_pred, axis=1), average='macro')
        print(f"Precision: {precision}, Recall: {recall}, F1_score: {F1_score}")
        f1_score_list.append(F1_score)
        precision_list.append(precision)
        recall_list.append(recall)        
    return oof, precision_list, recall_list, f1_score_list
    
def run_tree_model(model, train_x, train_y, val_x, val_y, seed):
    params = {'learning_rate': 0.2, 'depth': 6, 'bootstrap_type':'Bernoulli','random_seed':seed,
                      'od_type': 'Iter', 'od_wait': 100, 'allow_writing_files': False,
                      'loss_function': 'MultiClass'}        
    model = model(iterations=1000, **params)
    model.fit(train_x, train_y, eval_set=(val_x, val_y),
                metric_period=1000,
                use_best_model=True, 
                cat_features=[],
                verbose=1)
    val_pred  = model.predict_proba(val_x)

    F1_score = f1_score(val_y, np.argmax(val_pred, axis=1), average='macro')
    # precision 和 recall
    precision = precision_score(val_y, np.argmax(val_pred, axis=1), average='macro')
    recall = recall_score(val_y, np.argmax(val_pred, axis=1), average='macro')
    print(f"Precision: {precision}, Recall: {recall}, F1_score: {F1_score}")
    return val_pred, precision, recall, F1_score

    
def check_metric(y, y_hat):
    # 总体情况
    print("预测正确的标签数：", sum(y_hat == y))
    print("总体正确率：", sum(y_hat == y)/len(y_hat))

    # 分标签查看：
    print("真实标签为0样本的正确预测个数：", sum(y[y == 0] == y_hat[y == 0]))
    print("真实标签为1样本的正确预测个数：", sum(y[y == 1] == y_hat[y == 1]))
    print("真实标签为2样本的正确预测个数：", sum(y[y == 2] == y_hat[y == 2]))

    ## 我们更关心上涨下跌情况的预测
    # 所有不为1的标签的召回率（即仅考虑真实标签为上涨或下跌样本是否被正确分类）
    index = y != 1
    print("上涨下跌召回率：", sum(y_hat[index]==y[index])/sum((index)+1e-6))
    # 所有不为1的标签的准确率（即仅考虑预测为上涨或下跌样本是否是正确）
    index = y_hat != 1
    print("上涨下跌准确率：", sum(y_hat[index]==y[index])/sum((index)+1e-6))