# 说明
- 一共有1521个文件，total data shape: (3040479, 31)
- 自己直接按照4：1随机划分文件作为验证集和测试集，随机种子为42，进行数据分析：
    - 逐个concat:251s, 全部读取后一道concat:15s, ThreadPoolExecutor并行读取：6s
    - 训练集和验证集都没有数据缺失
    - train: 2426786 val: 613693
    - 标签（没有去掉前100个数据和最后N个数据时）
        在训练集中：
        标签为0的样本个数： 367407
        标签为1的样本个数： 1694936
        标签为2的样本个数： 364443
        在测试集中：
        标签为0的样本个数： 93397
        标签为1的样本个数： 427412
        标签为2的样本个数： 92884
    - 现在针对不同的N从头开始训练不同的模型，因此并没有用到α
    - 经过手动标签标注后发现对于每个文件（每天、每个上下午）的最后N的标签应该是随机的，不是接着下一个半天的数据来标的，不要使用。
    - 应当使用的训练数据格式应当是：100个tick数据+label_N（每个文件不要最后N天的数据）。核心应当是如何压缩这100个tick数据的信息。如果是深度学习模型，就直接喂进去；对于其他模型
    - 每个文件的数据行数都为1999

- 策略
    1. 没有做任何数据处理，采用树模型。
    2. 

- 运行指令：
    ```
    python -u run.py > log.txt
    ```

# TOOD
1. 检查涨跌停，处理盘口价格
2. factors_null_process：部分特征缺失值过多，如何处理
3. 为什么train, val, test样本的实际比例不对？？
4. cal_pnl计算流程是否正确？
5. 中间价和盘口价的原始数据都已经标准化了，之后还需要标准化吗？？（可能标准化的范围不一样:sym?date?）
6. 对犯的不同错误设置不同的损失：把recall降一降，提高precision和pnl
    将loss改为CostSensitiveLoss。原来loss的结果：
        [PyTorch] Epoch 27 train_loss=0.5993 val_f0.5_macro=0.5851 precision_macro=0.6230 recall_macro=0.5159
        Early stopping.
        MLP N=5 test f0.5_macro: 0.5915 precision_macro: 0.6323 recall_macro: 0.5186
        MLP N=5 PnL: -3,452 Return: -34.52% Sharpe: -5.87 WinRate: 36.54%
        {5: {'mlp': ({'accuracy': 0.7496720693554466, 'precision_macro': 0.6323089804773292, 'recall_macro': 0.5185705717380541, 'f0.5_macro': 0.5915046396277752, 'f1_macro': 0.5515886813964958, 'confusion_matrix': array([[ 3001,  5750,   607],
            [ 1973, 43715,  1601],
            [  663,  6009,  3006]]), 'pnl_total_pnl': np.float64(-3451.739667863193), 'pnl_total_return': np.float64(-0.3451739667863193), 'pnl_annual_return': np.float64(-0.9013763999765746), 'pnl_sharpe_ratio': np.float64(-5.873635639960603), 'pnl_max_drawdown': np.float64(-0.3619289194839038), 'pnl_win_rate': np.float64(0.3654312015503876), 'pnl_total_trades': np.float64(8256.0), 'pnl_winning_trades': np.int64(3017), 'pnl_losing_trades': np.int64(7586), 'pnl_avg_win': np.float64(0.0010396817763373323), 'pnl_avg_loss': np.float64(-0.000467510057216965), 'pnl_profit_loss_ratio': np.float64(2.2238703965566886), 'pnl_volatility': np.float64(0.0006336964351880505), 'pnl_long_only_return': np.float64(0.0421590267197649), 'pnl_long_only_sharpe': np.float64(1.0072271597954479), 'pnl_position_coverage': np.float64(0.1636034677723332), 'pnl_trading_frequency': np.float64(0.1244779494911421)}, './results/mlp_N5.pt')}}

7. 决策树的选择
    
    📌 总结对比
    | 特性     | XGBoost    | LightGBM    | CatBoost     |
    | ------ | ---------- | ----------- | ------------ |
    | 特点     | 稳定、全面、经典   | 更快、更轻、适合大数据 | 类别特征最强       |
    | 速度     | 快          | **最快**      | 中等           |
    | 调参难度   | 中等         | 较高（容易过拟合）   | **最简单**      |
    | 类别特征处理 | 弱          | 弱（要手动编码）    | **极强（自动处理）** |
    | 树构建方式  | Level-wise | Leaf-wise   | 对称树          |
    | 大规模训练  | 可以         | **非常好**     | 中等           |

    🧭 如何选择？
    如果你的数据是：

    - 类别特征多 → 直接用 CatBoost
    - 数据量大，追求速度 → LightGBM
    - 需要稳健的经典方案，调参空间大 → XGBoost


先替换 inf → 再填充 nan → 再 mad 去极值（基于 train） → 再 zscore 归一（基于 train）
