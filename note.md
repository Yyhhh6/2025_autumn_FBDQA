[Done] ./data/data_sym0: train=131, test=15
[Done] ./data/data_sym1: train=140, test=16
[Done] ./data/data_sym2: train=140, test=16
[Done] ./data/data_sym3: train=140, test=16
[Done] ./data/data_sym4: train=140, test=16
[Done] ./data/data_sym5: train=119, test=14
[Done] ./data/data_sym6: train=131, test=15
[Done] ./data/data_sym7: train=142, test=16
[Done] ./data/data_sym8: train=140, test=16
[Done] ./data/data_sym9: train=142, test=16
Train / Test split (copy) and summary finished.

## 分sym、参数搜索pipeline
1. bash run.sh：修改想要搜索的参数组合
2. python select_from_output.py：从output日志中选出最优参数组合，生成model_config.json
3. bash run3.sh：根据config.txt中的参数组合，逐个训练不同sym的数据集，并保存模型到对应目录
4. 将生成的所有model_symi文件夹中的模型文件提取出来（可直接使用下面的命令行指令）、model_config.json放入mmpc文件夹
    for i in {0..9}; do
        # 检查文件夹是否存在
        if [ -d "models_sym$i" ]; then
            # 将文件夹内的所有内容移动到当前目录
            mv models_sym$i/* .
            # 删除已经变空的文件夹
            rmdir models_sym$i
        fi
    done
5. zip -r yyh.zip mmpc


# /hdd/yyh/src/quant/models_ZZZ_0/model_20_all_20260101_235606.json
************target_confidence=0.5************
y:  [1 1 1 ... 1 1 1]
y shape: (293280,), test_labels shape: (293280,)
Results for N=20:
Precision: 0.4594
Recall:    0.3656
F0.5:      0.4370
Total PNL:   82.1969
Avg PNL:     0.001019
Trades:      68530
Win Rate:    0.734
Final Score:    7.685
************target_confidence=0.55************
y:  [1 1 1 ... 1 1 1]
y shape: (293280,), test_labels shape: (293280,)
Results for N=20:
Precision: 0.4975
Recall:    0.2870
F0.5:      0.4339
Total PNL:   70.1452
Avg PNL:     0.001200
Trades:      50080
Win Rate:    0.759
Final Score:    15.612
************target_confidence=0.6************
y:  [1 1 1 ... 1 1 1]
y shape: (293280,), test_labels shape: (293280,)
Results for N=20:
Precision: 0.5388
Recall:    0.2177
F0.5:      0.4161
Total PNL:   57.8367
Avg PNL:     0.001412
Trades:      35448
Win Rate:    0.786
Final Score:    27.426
************target_confidence=0.65************
y:  [1 1 1 ... 1 1 1]
y shape: (293280,), test_labels shape: (293280,)
Results for N=20:
Precision: 0.5831
Recall:    0.1583
F0.5:      0.3794
Total PNL:   45.4049
Avg PNL:     0.001650
Trades:      24048
Win Rate:    0.811
Final Score:    41.825
************target_confidence=0.7************
y:  [1 1 1 ... 1 1 1]
y shape: (293280,), test_labels shape: (293280,)
Results for N=20:
Precision: 0.6264
Recall:    0.1074
F0.5:      0.3185
Total PNL:   33.2729
Avg PNL:     0.001916
Trades:      15339
Win Rate:    0.835
Final Score:    55.124
************target_confidence=0.75************
y:  [1 1 1 ... 1 1 1]
y shape: (293280,), test_labels shape: (293280,)
Results for N=20:
Precision: 0.6667
Recall:    0.0664
F0.5:      0.2375
Total PNL:   22.3198
Avg PNL:     0.002212
Trades:      8964
Win Rate:    0.856
Final Score:    61.740
************target_confidence=0.8************
y:  [1 1 1 ... 1 1 1]
y shape: (293280,), test_labels shape: (293280,)
Results for N=20:
Precision: 0.7085
Recall:    0.0373
F0.5:      0.1541
Total PNL:   13.5757
Avg PNL:     0.002546
Trades:      4712
Win Rate:    0.882
Final Score:    58.361
************target_confidence=0.85************
y:  [1 1 1 ... 1 1 1]
y shape: (293280,), test_labels shape: (293280,)
Results for N=20:
Precision: 0.7412
Recall:    0.0180
F0.5:      0.0819
Total PNL:   7.0718
Avg PNL:     0.002882
Trades:      2128
Win Rate:    0.914
Final Score:    42.636
************target_confidence=0.9************
y:  [1 1 1 ... 1 1 1]
y shape: (293280,), test_labels shape: (293280,)
Results for N=20:
Precision: 0.7895
Recall:    0.0070
F0.5:      0.0340
Total PNL:   2.9654
Avg PNL:     0.003288
Trades:      760
Win Rate:    0.974
Final Score:    24.568
************target_confidence=0.95************
y:  [1 1 1 ... 1 1 1]
y shape: (293280,), test_labels shape: (293280,)
Results for N=20:
Precision: 0.8846
Recall:    0.0018
F0.5:      0.0089
Total PNL:   0.7413
Avg PNL:     0.003616
Trades:      183
Win Rate:    0.995
Final Score:    8.125
# /hdd/yyh/src/quant/models_ZZZ_0/model_20_all_20260102_001637.json
************target_confidence=0.5************
y:  [1 1 1 ... 1 1 1]
y shape: (293280,), test_labels shape: (293280,)
Results for N=20:
Precision: 0.4594
Recall:    0.3656
F0.5:      0.4370
Total PNL:   82.1969
Avg PNL:     0.001019
Trades:      68530
Win Rate:    0.734
Final Score:    7.685
************target_confidence=0.55************
y:  [1 1 1 ... 1 1 1]
y shape: (293280,), test_labels shape: (293280,)
Results for N=20:
Precision: 0.4975
Recall:    0.2870
F0.5:      0.4339
Total PNL:   70.1452
Avg PNL:     0.001200
Trades:      50080
Win Rate:    0.759
Final Score:    15.612
************target_confidence=0.6************
y:  [1 1 1 ... 1 1 1]
y shape: (293280,), test_labels shape: (293280,)
Results for N=20:
Precision: 0.5388
Recall:    0.2177
F0.5:      0.4161
Total PNL:   57.8367
Avg PNL:     0.001412
Trades:      35448
Win Rate:    0.786
Final Score:    27.426
************target_confidence=0.65************
y:  [1 1 1 ... 1 1 1]
y shape: (293280,), test_labels shape: (293280,)
Results for N=20:
Precision: 0.5831
Recall:    0.1583
F0.5:      0.3794
Total PNL:   45.4049
Avg PNL:     0.001650
Trades:      24048
Win Rate:    0.811
Final Score:    41.825
************target_confidence=0.7************
y:  [1 1 1 ... 1 1 1]
y shape: (293280,), test_labels shape: (293280,)
Results for N=20:
Precision: 0.6264
Recall:    0.1074
F0.5:      0.3185
Total PNL:   33.2729
Avg PNL:     0.001916
Trades:      15339
Win Rate:    0.835
Final Score:    55.124
************target_confidence=0.75************
y:  [1 1 1 ... 1 1 1]
y shape: (293280,), test_labels shape: (293280,)
Results for N=20:
Precision: 0.6667
Recall:    0.0664
F0.5:      0.2375
Total PNL:   22.3198
Avg PNL:     0.002212
Trades:      8964
Win Rate:    0.856
Final Score:    61.740
************target_confidence=0.8************
y:  [1 1 1 ... 1 1 1]
y shape: (293280,), test_labels shape: (293280,)
Results for N=20:
Precision: 0.7085
Recall:    0.0373
F0.5:      0.1541
Total PNL:   13.5757
Avg PNL:     0.002546
Trades:      4712
Win Rate:    0.882
Final Score:    58.361
************target_confidence=0.85************
y:  [1 1 1 ... 1 1 1]
y shape: (293280,), test_labels shape: (293280,)
Results for N=20:
Precision: 0.7412
Recall:    0.0180
F0.5:      0.0819
Total PNL:   7.0718
Avg PNL:     0.002882
Trades:      2128
Win Rate:    0.914
Final Score:    42.636
************target_confidence=0.9************
y:  [1 1 1 ... 1 1 1]
y shape: (293280,), test_labels shape: (293280,)
Results for N=20:
Precision: 0.7895
Recall:    0.0070
F0.5:      0.0340
Total PNL:   2.9654
Avg PNL:     0.003288
Trades:      760
Win Rate:    0.974
Final Score:    24.568
************target_confidence=0.95************
y:  [1 1 1 ... 1 1 1]
y shape: (293280,), test_labels shape: (293280,)
Results for N=20:
Precision: 0.8846
Recall:    0.0018
F0.5:      0.0089
Total PNL:   0.7413
Avg PNL:     0.003616
Trades:      183
Win Rate:    0.995
Final Score:    8.125