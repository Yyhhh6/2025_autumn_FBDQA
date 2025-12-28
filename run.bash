#!/bin/bash

# 定义参数组合列表
# 每个组合代表你之前列出的一行特定配置
configs=(
    "--weight1 1.0 --weight2 1.0 --weight3 1.0 --max_depth 3 --subsample 0.5 --colsample_bytree 0.48 --min_child_weight 18 --gamma 4.3"
    "--weight1 1.0 --weight2 0.5 --weight3 1.0 --max_depth 3 --subsample 0.5 --colsample_bytree 0.48 --min_child_weight 18 --gamma 4.3"
    "--weight1 1.5 --weight2 0.5 --weight3 1.5 --max_depth 3 --subsample 0.5 --colsample_bytree 0.48 --min_child_weight 18 --gamma 4.3"
    "--weight1 2.0 --weight2 0.5 --weight3 2.0 --max_depth 3 --subsample 0.5 --colsample_bytree 0.48 --min_child_weight 18 --gamma 4.3"
    "--weight1 1.0 --weight2 0.5 --weight3 1.0 --max_depth 4 --subsample 0.5 --colsample_bytree 0.48 --min_child_weight 18 --gamma 4.3"
    "--weight1 1.0 --weight2 0.5 --weight3 1.0 --max_depth 3 --subsample 0.5 --colsample_bytree 0.48 --min_child_weight 12 --gamma 4.3"
    "--weight1 1.0 --weight2 0.5 --weight3 1.0 --max_depth 3 --subsample 0.5 --colsample_bytree 0.48 --min_child_weight 10 --gamma 4.3"
    "--weight1 1.0 --weight2 0.5 --weight3 1.0 --max_depth 3 --subsample 0.5 --colsample_bytree 0.48 --min_child_weight 18 --gamma 3.0"
    "--weight1 1.0 --weight2 0.5 --weight3 1.0 --max_depth 3 --subsample 0.5 --colsample_bytree 0.48 --min_child_weight 18 --gamma 2.0"
    "--weight1 1.0 --weight2 0.5 --weight3 1.0 --max_depth 3 --subsample 0.5 --colsample_bytree 0.4 --min_child_weight 18 --gamma 4.3"
    "--weight1 1.0 --weight2 0.5 --weight3 1.0 --max_depth 3 --subsample 0.5 --colsample_bytree 0.35 --min_child_weight 18 --gamma 4.3"
    "--weight1 1.0 --weight2 0.5 --weight3 1.0 --max_depth 3 --subsample 0.6 --colsample_bytree 0.48 --min_child_weight 18 --gamma 4.3"
)

# 外层循环：sym 从 0 到 9
for i in {0..9}
do
    echo "================正在处理 data_sym$i ================"
    
    # 内层循环：遍历所有参数组合
    for cfg in "${configs[@]}"
    do
        echo "执行配置: $cfg"
        # 执行训练命令，并将 i 注入到 file_dir 中
        python -m mmpc.train --num_boost_round 8000 $cfg --file_dir ./data/data_sym$i | tee -a output$i.log
    done
done