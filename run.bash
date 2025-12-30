#!/bin/bash

# 定义参数组合列表
# 每个组合代表你之前列出的一行特定配置
configs=(
    "--weight1 1.0 --weight2 1.0 --weight3 1.0 --max_depth 3 --subsample 0.5 --colsample_bytree 0.48 --min_child_weight 18 --gamma 4.3 --penalty_scale 0.0"
    "--weight1 1.0 --weight2 1.0 --weight3 1.0 --max_depth 3 --subsample 0.5 --colsample_bytree 0.48 --min_child_weight 18 --gamma 4.3 --penalty_scale 10.0"
    "--weight1 1.0 --weight2 1.0 --weight3 1.0 --max_depth 3 --subsample 0.5 --colsample_bytree 0.48 --min_child_weight 18 --gamma 4.3 --penalty_scale 100.0"
    "--weight1 1.0 --weight2 0.5 --weight3 1.0 --max_depth 3 --subsample 0.5 --colsample_bytree 0.48 --min_child_weight 18 --gamma 4.3 --penalty_scale 0.0"
    "--weight1 1.0 --weight2 0.5 --weight3 1.0 --max_depth 3 --subsample 0.5 --colsample_bytree 0.48 --min_child_weight 18 --gamma 4.3 --penalty_scale 10.0"
    "--weight1 1.0 --weight2 0.5 --weight3 1.0 --max_depth 3 --subsample 0.5 --colsample_bytree 0.48 --min_child_weight 18 --gamma 4.3 --penalty_scale 100.0"
    "--weight1 1.5 --weight2 0.5 --weight3 1.5 --max_depth 3 --subsample 0.5 --colsample_bytree 0.48 --min_child_weight 18 --gamma 4.3 --penalty_scale 0.0"
    "--weight1 2.0 --weight2 0.5 --weight3 2.0 --max_depth 3 --subsample 0.5 --colsample_bytree 0.48 --min_child_weight 24 --gamma 4.3 --penalty_scale 0.0"
    "--weight1 2.0 --weight2 0.5 --weight3 2.0 --max_depth 3 --subsample 0.5 --colsample_bytree 0.48 --min_child_weight 18 --gamma 4.3 --penalty_scale 0.0"
    "--weight1 2.0 --weight2 0.5 --weight3 2.0 --max_depth 3 --subsample 0.5 --colsample_bytree 0.48 --min_child_weight 18 --gamma 4.3 --penalty_scale 10.0"
    "--weight1 2.0 --weight2 0.5 --weight3 2.0 --max_depth 3 --subsample 0.5 --colsample_bytree 0.48 --min_child_weight 18 --gamma 4.3 --penalty_scale 100.0"
    "--weight1 1.0 --weight2 0.5 --weight3 1.0 --max_depth 4 --subsample 0.5 --colsample_bytree 0.48 --min_child_weight 18 --gamma 4.3 --penalty_scale 0.0"
    "--weight1 1.0 --weight2 0.5 --weight3 1.0 --max_depth 3 --subsample 0.5 --colsample_bytree 0.48 --min_child_weight 12 --gamma 4.3 --penalty_scale 0.0"
    "--weight1 1.0 --weight2 0.5 --weight3 1.0 --max_depth 3 --subsample 0.5 --colsample_bytree 0.48 --min_child_weight 10 --gamma 4.3 --penalty_scale 0.0"
    "--weight1 1.0 --weight2 0.5 --weight3 1.0 --max_depth 3 --subsample 0.5 --colsample_bytree 0.48 --min_child_weight 18 --gamma 3.0 --penalty_scale 0.0 "
    "--weight1 1.0 --weight2 0.5 --weight3 1.0 --max_depth 3 --subsample 0.5 --colsample_bytree 0.48 --min_child_weight 18 --gamma 2.0 --penalty_scale 0.0"
    "--weight1 1.0 --weight2 0.5 --weight3 1.0 --max_depth 3 --subsample 0.5 --colsample_bytree 0.4 --min_child_weight 18 --gamma 4.3 --penalty_scale 0.0"
    "--weight1 1.0 --weight2 0.5 --weight3 1.0 --max_depth 3 --subsample 0.5 --colsample_bytree 0.35 --min_child_weight 18 --gamma 4.3 --penalty_scale 0.0"
    "--weight1 1.0 --weight2 0.5 --weight3 1.0 --max_depth 3 --subsample 0.6 --colsample_bytree 0.48 --min_child_weight 18 --gamma 4.3 --penalty_scale 0.0"
)

# 设定最大并发数
MAX_JOBS=4
# 创建一个临时管道
mkfifo /tmp/job_control
exec 6<>/tmp/job_control
rm /tmp/job_control

# 填充令牌
for ((m=0; m<$MAX_JOBS; m++)); do echo >&6; done

for i in {0..9}
do
    # 获取令牌 (如果没有令牌，会在这里阻塞)
    read -u 6
    
    (
        echo "Starting Sym $i..."
        for cfg in "${configs[@]}"
        do
            python -m mmpc.train --num_boost_round 3000 $cfg --sym $i --save_path ./models_sym$i --file_dir ./data/data_sym${i}_train >> output$i.log 2>&1
        done
        
        # 任务执行完，把令牌还回去
        echo >&6 
    ) & 
done

wait
exec 6>&- # 关闭管道

# # 外层循环：sym 从 0 到 9
# for i in {0..9}
# do
#     echo "================正在处理 data_sym$i ================"
    
#     # 内层循环：遍历所有参数组合
#     for cfg in "${configs[@]}"
#     do
#         echo "执行配置: $cfg"
#         # 执行训练命令，并将 i 注入到 file_dir 中
#         python -m mmpc.train --num_boost_round 3000 $cfg --sym $i --save_path ./models_sym$i --file_dir ./data/data_sym${i}_train | tee -a output$i.log
#     done
# done