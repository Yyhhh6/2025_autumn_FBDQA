#!/bin/bash

CONFIG_FILE="./base_model_config.json"

# 使用 Python 提取所有的 sym 键名
SYMS=$(python3 -c "import json; print(' '.join(json.load(open('$CONFIG_FILE')).keys()))")
# SYMS=${SYMS#*sym6 }

DIRS=("logs" "models")

for DIR in "${DIRS[@]}"; do
    if [ -d "$DIR" ]; then
        echo "Cleaning existing directory: $DIR"
        rm -rf "$DIR"/*
    else
        echo "Creating directory: $DIR"
        mkdir -p "$DIR"
    fi
done

for SYM in $SYMS; do
    echo "--------------------------------------------------"
    echo "Starting training for $SYM..."
    
    # 使用 Python 提取具体参数并存入变量
    # 这里定义一个辅助函数或直接调用
    get_param() {
        python3 -c "import json; d=json.load(open('$CONFIG_FILE')); print(d['$SYM']['train_args']['$1'])"
    }

    ROUND=$(get_param "num_boost_round")
    W1=$(get_param "weight1")
    W2=$(get_param "weight2")
    W3=$(get_param "weight3")
    DEPTH=$(get_param "max_depth")
    SUB=$(get_param "subsample")
    COL=$(get_param "colsample_bytree")
    MIN_W=$(get_param "min_child_weight")
    GAMMA=$(get_param "gamma")
    PEN=$(get_param "penalty_scale")
    
    # 后面执行 python 训练的代码保持不变...
    echo "Using parameters: Round=$ROUND, Depth=$DEPTH, Gamma=$GAMMA, W1=$W1, W2=$W2, W3=$W3, Subsample=$SUB, Colsample=$COL, MinChildWeight=$MIN_W, PenaltyScale=$PEN, FileDir=./data/data_$SYM, SavePath=./models/model_label20_$SYM"
    
    python -m yyh_fast.train --sym "$SYM" --num_boost_round "$ROUND" --weight1 "$W1" --weight2 "$W2" --weight3 "$W3" --max_depth "$DEPTH" --subsample "$SUB" --colsample_bytree "$COL" --min_child_weight "$MIN_W" --gamma "$GAMMA" --file_dir "./data/data_$SYM" --save_path "./models/model_label20_$SYM" --penalty_scale "$PEN" | tee -a "logs/output_$SYM.log"
done