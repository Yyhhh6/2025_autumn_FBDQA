import os
import re
import json
import shutil
import glob

def extract_best_config(log_content):
    """从日志内容中提取 Final Score 最大的配置参数"""
    # 匹配 target_confidence 到 Final Score 之间的块
    # 使用正则表达式匹配关键指标
    pattern = (
        r"target_confidence=([\d\.]+).*?"
        r"Precision:\s*([\d\.]+|nan).*?"
        r"Recall:\s*([\d\.]+|nan).*?"
        r"F0.5:\s*([\d\.]+|nan).*?"
        r"Total PNL:\s*([\d\.\-]+).*?"
        r"Avg PNL:\s*([\d\.\-]+|nan).*?"
        r"Trades:\s*(\d+).*?"
        r"Final Score:\s*([\d\.]+|nan)"
    )
    
    matches = re.findall(pattern, log_content, re.DOTALL)
    
    best_conf = None
    max_score = -float('inf')
    
    for conf, prec, rec, f05, tpnl, apnl, trades, score in matches:
        try:
            # 过滤掉无法计算 score 的配置（nan）
            if score == 'nan':
                continue
                
            current_score = float(score)
            if current_score > max_score:
                max_score = current_score
                best_conf = {
                    "best_confidence": float(conf),
                    "precision": float(prec) if prec != 'nan' else 0.0,
                    "recall": float(rec) if rec != 'nan' else 0.0,
                    "f0.5": float(f05) if f05 != 'nan' else 0.0,
                    "total_pnl": float(tpnl),
                    "avg_pnl": float(apnl) if apnl != 'nan' else 0.0,
                    "trades": int(trades),
                    "final_score": current_score
                }
        except ValueError:
            continue
            
    return best_conf

def main():
    # 1. 确保目标目录存在
    target_dir = "./yyh_fast"
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Created directory: {target_dir}")

    # 2. 提取日志信息
    all_configs = {}
    for i in range(10):
        log_path = f"./logs/output_sym{i}.log"
        if os.path.exists(log_path):
            with open(log_path, 'r', encoding='utf-8') as f:
                content = f.read()
                best = extract_best_config(content)
                if best:
                    all_configs[f"sym{i}"] = best
                    print(f"Processed sym{i}: Best Score {best['final_score']} at Confidence {best['best_confidence']}")
        else:
            print(f"Warning: {log_path} not found.")

    # 写入 JSON 配置文件
    config_output_path = os.path.join(target_dir, "confidence_config.json")
    with open(config_output_path, 'w', encoding='utf-8') as f:
        json.dump(all_configs, f, indent=4)
    print(f"Saved best configs to {config_output_path}")

    # 3. 复制模型文件夹中的 JSON 文件
    # 匹配模型目录：./models/model_label20_sym*
    model_dirs = glob.glob("./models/model_label20_sym*")
    for m_dir in model_dirs:
        # 查找文件夹下所有的 json 文件
        json_files = glob.glob(os.path.join(m_dir, "*.json"))
        for json_file in json_files:
            # 保持文件名不变，复制到 yyh_fast
            shutil.copy(json_file, target_dir)
            print(f"Copied {json_file} to {target_dir}")

if __name__ == "__main__":
    main()