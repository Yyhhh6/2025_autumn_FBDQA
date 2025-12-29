import re
import os
import json

def extract_to_config(log_dir, config_output="model_config.json"):
    config_map = {}
    
    # 修正后的正则：准确捕获参数块及其后续所有内容，直到遇到下一个参数块
    # 使用 (?=...) 前瞻断言来界定边界
    split_pattern = r"(===== Training Parameters =====.*?)(?====== Training Parameters =====|$)"
    
    # 提取参数名值对的正则
    param_kv_re = re.compile(r"(\w+): ([\d\./\w\-\[\]', ]+)")
    # 提取分数的正则
    score_re = re.compile(r"target_confidence=([\d\.]+).*?Final Score:\s+([\d\.\-]+)", re.DOTALL)

    for i in range(10):
        file_path = os.path.join(log_dir, f"output{i}.log")
        if not os.path.exists(file_path): continue
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 将日志按参数块切割
        units = re.findall(split_pattern, content, re.DOTALL)
        
        best_unit_score = -999999.0
        best_unit_params = {}
        best_unit_conf = 0.6

        for unit in units:
            # 1. 提取当前单元的参数
            current_params = {}
            # 只截取单元开头的参数部分（===== 之间）
            params_header = re.search(r"===== Training Parameters =====\n(.*?)\n=====", unit, re.DOTALL)
            if params_header:
                for line in params_header.group(1).strip().split('\n'):
                    if ': ' in line:
                        k, v = line.split(': ', 1)
                        try:
                            # 转换数字类型
                            current_params[k] = float(v) if '.' in v else (int(v) if v.isdigit() else v)
                        except:
                            current_params[k] = v

            # 2. 在当前单元内寻找最高分
            scores = score_re.findall(unit)
            for conf, score in scores:
                s_val = float(score)
                # 严格判断：只有在这个参数单元内找到更高分，才更新该 sym 的全局配置
                if s_val > best_unit_score:
                    best_unit_score = s_val
                    best_unit_conf = float(conf)
                    best_unit_params = current_params

        if best_unit_params:
            sym_key = f"sym{i}"
            config_map[sym_key] = {
                "best_confidence": best_unit_conf,
                "model_path": f"models_{sym_key}/best_model.json",
                "train_args": best_unit_params,
                "best_score": best_unit_score
            }

    with open(config_output, 'w', encoding='utf-8') as f:
        json.dump(config_map, f, indent=4, ensure_ascii=False)
    print(f"Verified config saved to {config_output}")

extract_to_config(".")