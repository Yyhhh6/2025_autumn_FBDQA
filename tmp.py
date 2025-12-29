import os
import shutil
import re

# 定义路径
src_dir = '/hdd/yyh/src/quant/data/data_raw'
base_dir = '/hdd/yyh/src/quant/data'

# 确保源路径存在
if not os.path.exists(src_dir):
    print(f"错误: 找不到源路径 {src_dir}")
else:
    # 遍历文件夹中的所有文件
    files_processed = 0
    for filename in os.listdir(src_dir):
        # 匹配文件名中的 sym 编号（例如 snapshot_sym0_... 中的 0）
        match = re.search(r'sym(\d+)', filename)
        
        if match:
            sym_id = match.group(1)
            target_folder_name = f'data_sym{sym_id}'
            target_dir = os.path.join(base_dir, target_folder_name)
            
            # 如果目标文件夹不存在，则创建
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            
            src_path = os.path.join(src_dir, filename)
            dst_path = os.path.join(target_dir, filename)
            
            # 执行复制操作
            shutil.copy2(src_path, dst_path) # copy2 会保留文件的元数据（如时间戳）
            files_processed += 1

    print(f"\n任务完成！共复制了 {files_processed} 个文件。")