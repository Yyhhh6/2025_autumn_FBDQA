import os
import random
import shutil

# 原始数据文件夹前缀
source_prefix = "./data/data_sym"
num_syms = 10
num_samples_per_sym = 15

# 目标文件夹
target_dir = "./data/data_test"
os.makedirs(target_dir, exist_ok=True)

for i in range(num_syms):
    folder_name = f"{source_prefix}{i}"
    if not os.path.exists(folder_name):
        print(f"Folder {folder_name} does not exist, skipping.")
        continue

    # 列出所有文件（只复制文件，不递归子文件夹）
    all_files = [f for f in os.listdir(folder_name) if os.path.isfile(os.path.join(folder_name, f))]
    if len(all_files) < num_samples_per_sym:
        print(f"Folder {folder_name} has less than {num_samples_per_sym} files, using all.")
        selected_files = all_files
    else:
        selected_files = random.sample(all_files, num_samples_per_sym)

    # 复制文件到目标文件夹
    for f in selected_files:
        src_path = os.path.join(folder_name, f)
        dst_path = os.path.join(target_dir, f)
        shutil.copy(src_path, dst_path)
        print(f"Copied {src_path} -> {dst_path}")

print("Sampling and copying finished.")
