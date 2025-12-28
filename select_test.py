import os
import random
import shutil
import math

# ================= 配置 =================
source_prefix = "./data/data_sym"
num_syms = 10
train_ratio = 0.9
random_seed = 42
# =======================================

random.seed(random_seed)

# 汇总目标文件夹
all_train_dir = f"{source_prefix}_train"
all_test_dir = f"{source_prefix}_test"
os.makedirs(all_train_dir, exist_ok=True)
os.makedirs(all_test_dir, exist_ok=True)

for i in range(num_syms):
    src_dir = f"{source_prefix}{i}"
    train_dir = f"{source_prefix}{i}_train"
    test_dir = f"{source_prefix}{i}_test"

    if not os.path.exists(src_dir):
        print(f"[Skip] {src_dir} does not exist.")
        continue

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 只取文件
    all_files = [
        f for f in os.listdir(src_dir)
        if os.path.isfile(os.path.join(src_dir, f))
    ]

    if len(all_files) == 0:
        print(f"[Skip] {src_dir} is empty.")
        continue

    random.shuffle(all_files)

    n_train = math.floor(len(all_files) * train_ratio)
    train_files = all_files[:n_train]
    test_files = all_files[n_train:]

    # 复制训练集
    for f in train_files:
        src = os.path.join(src_dir, f)
        dst = os.path.join(train_dir, f)
        shutil.copy(src, dst)
        # 汇总到统一train文件夹
        shutil.copy(src, os.path.join(all_train_dir, f))

    # 复制测试集
    for f in test_files:
        src = os.path.join(src_dir, f)
        dst = os.path.join(test_dir, f)
        shutil.copy(src, dst)
        # 汇总到统一test文件夹
        shutil.copy(src, os.path.join(all_test_dir, f))

    print(
        f"[Done] {src_dir}: "
        f"train={len(train_files)}, test={len(test_files)}"
    )

print("Train / Test split (copy) and summary finished.")
