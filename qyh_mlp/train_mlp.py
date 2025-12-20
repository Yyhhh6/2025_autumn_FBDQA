import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from dataset import LOBWindowDataset
from model import LOBMLP, Predictor
import os
import random

def split_csv_files(
    data_dir,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    # 读取所有 csv
    csv_files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".csv")
    ]

    csv_files.sort()  # 保证稳定
    random.seed(seed)
    random.shuffle(csv_files)

    n = len(csv_files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_files = csv_files[:n_train]
    val_files = csv_files[n_train:n_train + n_val]
    test_files = csv_files[n_train + n_val:]

    return train_files, val_files, test_files

def collate_fn(batch):
    """
    batch: List[(DataFrame, label)]
    """
    dfs = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return dfs, labels

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== 数据 =====
    data_dir = "data/data_raw"  # ← 放很多 csv 的目录
    train_csvs, val_csvs, test_csvs = split_csv_files(data_dir)

    print(f"Train CSVs: {len(train_csvs)}")
    print(f"Val CSVs:   {len(val_csvs)}")
    print(f"Test CSVs:  {len(test_csvs)}")

    train_dataset = LOBWindowDataset(
        csv_path=train_csvs,   # List[str]
        window_size=100
    )

    val_dataset = LOBWindowDataset(
        csv_path=val_csvs,     # List[str]
        window_size=100
    )

    test_dataset = LOBWindowDataset(
        csv_path=test_csvs,    # List[str]
        window_size=100
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    # ===== 模型 =====
    model = LOBMLP(input_size=43, hidden_sizes=[256, 512, 1024, 512, 256], output_size=3)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ===== Predictor（复用 preprocess）=====
    predictor = Predictor()
    predictor.device = device  # ⚠️ 很重要

    # ===== loss 日志文件 =====
    log_dir = "/user/qinyihua/codebase/2025_autumn_FBDQA/qyh_mlp/logs"
    os.makedirs(log_dir, exist_ok=True)
    train_loss_file = os.path.join(log_dir, "train_loss.txt")
    val_loss_file = os.path.join(log_dir, "val_loss.txt")

    num_epochs = 500
    save_every = 20  # 每20个epoch保存一次ckpt
    val_every = 10    # 每10个epoch计算一次验证 loss

    global_step = 0
    for epoch in range(num_epochs):
        # ===== 训练 =====
        model.train()
        total_train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Train", ncols=100)
        for dfs, labels in pbar:
            labels = labels.to(device)
            x_hat = predictor.preprocess(dfs)

            logits = model(x_hat)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            global_step += 1

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch+1}: avg_train_loss={avg_train_loss:.6f}")

        # 追加训练 loss 到 txt
        with open(train_loss_file, "a") as f:
            f.write(f"{epoch+1},{avg_train_loss:.6f}\n")

        # ===== 验证 =====
        if (epoch + 1) % val_every == 0:
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for dfs, labels in val_loader:
                    labels = labels.to(device)
                    x_hat = predictor.preprocess(dfs)
                    logits = model(x_hat)
                    loss = criterion(logits, labels)
                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            print(f"Epoch {epoch+1}: avg_val_loss={avg_val_loss:.6f}")

            # 追加验证 loss 到 txt
            with open(val_loss_file, "a") as f:
                f.write(f"{epoch+1},{avg_val_loss:.6f}\n")

        # ===== checkpoint =====
        if (epoch + 1) % save_every == 0:
            ckpt_dir = "/user/qinyihua/codebase/2025_autumn_FBDQA/qyh_mlp/checkpoints"
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"model_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    # ===== 最终模型 =====
    final_ckpt_path = os.path.join(ckpt_dir, "model_final.pth")
    torch.save(model.state_dict(), final_ckpt_path)
    print(f"Saved final model: {final_ckpt_path}")


if __name__ == "__main__":
    main()
