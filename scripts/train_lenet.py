"""
Pipeline Stage B — 模型训练
=============================
在 MNIST 上训练 LeNet-5 卷积网络，共 5 个 epoch。
每个 epoch 保存一份 checkpoint，为阶段 C 的训练时序哈希链
提供审计基础。

产物：checkpoints/lenet_epoch1~5.pth、training_curve.png
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import hashlib
import time
import json
import os
from pathlib import Path


# ========== LeNet-5 架构 ==========
class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)          # 28x28 -> 14x14

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)          # 14x14 -> 5x5

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x


# ========== 辅助函数 ==========
def compute_weight_hash(filepath: str) -> str:
    """计算 .pth 文件的 SHA-256"""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            sha.update(chunk)
    return sha.hexdigest()


# ========== 主流程 ==========
def main():
    # --- 参数 ---
    EPOCHS = 5
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-3
    DATA_DIR = "./data"
    CKPT_DIR = "./checkpoints"

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备] {device}")
    if device.type == "cuda":
        print(f"[GPU] {torch.cuda.get_device_name(0)}")

    # --- 数据加载 ---
    print("\n[*] 加载 MNIST...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_set  = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_set, batch_size=1000, shuffle=False)
    print(f"  训练集: {len(train_set)} 条, 测试集: {len(test_set)} 条")

    # --- 模型/优化器/损失 ---
    model = LeNet5(num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {total_params:,}")

    # --- 训练 ---
    os.makedirs(CKPT_DIR, exist_ok=True)
    history = {"epochs": [], "final_test_acc": 0, "total_time_sec": 0}

    print(f"\n[*] 训练 {EPOCHS} 个 epoch（每 epoch 保存 checkpoint）...")
    t_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        # 训练阶段
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(train_set)

        # 测试阶段
        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(dim=1)
                correct += pred.eq(y).sum().item()
        test_acc = correct / len(test_set)

        # 保存 checkpoint
        ckpt_path = os.path.join(CKPT_DIR, f"lenet_epoch{epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)
        weight_hash = compute_weight_hash(ckpt_path)

        print(f"  Epoch {epoch}/5 | loss={avg_loss:.4f} | test_acc={test_acc:.4f} | "
              f"hash={weight_hash[:16]}... | saved={ckpt_path}")

        history["epochs"].append({
            "epoch": epoch,
            "loss": round(avg_loss, 6),
            "test_acc": round(test_acc, 6),
            "checkpoint": ckpt_path,
            "weight_hash": weight_hash
        })

    t_total = time.time() - t_start
    history["total_time_sec"] = round(t_total, 2)
    history["final_test_acc"] = history["epochs"][-1]["test_acc"]

    # --- 画训练曲线 ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs_axis = [e["epoch"] for e in history["epochs"]]
        losses = [e["loss"] for e in history["epochs"]]
        accs   = [e["test_acc"] for e in history["epochs"]]
        hashes = [e["weight_hash"][:8] for e in history["epochs"]]

        fig, ax1 = plt.subplots(figsize=(10, 5))
        color1 = "#4A6B5F"  # 墨绿
        color2 = "#2E5C8A"  # 青花蓝

        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss", color=color1)
        ax1.plot(epochs_axis, losses, "o-", color=color1, linewidth=2, label="Train Loss")
        ax1.tick_params(axis="y", labelcolor=color1)

        ax2 = ax1.twinx()
        ax2.set_ylabel("Test Accuracy", color=color2)
        ax2.plot(epochs_axis, accs, "s--", color=color2, linewidth=2, label="Test Acc")
        ax2.tick_params(axis="y", labelcolor=color2)
        ax2.set_ylim(0, 1)

        # 在每个点标注权重哈希前 8 位
        for i, (ep, h) in enumerate(zip(epochs_axis, hashes)):
            ax1.annotate(f"H={h}", (ep, losses[i]),
                         textcoords="offset points", xytext=(0, 12),
                         fontsize=7, ha="center", color="#B7976F")

        fig.suptitle(f"LeNet-5 on MNIST — Training Curve\nDevice: {device} | "
                     f"Final Acc: {history['final_test_acc']:.4f}",
                     fontweight="bold")
        fig.tight_layout()
        curve_path = os.path.join(CKPT_DIR, "training_curve.png")
        plt.savefig(curve_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n 训练曲线已保存: {curve_path}")
    except Exception as e:
        print(f"\n 绘图跳过: {e}")

    # --- 汇总输出 ---
    print(f"\n{'='*60}")
    print(f"  训练完成")
    print(f"{'='*60}")
    print(f"  设备:        {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"  总耗时:      {t_total:.1f} 秒")
    print(f"  最终准确率:  {history['final_test_acc']:.4f} ({history['final_test_acc']*100:.2f}%)")
    print(f"  参数量:      {total_params:,}")
    for ep in history["epochs"]:
        print(f"  Epoch {ep['epoch']}: loss={ep['loss']:.4f}  acc={ep['test_acc']:.4f}  "
              f"H(W_{ep['epoch']})={ep['weight_hash']}")
    print(f"  产物目录:    {os.path.abspath(CKPT_DIR)}")
    print(f"{'='*60}")

    # 保存训练历史 JSON
    history_path = os.path.join(CKPT_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"\n 训练历史已保存: {history_path}")


if __name__ == "__main__":
    main()
