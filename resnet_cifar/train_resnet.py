"""
Pipeline Stage B（扩展实验）— ResNet18 + CIFAR-10 训练
=====================================================
在 CIFAR-10 上训练 ResNet18，共 10 个 epoch。
每个 epoch 保存一份 checkpoint，为训练时序哈希链提供审计基础。

本脚本与 scripts/train_lenet.py（LeNet-5 + MNIST 基线）对应，
用于验证框架对更大规模模型（约 1100 万参数）的可扩展性。

产物：
  resnet_cifar/checkpoints/resnet_epoch1~10.pth
  resnet_cifar/checkpoints/training_curve.png
  resnet_cifar/checkpoints/training_history.json

用法：
  cd 项目根目录
  python resnet_cifar/train_resnet.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import hashlib
import time
import json
import os
from pathlib import Path


# ========== 路径（基于脚本位置，不依赖 cwd） ==========
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
CKPT_DIR = os.path.join(SCRIPT_DIR, "checkpoints")


# ========== ResNet18 适配 CIFAR-10 ==========
def build_resnet18_cifar(num_classes: int = 10) -> nn.Module:
    """
    构造适配 CIFAR-10（32x32 彩色图）的 ResNet-18。

    标准 ResNet-18 的第一层是 7x7 stride-2 卷积 + maxpool，
    针对 ImageNet 224x224 设计；直接用于 32x32 会导致特征图过早缩小。
    此处将 conv1 改为 3x3 stride-1，并移除 maxpool，保留更多空间信息。
    """
    model = models.resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # 移除 3x3 maxpool
    return model


# ========== 辅助函数 ==========
def compute_weight_hash(filepath: str) -> str:
    """计算 .pth 文件的 SHA-256（流式读取，支持大 checkpoint）"""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            sha.update(chunk)
    return sha.hexdigest()


# ========== 主流程 ==========
def main():
    # --- 参数 ---
    EPOCHS = 10
    BATCH_SIZE = 128
    LEARNING_RATE = 0.1
    WEIGHT_DECAY = 5e-4
    MOMENTUM = 0.9

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备] {device}")
    if device.type == "cuda":
        print(f"[GPU] {torch.cuda.get_device_name(0)}")

    # --- 数据加载 ---
    print("\n[*] 加载 CIFAR-10...")
    # CIFAR-10 标准归一化统计量
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=transform_train)
    test_set = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=transform_test)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2)
    print(f"  训练集: {len(train_set)} 条, 测试集: {len(test_set)} 条")

    # --- 模型 / 优化器 / 损失 / 调度器 ---
    model = build_resnet18_cifar(num_classes=10).to(device)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE,
                          momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {total_params:,}")

    # --- 训练 ---
    os.makedirs(CKPT_DIR, exist_ok=True)
    history = {"epochs": [], "final_test_acc": 0, "total_time_sec": 0,
               "model": "ResNet18", "dataset": "CIFAR-10"}

    print(f"\n[*] 训练 {EPOCHS} 个 epoch（每 epoch 保存 checkpoint）...")
    t_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        # --- 训练阶段 ---
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

        # --- 测试阶段 ---
        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(dim=1)
                correct += pred.eq(y).sum().item()
        test_acc = correct / len(test_set)

        # 学习率步进
        cur_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # 保存 checkpoint
        ckpt_path = os.path.join(CKPT_DIR, f"resnet_epoch{epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)
        weight_hash = compute_weight_hash(ckpt_path)

        print(f"  Epoch {epoch}/{EPOCHS} | loss={avg_loss:.4f} | test_acc={test_acc:.4f} | "
              f"lr={cur_lr:.4f} | hash={weight_hash[:16]}... | saved={os.path.basename(ckpt_path)}")

        history["epochs"].append({
            "epoch": epoch,
            "loss": round(avg_loss, 6),
            "test_acc": round(test_acc, 6),
            "lr": round(cur_lr, 6),
            "checkpoint": ckpt_path,
            "weight_hash": weight_hash,
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
        accs = [e["test_acc"] for e in history["epochs"]]
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

        for i, (ep, h) in enumerate(zip(epochs_axis, hashes)):
            ax1.annotate(f"H={h}", (ep, losses[i]),
                         textcoords="offset points", xytext=(0, 12),
                         fontsize=7, ha="center", color="#B7976F")

        fig.suptitle(f"ResNet-18 on CIFAR-10 — Training Curve\nDevice: {device} | "
                     f"Final Acc: {history['final_test_acc']:.4f}",
                     fontweight="bold")
        fig.tight_layout()
        curve_path = os.path.join(CKPT_DIR, "training_curve.png")
        plt.savefig(curve_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n训练曲线已保存: {curve_path}")
    except Exception as e:
        print(f"\n[!] 绘图跳过: {e}")

    # --- 汇总输出 ---
    print(f"\n{'='*60}")
    print(f"  训练完成")
    print(f"{'='*60}")
    print(f"  模型:        ResNet-18 (CIFAR-10 适配版)")
    print(f"  数据集:      CIFAR-10")
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
    print(f"\n训练历史已保存: {history_path}")


if __name__ == "__main__":
    main()
