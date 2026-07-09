"""
Pipeline Stage C — Merkle Tree 构建
=====================================
为 MNIST 训练集构建二叉 Merkle 树。

叶子哈希：SHA-256(image_bytes || label_byte)
内部节点：Keccak-256（字典序排序后拼接，与 DataAnchor 合约一致）

产物：merkle_result.json
"""

import hashlib
import json
import time
import math
import os
from pathlib import Path
from torchvision import datasets
import numpy as np
from web3 import Web3


# ========== 哈希工具 ==========

def sha256_hex(data: bytes) -> str:
    """SHA-256 摘要，返回 64 位十六进制字符串（用于叶子哈希）"""
    return hashlib.sha256(data).hexdigest()


def keccak_pair(a: str, b: str) -> str:
    """Keccak-256 合并两个字典序排序后的哈希。

    与 OpenZeppelin 及 DataAnchor 合约一致：
    若 a < b 则 keccak(a||b)，否则 keccak(b||a)。
    输入为不带 0x 前缀的十六进制字符串。
    """
    if a < b:
        combined = bytes.fromhex(a) + bytes.fromhex(b)
    else:
        combined = bytes.fromhex(b) + bytes.fromhex(a)
    return Web3.keccak(combined).hex()


def leaf_hash(image: np.ndarray, label: int) -> str:
    """计算单条 MNIST 样本的叶子哈希 = SHA-256(img_bytes || label)"""
    img_bytes = image.astype(np.uint8).tobytes()
    label_byte = bytes([label])
    return sha256_hex(img_bytes + label_byte)


# ========== Merkle 树构建 ==========

def build_merkle_tree(leaves: list[str]) -> tuple[str, int, list[list[str]]]:
    """构建二叉 Merkle 树。

    返回 (root_hex, depth, layers)，其中 layers[0] 为叶子层，
    layers[-1] 仅含 root。奇数层通过复制末尾元素补齐（OpenZeppelin 惯例）。
    """
    tree_layers = [leaves[:]]
    layer = leaves[:]

    while len(layer) > 1:
        if len(layer) % 2 == 1:
            layer.append(layer[-1])
        new_layer = []
        for i in range(0, len(layer), 2):
            new_layer.append(keccak_pair(layer[i], layer[i + 1]))
        tree_layers.append(new_layer)
        layer = new_layer

    root = layer[0]
    return root, len(tree_layers), tree_layers


def get_merkle_proof(leaf_index: int, tree_layers: list[list[str]]) -> list[str]:
    """为指定索引的叶子生成 Merkle 证明路径"""
    proof = []
    idx = leaf_index
    for layer in tree_layers[:-1]:
        sibling_idx = idx + 1 if idx % 2 == 0 else idx - 1
        if sibling_idx < len(layer):
            proof.append(layer[sibling_idx])
        idx //= 2
    return proof


# ========== 主流程 ==========

def main():
    print("=" * 60)
    print("  Merkle Tree 构建 — MNIST 训练集")
    print("=" * 60)

    # 加载数据
    print("\n[*] 加载 MNIST 训练集...")
    t0 = time.time()
    train_set = datasets.MNIST("./data", train=True, download=True)
    n = len(train_set)
    print(f"  样本数: {n}")

    # 计算叶子哈希
    print(f"[*] 计算 {n} 条叶子哈希 (SHA-256)...")
    t1 = time.time()
    leaves = []
    for i, (img, label) in enumerate(train_set):
        img_array = np.array(img)
        leaves.append(leaf_hash(img_array, label))
        if (i + 1) % 10000 == 0:
            print(f"  进度: {i + 1}/{n}")

    leaf_time = time.time() - t1
    print(f"  叶子哈希计算耗时: {leaf_time:.1f} 秒")

    # 构建树
    print(f"\n[*] 构建 Merkle Tree（内部节点: keccak256）...")
    t2 = time.time()
    root, depth, tree_layers = build_merkle_tree(leaves)
    tree_time = time.time() - t2
    print(f"  Tree 深度: {depth} (理论: ceil(log2({n})) = {math.ceil(math.log2(n))})")
    print(f"  构建耗时: {tree_time:.1f} 秒")

    # 验证样例证明
    print(f"\n[*] 验证前 3 条样本的 Merkle Proof...")
    for idx in range(3):
        proof = get_merkle_proof(idx, tree_layers)
        leaf = leaves[idx]
        computed = leaf
        for sibling in proof:
            if computed < sibling:
                computed = Web3.keccak(
                    bytes.fromhex(computed) + bytes.fromhex(sibling)
                ).hex()
            else:
                computed = Web3.keccak(
                    bytes.fromhex(sibling) + bytes.fromhex(computed)
                ).hex()
        status = "通过" if computed == root else "失败"
        print(f"  idx={idx}: proof_len={len(proof)} {status}")

    # 取测试集样本（用于后续差集测试）
    test_set = datasets.MNIST("./data", train=False, download=True)
    test_img, test_label = test_set[0]
    test_leaf = leaf_hash(np.array(test_img), test_label)
    print(f"\n[*] 测试集样本叶子: {test_leaf[:32]}...（用于后续差集测试）")

    # 输出
    total_time = time.time() - t0
    output = {
        "dataset": "MNIST-train",
        "sample_count": n,
        "merkle_root": "0x" + root,
        "merkle_root_raw": root,
        "tree_depth": depth,
        "leaf_sample": leaves[:5],
        "leaf_hash_algorithm": "SHA-256",
        "internal_hash_algorithm": "keccak256",
        "leaf_calc_time_sec": round(leaf_time, 2),
        "tree_build_time_sec": round(tree_time, 2),
        "total_time_sec": round(total_time, 2),
        "test_sample_leaf": test_leaf,
    }

    out_path = "merkle_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"  构建完成")
    print(f"{'=' * 60}")
    print(f"  Merkle Root:  0x{root}")
    print(f"  Tree 深度:    {depth}")
    print(f"  总耗时:       {total_time:.1f} 秒")
    print(f"  输出文件:     {os.path.abspath(out_path)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
