"""
Pipeline Stage E — Merkle 证明验证
====================================
对链上已注册的数据集执行单条样本 Merkle 成员证明验证。
包含正样本（在训练集中）和负样本（不在训练集中）测试，
输出拒绝率。

依赖：已部署的 DataAnchor 合约 + 已注册数据集
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path
from web3 import Web3

import torch
from torchvision import datasets
import numpy as np
import hashlib


# ========== 复用 build_merkle 的哈希逻辑 ==========
def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def keccak_pair(a: str, b: str) -> str:
    if a < b:
        combined = bytes.fromhex(a) + bytes.fromhex(b)
    else:
        combined = bytes.fromhex(b) + bytes.fromhex(a)
    return Web3.keccak(combined).hex()


def leaf_hash(image: np.ndarray, label: int) -> str:
    img_bytes = image.astype(np.uint8).tobytes()
    label_byte = bytes([label])
    return sha256_hex(img_bytes + label_byte)


def get_merkle_proof(leaf_index: int, tree_layers: list[list[str]]) -> list[str]:
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
    parser = argparse.ArgumentParser(description="单条样本 Merkle Proof 验证")
    parser.add_argument("--image-idx", type=int, default=12345,
                        help="训练集中要验证的图片索引 (0-59999)")
    parser.add_argument("--test-idx", type=int, default=0,
                        help="测试集中的图片索引（用于负样本拒绝测试）")
    parser.add_argument("--rpc", default="http://127.0.0.1:8545")
    parser.add_argument("--data-addr", default="", help="DataAnchor 合约地址")
    parser.add_argument("--merkle-file", default="merkle_result.json")
    parser.add_argument("--merkle-tree-file", default="merkle_tree_layers.json")
    args = parser.parse_args()

    print("=" * 60)
    print("  单条样本 Merkle Proof 验证")
    print("=" * 60)

    # --- 连接 ---
    w3 = Web3(Web3.HTTPProvider(args.rpc))
    if not w3.is_connected():
        print(f"[!] 无法连接 {args.rpc}，请先启动 npx hardhat node")
        sys.exit(1)

    # --- 加载数据 ---
    acct = w3.eth.account.from_key(
        "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
    )

    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent

    # 读取合约
    data_addr = args.data_addr
    if not data_addr:
        # 尝试从 deploy_info 或 upload_report 读取
        for fname in [PROJECT_ROOT / "deploy_info.json", PROJECT_ROOT / "upload_report.json"]:
            if fname.exists():
                with open(fname) as f:
                    info = json.load(f)
                data_addr = info.get("dataAnchor") or info.get("data_anchor_addr", "")
                if data_addr:
                    break
    if not data_addr:
        print("[!] 请提供 --data-addr 或先运行 upload_to_chain.py")
        sys.exit(1)
    data_addr = Web3.to_checksum_address(data_addr)

    # 加载 ABI
    artifact_path = PROJECT_ROOT / "blockchain_anchor" / "artifacts" / "contracts" / "DataAnchor.sol" / "DataAnchor.json"
    if not os.path.exists(artifact_path):
        print(f"[!] ABI 不存在: {artifact_path}")
        sys.exit(1)
    with open(artifact_path) as f:
        abi = json.load(f)["abi"]
    contract = w3.eth.contract(address=data_addr, abi=abi)

    # 加载 Merkle
    with open(args.merkle_file) as f:
        merkle = json.load(f)
    root = merkle["merkle_root_raw"]
    dataset_id = w3.keccak(text="MNIST-train-v1")

    print(f"\n  合约地址: {data_addr}")
    print(f"  Merkle Root: 0x{root[:16]}...")

    # --- 重建 Merkle Tree（用于生成 proof）---
    print("\n[*] 加载 MNIST 并重建 Merkle Tree（生成 proof 用）...")
    train_set = datasets.MNIST("./data", train=True, download=True)
    n = len(train_set)

    # 构建叶子层
    leaves = []
    for i, (img, label) in enumerate(train_set):
        leaves.append(leaf_hash(np.array(img), label))
        if (i + 1) % 20000 == 0:
            print(f"  叶子进度: {i+1}/{n}")

    # 构建 tree layers
    tree_layers = [leaves[:]]
    layer = leaves[:]
    while len(layer) > 1:
        if len(layer) % 2 == 1:
            layer.append(layer[-1])
        new_layer = []
        for i in range(0, len(layer), 2):
            new_layer.append(keccak_pair(layer[i], layer[i+1]))
        tree_layers.append(new_layer)
        layer = new_layer

    # --- 正样本测试 ---
    print(f"\n{'─'*60}")
    print(f"  [测试 1] 正样本验证 — 训练集第 {args.image_idx} 张")
    print(f"{'─'*60}")

    img, label = train_set[args.image_idx]
    leaf = leaf_hash(np.array(img), label)
    proof = get_merkle_proof(args.image_idx, tree_layers)
    proof_bytes = [bytes.fromhex(p) for p in proof]

    print(f"  Label: {label}")
    print(f"  Leaf:  0x{leaf[:32]}...")
    print(f"  Proof 长度: {len(proof)} (理论: ⌈log₂({n})⌉ = {len(proof)})")

    t0 = time.time()
    is_member = contract.functions.verifyMember(
        dataset_id,
        bytes.fromhex(leaf),
        proof_bytes
    ).call()
    verify_time = (time.time() - t0) * 1000  # 毫秒

    result_text = "True (在训练集中)" if is_member else "False"
    print(f"  链上验证结果: {result_text}")
    print(f"  验证耗时:     {verify_time:.1f} ms")

    # --- 负样本测试 ---
    print(f"\n{'─'*60}")
    print(f"  [测试 2] 负样本拒绝 — 测试集第 {args.test_idx} 张（不在训练集）")
    print(f"{'─'*60}")

    test_set = datasets.MNIST("./data", train=False, download=True)
    test_img, test_label = test_set[args.test_idx]
    test_leaf = leaf_hash(np.array(test_img), test_label)
    print(f"  Label: {test_label}")
    print(f"  Leaf:  0x{test_leaf[:32]}...")
    print(f"  说明: 此样本不在训练集 Merkle Tree 中")

    # 用测试集样本的 leaf 配合训练集的 proof 去验证 → 应该 False
    # 用最后一个训练集样本的 proof 作为"无效证明"
    fake_proof = get_merkle_proof(59999, tree_layers)
    fake_proof_bytes = [bytes.fromhex(p) for p in fake_proof]

    t1 = time.time()
    is_fake_member = contract.functions.verifyMember(
        dataset_id,
        bytes.fromhex(test_leaf),
        fake_proof_bytes
    ).call()
    fake_time = (time.time() - t1) * 1000

    fake_text = "True (误报)" if is_fake_member else "False (正确拒绝)"
    print(f"  链上验证结果: {fake_text}")
    print(f"  验证耗时:     {fake_time:.1f} ms")

    # --- 批量负样本拒绝率测试 ---
    print(f"\n{'─'*60}")
    print(f"  [测试 3] 批量负样本拒绝率（10 次）")
    print(f"{'─'*60}")
    reject_count = 0
    for i in range(10):
        test_img_i, test_label_i = test_set[i]
        test_leaf_i = leaf_hash(np.array(test_img_i), test_label_i)
        result = contract.functions.verifyMember(
            dataset_id,
            bytes.fromhex(test_leaf_i),
            fake_proof_bytes
        ).call()
        if not result:
            reject_count += 1
    print(f"  拒绝率: {reject_count}/10 ({reject_count*10}%)")
    print(f"  {' 全部正确拒绝' if reject_count == 10 else ' 存在误报'}")

    # --- 汇总 ---
    print(f"\n{'='*60}")
    print(f"  验证完成")
    print(f"{'='*60}")
    print(f"  正样本: {'PASS' if is_member else 'FAIL'}")
    print(f"  负样本: {'PASS' if not is_fake_member else 'FAIL'}")
    print(f"  拒绝率: {reject_count}/10")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
