"""
批量 Merkle 成员证明（Multi-Proof）。

一次链上调用验证 k 条样本是否全部属于训练集，对比批量验证与
逐条验证的效率差异。
"""

import json
import os
import sys
import time
import random
from pathlib import Path
from web3 import Web3

import torch
from torchvision import datasets
import numpy as np
import hashlib

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

RPC_URL = "http://127.0.0.1:8545"
PRIVATE_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"

# 子集大小
SUBSET_SIZE = 20


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def leaf_hash(image: np.ndarray, label: int) -> str:
    img_bytes = image.astype(np.uint8).tobytes()
    return sha256_hex(img_bytes + bytes([label]))


def keccak_pair(a: str, b: str) -> str:
    if a < b:
        combined = bytes.fromhex(a) + bytes.fromhex(b)
    else:
        combined = bytes.fromhex(b) + bytes.fromhex(a)
    return Web3.keccak(combined).hex()


def build_merkle_tree(leaves: list) -> list[list[str]]:
    tree = [leaves[:]]
    layer = leaves[:]
    while len(layer) > 1:
        if len(layer) % 2 == 1:
            layer.append(layer[-1])
        new_layer = []
        for i in range(0, len(layer), 2):
            new_layer.append(keccak_pair(layer[i], layer[i + 1]))
        tree.append(new_layer)
        layer = new_layer
    return tree


def get_proof(idx: int, tree: list[list[str]]) -> list[str]:
    proof = []
    i = idx
    for layer in tree[:-1]:
        sibling = i + 1 if i % 2 == 0 else i - 1
        if sibling < len(layer):
            proof.append(layer[sibling])
        i //= 2
    return proof


def main():
    print("=" * 60)
    print("  实验 1：子集证明（Multi-Proof）")
    print("=" * 60)

    # 连接链
    w3 = Web3(Web3.HTTPProvider(RPC_URL))
    if not w3.is_connected():
        print("[!] Hardhat 本地链未启动")
        sys.exit(1)

    # 读合约
    deploy_info = PROJECT_ROOT / "deploy_info.json"
    if not deploy_info.exists():
        print("[!] deploy_info.json not found; run the pipeline first")
        sys.exit(1)
    with open(deploy_info) as f:
        info = json.load(f)
    data_addr = Web3.to_checksum_address(info["dataAnchor"])

    abi_path = PROJECT_ROOT / "blockchain_anchor" / "artifacts" / "contracts" / "DataAnchor.sol" / "DataAnchor.json"
    with open(abi_path) as f:
        abi = json.load(f)["abi"]
    contract = w3.eth.contract(address=data_addr, abi=abi)

    # 读 Merkle Root
    merkle = json.load(open(PROJECT_ROOT / "merkle_result.json"))
    root = merkle["merkle_root_raw"]
    dataset_id = w3.keccak(text="MNIST-train-v1")

    print(f"  Merkle Root: 0x{root[:16]}...")
    print(f"  子集大小:    k = {SUBSET_SIZE}")

    # 加载 MNIST + 构建树
    print("\n[*] 构建 Merkle Tree...")
    train_set = datasets.MNIST(str(PROJECT_ROOT / "data"), train=True, download=False)
    leaves = [leaf_hash(np.array(img), label) for img, label in train_set]
    tree = build_merkle_tree(leaves)

    # 随机选 k 个索引
    indices = sorted(random.sample(range(len(train_set)), SUBSET_SIZE))
    sample_leaves = [leaves[i] for i in indices]
    proofs = [get_proof(i, tree) for i in indices]
    proof_bytes_list = [[bytes.fromhex(p) for p in pf] for pf in proofs]

    print(f"  样本索引: {indices[:5]}... (共 {SUBSET_SIZE} 条)")
    print(f"  每条 proof 长度: {len(proofs[0])}")

    # 链上验证
    print("\n[*] 链上调用 verifySubsetAll...")
    t0 = time.time()
    acct = w3.eth.account.from_key(PRIVATE_KEY)

    # 先试 view 调用
    all_valid = contract.functions.verifySubsetAll(
        dataset_id,
        [bytes.fromhex(lh) for lh in sample_leaves],
        proof_bytes_list
    ).call()
    elapsed = (time.time() - t0) * 1000

    print(f"  结果:    {all_valid}")
    print(f"  耗时:    {elapsed:.1f} ms")

    # 对比：k 次单独验证
    print(f"\n[*] 对比：{SUBSET_SIZE} 次单独 verifyMember...")
    t1 = time.time()
    single_results = []
    for lh, pf in zip(sample_leaves, proof_bytes_list):
        single_results.append(
            contract.functions.verifyMember(
                dataset_id,
                bytes.fromhex(lh),
                pf
            ).call()
        )
    single_elapsed = (time.time() - t1) * 1000
    print(f"  全部通过: {all(single_results)}")
    print(f"  总耗时:   {single_elapsed:.1f} ms")
    print(f"  均耗时:   {single_elapsed / SUBSET_SIZE:.1f} ms/条")

    # 输出
    result = {
        "experiment": "subset_multi_proof",
        "subset_size": SUBSET_SIZE,
        "proof_length": len(proofs[0]),
        "verifySubsetAll_result": all_valid,
        "verifySubsetAll_time_ms": round(elapsed, 1),
        "single_verify_total_ms": round(single_elapsed, 1),
        "single_verify_avg_ms": round(single_elapsed / SUBSET_SIZE, 1),
        "sample_indices": indices,
    }
    out_path = OUTPUT_DIR / "subset_proof_result.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[✓] 结果已保存: {out_path}")


if __name__ == "__main__":
    main()
