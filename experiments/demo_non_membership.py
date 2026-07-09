"""
差集证明（Sparse Merkle Tree 非成员证明）。

使用深度 256 的稀疏哈希树证明某条样本不在训练集中。
"""

import json
import os
import sys
import time
from pathlib import Path
from web3 import Web3

from torchvision import datasets
import numpy as np
import hashlib

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

RPC_URL = "http://127.0.0.1:8545"
PRIVATE_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
DEPTH = 256


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def leaf_hash(image: np.ndarray, label: int) -> str:
    img_bytes = image.astype(np.uint8).tobytes()
    return sha256_hex(img_bytes + bytes([label]))


def h(left: bytes, right: bytes) -> bytes:
    return Web3.keccak(left + right)


def compute_defaults() -> list[bytes]:
    """预计算各级空子树的根哈希：default[k] = 深度 k 全空子树的根"""
    d = [b'\x00' * 32]
    for _ in range(DEPTH):
        d.append(h(d[-1], d[-1]))
    return d


def build_smt(leaf_data: dict[int, bytes], defaults: list[bytes]) -> dict:
    """
    构建 SMT，返回 {(level, position): hash} 的节点缓存。
    leaf_data: {full_path: leaf_hash}
    合约：level k 使用 bit (255-k) 决定左右。
    """
    nodes: dict[tuple[int, int], bytes] = {}  # (level, position_at_level) → hash

    # level=0: leaf level, position = full 256-bit path
    current: dict[int, bytes] = dict(leaf_data)

    for level in range(DEPTH):
        bit_pos = 255 - level
        next_level: dict[int, bytes] = {}
        processed = set()

        for pos in sorted(current.keys()):
            if pos in processed:
                continue
            sibling_pos = pos ^ (1 << bit_pos)
            bit = (pos >> bit_pos) & 1

            left_node = current.get(pos, defaults[level])
            right_node = current.get(sibling_pos, defaults[level])

            if bit == 0:
                # pos is left child, sibling_pos is right
                parent_hash = h(
                    current.get(pos, defaults[level]),
                    current.get(sibling_pos, defaults[level])
                )
            else:
                # pos is right child, sibling_pos is left
                parent_hash = h(
                    current.get(sibling_pos, defaults[level]),
                    current.get(pos, defaults[level])
                )

            parent_pos = pos & ~(1 << bit_pos)  # clear the bit
            next_level[parent_pos] = parent_hash
            processed.add(pos)
            processed.add(sibling_pos)

        # cache nodes at this level
        for pos, node_hash in current.items():
            nodes[(level, pos)] = node_hash
        current = next_level

    # cache root
    nodes[(DEPTH, 0)] = current.get(0, defaults[DEPTH])
    return nodes


def get_non_membership_proof(
    path: int, nodes: dict, defaults: list[bytes]
) -> list[bytes]:
    """为非成员 key 生成 256 个兄弟哈希"""
    proof = []
    for level in range(DEPTH):
        bit_pos = 255 - level
        sibling_pos = path ^ (1 << bit_pos)
        # 存储时高位已清除，查寻需同掩码
        mask = (1 << (256 - level)) - 1
        sibling_hash = nodes.get((level, sibling_pos & mask), defaults[level])
        proof.append(sibling_hash)
    return proof


def verify_offchain(path: int, proof: list[bytes], root: bytes) -> bool:
    """链下验证（模拟合约逻辑）"""
    computed = b'\x00' * 32
    for level in range(DEPTH):
        bit = (path >> (255 - level)) & 1
        sibling = proof[level]
        if bit == 1:
            computed = h(sibling, computed)
        else:
            computed = h(computed, sibling)
    return computed == root


def main():
    print("=" * 60)
    print("  实验 2：差集证明（Non-Membership Proof）")
    print("=" * 60)

    w3 = Web3(Web3.HTTPProvider(RPC_URL))
    if not w3.is_connected():
        print("[!] Hardhat 本地链未启动")
        sys.exit(1)

    # 读合约
    deploy_info = PROJECT_ROOT / "deploy_info.json"
    with open(deploy_info) as f:
        info = json.load(f)
    data_addr = Web3.to_checksum_address(info["dataAnchor"])
    abi_path = PROJECT_ROOT / "blockchain_anchor" / "artifacts" / "contracts" / "DataAnchor.sol" / "DataAnchor.json"
    with open(abi_path) as f:
        abi = json.load(f)["abi"]
    contract = w3.eth.contract(address=data_addr, abi=abi)

    # 加载 MNIST
    print("\n[*] 加载 MNIST...")
    train_set = datasets.MNIST(str(PROJECT_ROOT / "data"), train=True, download=False)
    test_set = datasets.MNIST(str(PROJECT_ROOT / "data"), train=False, download=False)

    DEMO_SIZE = 500
    print(f"[*] 构建 Sparse Merkle Tree（{DEMO_SIZE} 条样本）...")
    t0 = time.time()

    defaults = compute_defaults()

    # 构建叶子：path = uint256(keccak256(keccak256(leaf_hash)))
    # 合约中 _key 传入 keccak256(leaf_hash)，合约内部再 keccak256(_key) 得到 path
    leaf_data: dict[int, bytes] = {}
    train_keys = []  # 存 _key（传给合约的值），不是 path
    for i in range(DEMO_SIZE):
        img, label = train_set[i]
        lh = leaf_hash(np.array(img), label)
        key = Web3.keccak(bytes.fromhex(lh))  # _key = keccak256(leaf_hash)
        path = int.from_bytes(Web3.keccak(key), 'big')  # path = keccak256(_key)
        train_keys.append(key)
        leaf_data[path] = key  # leaf 值 = key 本身

    nodes = build_smt(leaf_data, defaults)
    smt_root = nodes[(DEPTH, 0)]
    build_time = time.time() - t0
    print(f"  SMT Root: 0x{smt_root.hex()[:32]}...")
    print(f"  构建耗时: {build_time:.1f} 秒")

    # 取测试集第一条（不在 SMT 中）
    print("\n[*] 生成非成员证明...")
    test_img, test_label = test_set[0]
    test_lh = leaf_hash(np.array(test_img), test_label)
    test_key = Web3.keccak(bytes.fromhex(test_lh))  # _key
    test_path = int.from_bytes(Web3.keccak(test_key), 'big')  # path = keccak256(_key)
    is_in = test_path in leaf_data
    print(f"  测试样本在 SMT 中: {is_in}")

    proof = get_non_membership_proof(test_path, nodes, defaults)
    print(f"  证明长度: {len(proof)}")

    # 链下验证
    local_ok = verify_offchain(test_path, proof, smt_root)
    print(f"  链下验证: {'通过' if local_ok else '失败'}")

    # 链上验证
    print("\n[*] 链上调用 verifyNonMembership...")
    t1 = time.time()
    proof_tuple = tuple(proof[:DEPTH])
    onchain_ok = contract.functions.verifyNonMembership(
        smt_root, test_key, proof_tuple
    ).call()
    onchain_time = (time.time() - t1) * 1000
    print(f"  链上结果: {'通过' if onchain_ok else '失败'}")
    print(f"  耗时:     {onchain_time:.1f} ms")

    result = {
        "experiment": "non_membership_smt",
        "smt_size": DEMO_SIZE,
        "smt_root": "0x" + smt_root.hex(),
        "proof_length": len(proof),
        "build_time_sec": round(build_time, 1),
        "local_verification": local_ok,
        "onchain_verification": onchain_ok,
        "onchain_time_ms": round(onchain_time, 1),
    }
    out_path = OUTPUT_DIR / "non_membership_result.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[✓] 结果已保存: {out_path}")


if __name__ == "__main__":
    main()
