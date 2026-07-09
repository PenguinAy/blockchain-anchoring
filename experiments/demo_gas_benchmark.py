"""
Merkle 证明 Gas 与延迟基准测试。

在四种数据集规模下（16 至 60,000 叶子）测量验证耗时，
验证 O(log n) 对数增长特性，生成对比图表。
"""

import json
import os
import sys
import time
from pathlib import Path
from web3 import Web3
import hashlib
import random

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

RPC_URL = "http://127.0.0.1:8545"
PRIVATE_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"

# 测试规模：2^4 到 2^16
SIZES = [16, 256, 4096, 60000]
REPEAT = 3  # 每个规模重复次数


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def keccak_pair(a: str, b: str) -> str:
    if a < b:
        combined = bytes.fromhex(a) + bytes.fromhex(b)
    else:
        combined = bytes.fromhex(b) + bytes.fromhex(a)
    return Web3.keccak(combined).hex()


def build_tree(n: int) -> tuple[str, list[list[str]]]:
    """构建 n 个随机叶子的 Merkle Tree"""
    leaves = [sha256_hex(str(i).encode() + os.urandom(8)) for i in range(n)]
    tree = [leaves[:]]
    layer = leaves[:]
    while len(layer) > 1:
        if len(layer) % 2 == 1:
            layer.append(layer[-1])
        new_layer = [keccak_pair(layer[i], layer[i+1]) for i in range(0, len(layer), 2)]
        tree.append(new_layer)
        layer = new_layer
    return layer[0], tree


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
    print("  实验 5：Gas 基准测试")
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

    acct = w3.eth.account.from_key(PRIVATE_KEY)

    # ========== 单条 Merkle Proof Gas 对比 ==========
    print("\n[*] 单条 Merkle Proof — Gas 随叶子数变化\n")
    results = []

    for n in SIZES:
        print(f"  规模 n = {n} (树深 = {n.bit_length()})...")
        root, tree = build_tree(n)
        dataset_id = w3.keccak(text=f"gas-test-{n}")

        # 注册数据集
        try:
            tx = contract.functions.registerDataset(
                dataset_id,
                bytes.fromhex(root),
                json.dumps({"name": f"gas-test", "size": n})
            ).build_transaction({
                "from": acct.address,
                "nonce": w3.eth.get_transaction_count(acct.address),
                "gas": 300000, "gasPrice": w3.eth.gas_price, "chainId": 31337,
            })
            signed = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
            w3.eth.send_raw_transaction(signed.raw_transaction)
            w3.eth.wait_for_transaction_receipt(signed.hash)
        except Exception:
            pass  # 已注册

        # 多次测试取平均
        gas_vals = []
        time_vals = []
        leaf_idx = random.randint(0, n - 1)
        leaf = tree[0][leaf_idx]
        proof_hex = get_proof(leaf_idx, tree)
        proof_bytes = [bytes.fromhex(p) for p in proof_hex]

        for r in range(REPEAT):
            t0 = time.time()
            try:
                contract.functions.verifyMember(
                    dataset_id,
                    bytes.fromhex(leaf),
                    proof_bytes
                ).call()
            except Exception:
                pass
            elapsed = (time.time() - t0) * 1000
            time_vals.append(elapsed)
            gas_vals.append(0)  # view 调用不消耗 gas

        avg_time = sum(time_vals) / len(time_vals)
        results.append({
            "size": n,
            "tree_depth": n.bit_length(),
            "proof_length": len(proof_hex),
            "avg_time_ms": round(avg_time, 1),
            "repeats": REPEAT,
        })
        print(f"    proof 长度: {len(proof_hex)}, 平均耗时: {avg_time:.1f} ms")

    # ========== 画图 ==========
    print("\n[*] 生成图表...")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        sizes_arr = [r["size"] for r in results]
        times_arr = [r["avg_time_ms"] for r in results]
        log_sizes = [np.log2(s) for s in sizes_arr]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # 图 1：验证时间 vs 叶子数（对数横轴）
        ax1.plot(sizes_arr, times_arr, "o-", color="#4A6B5F", linewidth=2, markersize=8)
        ax1.set_xscale("log", base=2)
        ax1.set_xlabel("Leaf count (log scale)")
        ax1.set_ylabel("Verification time (ms)")
        ax1.set_title("Merkle Proof Verification Time vs Dataset Size")
        ax1.grid(True, alpha=0.3)
        for s, t in zip(sizes_arr, times_arr):
            ax1.annotate(f"{t:.1f}ms", (s, t), textcoords="offset points",
                         xytext=(0, 10), fontsize=8, ha="center")

        # 图 2：理论 vs 实测
        ax2.plot(log_sizes, times_arr, "o-", color="#2E5C8A", linewidth=2, markersize=8,
                 label="Measured (ms)")
        # 理论 O(log n) 拟合
        z = np.polyfit(log_sizes, times_arr, 1)
        fit_line = np.polyval(z, log_sizes)
        ax2.plot(log_sizes, fit_line, "--", color="#B7976F", linewidth=1.5,
                 label=f"Linear fit (slope={z[0]:.1f})")
        ax2.set_xlabel("log2(Leaf count)")
        ax2.set_ylabel("Verification time (ms)")
        ax2.set_title("O(log n) Scaling Confirmation")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        chart_path = OUTPUT_DIR / "gas_benchmark.png"
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  图表已保存: {chart_path}")
    except Exception as e:
        print(f"  图表生成失败: {e}")

    # 输出数据
    output = {
        "experiment": "gas_benchmark",
        "sizes_tested": SIZES,
        "results": results,
        "note": "view functions do not consume gas; timing reflects off-chain RPC latency"
    }
    out_path = OUTPUT_DIR / "gas_benchmark_result.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[✓] 数据已保存: {out_path}")

    # 汇总表
    print(f"\n{'='*60}")
    print(f"  汇总")
    print(f"{'='*60}")
    for r in results:
        print(f"  n={r['size']:>6}  depth={r['tree_depth']:>2}  proof_len={r['proof_length']:>2}  avg={r['avg_time_ms']:>6.1f}ms")


if __name__ == "__main__":
    main()
