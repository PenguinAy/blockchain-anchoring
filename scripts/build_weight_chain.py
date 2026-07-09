"""
Pipeline Stage C — 训练时序哈希链
===================================
从逐 epoch checkpoint 构建防篡改哈希链：

    C_0 = SHA-256(W_0)
    C_t = SHA-256( SHA-256(W_t) || C_{t-1} )   for t = 1..E

链尾 C_E 通过 WeightAnchor.chainTip 上链锚定。
任何中间状态的篡改都会导致链尾变化。

输入：  checkpoints/lenet_epoch*.pth（来自阶段 B）
产物：  weight_chain.json
"""

import hashlib
import json
import os
import time
from pathlib import Path


def sha256_hex(data: bytes) -> str:
    """SHA-256 → 64 位十六进制"""
    return hashlib.sha256(data).hexdigest()


def sha256_bytes(data: bytes) -> bytes:
    """SHA-256 → bytes (32 字节)"""
    return hashlib.sha256(data).digest()


def build_training_chain(checkpoint_paths: list[str]) -> dict:
    """
    构建训练时序哈希链

    参数:
        checkpoint_paths: 按 epoch 顺序排列的 .pth 文件路径列表
                          [W_0, W_1, ..., W_E]

    返回:
        {
            "chain_tail": "最终 C_E（hex）",
            "chain": [C_0, C_1, ..., C_E],
            "weight_hashes": [H_0, H_1, ..., H_E],   # 每个 epoch 权重的 SHA-256
            "epochs": E+1
        }
    """
    E = len(checkpoint_paths)
    weight_hashes = []
    chain = []

    for t, path in enumerate(checkpoint_paths):
        # 读取权重文件
        with open(path, "rb") as f:
            weight_bytes = f.read()

        # 计算 H(W_t) = SHA-256(W_t)
        h_w = sha256_hex(weight_bytes)
        weight_hashes.append(h_w)

        # 计算 C_t
        if t == 0:
            c_t = sha256_hex(weight_bytes)  # C_0 = SHA-256(W_0)
        else:
            # C_t = SHA-256( SHA-256(W_t) || C_{t-1} )
            h_w_bytes = sha256_bytes(weight_bytes)   # 32 bytes
            c_prev_bytes = bytes.fromhex(chain[-1])   # 32 bytes (上一轮 C_{t-1})
            c_t = sha256_hex(h_w_bytes + c_prev_bytes)

        chain.append(c_t)
        print(f"  t={t}: H(W_{t})={h_w[:16]}...  C_{t}={c_t[:16]}...  ({os.path.basename(path)})")

    return {
        "chain_tail": chain[-1],
        "chain": chain,
        "weight_hashes": weight_hashes,
        "epochs": E,
    }


# ========== 验证函数（用于上传后验证） ==========
def verify_epoch(checkpoint_path: str, prev_chain_tip: str, rest_checkpoints: list[str]) -> str:
    """
    验证单个 epoch checkpoint 是否在训练链中。
    从给定 checkpoint 出发，结合后续 checkpoint，计算出最终 C_E 与链上比对。

    参数:
        checkpoint_path: 要验证的 .pth 文件路径
        prev_chain_tip: C_{t-1}（该 epoch 之前的链尾）
        rest_checkpoints: 后续 epoch 的 .pth 文件列表

    返回:
        计算出的 C_E（与链上 chainTip 比对）
    """
    with open(checkpoint_path, "rb") as f:
        w_bytes = f.read()
    h_w = sha256_bytes(w_bytes)
    c_prev = bytes.fromhex(prev_chain_tip)
    c = sha256_hex(h_w + c_prev)

    for rest_path in rest_checkpoints:
        with open(rest_path, "rb") as f:
            w_rest = f.read()
        h_rest = sha256_bytes(w_rest)
        c = sha256_hex(h_rest + bytes.fromhex(c))

    return c


# ========== 主流程 ==========
def main():
    CKPT_DIR = "./checkpoints"

    print("=" * 60)
    print("  训练时序哈希链构建")
    print("  C_t = SHA-256( SHA-256(W_t) || C_{t-1} )")
    print("=" * 60)

    # --- 收集 checkpoint 文件 ---
    ckpt_files = sorted(
        [f for f in os.listdir(CKPT_DIR) if f.startswith("lenet_epoch") and f.endswith(".pth")],
        key=lambda x: int(x.replace("lenet_epoch", "").replace(".pth", ""))
    )

    if not ckpt_files:
        print(f"\n[!] 在 {os.path.abspath(CKPT_DIR)} 下没有找到 checkpoint 文件")
        print("    请先运行 train_lenet.py 生成 checkpoints")
        return

    ckpt_paths = [os.path.join(CKPT_DIR, f) for f in ckpt_files]
    print(f"\n[*] 找到 {len(ckpt_paths)} 个 checkpoint:")
    for p in ckpt_paths:
        size_kb = os.path.getsize(p) / 1024
        print(f"    {os.path.basename(p)} ({size_kb:.1f} KB)")

    # --- 构建链 ---
    print(f"\n[*] 构建时序哈希链...")
    t0 = time.time()
    result = build_training_chain(ckpt_paths)
    elapsed = time.time() - t0

    # --- 输出 ---
    result["build_time_sec"] = round(elapsed, 2)
    result["checkpoint_files"] = ckpt_files
    result["checkpoint_sizes_kb"] = [round(os.path.getsize(p)/1024, 1) for p in ckpt_paths]
    result["formula"] = "C_t = SHA-256(SHA-256(W_t) || C_{t-1})"

    out_path = "weight_chain.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # --- 验证：任意篡改一个字节就会变 ---
    print(f"\n[*] 防篡改验证：修改 Epoch 3 的一个字节...")
    test_path = ckpt_paths[2]  # epoch 3 (index 2)
    with open(test_path, "rb") as f:
        original_bytes = f.read()
    tampered_bytes = original_bytes[:1000] + bytes([255]) + original_bytes[1001:]
    tampered_path = test_path + ".tampered"
    with open(tampered_path, "wb") as f:
        f.write(tampered_bytes)

    # 用被篡改的 epoch 3 重新计算链
    tampered_ckpts = ckpt_paths[:]
    tampered_ckpts[2] = tampered_path
    tampered_result = build_training_chain(tampered_ckpts)
    os.remove(tampered_path)

    print(f"  原始链尾 C_5:  {result['chain_tail'][:32]}...")
    print(f"  篡改后链尾:    {tampered_result['chain_tail'][:32]}...")
    print(f"  原始链尾与篡改后链尾一致: {result['chain_tail'] == tampered_result['chain_tail']}")
    print(f"  篡改检测: {'通过' if result['chain_tail'] != tampered_result['chain_tail'] else '失败'}")

    print(f"\n{'='*60}")
    print(f"  构建完成")
    print(f"{'='*60}")
    print(f"  链尾 C_{result['epochs']}:  0x{result['chain_tail']}")
    for i, (h, c) in enumerate(zip(result["weight_hashes"], result["chain"])):
        print(f"  Epoch {i}: H(W)={h[:16]}...  C={c[:16]}...")
    print(f"  耗时:       {elapsed:.1f} 秒")
    print(f"  输出文件:   {os.path.abspath(out_path)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
