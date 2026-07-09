"""
Pipeline Stage E — 正向溯源（数据集 → 模型）
===============================================
给定数据集 ID，从 WeightAnchor 合约查询所有使用该数据集训练的模型。
"""

import json
import os
import sys
from pathlib import Path
from web3 import Web3


def main():
    RPC_URL = "http://127.0.0.1:8545"
    w3 = Web3(Web3.HTTPProvider(RPC_URL))
    if not w3.is_connected():
        print("[!] 无法连接本地链，请先启动 npx hardhat node")
        sys.exit(1)

    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent

    # --- 读合约地址 ---
    weight_addr = ""
    for fname in [PROJECT_ROOT / "deploy_info.json", PROJECT_ROOT / "upload_report.json"]:
        if fname.exists():
            with open(fname) as f:
                info = json.load(f)
            weight_addr = info.get("weightAnchor") or info.get("weight_anchor_addr", "")
            if weight_addr:
                break
    if not weight_addr:
        print("[!] 请先运行 upload_to_chain.py")
        sys.exit(1)
    weight_addr = Web3.to_checksum_address(weight_addr)

    # --- 加载 ABI ---
    artifact_path = PROJECT_ROOT / "blockchain_anchor" / "artifacts" / "contracts" / "WeightAnchor.sol" / "WeightAnchor.json"
    with open(artifact_path) as f:
        abi = json.load(f)["abi"]
    contract = w3.eth.contract(address=weight_addr, abi=abi)

    # --- 查询 ---
    dataset_id = w3.keccak(text="MNIST-train-v1")
    print(f"数据集 ID: 0x{dataset_id.hex()}")
    print(f"合约地址:  {weight_addr}")

    models = contract.functions.getModelsByDataset(dataset_id).call()
    print(f"\n该数据集训练的模型数: {len(models)}")
    ZERO32 = b'\x00' * 32
    for i, mid in enumerate(models):
        model = contract.functions.getModel(mid).call()
        is_root = model[3] == ZERO32
        parent_str = "根模型" if is_root else "0x" + model[3].hex()
        print(f"\n  模型 {i+1}: 0x{mid.hex()}")
        print(f"    权重哈希:  0x{model[0].hex()}")
        print(f"    训练链尾:  0x{model[2].hex()[:16]}...")
        print(f"    父模型:    {parent_str}")
        print(f"    版本:      {model[6]}")
        print(f"    注册时间:  {model[5]}")


if __name__ == "__main__":
    main()
