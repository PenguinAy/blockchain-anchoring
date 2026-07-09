"""
Pipeline Stage E — 反向溯源（模型 → 数据集 + 版本链）
========================================================
给定模型 ID，从 WeightAnchor 合约查询其训练数据集、父模型链和
子模型列表。
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
    data_addr = ""
    for fname in [PROJECT_ROOT / "deploy_info.json", PROJECT_ROOT / "upload_report.json"]:
        if fname.exists():
            with open(fname) as f:
                info = json.load(f)
            weight_addr = info.get("weightAnchor") or info.get("weight_anchor_addr", "")
            data_addr = info.get("dataAnchor") or info.get("data_anchor_addr", "")
            if weight_addr and data_addr:
                break
    if not weight_addr:
        print("[!] 请先运行 upload_to_chain.py")
        sys.exit(1)
    weight_addr = Web3.to_checksum_address(weight_addr)
    data_addr = Web3.to_checksum_address(data_addr)

    # --- 加载 ABI ---
    base = PROJECT_ROOT / "blockchain_anchor" / "artifacts" / "contracts"
    with open(base / "WeightAnchor.sol" / "WeightAnchor.json") as f:
        weight_abi = json.load(f)["abi"]
    with open(base / "DataAnchor.sol" / "DataAnchor.json") as f:
        data_abi = json.load(f)["abi"]

    weight_contract = w3.eth.contract(address=weight_addr, abi=weight_abi)
    data_contract = w3.eth.contract(address=data_addr, abi=data_abi)

    # --- 查询 ---
    model_id = w3.keccak(text="LeNet5-v1")
    print(f"模型 ID: 0x{model_id.hex()}")

    model = weight_contract.functions.getModel(model_id).call()
    dataset_id = model[1]  # datasetId 字段

    print(f"\n{'='*60}")
    print(f"  模型详情")
    print(f"{'='*60}")
    print(f"  权重哈希:    0x{model[0].hex()}")
    print(f"  数据集 ID:   0x{dataset_id.hex()}")
    print(f"  训练链尾:    0x{model[2].hex()[:32]}...")
    ZERO32 = b'\x00' * 32
    parent_str = "根模型 (无父)" if model[3] == ZERO32 else "0x" + model[3].hex()
    print(f"  父模型:      {parent_str}")
    print(f"  注册者:      {model[4]}")
    print(f"  注册时间:    {model[5]}")
    print(f"  版本:        {model[6]}")

    # 反向查数据集
    print(f"\n{'─'*60}")
    print(f"  反向溯源：模型 → 训练数据")
    print(f"{'─'*60}")
    dataset = data_contract.functions.getDataset(dataset_id).call()
    print(f"  Merkle Root: 0x{dataset[0].hex()}")
    print(f"  元数据:      {dataset[2]}")
    print(f"  注册者:      {dataset[3]}")
    print(f"  注册时间:    {dataset[4]}")

    # 版本链血缘
    print(f"\n{'─'*60}")
    print(f"  版本链血缘")
    print(f"{'─'*60}")
    lineage = weight_contract.functions.getLineage(model_id).call()
    print(f"  祖先模型数:  {len(lineage)}")
    for i, ancestor in enumerate(lineage):
        print(f"    祖先 {i+1}: 0x{ancestor.hex()}")

    children = weight_contract.functions.getChildren(model_id).call()
    print(f"  子模型数:    {len(children)}")
    for i, child in enumerate(children):
        print(f"    子 {i+1}: 0x{child.hex()}")


if __name__ == "__main__":
    main()
