"""
Pipeline Stage D — 链上锚定
=============================
将数据集 Merkle 根（DataAnchor）和模型权重哈希 + 训练链尾
（WeightAnchor）注册到 Hardhat 本地链。

依赖：已编译部署的合约、merkle_result.json、weight_chain.json、
      deploy_info.json
产物：upload_report.json
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path
from web3 import Web3


# ========== 默认配置 ==========
RPC_URL = "http://127.0.0.1:8545"
CHAIN_ID = 31337

# Hardhat 本地链 Account #0 私钥（固定值，预存 10000 ETH，仅本地测试用）
HARDHAT_PRIVATE_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"

# 合约地址默认值（部署后填入）
DATA_ANCHOR_ADDR = ""
WEIGHT_ANCHOR_ADDR = ""

# 关键路径（以脚本自身位置为基准，不受 cwd 影响）
SCRIPT_DIR = Path(__file__).resolve().parent          # scripts/
PROJECT_ROOT = SCRIPT_DIR.parent                       # 项目根
ARTIFACTS_DIR = PROJECT_ROOT / "blockchain_anchor" / "artifacts"


def load_abi(contract_name: str) -> list:
    """从 Hardhat 编译产物中加载 ABI"""
    abi_path = ARTIFACTS_DIR / "contracts" / f"{contract_name}.sol" / f"{contract_name}.json"
    if not abi_path.exists():
        print(f"[!] ABI 文件不存在: {abi_path}")
        print(f"    请先运行: cd blockchain_anchor && npx hardhat compile")
        sys.exit(1)
    with open(abi_path, "r") as f:
        artifact = json.load(f)
    return artifact["abi"]


def load_saved_contracts() -> dict | None:
    """尝试从项目根或 scripts/ 读取 deploy_info.json"""
    for p in [PROJECT_ROOT / "deploy_info.json",
              SCRIPT_DIR / "deploy_info.json"]:
        if p.exists():
            with open(p, "r") as f:
                return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser(description="区块链锚定 — 数据与权重上链")
    parser.add_argument("--rpc", default=RPC_URL, help=f"RPC URL (默认: {RPC_URL})")
    parser.add_argument("--data-addr", default=DATA_ANCHOR_ADDR, help="DataAnchor 合约地址")
    parser.add_argument("--weight-addr", default=WEIGHT_ANCHOR_ADDR, help="WeightAnchor 合约地址")
    parser.add_argument("--private-key", default=HARDHAT_PRIVATE_KEY, help="部署账户私钥")
    parser.add_argument("--dataset-id", default="MNIST-train-v1", help="数据集标识符")
    parser.add_argument("--model-id", default="LeNet5-v1", help="模型标识符")
    parser.add_argument("--merkle-file", default="merkle_result.json", help="Merkle 结果文件")
    parser.add_argument("--chain-file", default="weight_chain.json", help="权重链结果文件")
    args = parser.parse_args()

    # --- 连接本地链 ---
    print("=" * 60)
    print("  区块链锚定 — 上链")
    print("=" * 60)
    print(f"\n[*] 连接本地链: {args.rpc}")
    w3 = Web3(Web3.HTTPProvider(args.rpc))
    if not w3.is_connected():
        print(f"[!] 无法连接 {args.rpc}")
        print("    请先启动本地链: cd blockchain_anchor && npx hardhat node")
        sys.exit(1)
    print(f"  链 ID: {w3.eth.chain_id}")
    print(f"  当前区块: {w3.eth.block_number}")

    # --- 账户 ---
    acct = w3.eth.account.from_key(args.private_key)
    balance = w3.eth.get_balance(acct.address)
    print(f"  账户: {acct.address}")
    print(f"  余额: {w3.from_wei(balance, 'ether')} ETH")

    # --- 合约地址 ---
    data_addr = args.data_addr
    weight_addr = args.weight_addr

    # 尝试从 deploy_info.json 读取
    saved = load_saved_contracts()
    if saved:
        if not data_addr:
            data_addr = saved.get("dataAnchor", "")
        if not weight_addr:
            weight_addr = saved.get("weightAnchor", "")
        if saved:
            print(f"\n[*] 从 deploy_info.json 读取合约地址")

    if not data_addr or not weight_addr:
        print("\n[!] 缺少合约地址，请通过以下方式之一提供：")
        print("    1. --data-addr 和 --weight-addr 命令行参数")
        print("    2. 运行部署脚本后，将输出保存为 deploy_info.json")
        print("    3. 直接修改本脚本顶部的 DATA_ANCHOR_ADDR / WEIGHT_ANCHOR_ADDR")
        sys.exit(1)

    # viem 输出小写地址，web3.py 要求 checksum 格式
    data_addr = Web3.to_checksum_address(data_addr)
    weight_addr = Web3.to_checksum_address(weight_addr)
    print(f"  DataAnchor:   {data_addr}")
    print(f"  WeightAnchor: {weight_addr}")

    # --- 加载 ABI ---
    print(f"\n[*] 加载合约 ABI...")
    data_abi = load_abi("DataAnchor")
    weight_abi = load_abi("WeightAnchor")
    data_contract = w3.eth.contract(address=data_addr, abi=data_abi)
    weight_contract = w3.eth.contract(address=weight_addr, abi=weight_abi)
    print("  ABI 加载成功")

    # --- 读取本地数据 ---
    print(f"\n[*] 读取本地计算结果...")
    merkle_path = PROJECT_ROOT / args.merkle_file
    chain_path = PROJECT_ROOT / args.chain_file
    if not merkle_path.exists():
        print(f"[!] {merkle_path} 不存在，请先运行 build_merkle.py")
        sys.exit(1)
    if not chain_path.exists():
        print(f"[!] {chain_path} 不存在，请先运行 build_weight_chain.py")
        sys.exit(1)

    with open(merkle_path, "r") as f:
        merkle = json.load(f)
    with open(chain_path, "r") as f:
        weight_chain = json.load(f)

    merkle_root_hex = merkle["merkle_root_raw"]  # 不带 0x 的 hex
    chain_tail_hex = weight_chain["chain_tail"]
    weight_hash_hex = weight_chain["weight_hashes"][-1]  # 最终 epoch 的 H(W_E)
    print(f"  Merkle Root (前16位):  0x{merkle_root_hex[:16]}...")
    print(f"  最终权重哈希 (前16位):  {weight_hash_hex[:16]}...")
    print(f"  训练链尾 C_5 (前16位):  {chain_tail_hex[:16]}...")

    # --- 生成 ID ---
    dataset_id = w3.keccak(text=args.dataset_id)
    model_id = w3.keccak(text=args.model_id)
    print(f"\n  datasetId: 0x{dataset_id.hex()}")
    print(f"  modelId:   0x{model_id.hex()}")

    # ===================== 1. 注册数据集 =====================
    print(f"\n{'─'*60}")
    print("  [1/3] 注册数据集到 DataAnchor")
    print(f"{'─'*60}")

    merkle_root_bytes = bytes.fromhex(merkle_root_hex)
    dataset_meta = json.dumps({
        "name": "MNIST", "size": merkle["sample_count"],
        "tree_depth": merkle["tree_depth"],
    })
    data_tx_hash = "SKIPPED"
    data_gas = 0
    try:
        tx = data_contract.functions.registerDataset(
            dataset_id, merkle_root_bytes, dataset_meta
        ).build_transaction({
            "from": acct.address,
            "nonce": w3.eth.get_transaction_count(acct.address),
            "gas": 300000, "gasPrice": w3.eth.gas_price, "chainId": CHAIN_ID,
        })
        signed = w3.eth.account.sign_transaction(tx, args.private_key)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        data_tx_hash = tx_hash.hex()
        data_gas = receipt.gasUsed
        print(f"   数据集已注册")
        print(f"    TX Hash:    {data_tx_hash}")
        print(f"    Gas Used:   {data_gas}")
        print(f"    Block:      {receipt.blockNumber}")
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"   数据集已注册，跳过")
        else:
            raise

    # ===================== 2. 注册模型权重 =====================
    print(f"\n{'─'*60}")
    print("  [2/3] 注册模型权重到 WeightAnchor")
    print(f"{'─'*60}")

    weight_hash_bytes = bytes.fromhex(weight_hash_hex)
    chain_tip_bytes = bytes.fromhex(chain_tail_hex)
    parent_id = b'\x00' * 32  # 根模型，无父
    model_meta = json.dumps({
        "arch": "LeNet-5", "epochs": weight_chain["epochs"],
        "optimizer": "Adam", "lr": 1e-3, "batch_size": 128,
    })
    weight_tx_hash = "SKIPPED"
    weight_gas = 0
    try:
        tx = weight_contract.functions.registerModel(
            model_id, weight_hash_bytes, dataset_id,
            chain_tip_bytes, parent_id, model_meta, "v1.0"
        ).build_transaction({
            "from": acct.address,
            "nonce": w3.eth.get_transaction_count(acct.address),
            "gas": 500000, "gasPrice": w3.eth.gas_price, "chainId": CHAIN_ID,
        })
        signed = w3.eth.account.sign_transaction(tx, args.private_key)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        weight_tx_hash = tx_hash.hex()
        weight_gas = receipt.gasUsed
        print(f"    模型已注册")
        print(f"    TX Hash:    {weight_tx_hash}")
        print(f"    Gas Used:   {weight_gas}")
        print(f"    Block:      {receipt.blockNumber}")
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"  模型已注册，跳过")
        else:
            raise

    # ===================== 3. 验证 =====================
    print(f"\n{'─'*60}")
    print("  [3/3] 链上验证")
    print(f"{'─'*60}")

    # 3.1 数据集完整性
    is_valid_data = data_contract.functions.verifyDataset(
        dataset_id,
        bytes.fromhex(merkle_root_hex)
    ).call()
    print(f"  数据完整性验证:  {' True' if is_valid_data else ' False'}")

    # 3.2 权重完整性
    is_valid_weight = weight_contract.functions.verifyModel(
        model_id,
        bytes.fromhex(weight_hash_hex)
    ).call()
    print(f"  权重完整性验证:  {' True' if is_valid_weight else ' False'}")

    # 3.3 训练链验证
    is_valid_chain = weight_contract.functions.verifyTrainingChain(
        model_id,
        bytes.fromhex(chain_tail_hex)
    ).call()
    print(f"  训练链尾验证:    {' True' if is_valid_chain else ' False'}")

    # 3.4 篡改测试
    fake_root = b'\x00' * 32
    is_fake = data_contract.functions.verifyDataset(dataset_id, fake_root).call()
    print(f"  篡改检测 (全零): {' False (正确拒绝)' if not is_fake else ' 未检测到'}")

    # 3.5 血缘查询
    lineage = weight_contract.functions.getLineage(model_id).call()
    print(f"  模型血缘链:      {len(lineage)} 个祖先 (根模型应为 0)")

    # ===================== 汇总 =====================
    print(f"\n{'='*60}")
    print(f"  上链完成")
    print(f"{'='*60}")
    print(f"  数据集 TX Hash:  {data_tx_hash}")
    print(f"  数据集注册 Gas:  {data_gas}")
    print(f"  模型 TX Hash:    {weight_tx_hash}")
    print(f"  模型注册 Gas:    {weight_gas}")
    print(f"  数据验证:        {is_valid_data}")
    print(f"  权重验证:        {is_valid_weight}")
    print(f"  训练链验证:      {is_valid_chain}")
    print(f"  篡改检测:        {not is_fake}")
    print(f"{'='*60}")

    # --- 输出 JSON（供论文填数） ---
    report = {
        "data_anchor_addr": data_addr,
        "weight_anchor_addr": weight_addr,
        "dataset_id": "0x" + dataset_id.hex(),
        "model_id": "0x" + model_id.hex(),
        "data_tx_hash": data_tx_hash,
        "data_gas_used": data_gas,
        "weight_tx_hash": weight_tx_hash,
        "weight_gas_used": weight_gas,
        "data_verification": is_valid_data,
        "weight_verification": is_valid_weight,
        "chain_verification": is_valid_chain,
        "tamper_detection": not is_fake,
        "network": f"localhost:{CHAIN_ID}",
    }
    report_path = PROJECT_ROOT / "upload_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n 报告已保存: {report_path}")


if __name__ == "__main__":
    main()
