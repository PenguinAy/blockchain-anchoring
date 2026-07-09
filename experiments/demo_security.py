"""
安全对抗测试。

- 重放攻击：尝试两次注册相同 datasetId / modelId
- Sybil 攻击：使用非授权账户尝试注册
"""

import json
import os
import sys
from pathlib import Path
from web3 import Web3

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

RPC_URL = "http://127.0.0.1:8545"
DEPLOYER_PK = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
# Hardhat Account #1 — 模拟攻击者
ATTACKER_PK = "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d"


def main():
    print("=" * 60)
    print("  实验 4：安全对抗测试")
    print("=" * 60)

    w3 = Web3(Web3.HTTPProvider(RPC_URL))
    if not w3.is_connected():
        print("[!] Hardhat 本地链未启动")
        sys.exit(1)

    deployer = w3.eth.account.from_key(DEPLOYER_PK)
    attacker = w3.eth.account.from_key(ATTACKER_PK)

    # 读合约
    deploy_info = PROJECT_ROOT / "deploy_info.json"
    with open(deploy_info) as f:
        info = json.load(f)
    data_addr = Web3.to_checksum_address(info["dataAnchor"])
    weight_addr = Web3.to_checksum_address(info["weightAnchor"])

    abi_base = PROJECT_ROOT / "blockchain_anchor" / "artifacts" / "contracts"
    with open(abi_base / "DataAnchor.sol" / "DataAnchor.json") as f:
        data_abi = json.load(f)["abi"]
    with open(abi_base / "WeightAnchor.sol" / "WeightAnchor.json") as f:
        weight_abi = json.load(f)["abi"]

    data_contract = w3.eth.contract(address=data_addr, abi=data_abi)
    weight_contract = w3.eth.contract(address=weight_addr, abi=weight_abi)

    results = {}

    # ========== 测试 1：重放攻击 ==========
    print(f"\n{'─'*60}")
    print("  测试 1：重放攻击（重复注册同一 datasetId）")
    print(f"{'─'*60}")

    dataset_id = w3.keccak(text="security-replay-test-v1")
    merkle_root = w3.keccak(text="test-root-replay")

    # 第一次注册（用全新的 ID，避免与 pipeline 已注册的冲突）
    print("  第一次注册...")
    tx = data_contract.functions.registerDataset(
        dataset_id,
        merkle_root,
        '{"name":"replay-test"}'
    ).build_transaction({
        "from": deployer.address,
        "nonce": w3.eth.get_transaction_count(deployer.address),
        "gas": 300000, "gasPrice": w3.eth.gas_price, "chainId": 31337,
    })
    signed = w3.eth.account.sign_transaction(tx, DEPLOYER_PK)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    w3.eth.wait_for_transaction_receipt(tx_hash)
    print(f"  第一次注册: 成功 (TX: {tx_hash.hex()[:16]}...)")

    # 第二次注册（重放）
    print("  第二次注册（重放攻击）...")
    replay_blocked = False
    try:
        tx2 = data_contract.functions.registerDataset(
            dataset_id, merkle_root, '{"name":"replay-attack"}'
        ).build_transaction({
            "from": deployer.address,
            "nonce": w3.eth.get_transaction_count(deployer.address),
            "gas": 300000, "gasPrice": w3.eth.gas_price, "chainId": 31337,
        })
        signed2 = w3.eth.account.sign_transaction(tx2, DEPLOYER_PK)
        tx_hash2 = w3.eth.send_raw_transaction(signed2.raw_transaction)
        w3.eth.wait_for_transaction_receipt(tx_hash2)
    except Exception as e:
        if "already exists" in str(e).lower() or "revert" in str(e).lower():
            replay_blocked = True
            print(f"  第二次注册: 被合约拒绝（符合预期）")
        else:
            print(f"  异常: {e}")

    results["replay_attack_blocked"] = replay_blocked

    # ========== 测试 2：Sybil 攻击 ==========
    print(f"\n{'─'*60}")
    print("  测试 2：Sybil 攻击（非授权账户注册）")
    print(f"{'─'*60}")

    # Account #1 尝试注册一个"伪造"数据集
    fake_id = w3.keccak(text="fake-dataset-sybil")
    fake_root = w3.keccak(text="fake-data")

    print(f"  攻击者: {attacker.address}")
    print(f"  尝试注册伪造数据集...")
    sybil_succeeded = False
    try:
        tx3 = data_contract.functions.registerDataset(
            fake_id, fake_root, '{"name":"sybil-attack"}'
        ).build_transaction({
            "from": attacker.address,
            "nonce": w3.eth.get_transaction_count(attacker.address),
            "gas": 300000, "gasPrice": w3.eth.gas_price, "chainId": 31337,
        })
        signed3 = w3.eth.account.sign_transaction(tx3, ATTACKER_PK)
        tx_hash3 = w3.eth.send_raw_transaction(signed3.raw_transaction)
        receipt3 = w3.eth.wait_for_transaction_receipt(tx_hash3)
        sybil_succeeded = True

        # 验证伪造数据已上链
        record = data_contract.functions.datasets(fake_id).call()
        # record[2] = registrar (address)
        print(f"  注册成功!")
        print(f"  注册者:   {record[2]}")
        print(f"  Merkle Root: 0x{record[0].hex()[:16]}...")
    except Exception as e:
        print(f"  注册被拒绝: {str(e)[:80]}...")

    results["sybil_attack_possible"] = sybil_succeeded

    # ========== 测试 3：模型重放攻击 ==========
    print(f"\n{'─'*60}")
    print("  测试 3：模型重放攻击（重复注册同一 modelId）")
    print(f"{'─'*60}")

    model_id = w3.keccak(text="security-test-model")
    # 先注册
    tx4 = weight_contract.functions.registerModel(
        model_id, w3.keccak(text="w1"), dataset_id,
        w3.keccak(text="chain1"), b'\x00' * 32,
        '{}', "v1.0"
    ).build_transaction({
        "from": deployer.address,
        "nonce": w3.eth.get_transaction_count(deployer.address),
        "gas": 500000, "gasPrice": w3.eth.gas_price, "chainId": 31337,
    })
    signed4 = w3.eth.account.sign_transaction(tx4, DEPLOYER_PK)
    w3.eth.send_raw_transaction(signed4.raw_transaction)
    w3.eth.wait_for_transaction_receipt(signed4.hash)
    print(f"  第一次注册: 成功")

    model_replay_blocked = False
    try:
        tx5 = weight_contract.functions.registerModel(
            model_id, w3.keccak(text="w2"), dataset_id,
            w3.keccak(text="chain2"), b'\x00' * 32,
            '{}', "v2.0"
        ).build_transaction({
            "from": deployer.address,
            "nonce": w3.eth.get_transaction_count(deployer.address),
            "gas": 500000, "gasPrice": w3.eth.gas_price, "chainId": 31337,
        })
        signed5 = w3.eth.account.sign_transaction(tx5, DEPLOYER_PK)
        w3.eth.send_raw_transaction(signed5.raw_transaction)
        w3.eth.wait_for_transaction_receipt(signed5.hash)
    except Exception:
        model_replay_blocked = True
        print(f"  第二次注册: 被合约拒绝（符合预期）")

    results["model_replay_blocked"] = model_replay_blocked

    # 汇总
    print(f"\n{'='*60}")
    print(f"  安全测试汇总")
    print(f"{'='*60}")
    print(f"  重放攻击（数据集）: {'已阻止' if results['replay_attack_blocked'] else '未阻止'}")
    print(f"  Sybil 攻击:       {'可能（无权限控制）' if results['sybil_attack_possible'] else '已阻止'}")
    print(f"  重放攻击（模型）:   {'已阻止' if results['model_replay_blocked'] else '未阻止'}")

    out_path = OUTPUT_DIR / "security_test_result.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[✓] 结果已保存: {out_path}")


if __name__ == "__main__":
    main()
