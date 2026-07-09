"""
模型版本链 DAG 演示。

在链上注册三代有父子关系的模型，验证正反向血缘查询功能。
"""

import json
import os
import sys
import hashlib
from pathlib import Path
from web3 import Web3

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

RPC_URL = "http://127.0.0.1:8545"
PRIVATE_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"

# 版本链定义
LINEAGE = [
    {"name": "LeNet-v1",       "version": "v1.0", "parent": None,  "epochs": 5,  "lr": 1e-3},
    {"name": "LeNet-v2-finetuned", "version": "v2.0", "parent": "LeNet-v1", "epochs": 3, "lr": 5e-4},
    {"name": "LeNet-v2.1-pruned",  "version": "v2.1", "parent": "LeNet-v2-finetuned", "epochs": 2, "lr": 5e-4},
]


def sha256_file(path: str) -> bytes:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            sha.update(chunk)
    return sha.digest()


def main():
    print("=" * 60)
    print("  实验 3：版本链 DAG 演示")
    print("=" * 60)

    w3 = Web3(Web3.HTTPProvider(RPC_URL))
    if not w3.is_connected():
        print("[!] Hardhat 本地链未启动")
        sys.exit(1)

    acct = w3.eth.account.from_key(PRIVATE_KEY)
    print(f"  部署账户: {acct.address}")

    # 读合约
    deploy_info = PROJECT_ROOT / "deploy_info.json"
    with open(deploy_info) as f:
        info = json.load(f)
    weight_addr = Web3.to_checksum_address(info["weightAnchor"])

    abi_path = PROJECT_ROOT / "blockchain_anchor" / "artifacts" / "contracts" / "WeightAnchor.sol" / "WeightAnchor.json"
    with open(abi_path) as f:
        abi = json.load(f)["abi"]
    contract = w3.eth.contract(address=weight_addr, abi=abi)

    # 读已有数据
    merkle = json.load(open(PROJECT_ROOT / "merkle_result.json"))
    weight_chain = json.load(open(PROJECT_ROOT / "weight_chain.json"))
    dataset_id = w3.keccak(text="MNIST-train-v1")

    # 注册 3 个模型
    ckpt_base = PROJECT_ROOT / "checkpoints" / "lenet_epoch5.pth"
    if not ckpt_base.exists():
        print("[!] checkpoint not found; run train_lenet.py first")
        sys.exit(1)

    model_ids = {}
    for i, entry in enumerate(LINEAGE):
        name = entry["name"]
        parent_name = entry["parent"]

        # 用同一个权重文件（演示版本链结构，权重内容不影响 DAG 逻辑）
        weight_hash = sha256_file(str(ckpt_base))
        model_id = w3.keccak(text=name)
        model_ids[name] = model_id
        parent_id = model_ids[parent_name] if parent_name else b'\x00' * 32
        chain_tip = weight_chain["chain_tail"]

        print(f"\n  注册 {name}...")
        print(f"    modelId:      0x{model_id.hex()[:16]}...")
        print(f"    parentModelId: {'0 (根模型)' if parent_name is None else '0x' + parent_id.hex()[:16] + '...'}")
        print(f"    weightHash:   0x{weight_hash.hex()[:16]}...")
        print(f"    chainTip:     0x{bytes.fromhex(chain_tip).hex()[:16]}...")

        metadata = json.dumps({
            "arch": "LeNet-5", "epochs": entry["epochs"], "lr": entry["lr"],
        })

        try:
            tx = contract.functions.registerModel(
                model_id,
                weight_hash,
                dataset_id,
                bytes.fromhex(chain_tip),
                parent_id,
                metadata,
                entry["version"]
            ).build_transaction({
                "from": acct.address,
                "nonce": w3.eth.get_transaction_count(acct.address),
                "gas": 500000, "gasPrice": w3.eth.gas_price, "chainId": 31337,
            })
            signed = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
            tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
            print(f"    TX Hash:      {tx_hash.hex()}")
            print(f"    Gas Used:     {receipt.gasUsed}")
        except Exception as e:
            if "already exists" in str(e).lower():
                print(f"    已注册，跳过")
            else:
                raise

    # 验证血缘查询
    print(f"\n{'='*60}")
    print(f"  血缘查询验证")
    print(f"{'='*60}")

    # getLineage: 从叶子回溯到根
    leaf_name = LINEAGE[-1]["name"]
    leaf_id = model_ids[leaf_name]
    lineage = contract.functions.getLineage(leaf_id).call()
    print(f"\n  getLineage({leaf_name}):")
    print(f"    祖先数量: {len(lineage)}")
    for j, ancestor in enumerate(lineage):
        print(f"    [{j}] 0x{ancestor.hex()}")

    # verifyLineage
    root_name = LINEAGE[0]["name"]
    root_id = model_ids[root_name]
    is_child = contract.functions.verifyLineage(leaf_id, root_id).call()
    print(f"\n  verifyLineage({leaf_name}, {root_name}): {is_child}")

    # getChildren
    mid_name = LINEAGE[1]["name"]
    mid_id = model_ids[mid_name]
    children = contract.functions.getChildren(mid_id).call()
    print(f"\n  getChildren({mid_name}):")
    print(f"    子模型数量: {len(children)}")
    for child in children:
        print(f"    0x{child.hex()}")

    # 汇总输出
    print(f"\n{'='*60}")
    print(f"  版本链结构:")
    print(f"  {LINEAGE[0]['name']}")
    print(f"    └── {LINEAGE[1]['name']}")
    print(f"          └── {LINEAGE[2]['name']}")

    result = {
        "experiment": "model_lineage_chain",
        "models": [
            {"name": m["name"], "version": m["version"],
             "modelId": "0x" + model_ids[m["name"]].hex(),
             "parentModelId": "0x" + (model_ids[m["parent"]].hex() if m["parent"] else "0"*64)}
            for m in LINEAGE
        ],
        "lineage_length": len(lineage),
        "verifyLineage": is_child,
        "children_count": len(children),
    }
    out_path = OUTPUT_DIR / "lineage_chain_result.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[✓] 结果已保存: {out_path}")


if __name__ == "__main__":
    main()
