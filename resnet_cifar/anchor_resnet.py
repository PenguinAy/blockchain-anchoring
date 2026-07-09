"""
ResNet18 + CIFAR-10 端到端链上锚定
==================================
将扩展实验的完整流程集成在一个脚本中：

  1. 从 CIFAR-10 训练集构建 Merkle 树（50 000 叶子）
  2. 读取训练时序哈希链（weight_chain_resnet.json）
  3. 连接 Hardhat 本地链，加载 DataAnchor / WeightAnchor 合约
  4. 注册数据集 + 模型（根模型）
  5. 链上验证（数据完整性 / 权重 / 训练链 / 篡改检测）
  6. 注册三代版本链 DAG（ResNet-v1 → v2-finetuned → v3-pruned）
  7. 血缘查询验证（getLineage / verifyLineage / getChildren）
  8. 输出报告 JSON

前置条件：
  - 已运行 train_resnet.py（生成 checkpoints）
  - 已运行 build_weight_chain.py（生成 weight_chain_resnet.json）
  - 已启动 Hardhat 本地链并部署合约（deploy_info.json 存在）

用法：
  # 终端 1：启动本地链
  cd blockchain_anchor && npx hardhat node

  # 终端 2：部署合约（若 deploy_info.json 已存在可跳过）
  cd blockchain_anchor && npx hardhat run scripts/deploy.ts --network localhost

  # 终端 2：运行本脚本
  python resnet_cifar/anchor_resnet.py
"""

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from torchvision import datasets
from web3 import Web3


# ========== 路径 ==========
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = SCRIPT_DIR / "data"
CKPT_DIR = SCRIPT_DIR / "checkpoints"
ARTIFACTS_DIR = PROJECT_ROOT / "blockchain_anchor" / "artifacts"

# ========== 链配置 ==========
RPC_URL = "http://127.0.0.1:8545"
CHAIN_ID = 31337
# Hardhat Account #0（固定值，仅本地测试）
PRIVATE_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"

# ========== 版本链定义（三代 ResNet） ==========
LINEAGE = [
    {"name": "ResNet-v1",            "version": "v1.0", "parent": None,                  "epochs": 10, "lr": 0.1},
    {"name": "ResNet-v2-finetuned",  "version": "v2.0", "parent": "ResNet-v1",           "epochs": 5,  "lr": 0.01},
    {"name": "ResNet-v3-pruned",     "version": "v3.0", "parent": "ResNet-v2-finetuned", "epochs": 3,  "lr": 0.005},
]

DATASET_ID_STR = "CIFAR10-train-v1"


# ============================================================
# 哈希工具
# ============================================================

def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> bytes:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            sha.update(chunk)
    return sha.digest()


def keccak_pair(a: str, b: str) -> str:
    """Keccak-256 合并两个字典序排序后的哈希（与 DataAnchor 合约一致）。"""
    if a < b:
        combined = bytes.fromhex(a) + bytes.fromhex(b)
    else:
        combined = bytes.fromhex(b) + bytes.fromhex(a)
    return Web3.keccak(combined).hex()


def leaf_hash_cifar(image: np.ndarray, label: int) -> str:
    """CIFAR-10 叶子哈希 = SHA-256(img_bytes || label_byte)。"""
    img_bytes = image.astype(np.uint8).tobytes()
    return sha256_hex(img_bytes + bytes([label]))


# ============================================================
# Merkle 树构建
# ============================================================

def build_merkle_tree(leaves: list[str]) -> tuple[str, int, list[list[str]]]:
    """构建二叉 Merkle 树（奇数层复制末尾元素补齐，OpenZeppelin 惯例）。"""
    tree_layers = [leaves[:]]
    layer = leaves[:]
    while len(layer) > 1:
        if len(layer) % 2 == 1:
            layer.append(layer[-1])
        new_layer = [keccak_pair(layer[i], layer[i + 1]) for i in range(0, len(layer), 2)]
        tree_layers.append(new_layer)
        layer = new_layer
    return layer[0], len(tree_layers), tree_layers


def get_merkle_proof(leaf_index: int, tree_layers: list[list[str]]) -> list[str]:
    """为指定索引的叶子生成 Merkle 证明路径。"""
    proof = []
    idx = leaf_index
    for layer in tree_layers[:-1]:
        sibling_idx = idx + 1 if idx % 2 == 0 else idx - 1
        if sibling_idx < len(layer):
            proof.append(layer[sibling_idx])
        idx //= 2
    return proof


# ============================================================
# 链上交易辅助
# ============================================================

def send_tx(w3, acct, contract, fn_name, args, gas=500_000):
    """构建、签名、发送交易，返回 (tx_hash, gas_used, block)。"""
    tx = contract.functions[fn_name](*args).build_transaction({
        "from": acct.address,
        "nonce": w3.eth.get_transaction_count(acct.address),
        "gas": gas,
        "gasPrice": w3.eth.gas_price,
        "chainId": CHAIN_ID,
    })
    signed = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    return tx_hash.hex(), receipt.gasUsed, receipt.blockNumber


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 60)
    print("  ResNet-18 + CIFAR-10 端到端链上锚定")
    print("=" * 60)

    # ---- 1. 读取训练哈希链 ----
    chain_path = SCRIPT_DIR / "weight_chain_resnet.json"
    if not chain_path.exists():
        print(f"[!] {chain_path} 不存在，请先运行 build_weight_chain.py")
        sys.exit(1)
    weight_chain = json.load(open(chain_path))
    chain_tail = weight_chain["chain_tail"]
    weight_hash_hex = weight_chain["weight_hashes"][-1]
    epochs = weight_chain["epochs"]
    print(f"\n[*] 训练哈希链已加载: {epochs} epoch, 链尾 C_E={chain_tail[:16]}...")

    # ---- 2. 构建 CIFAR-10 Merkle 树 ----
    print(f"\n[*] 加载 CIFAR-10 训练集，构建 Merkle 树...")
    t0 = time.time()
    train_set = datasets.CIFAR10(str(DATA_DIR), train=True, download=True)
    n = len(train_set)
    print(f"  样本数: {n}")

    t1 = time.time()
    leaves = []
    for i, (img, label) in enumerate(train_set):
        leaves.append(leaf_hash_cifar(np.array(img), label))
        if (i + 1) % 10000 == 0:
            print(f"  叶子哈希进度: {i + 1}/{n}")
    leaf_time = time.time() - t1

    t2 = time.time()
    merkle_root, depth, tree_layers = build_merkle_tree(leaves)
    tree_time = time.time() - t2
    merkle_total = time.time() - t0
    print(f"  Merkle Root:  0x{merkle_root}")
    print(f"  树深度: {depth} (理论 ceil(log2({n}))={math.ceil(math.log2(n))})")
    print(f"  叶子哈希耗时: {leaf_time:.1f}s, 建树耗时: {tree_time:.1f}s, 合计 {merkle_total:.1f}s")

    # 验证一条 Merkle 证明
    proof = get_merkle_proof(0, tree_layers)
    computed = leaves[0]
    for sib in proof:
        computed = keccak_pair(computed, sib)
    proof_ok = computed == merkle_root
    print(f"  Merkle Proof 验证 (idx=0): {'通过' if proof_ok else '失败'}")

    # ---- 3. 连接本地链 ----
    print(f"\n[*] 连接 Hardhat 本地链: {RPC_URL}")
    w3 = Web3(Web3.HTTPProvider(RPC_URL))
    if not w3.is_connected():
        print("[!] 无法连接，请先启动: cd blockchain_anchor && npx hardhat node")
        sys.exit(1)
    acct = w3.eth.account.from_key(PRIVATE_KEY)
    print(f"  链 ID: {w3.eth.chain_id}, 账户: {acct.address}")
    print(f"  余额: {w3.from_wei(w3.eth.get_balance(acct.address), 'ether')} ETH")

    # ---- 4. 加载合约 ----
    deploy_info_path = PROJECT_ROOT / "deploy_info.json"
    if not deploy_info_path.exists():
        print("[!] deploy_info.json 不存在，请先部署合约:")
        print("    cd blockchain_anchor && npx hardhat run scripts/deploy.ts --network localhost")
        sys.exit(1)
    deploy_info = json.load(open(deploy_info_path))
    data_addr = Web3.to_checksum_address(deploy_info["dataAnchor"])
    weight_addr = Web3.to_checksum_address(deploy_info["weightAnchor"])
    print(f"  DataAnchor:   {data_addr}")
    print(f"  WeightAnchor: {weight_addr}")

    data_abi = json.load(open(ARTIFACTS_DIR / "contracts" / "DataAnchor.sol" / "DataAnchor.json"))["abi"]
    weight_abi = json.load(open(ARTIFACTS_DIR / "contracts" / "WeightAnchor.sol" / "WeightAnchor.json"))["abi"]
    data_contract = w3.eth.contract(address=data_addr, abi=data_abi)
    weight_contract = w3.eth.contract(address=weight_addr, abi=weight_abi)

    # ---- 5. 注册数据集 ----
    dataset_id = w3.keccak(text=DATASET_ID_STR)
    merkle_root_bytes = bytes.fromhex(merkle_root)
    dataset_meta = json.dumps({"name": "CIFAR-10", "size": n, "tree_depth": depth})

    print(f"\n{'─'*60}")
    print("  [1/4] 注册 CIFAR-10 数据集")
    print(f"{'─'*60}")
    try:
        data_tx, data_gas, data_block = send_tx(
            w3, acct, data_contract, "registerDataset",
            [dataset_id, merkle_root_bytes, dataset_meta], gas=300_000)
        print(f"已注册  Gas={data_gas}  Block={data_block}")
    except Exception as e:
        if "already" in str(e).lower():
            print(f"已注册，跳过  (data_gas=0)")
            data_gas = 0
        else:
            raise

    # ---- 6. 注册根模型 ResNet-v1 ----
    model_id = w3.keccak(text=LINEAGE[0]["name"])
    weight_hash_bytes = bytes.fromhex(weight_hash_hex)
    chain_tip_bytes = bytes.fromhex(chain_tail)
    parent_id = b"\x00" * 32
    model_meta = json.dumps({
        "arch": "ResNet-18", "epochs": epochs, "optimizer": "SGD",
        "lr": 0.1, "batch_size": 128, "params": 11173962,
    })

    print(f"\n{'─'*60}")
    print(f"  [2/4] 注册根模型 {LINEAGE[0]['name']}")
    print(f"{'─'*60}")
    try:
        weight_tx, weight_gas, weight_block = send_tx(
            w3, acct, weight_contract, "registerModel",
            [model_id, weight_hash_bytes, dataset_id, chain_tip_bytes,
             parent_id, model_meta, LINEAGE[0]["version"]], gas=500_000)
        print(f"已注册  Gas={weight_gas}  Block={weight_block}")
    except Exception as e:
        if "already" in str(e).lower():
            print(f"已注册，跳过  (weight_gas=0)")
            weight_gas = 0
        else:
            raise

    # ---- 7. 链上验证 ----
    print(f"\n{'─'*60}")
    print("  [3/4] 链上验证")
    print(f"{'─'*60}")

    v_data = data_contract.functions.verifyDataset(dataset_id, merkle_root_bytes).call()
    v_weight = weight_contract.functions.verifyModel(model_id, weight_hash_bytes).call()
    v_chain = weight_contract.functions.verifyTrainingChain(model_id, chain_tip_bytes).call()
    fake_root = b"\x00" * 32
    v_tamper = not data_contract.functions.verifyDataset(dataset_id, fake_root).call()

    print(f"  数据集完整性:   {'True' if v_data else 'False'}")
    print(f"  权重完整性:     {'True' if v_weight else 'False'}")
    print(f"  训练链尾:       {'True' if v_chain else 'False'}")
    print(f"  篡改检测(全零): {'已拒绝' if v_tamper else '未检测'}")

    # ---- 8. 三代版本链 DAG ----
    print(f"\n{'─'*60}")
    print("  [4/4] 注册三代版本链 DAG")
    print(f"{'─'*60}")

    # 用最终权重哈希演示 DAG 结构（权重内容不影响版本链逻辑）
    ckpt_final = CKPT_DIR / f"resnet_epoch{epochs}.pth"
    base_wh = sha256_file(str(ckpt_final))

    model_ids = {}
    model_ids[LINEAGE[0]["name"]] = model_id  # 根模型已注册

    lineage_gases = []
    for entry in LINEAGE:
        name = entry["name"]
        parent_name = entry["parent"]
        mid = w3.keccak(text=name)
        model_ids[name] = mid
        pid = model_ids[parent_name] if parent_name else b"\x00" * 32

        meta = json.dumps({"arch": "ResNet-18", "epochs": entry["epochs"], "lr": entry["lr"]})
        rel = "根模型" if parent_name is None else f"parent={parent_name}"
        print(f"\n  注册 {name} ({entry['version']}) — {rel}")
        print(f"    modelId:      0x{mid.hex()[:16]}...")
        print(f"    parentModelId: {'0 (根)' if parent_name is None else '0x' + pid.hex()[:16] + '...'}")

        if parent_name is None:
            # 根模型已在上一步注册
            print(f"    (已注册，跳过)")
            lineage_gases.append(0)
            continue

        try:
            tx, gas, blk = send_tx(
                w3, acct, weight_contract, "registerModel",
                [mid, base_wh, dataset_id, chain_tip_bytes, pid, meta, entry["version"]],
                gas=500_000)
            print(f"Gas={gas}  Block={blk}")
            lineage_gases.append(gas)
        except Exception as e:
            if "already" in str(e).lower():
                print(f"已注册，跳过")
                lineage_gases.append(0)
            else:
                raise

    # 血缘查询
    print(f"\n[*] 血缘查询验证...")
    leaf_id = model_ids[LINEAGE[-1]["name"]]
    root_id = model_ids[LINEAGE[0]["name"]]
    mid_id = model_ids[LINEAGE[1]["name"]]

    lineage_chain = weight_contract.functions.getLineage(leaf_id).call()
    is_ancestor = weight_contract.functions.verifyLineage(leaf_id, root_id).call()
    children = weight_contract.functions.getChildren(mid_id).call()

    print(f"  getLineage({LINEAGE[-1]['name']}): {len(lineage_chain)} 个祖先")
    print(f"  verifyLineage({LINEAGE[-1]['name']}, {LINEAGE[0]['name']}): {is_ancestor}")
    print(f"  getChildren({LINEAGE[1]['name']}): {len(children)} 个子模型")

    print(f"\n  版本链结构:")
    print(f"  {LINEAGE[0]['name']} ({LINEAGE[0]['version']})")
    print(f"    └── {LINEAGE[1]['name']} ({LINEAGE[1]['version']})")
    print(f"          └── {LINEAGE[2]['name']} ({LINEAGE[2]['version']})")

    # ---- 汇总 ----
    print(f"\n{'='*60}")
    print(f"  端到端锚定完成")
    print(f"{'='*60}")
    print(f"  数据集:     CIFAR-10 ({n} 样本)")
    print(f"  Merkle Root: 0x{merkle_root}")
    print(f"  Merkle 深度: {depth}, 构建耗时: {merkle_total:.1f}s")
    print(f"  数据集注册 Gas: {data_gas}")
    print(f"  模型注册 Gas:   {weight_gas}")
    print(f"  数据验证: {v_data} | 权重验证: {v_weight} | 链验证: {v_chain} | 篡改: {v_tamper}")
    print(f"  版本链: {len(lineage_chain)} 代, verifyLineage={is_ancestor}")
    print(f"{'='*60}")

    # ---- 输出报告 ----
    report = {
        "experiment": "resnet_cifar_e2e_anchor",
        "model": "ResNet-18",
        "dataset": "CIFAR-10",
        "dataset_sample_count": n,
        "merkle_root": "0x" + merkle_root,
        "merkle_depth": depth,
        "merkle_build_time_sec": round(merkle_total, 2),
        "merkle_proof_verified": proof_ok,
        "training_epochs": epochs,
        "training_chain_tail": "0x" + chain_tail,
        "final_weight_hash": "0x" + weight_hash_hex,
        "dataset_id": "0x" + dataset_id.hex(),
        "root_model_id": "0x" + model_id.hex(),
        "data_gas_used": data_gas,
        "weight_gas_used": weight_gas,
        "data_verification": v_data,
        "weight_verification": v_weight,
        "chain_verification": v_chain,
        "tamper_detection": v_tamper,
        "lineage": [
            {"name": m["name"], "version": m["version"],
             "modelId": "0x" + model_ids[m["name"]].hex(),
             "parentModelId": "0x" + (model_ids[m["parent"]].hex() if m["parent"] else "0" * 64)}
            for m in LINEAGE
        ],
        "lineage_length": len(lineage_chain),
        "verifyLineage": is_ancestor,
        "children_count": len(children),
        "network": f"localhost:{CHAIN_ID}",
    }
    report_path = SCRIPT_DIR / "anchor_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n报告已保存: {report_path}")


if __name__ == "__main__":
    main()
