# Blockchain Anchoring for Deep Learning Models

### 基于哈希树的深度学习模型数据与权重链上锚定

[English](#english) | [中文](#中文)

A model-agnostic and dataset-agnostic framework for anchoring deep learning
training data and model weights to EVM-compatible blockchains via
Merkle-tree-based cryptographic commitments.  This repository accompanies the
paper *"基于哈希树的深度学习模型数据与权重链上锚定研究"* and provides the
complete implementation, experiment scripts, and a reusable Python library.

---

## English

### Why This Matters

In 2023, artists sued Stability AI for training Stable Diffusion on their
work without permission.  In 2022, developers sued GitHub Copilot for using
open-source code without attribution.  In 2023, Meta's LLaMA weights leaked
onto 4chan and spawned thousands of derivative models with no traceable
lineage.  In 2024, the EU AI Act made training-data transparency a legal
obligation for general-purpose AI providers.

Every one of these cases hits the same wall: **nobody can produce
cryptographic evidence of what data was used, whether weights have been
swapped, or where a derivative model came from.**  The current state of the
art is self-reported training logs stored on a company's own server — useless
in a court of law.

This repository provides a lightweight, practical solution:

| Problem | How This Repo Solves It |
|---------|-------------------------|
| *Was my image used to train this model?* | Merkle inclusion proof — 16 hashes, verified on-chain, no trust required |
| *Were the model weights secretly replaced?* | SHA-256 weight hash + training chain tip committed on-chain |
| *Is this model derived from a leaked parent model?* | Immutable model lineage DAG via `parentModelId` on-chain |
| *Can the author forge the training log after the fact?* | No — blockchain immutability + hash chain tamper evidence |
| *Does this scale to large datasets?* | Yes — O(log n) proof size, constant 32-byte on-chain storage |

### Who Is This For

- **AI researchers** who want to make their training process auditable
- **Model providers** who need to demonstrate compliance with the EU AI Act
- **Copyright holders** who need cryptographic evidence for litigation
- **Blockchain developers** exploring real-world smart contract applications
- **Students and educators** studying the intersection of AI and cryptography

### How It Works

1. **Hash the dataset** into a single 32-byte Merkle root
2. **Chain training checkpoints** into a tamper-evident hash chain
   Cₜ = H(Wₜ ‖ Cₜ₋₁)
3. **Commit both to a blockchain** (any EVM-compatible chain)
4. **Anyone can verify** — without trusting the model provider —
   whether a specific sample was in the training set, whether the weights
   have been altered, or what parent model a derivative was fine-tuned from

### Project Structure

```
├── frame/                       # Reusable library 
│   ├── hashing.py               #   SHA-256 / Keccak-256 utilities
│   ├── merkle.py                #   Merkle Tree, inclusion proofs, subsetproofs
│   ├── sparse_merkle.py         #   Sparse Merkle Tree, non-membership proofs
│   ├── training_chain.py        #   Epoch-wise training hash chain
│   ├── contracts.py             #   On-chain interaction client
│   ├── deployment.py            #   Contract deployment (pure Python)
│   ├── lineage.py               #   Model version DAG
│   ├── security.py              #   Replay / Sybil attack tests
│   ├── benchmark.py             #   Gas and latency benchmarks
│   └── pipeline.py              #   End-to-end orchestration
│
├── blockchain_anchor/           # Solidity contracts + Hardhat project
│   ├── contracts/
│   │   ├── DataAnchor.sol       #   Dataset Merkle root registry
│   │   └── WeightAnchor.sol     #   Model weight + lineage registry
│   ├── scripts/deploy.ts
│   └── hardhat.config.ts
│
├── scripts/                     # Reference implementation (LeNet + MNIST)
│   ├── train_lenet.py           #   Train LeNet-5, save per-epoch checkpoints
│   ├── build_merkle.py          #   Build Merkle tree from dataset
│   ├── build_weight_chain.py    #   Build training hash chain
│   ├── upload_to_chain.py       #   Register on blockchain
│   ├── verify_single.py         #   Single-sample Merkle proof
│   ├── query_models_by_dataset.py  #   Forward provenance
│   ├── query_dataset_by_model.py   #   Reverse provenance + lineage
│   └── run_pipeline.py          #   One-click A->B->C->D->E pipeline
│
├── resnet_cifar/                # Extended experiment (ResNet-18 + CIFAR-10, ~11M params)
│   ├── train_resnet.py          #   Train ResNet-18, per-epoch checkpoints
│   ├── build_weight_chain.py    #   Build training hash chain
│   └── anchor_resnet.py         #   End-to-end anchoring + 3-gen lineage DAG
│
└── experiments/                 # Experiment demos (subset / non-membership / lineage / security / gas)
    ├── demo_subset_proof.py     #   Batch Merkle proof (k=20)
    ├── demo_non_membership.py   #   Sparse Merkle Tree: proving absence
    ├── demo_lineage_chain.py    #   3-generation model lineage DAG
    ├── demo_security.py         #   Replay + Sybil attack tests
    └── demo_gas_benchmark.py    #   O(log n) scaling benchmark
```

### Quick Start

#### 1. Environment Setup

```bash
# Create conda environment
conda create -n blockchain python=3.10 -y
conda activate blockchain

# Install Python dependencies
pip install -r requirements.txt

# Install Node.js (from https://nodejs.org, LTS version)
# Then install Hardhat:
cd blockchain_anchor
npm install
npx hardhat compile
cd ..
```

Verify GPU (optional — CPU works fine for LeNet + MNIST):
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

#### 2. Run the Full Pipeline

**Terminal 1** — Start local blockchain (keep it running):
```bash
cd blockchain_anchor
npx hardhat node
```

**Terminal 2** — One-command pipeline:
```bash
conda activate blockchain
python scripts/run_pipeline.py --force
```

The pipeline executes five stages automatically:
- **Stage A** — Environment check (dependencies, directories, Hardhat node)
- **Stage B** — Train LeNet-5 on MNIST (5 epochs, ~68 seconds on GPU, ~2 min on CPU)
- **Stage C** — Build Merkle tree (60,000 leaves → 1 root) and training hash chain
- **Stage D** — Compile + deploy Solidity contracts, register dataset and model on-chain
- **Stage E** — Verify Merkle proof, forward/reverse provenance queries

#### 3. Run the Experiments

```bash
python experiments/demo_subset_proof.py       # Batch proof (20 samples)
python experiments/demo_non_membership.py     # SMT non-membership proof
python experiments/demo_lineage_chain.py      # 3-generation model DAG
python experiments/demo_security.py           # Replay + Sybil tests
python experiments/demo_gas_benchmark.py      # O(log n) scaling chart
```

Each script outputs results to `experiments/outputs/`.

#### 4. Use the Framework in Your Own Project

```python
from frame import (
    MerkleTree, SparseMerkleTree, TrainingChain,
    AnchorClient, ModelLineage, SecurityTester, BenchmarkRunner,
    leaf_hash, keccak256,
)

# --- Build Merkle commitment from any dataset ---
# Caller serialises data to bytes (label concatenation etc. is up to you)
items = [img_bytes + bytes([label]) for ...]
tree = MerkleTree.from_iterable(items)
print(f"Merkle Root: 0x{tree.root.hex()}")

# --- Build training hash chain from checkpoints ---
chain = TrainingChain.from_checkpoints("checkpoints/")
print(f"Chain tail: {chain.chain_tail[:32]}...")

# --- Anchor to any EVM chain ---
client = AnchorClient(
    rpc_url="https://sepolia.infura.io/v3/YOUR_KEY",
    private_key="0x...",
    data_anchor_addr="0x...",
    weight_anchor_addr="0x...",
)
from web3 import Web3
dataset_id = Web3.keccak(text="my-dataset-v1")
client.register_dataset(dataset_id, tree.root, '{"name":"my-data"}')

# --- Verify ---
proof = tree.get_proof(index=42)
is_member = client.verify_member(dataset_id, proof.leaf_hash, proof.siblings)

# --- Non-membership (prove something is NOT in the training set) ---
smt = SparseMerkleTree.from_iterable(items)
stranger_key = keccak256(b"not-in-training-set")
proof_nm = smt.get_non_membership_proof(stranger_key)
assert proof_nm.verify()
```

### Solidity Contracts

| Contract | Purpose |
|----------|---------|
| `DataAnchor` | Register dataset Merkle roots; verify single, subset, and non-membership proofs |
| `WeightAnchor` | Register model weight hashes + training chain tips + parent model IDs; query lineage DAG |

Target: Solidity `^0.8.28`, EVM Cancun. Compatible with Hardhat localnet, Sepolia, Ethereum mainnet, and any EVM-compatible L2.

### Key Results

#### Baseline: LeNet-5 + MNIST
| Metric | Value |
|--------|-------|
| Training accuracy | 98.83% (5 epochs, 68 sec) |
| Merkle tree depth | 17 (60,000 leaves) |
| Merkle root | 32 bytes |
| Merkle proof length | 16 siblings |
| Single proof verification | ~18 ms (on-chain, view call) |
| Subset proof (k=20) | ~57 ms |
| Non-membership proof (SMT, depth 256) | ~37 ms |
| Dataset registration gas | ~206,000 |
| Model registration gas | ~343,000 |
| Replay attack | Blocked by contract |
| Tamper detection | 100% (any single-byte change detected) |

#### Extended: ResNet-18 + CIFAR-10

| Metric | Value |
|--------|-------|
| Training accuracy | ~93% (10 epochs, ~15 min) |
| Merkle tree depth | 17 (50,000 leaves) |
| Merkle tree build time | ~3 sec |
| Dataset registration gas | ~206,000 |
| Model registration gas | ~343,000 |
| Lineage | 3-generation DAG (v1 -> v2-finetuned -> v3-pruned) |

> ResNet-18 checkpoints are ~44 MB vs. LeNet-5's ~250 KB.  On-chain gas is identical,
> confirming the framework's scale-independence.  See `resnet_cifar/` for scripts.

### License

MIT — see [LICENSE](LICENSE).

---

## 中文

### 为什么这个仓库重要

2023 年，艺术家集体起诉 Stability AI 未经许可使用其作品训练 Stable Diffusion；
2022 年，开发者起诉 GitHub Copilot 使用开源代码未署名；
2023 年，Meta 的 LLaMA 权重在 4chan 泄漏后衍生出数千个无法溯源的微调模型；
2024 年，欧盟 AI Act 将训练数据透明度上升为法律义务。

这些事件的共同困境是：**没有人能拿出密码学级别的证据，证明训练数据是什么、模型权重是否被替换、衍生模型来自哪里。** 目前行业做法是公司自己服务器上存一份训练日志——在法律上毫无说服力。

本仓库提供了一个轻量级、可工程化的解决方案：

| 问题 | 如何解决 |
|------|---------|
| *我的图是否被用于训练此模型？* | Merkle 成员证明 — 16 个哈希，链上验证，无需信任 |
| *模型权重是否被偷偷替换？* | 权重 SHA-256 + 训练链尾上链锚定 |
| *此模型是否来自某个泄漏的父模型？* | 不可篡改的模型版本链 DAG（parentModelId）|
| *训练方能否事后伪造日志？* | 不能 — 区块链不可逆 + 哈希链防篡改 |
| *大数据集能扛住吗？* | 能 — O(log n) 证明长度，链上始终只存 32 字节 |

### 面向谁

- **AI 研究者**——希望训练过程可被第三方审计
- **模型厂商**——需要证明符合 EU AI Act 合规要求
- **版权方**——需要密码学证据用于诉讼
- **区块链开发者**——寻找智能合约的真实应用场景
- **学生和教学者**——研究 AI 与密码学的交叉领域

### 项目原理

1. **将数据集哈希压缩**为单一 32 字节 Merkle 根
2. **将训练过程的每个 epoch checkpoint 串成哈希链** Cₜ = H(Wₜ ‖ Cₜ₋₁)
3. **将 Merkle 根和链尾提交到区块链**（任何 EVM 兼容链）
4. **任何人均可独立验证**——不依赖模型提供方——某张图片是否在训练集中、
   模型权重是否被替换、衍生模型来自哪个父模型

### 项目结构

```
├── frame/                       # 通用框架库，不绑定特定模型或数据集
├── blockchain_anchor/           # Solidity 合约 + Hardhat 配置
│   └── contracts/               #   DataAnchor.sol / WeightAnchor.sol
├── scripts/                     # 参考实现（LeNet-5 + MNIST 基线）
├── resnet_cifar/                # 扩展实验（ResNet-18 + CIFAR-10，~11M 参数）
└── experiments/                 # 独立实验演示（子集/差集/版本链/安全/Gas）
```

### 环境配置

```bash
# 创建 conda 环境
conda create -n blockchain python=3.10 -y
conda activate blockchain

# 安装 Python 依赖
pip install -r requirements.txt

# 安装 Node.js（从 https://nodejs.org 下载 LTS 版本）
# 安装 Hardhat：
cd blockchain_anchor
npm install
npx hardhat compile
cd ..
```

验证 GPU（可选——LeNet + MNIST 用 CPU 也完全够）：
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### 一键运行全流程

**终端 1**——启动本地区块链（保持运行）：
```bash
cd blockchain_anchor
npx hardhat node
```

**终端 2**——一键流程：
```bash
conda activate blockchain
python scripts/run_pipeline.py --force
```

流水线自动执行五个阶段：
- **阶段 A**——环境自检（依赖库、目录结构、Hardhat 节点）
- **阶段 B**——训练 LeNet-5（MNIST，5 epoch，GPU 约 68 秒，CPU 约 2 分钟）
- **阶段 C**——构建 Merkle 树（60000 叶子→1 个根哈希）+ 训练时序哈希链
- **阶段 D**——编译部署 Solidity 合约，数据集和模型上链锚定
- **阶段 E**——Merkle 证明验证、正向/反向溯源查询

如果中途中断，可从断点恢复：
```bash
python scripts/run_pipeline.py --from D --force   # 从阶段 D 继续
```

### 运行实验

```bash
python experiments/demo_subset_proof.py       # 批量证明（20 条样本）
python experiments/demo_non_membership.py     # 差集证明（Sparse Merkle Tree）
python experiments/demo_lineage_chain.py      # 三代模型版本链 DAG
python experiments/demo_security.py           # 重放攻击 + Sybil 攻击测试
python experiments/demo_gas_benchmark.py      # O(log n) 性能曲线
```

每个脚本将结果输出到 `experiments/outputs/`。

### 框架使用方法

```python
from frame import (
    MerkleTree, SparseMerkleTree, TrainingChain,
    AnchorClient, ModelLineage, SecurityTester, BenchmarkRunner,
    leaf_hash, keccak256,
)

# --- 1. 从任意数据集构建 Merkle 承诺 ---
# 调用方负责将原始数据序列化为 bytes（标签拼接等由调用方处理）
items = [img_bytes + bytes([label]) for ...]
tree = MerkleTree.from_iterable(items)

# --- 2. 从 checkpoint 构建训练哈希链 ---
chain = TrainingChain.from_checkpoints("checkpoints/")

# --- 3. 锚定到任意 EVM 链 ---
client = AnchorClient(rpc_url, private_key, data_addr, weight_addr)
from web3 import Web3
dataset_id = Web3.keccak(text="my-dataset-v1")
client.register_dataset(dataset_id, tree.root, '{"name":"my-data"}')

# --- 4. 验证 ---
proof = tree.get_proof(index=42)
is_member = client.verify_member(dataset_id, proof.leaf_hash, proof.siblings)

# --- 5. 差集证明（"我的训练集里没有这张图"）---
smt = SparseMerkleTree.from_iterable(items)
stranger_key = keccak256(b"not-in-set")
proof_nm = smt.get_non_membership_proof(stranger_key)
assert proof_nm.verify()
```

框架不绑定任何特定模型或数据集——换用 ResNet、CIFAR-10、LLaMA 只需改动数据加载部分的几行代码，
`frame/` 下的所有模块保持通用。

### Solidity 合约

| 合约 | 功能 |
|------|------|
| `DataAnchor` | 注册数据集 Merkle 根；验证单条、子集、差集证明 |
| `WeightAnchor` | 注册模型权重哈希 + 训练链尾 + 父模型 ID；查询版本链 DAG |

编译目标：Solidity `^0.8.28`，EVM Cancun。兼容 Hardhat 本地链、Sepolia 测试网、以太坊主网及任何 EVM L2。

### 核心实验结果

#### 基线：LeNet-5 + MNIST
| 指标 | 数值 |
|------|------|
| 训练准确率 | 98.83%（5 epoch，68 秒） |
| Merkle 树深度 | 17（60,000 叶子） |
| Merkle 根 | 32 字节 |
| 单条 Merkle 证明长度 | 16 个兄弟哈希 |
| 单条证明验证耗时 | ~18 ms（链上 view 调用） |
| 子集证明（k=20） | ~57 ms |
| 差集证明（SMT，深度 256） | ~37 ms |
| 数据集注册 Gas | ~206,000 |
| 模型注册 Gas | ~343,000 |
| 重放攻击 | 合约拒绝 |
| 篡改检测 | 100%（任意单字节篡改均可检出） |

#### 扩展：ResNet-18 + CIFAR-10

| 指标 | 数值 |
|------|------|
| 训练准确率 | ~93%（10 epoch，~15 分钟） |
| Merkle 树深度 | 17（50,000 叶子） |
| Merkle 树构建耗时 | ~3 秒 |
| 数据集注册 Gas | ~206,000 |
| 模型注册 Gas | ~343,000 |
| 版本链 | 三代 DAG（v1 → v2-finetuned → v3-pruned） |

> ResNet-18 的 checkpoint 文件约 44 MB，LeNet-5 约 250 KB。链上 Gas 开销完全一致，
> 框架的规模无关性得到验证。扩展实验脚本位于 `resnet_cifar/` 目录。


### 许可证

MIT — 详见 [LICENSE](LICENSE)。
