"""
Blockchain Anchoring Framework for Deep Learning Models
========================================================

A generic, model-agnostic and dataset-agnostic framework for anchoring
deep learning training data and model weights to EVM-compatible blockchains
via Merkle-tree-based cryptographic commitments.

Modules
-------
hashing         — SHA-256 and Keccak-256 hash utilities
merkle          — Merkle Tree construction, proof generation, verification
sparse_merkle   — Sparse Merkle Tree with non-membership proofs
training_chain  — Epoch-wise training hash chain C_t = H(W_t || C_{t-1})
contracts       — On-chain interaction via DataAnchor and WeightAnchor
lineage         — Model version DAG management
security        — Replay and Sybil attack tests
pipeline        — End-to-end orchestration

Usage
-----
    from frame import MerkleTree, SparseMerkleTree, TrainingChain, AnchorClient

    # 1. Build Merkle commitment for a dataset
    tree = MerkleTree.from_iterable(dataset, hash_fn=leaf_hash)
    root = tree.root

    # 2. Build training hash chain from checkpoints
    chain = TrainingChain.from_checkpoints("checkpoints/")
    chain_tip = chain.tail

    # 3. Anchor to blockchain
    client = AnchorClient(rpc_url, private_key)
    client.register_dataset(dataset_id, root, metadata)
    client.register_model(model_id, weight_hash, dataset_id, chain_tip)

    # 4. Verify membership
    proof = tree.get_proof(index=42)
    is_valid = client.verify_member(dataset_id, leaf, proof)

    # 5. Verify model lineage
    ancestors = client.get_lineage(model_id)
"""

__version__ = "1.0.0"
__author__ = "Blockchain Anchoring Research Project"

from .hashing import (
    sha256_hex, keccak256, keccak256_hex,
    leaf_hash, streaming_sha256, streaming_sha256_hex,
    merge_shard_hashes, merge_shard_hashes_hex, pair_hash,
    HashFunction,
)
from .merkle import MerkleTree, MerkleProof
from .sparse_merkle import SparseMerkleTree, NonMembershipProof
from .training_chain import TrainingChain, CheckpointInfo
from .contracts import AnchorClient, ContractAddresses
from .lineage import ModelLineage, LineageNode
from .security import SecurityTester
from .pipeline import Pipeline, PipelineConfig
from .deployment import deploy_contract, deploy_anchors
from .benchmark import BenchmarkRunner, BenchmarkResult

__all__ = [
    # hashing
    "sha256_hex",
    "keccak256",
    "keccak256_hex",
    "leaf_hash",
    "streaming_sha256",
    "streaming_sha256_hex",
    "merge_shard_hashes",
    "merge_shard_hashes_hex",
    "pair_hash",
    "HashFunction",
    # merkle
    "MerkleTree",
    "MerkleProof",
    # sparse merkle
    "SparseMerkleTree",
    "NonMembershipProof",
    # training
    "TrainingChain",
    "CheckpointInfo",
    # contracts
    "AnchorClient",
    "ContractAddresses",
    # lineage
    "ModelLineage",
    "LineageNode",
    # security
    "SecurityTester",
    # pipeline
    "Pipeline",
    "PipelineConfig",
    # deployment
    "deploy_contract",
    "deploy_anchors",
    # benchmark
    "BenchmarkRunner",
    "BenchmarkResult",
]
