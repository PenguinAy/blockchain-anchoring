"""
End-to-end anchoring pipeline.

Orchestrates the full workflow: training → Merkle tree → training chain →
contract deployment → on-chain registration → verification.

Designed as a reusable entry point that can be configured for any model
and any dataset by passing appropriate callables.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .hashing import leaf_hash, sha256_hex, keccak256
from .merkle import MerkleTree
from .sparse_merkle import SparseMerkleTree
from .training_chain import TrainingChain
from .contracts import AnchorClient, RegistrationResult
from .lineage import ModelLineage, LineageNode, make_model_id, make_dataset_id
from .security import SecurityTester, SecurityReport


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Configuration for an anchoring pipeline run."""

    # Chain
    rpc_url: str = "http://127.0.0.1:8545"
    private_key: str = ""
    data_anchor_addr: str = ""
    weight_anchor_addr: str = ""
    chain_id: int = 31337

    # Model
    model_name: str = "model"
    model_version: str = "v1.0"
    model_meta: str = "{}"
    checkpoint_dir: str = "checkpoints"
    final_checkpoint: str = ""

    # Dataset
    dataset_name: str = "dataset"
    dataset_metadata: str = "{}"

    # Security
    attacker_private_key: str = ""

    # Output
    output_dir: str = "outputs"


# ---------------------------------------------------------------------------
# Pipeline results
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Aggregated output of a pipeline run."""

    merkle_root: str = ""
    merkle_depth: int = 0
    merkle_leaf_count: int = 0
    merkle_build_time_sec: float = 0.0

    chain_tail: str = ""
    chain_epochs: int = 0

    smt_root: str = ""
    smt_size: int = 0

    dataset_registration: Optional[RegistrationResult] = None
    model_registration: Optional[RegistrationResult] = None

    data_verification: bool = False
    weight_verification: bool = False
    chain_verification: bool = False
    tamper_detection: bool = False

    member_proof_valid: bool = False
    non_member_proof_valid: bool = False
    subset_proof_valid: bool = False

    lineage: Optional[ModelLineage] = None
    security_report: Optional[SecurityReport] = None

    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "merkle_root": self.merkle_root,
            "merkle_depth": self.merkle_depth,
            "merkle_leaf_count": self.merkle_leaf_count,
            "merkle_build_time_sec": self.merkle_build_time_sec,
            "chain_tail": self.chain_tail,
            "chain_epochs": self.chain_epochs,
            "smt_root": self.smt_root,
            "smt_size": self.smt_size,
            "dataset_tx_hash": (
                self.dataset_registration.tx_hash
                if self.dataset_registration
                else ""
            ),
            "dataset_gas_used": (
                self.dataset_registration.gas_used
                if self.dataset_registration
                else 0
            ),
            "model_tx_hash": (
                self.model_registration.tx_hash
                if self.model_registration
                else ""
            ),
            "model_gas_used": (
                self.model_registration.gas_used
                if self.model_registration
                else 0
            ),
            "data_verification": self.data_verification,
            "weight_verification": self.weight_verification,
            "chain_verification": self.chain_verification,
            "tamper_detection": self.tamper_detection,
            "member_proof_valid": self.member_proof_valid,
            "non_member_proof_valid": self.non_member_proof_valid,
            "subset_proof_valid": self.subset_proof_valid,
            "errors": self.errors,
        }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class Pipeline:
    """
    End-to-end blockchain anchoring pipeline.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration.

    Examples
    --------
    >>> config = PipelineConfig(
    ...     model_name="LeNet5-MNIST",
    ...     checkpoint_dir="checkpoints",
    ...     rpc_url="http://127.0.0.1:8545",
    ...     private_key="0xac09...",
    ... )
    >>> pipeline = Pipeline(config)
    >>> result = pipeline.run(items)
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.result = PipelineResult()
        self._client: Optional[AnchorClient] = None

    # ---- client ------------------------------------------------------

    @property
    def client(self) -> AnchorClient:
        if self._client is None:
            self._client = AnchorClient(
                rpc_url=self.config.rpc_url,
                private_key=self.config.private_key,
                data_anchor_addr=self.config.data_anchor_addr,
                weight_anchor_addr=self.config.weight_anchor_addr,
            )
        return self._client

    # ---- stages ------------------------------------------------------

    def build_merkle(
        self, items: list[bytes]
    ) -> MerkleTree:
        """Stage C-1: build Merkle tree from pre-serialised bytes."""
        t0 = time.time()
        leaves = [leaf_hash(item) for item in items]
        tree = MerkleTree(leaves)
        elapsed = time.time() - t0

        self.result.merkle_root = "0x" + tree.root.hex()
        self.result.merkle_depth = tree.depth
        self.result.merkle_leaf_count = tree.leaf_count
        self.result.merkle_build_time_sec = round(elapsed, 2)
        return tree

    def build_smt(self, items: list[bytes]) -> SparseMerkleTree:
        """Stage C-2: build Sparse Merkle tree for non-membership proofs."""
        n = len(items)
        if n > 50_000:
            print(
                f"  [*] Building SMT with {n} items — this may take a while "
                f"in pure Python.  For production datasets, consider a native "
                f"SMT library (see sparse_merkle.py docstring)."
            )
        smt = SparseMerkleTree()
        for item in items:
            lh = leaf_hash(item)
            key = keccak256(lh)
            smt.insert(key)
        smt.build()
        self.result.smt_root = "0x" + smt.root.hex()
        self.result.smt_size = smt.size
        return smt

    def build_training_chain(self) -> TrainingChain:
        """Stage C-3: build training hash chain from checkpoints."""
        chain = TrainingChain.from_checkpoints(self.config.checkpoint_dir)
        self.result.chain_tail = chain.chain_tail
        self.result.chain_epochs = chain.epochs
        return chain

    def anchor(
        self,
        tree: MerkleTree,
        chain: TrainingChain,
        weight_hash: bytes,
        parent_model_id: bytes = None,
    ) -> None:
        """Stage D: register dataset and model on-chain."""
        dataset_id = make_dataset_id(self.config.dataset_name)
        model_id = make_model_id(self.config.model_name)
        if parent_model_id is None:
            parent_model_id = b"\x00" * 32

        # Dataset
        try:
            self.result.dataset_registration = (
                self.client.register_dataset(
                    dataset_id,
                    tree.root,
                    self.config.dataset_metadata,
                )
            )
        except Exception as e:
            self.result.errors.append(f"dataset_registration: {e}")

        # Model
        try:
            self.result.model_registration = (
                self.client.register_model(
                    model_id,
                    weight_hash,
                    dataset_id,
                    bytes.fromhex(chain.chain_tail),
                    parent_model_id,
                    self.config.model_meta,
                    self.config.model_version,
                )
            )
        except Exception as e:
            self.result.errors.append(f"model_registration: {e}")

    def verify(self, tree: MerkleTree, chain: TrainingChain) -> None:
        """Stage E: on-chain verification."""
        dataset_id = make_dataset_id(self.config.dataset_name)
        model_id = make_model_id(self.config.model_name)

        try:
            self.result.data_verification = self.client.verify_dataset(
                dataset_id, tree.root
            )
        except Exception as e:
            self.result.errors.append(f"data_verification: {e}")

        try:
            self.result.weight_verification = self.client.verify_model(
                model_id, bytes.fromhex(chain.weight_hashes[-1])
            )
        except Exception as e:
            self.result.errors.append(f"weight_verification: {e}")

        try:
            self.result.chain_verification = (
                self.client.verify_training_chain(
                    model_id, bytes.fromhex(chain.chain_tail)
                )
            )
        except Exception as e:
            self.result.errors.append(f"chain_verification: {e}")

        # Tamper detection
        try:
            fake_root = b"\x00" * 32
            self.result.tamper_detection = not self.client.verify_dataset(
                dataset_id, fake_root
            )
        except Exception:
            pass

    def run(
        self, items: list[bytes]
    ) -> PipelineResult:
        """
        Execute the full pipeline.

        Parameters
        ----------
        items : list[bytes]
            Pre-serialised data items (caller handles label/metadata encoding).

        Returns
        -------
        PipelineResult
        """
        t_start = time.time()

        # Stage C
        tree = self.build_merkle(items)
        self.build_smt(items)
        chain = self.build_training_chain()

        # Stage D
        weight_hash = bytes.fromhex(chain.weight_hashes[-1])
        self.anchor(tree, chain, weight_hash)

        # Stage E
        self.verify(tree, chain)

        elapsed = time.time() - t_start
        return self.result
