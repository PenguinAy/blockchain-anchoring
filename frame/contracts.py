"""
On-chain anchoring via DataAnchor and WeightAnchor contracts.

Provides a high-level client for registering datasets and models,
verifying Merkle proofs, and querying model lineage on any EVM-compatible
blockchain (Hardhat localnet, Sepolia testnet, Ethereum mainnet, L2s).
"""

import json
import sys
from pathlib import Path
from typing import NamedTuple, Optional

from web3 import Web3
from web3.types import TxReceipt


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class ContractAddresses(NamedTuple):
    """Deployed contract addresses."""

    data_anchor: str
    """Address of the DataAnchor contract."""

    weight_anchor: str
    """Address of the WeightAnchor contract."""


class RegistrationResult(NamedTuple):
    """Result of an on-chain registration transaction."""

    tx_hash: str
    """Transaction hash (hex)."""

    gas_used: int
    """Gas consumed by the transaction."""

    block_number: int
    """Block in which the transaction was included."""


# ---------------------------------------------------------------------------
# Anchor Client
# ---------------------------------------------------------------------------

class AnchorClient:
    """
    Client for interacting with the DataAnchor and WeightAnchor contracts.

    Parameters
    ----------
    rpc_url : str
        JSON-RPC endpoint (e.g. ``"http://127.0.0.1:8545"``).
    private_key : str
        Hex-encoded private key of the account used to sign transactions.
    data_anchor_addr : str
        Deployed address of the DataAnchor contract.
    weight_anchor_addr : str
        Deployed address of the WeightAnchor contract.
    artifacts_dir : str or Path, optional
        Path to the Hardhat build artifacts directory.
        Default: ``"blockchain_anchor/artifacts"`` relative to the
        project root (auto-detected).

    Attributes
    ----------
    w3 : Web3
        The underlying Web3 connection.
    account : LocalAccount
        The signing account.
    data_anchor : Contract
        Bound DataAnchor contract instance.
    weight_anchor : Contract
        Bound WeightAnchor contract instance.

    Examples
    --------
    >>> client = AnchorClient(
    ...     rpc_url="http://127.0.0.1:8545",
    ...     private_key="0xac09...",
    ...     data_anchor_addr="0x5FbD...",
    ...     weight_anchor_addr="0xe7f1...",
    ... )
    >>> client.register_dataset(dataset_id, merkle_root, metadata)
    """

    def __init__(
        self,
        rpc_url: str,
        private_key: str,
        data_anchor_addr: str,
        weight_anchor_addr: str,
        artifacts_dir: Optional[str | Path] = None,
    ) -> None:
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        if not self.w3.is_connected():
            raise ConnectionError(f"Cannot connect to RPC endpoint: {rpc_url}")

        self.account = self.w3.eth.account.from_key(private_key)
        self.chain_id = self.w3.eth.chain_id

        # Resolve artifact paths
        if artifacts_dir is None:
            artifacts_dir = self._find_artifacts()
        self._artifacts = Path(artifacts_dir)

        data_abi = self._load_abi("DataAnchor")
        weight_abi = self._load_abi("WeightAnchor")

        self.data_addr = Web3.to_checksum_address(data_anchor_addr)
        self.weight_addr = Web3.to_checksum_address(weight_anchor_addr)

        self.data_anchor = self.w3.eth.contract(
            address=self.data_addr, abi=data_abi
        )
        self.weight_anchor = self.w3.eth.contract(
            address=self.weight_addr, abi=weight_abi
        )

    @staticmethod
    def _find_artifacts() -> Path:
        """Heuristic to locate Hardhat artifacts from known locations."""
        candidates = [
            Path("blockchain_anchor/artifacts"),
            Path("../blockchain_anchor/artifacts"),
            Path(__file__).resolve().parent.parent
            / "blockchain_anchor" / "artifacts",
        ]
        for p in candidates:
            if (p / "contracts" / "DataAnchor.sol" / "DataAnchor.json").exists():
                return p
        raise FileNotFoundError(
            "Cannot find Hardhat artifacts.  "
            "Pass artifacts_dir=... or run 'npx hardhat compile'."
        )

    def _load_abi(self, name: str) -> list:
        path = (
            self._artifacts / "contracts" / f"{name}.sol" / f"{name}.json"
        )
        if not path.exists():
            raise FileNotFoundError(f"ABI not found: {path}")
        with open(path) as f:
            return json.load(f)["abi"]

    def _build_and_send(self, tx_fn, gas: int = 300_000) -> RegistrationResult:
        """Build, sign, send a transaction, and return its receipt."""
        nonce = self.w3.eth.get_transaction_count(self.account.address)
        tx = tx_fn.build_transaction({
            "from": self.account.address,
            "nonce": nonce,
            "gas": gas,
            "gasPrice": self.w3.eth.gas_price,
            "chainId": self.chain_id,
        })
        signed = self.w3.eth.account.sign_transaction(tx, self.account.key)
        tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        return RegistrationResult(
            tx_hash=tx_hash.hex(),
            gas_used=receipt.gasUsed,
            block_number=receipt.blockNumber,
        )

    # === Dataset operations ===========================================

    def register_dataset(
        self,
        dataset_id: bytes,
        merkle_root: bytes,
        metadata: str,
    ) -> RegistrationResult:
        """
        Register a dataset Merkle root on-chain.

        Parameters
        ----------
        dataset_id : bytes
            32-byte unique identifier (e.g. ``keccak256("MNIST-train-v1")``).
        merkle_root : bytes
            32-byte Merkle root of the dataset.
        metadata : str
            JSON string with dataset description (name, size, etc.).

        Returns
        -------
        RegistrationResult

        Raises
        ------
        Web3RPCError
            If the dataset is already registered.
        """
        return self._build_and_send(
            self.data_anchor.functions.registerDataset(
                dataset_id, merkle_root, metadata
            ),
            gas=300_000,
        )

    def verify_dataset(self, dataset_id: bytes, local_root: bytes) -> bool:
        """Check that *local_root* matches the on-chain Merkle root."""
        return self.data_anchor.functions.verifyDataset(
            dataset_id, local_root
        ).call()

    def verify_member(
        self,
        dataset_id: bytes,
        leaf: bytes,
        proof: list[bytes],
    ) -> bool:
        """
        Verify a single Merkle inclusion proof on-chain.

        Parameters
        ----------
        dataset_id : bytes
            The dataset identifier.
        leaf : bytes
            32-byte leaf hash.
        proof : list[bytes]
            Ordered sibling hashes from the Merkle proof.

        Returns
        -------
        bool
        """
        return self.data_anchor.functions.verifyMember(
            dataset_id, leaf, proof
        ).call()

    def verify_subset_all(
        self,
        dataset_id: bytes,
        leaves: list[bytes],
        proofs: list[list[bytes]],
    ) -> bool:
        """
        Verify multiple inclusion proofs in a single call.

        Parameters
        ----------
        dataset_id : bytes
        leaves : list[bytes]
            Leaf hashes to verify.
        proofs : list[list[bytes]]
            Corresponding Merkle proof for each leaf.

        Returns
        -------
        bool
            True only if every leaf is verified.
        """
        return self.data_anchor.functions.verifySubsetAll(
            dataset_id, leaves, proofs
        ).call()

    def verify_non_membership(
        self,
        smt_root: bytes,
        key: bytes,
        siblings: tuple[bytes, ...],
    ) -> bool:
        """
        Verify a Sparse Merkle Tree non-membership proof on-chain.

        Parameters
        ----------
        smt_root : bytes
            Root of the Sparse Merkle Tree.
        key : bytes
            32-byte key to prove absent.
        siblings : tuple[bytes, ...]
            256 sibling hashes (fixed-size Solidity array).

        Returns
        -------
        bool
        """
        return self.data_anchor.functions.verifyNonMembership(
            smt_root, key, siblings
        ).call()

    # === Model operations =============================================

    def register_model(
        self,
        model_id: bytes,
        weight_hash: bytes,
        dataset_id: bytes,
        chain_tip: bytes,
        parent_model_id: bytes,
        model_meta: str,
        version: str,
    ) -> RegistrationResult:
        """
        Register a model on-chain.

        Parameters
        ----------
        model_id : bytes
            32-byte unique model identifier.
        weight_hash : bytes
            SHA-256 of the final model weights.
        dataset_id : bytes
            Identifier of the training dataset (must be registered).
        chain_tip : bytes
            Training chain tail C_E (see ``TrainingChain``).
        parent_model_id : bytes
            Parent model ID, or ``b'\\x00'*32`` for a root model.
        model_meta : str
            JSON string with model metadata (architecture, hyperparameters).
        version : str
            Semantic version string.

        Returns
        -------
        RegistrationResult
        """
        return self._build_and_send(
            self.weight_anchor.functions.registerModel(
                model_id,
                weight_hash,
                dataset_id,
                chain_tip,
                parent_model_id,
                model_meta,
                version,
            ),
            gas=500_000,
        )

    def verify_model(self, model_id: bytes, local_weight_hash: bytes) -> bool:
        """Check that *local_weight_hash* matches the on-chain record."""
        return self.weight_anchor.functions.verifyModel(
            model_id, local_weight_hash
        ).call()

    def verify_training_chain(
        self, model_id: bytes, local_chain_tip: bytes
    ) -> bool:
        """Check that *local_chain_tip* matches the on-chain chainTip."""
        return self.weight_anchor.functions.verifyTrainingChain(
            model_id, local_chain_tip
        ).call()

    # === Lineage queries ==============================================

    def get_lineage(self, model_id: bytes) -> list[bytes]:
        """
        Return the full ancestor chain of a model.

        Walk upward from *model_id* following ``parentModelId`` until
        reaching a root model (``parentModelId == 0``).
        """
        return self.weight_anchor.functions.getLineage(model_id).call()

    def get_children(self, model_id: bytes) -> list[bytes]:
        """Return the direct child models of *model_id*."""
        return self.weight_anchor.functions.getChildren(model_id).call()

    def verify_lineage(self, child_id: bytes, ancestor_id: bytes) -> bool:
        """Check that *ancestor_id* is in the lineage of *child_id*."""
        return self.weight_anchor.functions.verifyLineage(
            child_id, ancestor_id
        ).call()

    def get_models_by_dataset(self, dataset_id: bytes) -> list[bytes]:
        """Return all model IDs trained on a given dataset."""
        return self.weight_anchor.functions.getModelsByDataset(
            dataset_id
        ).call()

    # === Helpers ======================================================

    @property
    def address(self) -> str:
        return self.account.address

    @property
    def balance_eth(self) -> float:
        return float(
            self.w3.from_wei(
                self.w3.eth.get_balance(self.account.address), "ether"
            )
        )

    def __repr__(self) -> str:
        return (
            f"AnchorClient(account={self.address[:10]}..., "
            f"chain_id={self.chain_id})"
        )
