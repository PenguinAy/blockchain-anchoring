"""
Model version lineage (DAG) management.

Tracks parent-child relationships between registered models, forming
an immutable directed acyclic graph (DAG) anchored on-chain via the
``parentModelId`` field in WeightAnchor.

The DAG supports the full lifecycle of model derivation:
fine-tuning, distillation, pruning, merging, etc.
"""

from dataclasses import dataclass, field
from typing import Optional

from web3 import Web3


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LineageNode:
    """A single node in the model lineage DAG."""

    model_id: bytes
    """32-byte on-chain model identifier."""

    name: str
    """Human-readable model name."""

    version: str
    """Semantic version string."""

    parent_model_id: bytes
    """32-byte parent identifier, or ``b'\\x00'*32`` for root."""

    dataset_id: bytes
    """32-byte training dataset identifier."""

    weight_hash: bytes
    """SHA-256 of the final weights."""

    chain_tip: bytes
    """Training hash chain tip C_E."""

    metadata: str
    """JSON metadata string."""

    children: list["LineageNode"] = field(default_factory=list)
    """Direct child nodes (populated during traversal)."""

    @property
    def is_root(self) -> bool:
        """True if this model has no parent."""
        return self.parent_model_id == b"\x00" * 32

    @property
    def model_id_hex(self) -> str:
        return "0x" + self.model_id.hex()


# ---------------------------------------------------------------------------
# Model Lineage
# ---------------------------------------------------------------------------

class ModelLineage:
    """
    Manages a collection of registered models and their parent-child
    relationships.

    Models are registered on-chain through ``AnchorClient``; this class
    provides the in-memory DAG representation and traversal helpers.

    Parameters
    ----------
    models : list[LineageNode], optional
        Initial set of lineage nodes.

    Examples
    --------
    >>> lineage = ModelLineage()
    >>> lineage.add(root_node)
    >>> lineage.add(child_node)   # parent_model_id points to root
    >>> ancestors = lineage.ancestors(child_node.model_id)
    """

    def __init__(self, models: Optional[list[LineageNode]] = None) -> None:
        self._nodes: dict[bytes, LineageNode] = {}
        self._children: dict[bytes, list[bytes]] = {}
        if models:
            for m in models:
                self.add(m)

    # ---- mutations ---------------------------------------------------

    def add(self, node: LineageNode) -> None:
        """Register a lineage node and update parent-child links."""
        self._nodes[node.model_id] = node
        if node.parent_model_id != b"\x00" * 32:
            self._children.setdefault(node.parent_model_id, []).append(
                node.model_id
            )

    # ---- queries -----------------------------------------------------

    def get(self, model_id: bytes) -> Optional[LineageNode]:
        """Return the node for *model_id*, or None."""
        return self._nodes.get(model_id)

    def ancestors(self, model_id: bytes) -> list[LineageNode]:
        """
        Return the ancestry chain from *model_id* up to the root.

        The result is ordered from parent upward (first element is the
        immediate parent, last is the root).
        """
        result: list[LineageNode] = []
        current = self._nodes.get(model_id)
        while current is not None and not current.is_root:
            parent = self._nodes.get(current.parent_model_id)
            if parent is None:
                break
            result.append(parent)
            current = parent
        return result

    def children(self, model_id: bytes) -> list[LineageNode]:
        """Return the direct children of *model_id*."""
        child_ids = self._children.get(model_id, [])
        return [self._nodes[cid] for cid in child_ids if cid in self._nodes]

    def descendants(self, model_id: bytes) -> list[LineageNode]:
        """
        Return all descendants of *model_id* (BFS traversal).

        The result excludes *model_id* itself.
        """
        result: list[LineageNode] = []
        queue = list(self._children.get(model_id, []))
        while queue:
            cid = queue.pop(0)
            node = self._nodes.get(cid)
            if node is None:
                continue
            result.append(node)
            queue.extend(self._children.get(cid, []))
        return result

    def is_ancestor(self, ancestor_id: bytes, child_id: bytes) -> bool:
        """True if *ancestor_id* is in the lineage of *child_id*."""
        anc = self.ancestors(child_id)
        return any(a.model_id == ancestor_id for a in anc)

    def roots(self) -> list[LineageNode]:
        """Return all root models (no parent)."""
        return [n for n in self._nodes.values() if n.is_root]

    @property
    def size(self) -> int:
        return len(self._nodes)

    # ---- export ------------------------------------------------------

    def to_dict(self) -> dict:
        """Export the lineage as a JSON-serialisable dict."""
        nodes = {}
        for mid, node in self._nodes.items():
            nodes[mid.hex()] = {
                "name": node.name,
                "version": node.version,
                "parent_model_id": "0x" + node.parent_model_id.hex(),
                "dataset_id": "0x" + node.dataset_id.hex(),
                "weight_hash": "0x" + node.weight_hash.hex(),
                "chain_tip": "0x" + node.chain_tip.hex(),
                "children": [c.hex() for c in self._children.get(mid, [])],
            }
        return {
            "size": self.size,
            "roots": [r.model_id.hex() for r in self.roots()],
            "nodes": nodes,
        }

    def __repr__(self) -> str:
        return f"ModelLineage(models={self.size}, roots={len(self.roots())})"


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def make_model_id(name: str) -> bytes:
    """
    Derive a deterministic 32-byte model ID from a name.

    Uses ``keccak256(name)``, matching the convention in the Solidity
    contracts and upload scripts.
    """
    return Web3.keccak(text=name)


def make_dataset_id(name: str) -> bytes:
    """Derive a deterministic 32-byte dataset ID from a name."""
    return Web3.keccak(text=name)
