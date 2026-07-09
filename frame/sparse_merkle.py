"""
Sparse Merkle Tree (SMT) with non-membership proofs.

A Sparse Merkle Tree of fixed depth 256 maps each key to a leaf position
determined by ``keccak256(key)``.  Empty leaves default to ``bytes32(0)``.
This supports proofs that a key is **not** present — a capability ordinary
Merkle trees cannot provide.

Builds the tree bottom-up in batches: at each level, only non-empty
subtrees are paired, so the work per level is proportional to the number
of active nodes rather than the full 2^256 space.  Total cost is
O(n * 256) hash operations in the worst case, but for randomly
distributed keys the effective cost is closer to O(n * log n) since path
divergence is shallow.

For production datasets with millions of entries, consider replacing this
module with a native SMT library (e.g. Jellyfish Merkle) for better
performance.

The implementation matches the ``verifyNonMembership`` function in the
DataAnchor Solidity contract.
"""

from typing import NamedTuple

from .hashing import keccak256

DEPTH = 256
"""Fixed depth of the Sparse Merkle Tree."""


# ---------------------------------------------------------------------------
# Default hashes for empty subtrees
# ---------------------------------------------------------------------------

def _compute_defaults() -> list[bytes]:
    """Pre-compute hashes of empty subtrees at every depth."""
    d = [b"\x00" * 32]  # depth 0: empty leaf
    for _ in range(DEPTH):
        d.append(keccak256(d[-1] + d[-1]))
    return d


_DEFAULTS = _compute_defaults()
"""Cached default empty-subtree hashes for depths 0..256."""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class NonMembershipProof(NamedTuple):
    """A Sparse Merkle Tree non-membership proof."""

    key: bytes
    """The key being proved absent (32 bytes)."""

    siblings: list[bytes]
    """Ordered list of 256 sibling hashes, from leaf-level to root-level,
    matching the contract's ``verifyNonMembership`` iteration order."""

    root: bytes
    """The SMT root this proof verifies against."""

    def verify(self) -> bool:
        """
        Verify this proof locally (off-chain).

        Returns
        -------
        bool
            True if the proof is valid.
        """
        path = int.from_bytes(keccak256(self.key), "big")
        computed = b"\x00" * 32
        for level in range(DEPTH):
            bit_pos = 255 - level
            bit = (path >> bit_pos) & 1
            sibling = self.siblings[level]
            if bit == 1:
                computed = keccak256(sibling + computed)
            else:
                computed = keccak256(computed + sibling)
        return computed == self.root


# ---------------------------------------------------------------------------
# Sparse Merkle Tree
# ---------------------------------------------------------------------------

class SparseMerkleTree:
    """
    A Sparse Merkle Tree of fixed depth 256.

    The tree maps each inserted key to a leaf position determined by
    ``uint256(keccak256(key))``.  Empty leaves default to ``bytes32(0)``.

    Construction is O(k · log k) where k is the number of inserted keys.

    Parameters
    ----------
    keys : list[bytes], optional
        Initial keys to insert.

    Attributes
    ----------
    root : bytes
        32-byte Merkle root of the tree.
    size : int
        Number of keys in the tree.

    Examples
    --------
    >>> from frame import SparseMerkleTree
    >>> smt = SparseMerkleTree()
    >>> smt.insert(keccak256(b"sample-key"))
    >>> smt.build()
    >>> proof = smt.get_non_membership_proof(keccak256(b"other-key"))
    >>> proof.verify()
    True
    """

    def __init__(self, keys: list[bytes] | None = None) -> None:
        self._leaves: dict[int, bytes] = {}
        self._nodes: dict[tuple[int, int], bytes] = {}
        self.root = _DEFAULTS[DEPTH]
        if keys:
            for key in keys:
                self._insert_leaf(key)
            self._build()

    # ---- key management ---------------------------------------------

    def insert(self, key: bytes) -> None:
        """Insert a key.  Call ``build()`` after bulk insertion."""
        if len(key) != 32:
            raise ValueError("key must be exactly 32 bytes")
        path = int.from_bytes(keccak256(key), "big")
        self._leaves[path] = key

    def _insert_leaf(self, key: bytes) -> None:
        """Internal: insert without validation (used in __init__)."""
        path = int.from_bytes(keccak256(key), "big")
        self._leaves[path] = key

    def __contains__(self, key: bytes) -> bool:
        path = int.from_bytes(keccak256(key), "big")
        return path in self._leaves

    @property
    def size(self) -> int:
        return len(self._leaves)

    # ---- build ------------------------------------------------------

    def build(self) -> bytes:
        """Build the tree and return the root."""
        self._build()
        return self.root

    def _build(self) -> None:
        """
        Build the SMT bottom-up, grouping nodes by prefix at each level.

        For each level k (leaf=0, root=256), nodes are paired if they
        share the same prefix after clearing bit (255-k).  Only non-empty
        subtrees are stored, so the work per level is proportional to the
        number of active nodes at that level — not to the full 2^256 space.
        """
        self._nodes.clear()
        current: dict[int, bytes] = dict(self._leaves)

        for level in range(DEPTH):
            bit_pos = 255 - level

            # Cache this level's nodes for proof generation
            for pos, h in current.items():
                self._nodes[(level, pos)] = h

            if not current:
                break

            # Group nodes by their prefix (position with bit_pos cleared).
            # Every node in a group shares the same parent position.
            groups: dict[int, dict[int, bytes]] = {}
            for pos, h in current.items():
                prefix = pos & ~(1 << bit_pos)
                groups.setdefault(prefix, {})[pos] = h

            next_level: dict[int, bytes] = {}
            for prefix, group in groups.items():
                pos0 = prefix                       # bit = 0
                pos1 = prefix | (1 << bit_pos)      # bit = 1
                n0 = group.get(pos0, _DEFAULTS[level])
                n1 = group.get(pos1, _DEFAULTS[level])
                next_level[prefix] = keccak256(n0 + n1)

            current = next_level

        self.root = current.get(0, _DEFAULTS[DEPTH])
        self._nodes[(DEPTH, 0)] = self.root

    # ---- proof ------------------------------------------------------

    def get_non_membership_proof(self, key: bytes) -> NonMembershipProof:
        """
        Generate a non-membership proof for a key.

        The key must **not** be in the tree.  Returns a proof that the
        leaf at the key's position is empty.

        Parameters
        ----------
        key : bytes
            32-byte key to prove absent.

        Returns
        -------
        NonMembershipProof

        Raises
        ------
        ValueError
            If the key *is* present in the tree.
        """
        if key in self:
            raise ValueError("key is present in the tree; cannot prove non-membership")
        if not self._nodes:
            self._build()

        path = int.from_bytes(keccak256(key), "big")
        siblings: list[bytes] = []

        for level in range(DEPTH):
            bit_pos = 255 - level
            sibling_pos = path ^ (1 << bit_pos)
            # Match the position masking used during tree construction
            mask = (1 << (256 - level)) - 1
            sibling_hash = self._nodes.get(
                (level, sibling_pos & mask), _DEFAULTS[level]
            )
            siblings.append(sibling_hash)

        return NonMembershipProof(key, siblings, self.root)

    def verify_non_membership(self, proof: NonMembershipProof) -> bool:
        """Verify a non-membership proof (delegates to ``proof.verify()``)."""
        return proof.verify()

    # ---- helpers ----------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"SparseMerkleTree(size={self.size}, "
            f"root=0x{self.root[:8].hex()}...)"
        )
