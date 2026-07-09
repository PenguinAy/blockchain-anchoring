"""
Merkle Tree: construction, proof generation, and verification.

Builds a binary Merkle tree from an ordered list of leaf hashes.
Internal nodes use Keccak-256 with sorted-pair convention (matching
OpenZeppelin and the DataAnchor Solidity contract).

Construction is O(n) in time and O(n) in space.  For datasets too large
to hold in memory, iterate over the data in batches, collect leaf hashes
incrementally, then pass the finished leaf list to the constructor.
"""

from typing import Iterable, NamedTuple
from .hashing import pair_hash


# ---------------------------------------------------------------------------
# Proof data structures
# ---------------------------------------------------------------------------

class MerkleProof(NamedTuple):
    """A Merkle inclusion proof for a single leaf."""

    leaf_index: int
    leaf_hash: bytes
    siblings: list[bytes]
    root: bytes

    def verify(self) -> bool:
        """Verify this proof against its own root."""
        computed = self.leaf_hash
        for sib in self.siblings:
            left, right = (computed, sib) if computed < sib else (sib, computed)
            computed = pair_hash(left, right)
        return computed == self.root


class SubsetProof(NamedTuple):
    """A batch proof for multiple leaves."""

    leaves: list[tuple[int, bytes]]
    proofs: list[list[bytes]]
    root: bytes

    def verify_all(self) -> list[bool]:
        """Verify every proof; returns a list of booleans."""
        return [
            MerkleProof(-1, lh, pf, self.root).verify()
            for (_, lh), pf in zip(self.leaves, self.proofs)
        ]


# ---------------------------------------------------------------------------
# Merkle Tree
# ---------------------------------------------------------------------------

class MerkleTree:
    """
    A binary Merkle tree built over an ordered list of leaf hashes.

    Construction is O(n) in both time and space.  An odd number of leaves
    is handled by duplicating the last leaf (matching the OpenZeppelin
    convention).

    Parameters
    ----------
    leaves : list[bytes]
        Ordered leaf hashes, each 32 bytes.

    Attributes
    ----------
    root : bytes
        32-byte Merkle root.
    depth : int
        Number of levels including the leaf layer (= ceil(log2(n)) + 1).
    leaf_count : int
        Number of leaves in the tree.

    Examples
    --------
    >>> from frame.hashing import sha256, leaf_hash_fn
    >>> items = [(b"data0", b"0"), (b"data1", b"1"), (b"data2", b"2")]
    >>> leaves = [leaf_hash_fn(d, l) for d, l in items]
    >>> tree = MerkleTree(leaves)
    >>> proof = tree.get_proof(1)
    >>> proof.verify()
    True
    """

    def __init__(self, leaves: list[bytes]) -> None:
        if not leaves:
            raise ValueError("leaf list must not be empty")
        self.leaf_count = len(leaves)
        self._layers = self._build_layers(leaves)
        self.root = self._layers[-1][0]
        self.depth = len(self._layers)

    # ---- construction -----------------------------------------------

    @staticmethod
    def _build_layers(leaves: list[bytes]) -> list[list[bytes]]:
        layers = [leaves[:]]
        current = leaves[:]
        while len(current) > 1:
            if len(current) % 2 == 1:
                current.append(current[-1])
            next_layer: list[bytes] = []
            for i in range(0, len(current), 2):
                next_layer.append(pair_hash(current[i], current[i + 1]))
            layers.append(next_layer)
            current = next_layer
        return layers

    # ---- proof generation -------------------------------------------

    def get_proof(self, leaf_index: int) -> MerkleProof:
        """
        Generate an inclusion proof for the leaf at ``leaf_index``.

        Parameters
        ----------
        leaf_index : int
            0-based index into the original leaf list.

        Returns
        -------
        MerkleProof

        Raises
        ------
        IndexError
            If ``leaf_index`` is out of range.
        """
        if leaf_index < 0 or leaf_index >= self.leaf_count:
            raise IndexError(
                f"leaf_index {leaf_index} out of range [0, {self.leaf_count})"
            )

        siblings: list[bytes] = []
        idx = leaf_index
        for layer in self._layers[:-1]:
            sibling_idx = idx + 1 if idx % 2 == 0 else idx - 1
            if sibling_idx < len(layer):
                siblings.append(layer[sibling_idx])
            idx //= 2

        return MerkleProof(
            leaf_index=leaf_index,
            leaf_hash=self._layers[0][leaf_index],
            siblings=siblings,
            root=self.root,
        )

    def get_subset_proof(self, indices: list[int]) -> SubsetProof:
        """
        Generate inclusion proofs for multiple leaves at once.

        Parameters
        ----------
        indices : list[int]
            Sorted list of leaf indices.

        Returns
        -------
        SubsetProof
        """
        entries: list[tuple[int, bytes]] = []
        proofs: list[list[bytes]] = []
        for i in indices:
            p = self.get_proof(i)
            entries.append((i, p.leaf_hash))
            proofs.append(p.siblings)
        return SubsetProof(entries, proofs, self.root)

    # ---- helpers ----------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"MerkleTree(leaf_count={self.leaf_count}, depth={self.depth}, "
            f"root=0x{self.root[:8].hex()}...)"
        )

    @classmethod
    def from_iterable(
        cls,
        items: Iterable[bytes],
        hash_fn: callable = None,
    ) -> "MerkleTree":
        """
        Build a Merkle tree from an iterable of byte strings.

        Each item is hashed with *hash_fn* to produce a leaf.  The caller
        is responsible for serialising structured data into bytes before
        calling this method.

        For very large datasets, construct the leaf list incrementally
        and pass it to the constructor directly.

        Parameters
        ----------
        items : Iterable[bytes]
            Iterable of pre-serialised byte strings.
        hash_fn : callable, optional
            Leaf hash function.  Defaults to ``frame.hashing.leaf_hash``
            which uses SHA-256.

        Returns
        -------
        MerkleTree
        """
        if hash_fn is None:
            from .hashing import leaf_hash as _default
            hash_fn = _default
        leaves = [hash_fn(item) for item in items]
        return cls(leaves)
