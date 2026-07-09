"""
Cryptographic hash utilities.

Provides SHA-256 and Keccak-256 (Solidity-compatible) hash functions,
streaming file hashing for large models, and helpers for constructing
Merkle tree leaf hashes from arbitrary byte sequences.
"""

import hashlib
from typing import Callable, Protocol
from web3 import Web3

CHUNK_SIZE = 8192
"""Default chunk size in bytes for streaming file reads."""


# ---------------------------------------------------------------------------
# Type definitions
# ---------------------------------------------------------------------------

HashFunction = Callable[[bytes], bytes]
"""A hash function that maps arbitrary bytes to a fixed-length digest."""


class HashableDataset(Protocol):
    """Protocol for datasets yielding byte sequences suitable for hashing."""

    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> bytes: ...


# ---------------------------------------------------------------------------
# Raw hash functions
# ---------------------------------------------------------------------------

def sha256(data: bytes) -> bytes:
    """Compute SHA-256 digest of *data*.  Returns 32 bytes."""
    return hashlib.sha256(data).digest()


def sha256_hex(data: bytes) -> str:
    """SHA-256 digest as a 64-char hex string without the '0x' prefix."""
    return hashlib.sha256(data).hexdigest()


def keccak256(data: bytes) -> bytes:
    """
    Compute Keccak-256 digest matching Solidity's ``keccak256``.

    Uses ``Web3.keccak`` which is equivalent to ``keccak256(abi.encodePacked(...))``.
    Returns 32 bytes.
    """
    return Web3.keccak(data)


def keccak256_hex(data: bytes) -> str:
    """Keccak-256 digest as a 64-char hex string without the '0x' prefix."""
    return Web3.keccak(data).hex()


# ---------------------------------------------------------------------------
# Streaming file hashing (for large model checkpoints)
# ---------------------------------------------------------------------------

def streaming_sha256(filepath: str, chunk_size: int = CHUNK_SIZE) -> bytes:
    """
    Compute the SHA-256 digest of a file by reading it in chunks.

    Unlike ``sha256(open(path, 'rb').read())``, this never loads the entire
    file into memory, making it safe for multi-GB model checkpoints.

    Parameters
    ----------
    filepath : str
        Path to the file to hash.
    chunk_size : int
        Read buffer size in bytes (default 8192).

    Returns
    -------
    bytes
        32-byte SHA-256 digest.
    """
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            sha.update(chunk)
    return sha.digest()


def streaming_sha256_hex(filepath: str, chunk_size: int = CHUNK_SIZE) -> str:
    """Like ``streaming_sha256`` but returns a 64-char hex string."""
    return streaming_sha256(filepath, chunk_size).hex()


# ---------------------------------------------------------------------------
# Multi-shard checkpoint hashing
# ---------------------------------------------------------------------------

def merge_shard_hashes(shard_paths: list[str], chunk_size: int = CHUNK_SIZE) -> bytes:
    """
    Compute a single weight hash from multiple shard files.

    Concatenates the SHA-256 digests of each shard in order and hashes
    the result, producing a single 32-byte commitment that covers all
    shards without loading them all into memory at once.

    Parameters
    ----------
    shard_paths : list[str]
        Ordered list of paths to shard files.
    chunk_size : int
        Read buffer size for each shard.

    Returns
    -------
    bytes
        32-byte combined digest.
    """
    sha = hashlib.sha256()
    for path in shard_paths:
        sha.update(streaming_sha256(path, chunk_size))
    return sha.digest()


def merge_shard_hashes_hex(shard_paths: list[str]) -> str:
    """Like ``merge_shard_hashes`` but returns a 64-char hex string."""
    return merge_shard_hashes(shard_paths).hex()


# ---------------------------------------------------------------------------
# Leaf hash (single-bytes interface — no label assumption)
# ---------------------------------------------------------------------------

def leaf_hash(data: bytes, hash_fn: HashFunction = sha256) -> bytes:
    """
    Hash a single data item for use as a Merkle tree leaf.

    The caller is responsible for encoding structured data (labels,
    metadata, etc.) into *data* before calling this function.  This
    keeps the framework model- and modality-agnostic:

        # Image classification
        leaf = leaf_hash(img_bytes + bytes([label]))

        # Text corpus
        leaf = leaf_hash(document.encode())

    Parameters
    ----------
    data : bytes
        Raw bytes to hash.
    hash_fn : HashFunction
        Hash function (default SHA-256).

    Returns
    -------
    bytes
        32-byte leaf hash.
    """
    return hash_fn(data)


# ---------------------------------------------------------------------------
# Internal node hashing (Merkle tree)
# ---------------------------------------------------------------------------

def pair_hash(
    left: bytes,
    right: bytes,
    hash_fn: HashFunction = keccak256,
) -> bytes:
    """
    Hash two sibling nodes into their parent node.

    Siblings are sorted lexicographically before concatenation, matching
    the OpenZeppelin MerkleProof convention and the DataAnchor contract's
    ``verifyMember`` implementation:

        left < right ? keccak256(left || right) : keccak256(right || left)

    Parameters
    ----------
    left : bytes
        Left sibling hash (32 bytes).
    right : bytes
        Right sibling hash (32 bytes).
    hash_fn : HashFunction
        Hash function (default Keccak-256, matching Solidity).

    Returns
    -------
    bytes
        32-byte parent node hash.
    """
    if left < right:
        return hash_fn(left + right)
    return hash_fn(right + left)
