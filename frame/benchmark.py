"""
Performance and gas benchmark utilities.

Measures Merkle proof verification time and cost across varying dataset
sizes to generate the scaling data required by the accompanying paper
(Figure 5-2: Gas vs. leaf count, Figure 5-3: proof type comparison).
"""

import json
import time
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

from .hashing import leaf_hash
from .merkle import MerkleTree
from .sparse_merkle import SparseMerkleTree
from .contracts import AnchorClient


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkPoint:
    """A single benchmark data point."""

    leaf_count: int
    """Number of leaves in the tree."""

    tree_depth: int
    """Actual tree depth (= ceil(log2(n)) + 1)."""

    proof_length: int
    """Number of sibling hashes in a single Merkle proof."""

    avg_time_ms: float
    """Average verification time over repeated runs."""

    repeats: int = 3
    """Number of repeated measurements."""


@dataclass
class ProofTypeComparison:
    """Comparison of single, subset, and non-membership proofs at one scale."""

    leaf_count: int

    single_time_ms: float = 0.0
    single_passed: bool = False

    subset_size: int = 0
    subset_time_ms: float = 0.0
    subset_passed: bool = False

    non_member_time_ms: float = 0.0
    non_member_passed: bool = False


@dataclass
class BenchmarkResult:
    """Aggregated benchmark results."""

    merkle_scaling: list[BenchmarkPoint] = field(default_factory=list)
    proof_comparison: list[ProofTypeComparison] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "merkle_scaling": [
                {
                    "leaf_count": p.leaf_count,
                    "tree_depth": p.tree_depth,
                    "proof_length": p.proof_length,
                    "avg_time_ms": p.avg_time_ms,
                }
                for p in self.merkle_scaling
            ],
            "proof_comparison": [
                {
                    "leaf_count": c.leaf_count,
                    "single_ms": c.single_time_ms,
                    "single_ok": c.single_passed,
                    "subset_ms": c.subset_time_ms,
                    "subset_ok": c.subset_passed,
                    "non_member_ms": c.non_member_time_ms,
                    "non_member_ok": c.non_member_passed,
                }
                for c in self.proof_comparison
            ],
        }


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """
    Run benchmark measurements for the anchoring framework.

    Parameters
    ----------
    client : AnchorClient
        Connected client with deployed contracts.
    """

    def __init__(self, client: AnchorClient) -> None:
        self.client = client

    def measure_merkle_scaling(
        self,
        sizes: list[int] = (16, 256, 4096, 60000),
        repeats: int = 3,
    ) -> list[BenchmarkPoint]:
        """
        Measure Merkle proof verification time as a function of leaf count.

        Constructs a Merkle tree of each size, registers it on-chain, and
        measures the mean ``verifyMember`` call time over *repeats* trials.

        Parameters
        ----------
        sizes : list[int]
            Leaf counts to benchmark.
        repeats : int
            Measurements per size.

        Returns
        -------
        list[BenchmarkPoint]
        """
        points: list[BenchmarkPoint] = []
        for n in sizes:
            # Generate synthetic leaves
            leaves = [
                leaf_hash(f"item-{i}-{n}".encode() + str(i).encode())
                for i in range(n)
            ]
            tree = MerkleTree(leaves)

            dataset_id = self.client.w3.keccak(text=f"bench-{n}-{time.time()}")
            try:
                self.client.register_dataset(
                    dataset_id, tree.root, json.dumps({"name": "bench", "size": n})
                )
            except Exception:
                pass  # may already exist

            leaf = tree._layers[0][0]
            proof = tree.get_proof(0)

            times: list[float] = []
            for _ in range(repeats):
                t0 = time.time()
                self.client.verify_member(
                    dataset_id, leaf, [bytes(s) for s in proof.siblings]
                )
                times.append((time.time() - t0) * 1000)

            avg = sum(times) / len(times)
            points.append(
                BenchmarkPoint(
                    leaf_count=n,
                    tree_depth=tree.depth,
                    proof_length=len(proof.siblings),
                    avg_time_ms=round(avg, 1),
                    repeats=repeats,
                )
            )
        return points

    def compare_proof_types(
        self,
        sizes: list[int] = (1000, 10000, 60000),
        subset_size: int = 20,
    ) -> list[ProofTypeComparison]:
        """
        Compare single, subset, and non-membership proof performance.

        Parameters
        ----------
        sizes : list[int]
            Dataset sizes to benchmark.
        subset_size : int
            Number of leaves in the subset proof.

        Returns
        -------
        list[ProofTypeComparison]
        """
        comparisons: list[ProofTypeComparison] = []
        for n in sizes:
            comp = ProofTypeComparison(leaf_count=n, subset_size=subset_size)

            leaves = [
                leaf_hash(f"item-{i}-{n}".encode() + str(i).encode())
                for i in range(n)
            ]
            tree = MerkleTree(leaves)

            # Single proof
            proof = tree.get_proof(0)
            t0 = time.time()
            comp.single_passed = proof.verify()
            comp.single_time_ms = round((time.time() - t0) * 1000, 1)

            # Subset proof
            import random
            random.seed(42)
            indices = sorted(random.sample(range(n), min(subset_size, n)))
            subset = tree.get_subset_proof(indices)
            t1 = time.time()
            results = subset.verify_all()
            comp.subset_passed = all(results)
            comp.subset_time_ms = round((time.time() - t1) * 1000, 1)

            # Non-membership proof (SMT).
            # Cap at 500: the pure-Python SMT implementation is O(n*256)
            # hash operations and is not intended for production-scale data.
            # See sparse_merkle.py docstring for details.
            smt = SparseMerkleTree()
            for i in range(min(n, 500)):
                key = leaf_hash(f"item-{i}-{n}".encode() + str(i).encode())
                smt.insert(key)
            smt.build()
            stranger = leaf_hash(b"not-in-set" + b"0")
            t2 = time.time()
            proof_nm = smt.get_non_membership_proof(stranger)
            comp.non_member_passed = proof_nm.verify()
            comp.non_member_time_ms = round((time.time() - t2) * 1000, 1)

            comparisons.append(comp)
        return comparisons

    def run(self) -> BenchmarkResult:
        """Run full benchmark suite."""
        result = BenchmarkResult()
        result.merkle_scaling = self.measure_merkle_scaling()
        result.proof_comparison = self.compare_proof_types()
        return result
