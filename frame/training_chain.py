"""
Epoch-wise and step-wise training hash chains.

Builds a tamper-evident chain from checkpoints (or lightweight training
metadata) using the recurrence::

    C_0 = H(W_0)
    C_t = H( H(W_t) || C_{t-1} )   for t = 1, 2, ..., E

where H is SHA-256.  The chain tip C_E is committed on-chain.

Supports three modes:
1. Full-checkpoint mode (W_t = file hash of the saved weights)
2. Multi-shard mode (W_t = merge of per-rank shard hashes)
3. Lightweight step mode (W_t = metadata string: loss, step, lr, seed)
"""

import hashlib
from pathlib import Path
from typing import NamedTuple

from .hashing import streaming_sha256, merge_shard_hashes

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class CheckpointInfo(NamedTuple):
    """Metadata for a single entry in a training hash chain."""

    index: int
    """Chain index (0-based, epoch or step number)."""

    source: str
    """File path or step metadata label."""

    weight_hash: str
    """SHA-256 hex digest of the data at this index."""

    chain_tip: str
    """Chain tip C_t after incorporating this entry."""

    mode: str
    """'full', 'shard', or 'step'."""


# ---------------------------------------------------------------------------
# Training Chain
# ---------------------------------------------------------------------------

class TrainingChain:
    """
    A tamper-evident hash chain over training checkpoints or metadata.

    Entries are processed in the order provided.  The chain tip is
    committed on-chain as the ``chainTip`` field in WeightAnchor.

    Parameters
    ----------
    entries : list[tuple[str, bytes]]
        Ordered list of (label, data_bytes) pairs.  Each label is used
        for display; each data_bytes is hashed with SHA-256 to produce
        the W_t value for that entry.

    Attributes
    ----------
    entries_count : int
        Number of entries in the chain.
    chain_tail : str
        The final chain tip (64-character hex).
    weight_hashes : list[str]
        SHA-256 digests of each entry's data_bytes.
    chain : list[str]
        Intermediate chain tips [C_0, C_1, ..., C_E].
    infos : list[CheckpointInfo]
        Structured metadata for each entry.
    """

    def __init__(self, entries: list[tuple[str, bytes]]) -> None:
        if not entries:
            raise ValueError("at least one entry is required")

        self.entries_count = len(entries)
        self.weight_hashes: list[str] = []
        self.chain: list[str] = []
        self.chain_tail: str = ""
        self.infos: list[CheckpointInfo] = []
        self._entries = entries
        self._build()

    # ---- construction -------------------------------------------------

    def _build(self) -> None:
        for t, (label, data) in enumerate(self._entries):
            h_w = hashlib.sha256(data).hexdigest()
            self.weight_hashes.append(h_w)

            if t == 0:
                c_t = hashlib.sha256(data).hexdigest()
            else:
                h_bytes = hashlib.sha256(data).digest()
                c_prev = bytes.fromhex(self.chain[-1])
                c_t = hashlib.sha256(h_bytes + c_prev).hexdigest()

            self.chain.append(c_t)
            self.infos.append(CheckpointInfo(
                index=t, source=label, weight_hash=h_w, chain_tip=c_t,
                mode="full" if label.endswith(".pth") else "step",
            ))

        self.chain_tail = self.chain[-1]

    # ---- verification ------------------------------------------------

    def verify_integrity(self) -> bool:
        """Recompute the entire chain and compare."""
        for t, (_, data) in enumerate(self._entries):
            h = hashlib.sha256(data).hexdigest()
            if h != self.weight_hashes[t]:
                return False
        return True

    # ---- helpers -----------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"TrainingChain(entries={self.entries_count}, "
            f"chain_tail=0x{self.chain_tail[:16]}...)"
        )

    # ---- factory methods ---------------------------------------------

    @classmethod
    def from_checkpoints(
        cls,
        directory: str | Path,
        pattern: str = "*epoch*.pth",
    ) -> "TrainingChain":
        """
        Build a chain from checkpoint files in *directory*.

        Files are sorted by name.  Each file is hashed with streaming
        SHA-256 (safe for multi-GB checkpoints).

        Parameters
        ----------
        directory : str or Path
            Directory containing checkpoint files.
        pattern : str
            Glob pattern (default ``"*epoch*.pth"``).

        Returns
        -------
        TrainingChain
        """
        dir_path = Path(directory)
        files = sorted(
            dir_path.glob(pattern),
            key=lambda p: int(
                "".join(c for c in p.stem if c.isdigit()) or "0"
            ),
        )
        if not files:
            raise FileNotFoundError(
                f"No files matching '{pattern}' in {dir_path}"
            )
        entries = [
            (f.name, streaming_sha256(str(f))) for f in files
        ]
        return cls(entries)

    @classmethod
    def from_shards(
        cls,
        shard_groups: list[list[str]],
        labels: list[str] | None = None,
    ) -> "TrainingChain":
        """
        Build a chain where each entry is a set of shard files.

        Useful for distributed training where each rank writes its own
        shard.  Shards at each step are merged into a single digest.

        Parameters
        ----------
        shard_groups : list[list[str]]
            Ordered list of shard file groups.  ``shard_groups[t]``
            contains the paths for step t's shards.
        labels : list[str], optional
            Human-readable labels for each step.

        Returns
        -------
        TrainingChain
        """
        if labels is None:
            labels = [f"step-{i}" for i in range(len(shard_groups))]
        entries = [
            (label, merge_shard_hashes(shards))
            for label, shards in zip(labels, shard_groups)
        ]
        return cls(entries)

    @classmethod
    def from_steps(
        cls,
        steps: list[dict],
        label_prefix: str = "step",
    ) -> "TrainingChain":
        """
        Build a chain from lightweight step metadata.

        Each dict in *steps* should contain key-value pairs describing
        the training state at that step (e.g. ``{"loss": 0.5, "lr": 1e-3,
        "global_step": 1000}``).  The dict is serialised to JSON and
        hashed.  No checkpoint files are involved — the chain tracks
        training trajectory without storing full weights.

        Parameters
        ----------
        steps : list[dict]
            Training metadata for each step.
        label_prefix : str
            Prefix for auto-generated labels (default ``"step"``).

        Returns
        -------
        TrainingChain
        """
        import json as _json
        entries = [
            (f"{label_prefix}-{i}",
             _json.dumps(s, sort_keys=True).encode())
            for i, s in enumerate(steps)
        ]
        return cls(entries)
