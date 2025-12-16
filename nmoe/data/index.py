"""
Index file format for nmoe data shards.

Format (.idx):
  Header (32 bytes):
    magic:      b"NMOEIDX\\x00"    (8 bytes)
    version:    uint64            (8 bytes) = 1
    num_docs:   uint64            (8 bytes)
    reserved:   uint64            (8 bytes) = 0

  Body:
    doc_boundaries: uint64[num_docs * 2]  # (start_idx, end_idx) pairs

Compatible with OLMo-core style document indices but with explicit header.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Iterator

import numpy as np

# Magic bytes for nmoe index files
INDEX_MAGIC = b"NMOEIDX\x00"
INDEX_VERSION = 1
HEADER_SIZE = 32  # 8 + 8 + 8 + 8 bytes


@dataclass
class IndexHeader:
    """Index file header."""
    version: int
    num_docs: int

    def to_bytes(self) -> bytes:
        return struct.pack(
            "<8sQQQ",
            INDEX_MAGIC,
            self.version,
            self.num_docs,
            0,  # reserved
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "IndexHeader":
        magic, version, num_docs, _ = struct.unpack("<8sQQQ", data[:HEADER_SIZE])
        if magic != INDEX_MAGIC:
            raise ValueError(f"Invalid index magic: {magic!r}, expected {INDEX_MAGIC!r}")
        if version != INDEX_VERSION:
            raise ValueError(f"Unsupported index version: {version}, expected {INDEX_VERSION}")
        return cls(version=version, num_docs=num_docs)


class IndexWriter:
    """Write document boundary indices to .idx file."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._boundaries: List[Tuple[int, int]] = []

    def add_document(self, start_idx: int, end_idx: int) -> None:
        """Add a document boundary (start inclusive, end exclusive)."""
        self._boundaries.append((start_idx, end_idx))

    def finalize(self) -> int:
        """Write index file and return number of documents."""
        header = IndexHeader(version=INDEX_VERSION, num_docs=len(self._boundaries))

        # Convert to numpy array
        boundaries = np.array(self._boundaries, dtype=np.uint64).flatten()

        # Write atomically via temp file
        tmp_path = self.path.with_suffix(".idx.tmp")
        with open(tmp_path, "wb") as f:
            f.write(header.to_bytes())
            f.write(boundaries.tobytes())

        tmp_path.replace(self.path)
        return len(self._boundaries)


class IndexReader:
    """Read document boundary indices from .idx file."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._header: IndexHeader | None = None
        self._boundaries: np.ndarray | None = None

    @property
    def header(self) -> IndexHeader:
        if self._header is None:
            with open(self.path, "rb") as f:
                self._header = IndexHeader.from_bytes(f.read(HEADER_SIZE))
        return self._header

    @property
    def num_docs(self) -> int:
        return self.header.num_docs

    def _load_boundaries(self) -> np.ndarray:
        if self._boundaries is None:
            with open(self.path, "rb") as f:
                f.seek(HEADER_SIZE)
                data = f.read()
            self._boundaries = np.frombuffer(data, dtype=np.uint64).reshape(-1, 2)
        return self._boundaries

    def get_document(self, doc_idx: int) -> Tuple[int, int]:
        """Get (start_idx, end_idx) for document."""
        boundaries = self._load_boundaries()
        if doc_idx < 0 or doc_idx >= len(boundaries):
            raise IndexError(f"Document index {doc_idx} out of range [0, {len(boundaries)})")
        return int(boundaries[doc_idx, 0]), int(boundaries[doc_idx, 1])

    def iter_documents(self) -> Iterator[Tuple[int, int]]:
        """Iterate over all document boundaries."""
        boundaries = self._load_boundaries()
        for i in range(len(boundaries)):
            yield int(boundaries[i, 0]), int(boundaries[i, 1])

    def to_array(self) -> np.ndarray:
        """Return all boundaries as (N, 2) array."""
        return self._load_boundaries().copy()


def index_path_for_shard(shard_path: Path) -> Path:
    """Get index path for a shard (.npy → .idx)."""
    return shard_path.with_suffix(".idx")


def regenerate_index_from_shard(
    shard_path: Path,
    eos_token_id: int,
    dtype: type = np.uint32,
) -> Path:
    """Regenerate .idx file from .npy shard by scanning for EOS tokens.

    Useful for recovery if index file is lost/corrupted.
    """
    tokens = np.load(shard_path, mmap_mode="r")
    if tokens.dtype != dtype:
        tokens = tokens.astype(dtype)

    # Find all EOS positions
    eos_positions = np.where(tokens == eos_token_id)[0]

    # Build document boundaries
    idx_path = index_path_for_shard(shard_path)
    writer = IndexWriter(idx_path)

    start = 0
    for eos_pos in eos_positions:
        end = int(eos_pos) + 1  # Include EOS in document
        writer.add_document(start, end)
        start = end

    writer.finalize()
    return idx_path
