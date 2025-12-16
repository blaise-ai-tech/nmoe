"""
Data source adapters for streaming from HuggingFace Datasets.

Supports:
- HuggingFace Hub datasets (streaming mode)
- Local JSONL files
- Local text files

All sources yield (doc_id, text) pairs for downstream processing.
"""
from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Tuple, Any, Dict, Optional, List

# Lazy imports for optional dependencies
_datasets = None
_pyarrow = None


def _get_datasets():
    global _datasets
    if _datasets is None:
        import datasets
        _datasets = datasets
    return _datasets


def _get_pyarrow():
    global _pyarrow
    if _pyarrow is None:
        import pyarrow as pa
        _pyarrow = pa
    return _pyarrow


@dataclass
class Document:
    """A document with ID and text content."""
    doc_id: str
    text: str
    metadata: Dict[str, Any] | None = None

    def __post_init__(self):
        if not self.doc_id:
            # Generate ID from text hash if not provided
            self.doc_id = hashlib.md5(self.text.encode("utf-8")).hexdigest()[:16]


class DataSource(ABC):
    """Abstract base class for data sources."""

    @abstractmethod
    def __iter__(self) -> Iterator[Document]:
        """Yield documents from the source."""
        pass

    @abstractmethod
    def estimate_size(self) -> int | None:
        """Estimate number of documents (None if unknown)."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Source name for logging and manifests."""
        pass


class HuggingFaceSource(DataSource):
    """Stream documents from a HuggingFace dataset.

    Example:
        # Load entire dataset
        source = HuggingFaceSource(
            dataset="HuggingFaceFW/fineweb-edu",
            split="train",
            text_field="text",
        )

        # Load specific files using data_files pattern
        source = HuggingFaceSource(
            dataset="allenai/dolma3_mix-6T-1025",
            split="train",
            text_field="text",
            data_files="data/common_crawl-*/**/*.jsonl.zst",
        )

        for doc in source:
            print(doc.text[:100])
    """

    def __init__(
        self,
        dataset: str,
        split: str = "train",
        text_field: str = "text",
        id_field: str | None = None,
        subset: str | None = None,
        streaming: bool = True,
        trust_remote_code: bool = False,
        num_proc: int | None = None,
        data_files: str | List[str] | None = None,
    ):
        self.dataset = dataset
        self.split = split
        self.text_field = text_field
        self.id_field = id_field
        self.subset = subset
        self.streaming = streaming
        self.trust_remote_code = trust_remote_code
        self.num_proc = num_proc
        self.data_files = data_files
        self._size: int | None = None
        self._shard_num_shards: int | None = None
        self._shard_index: int | None = None

    def shard(self, *, num_shards: int, index: int) -> "HuggingFaceSource":
        """Return a deterministically sharded view of this source.

        This is designed for K8s Indexed Jobs: each completion uses the same
        config, but a different `index` and a shared `num_shards`.
        """
        if num_shards <= 0:
            raise ValueError(f"num_shards must be > 0 (got {num_shards})")
        if index < 0 or index >= num_shards:
            raise ValueError(f"index out of range: index={index} num_shards={num_shards}")

        out = HuggingFaceSource(
            dataset=self.dataset,
            split=self.split,
            text_field=self.text_field,
            id_field=self.id_field,
            subset=self.subset,
            streaming=self.streaming,
            trust_remote_code=self.trust_remote_code,
            num_proc=self.num_proc,
            data_files=self.data_files,
        )
        out._size = self._size
        out._shard_num_shards = num_shards
        out._shard_index = index
        return out

    @property
    def name(self) -> str:
        if self.subset:
            return f"{self.dataset}/{self.subset}"
        if self.data_files:
            # Extract source name from data_files pattern
            # e.g., "data/common_crawl-*/**/*.jsonl.zst" -> "common_crawl"
            pattern = self.data_files if isinstance(self.data_files, str) else self.data_files[0]
            parts = pattern.replace("data/", "").split("-")[0].split("/")[0]
            return f"{self.dataset}:{parts}"
        return self.dataset

    def estimate_size(self) -> int | None:
        return self._size

    def __iter__(self) -> Iterator[Document]:
        datasets = _get_datasets()

        # Build load_dataset kwargs
        load_kwargs = {
            "path": self.dataset,
            "split": self.split,
            "streaming": self.streaming,
            "trust_remote_code": self.trust_remote_code,
        }

        if self.subset:
            load_kwargs["name"] = self.subset
        if self.data_files:
            load_kwargs["data_files"] = self.data_files
        if self.num_proc and not self.streaming:
            load_kwargs["num_proc"] = self.num_proc

        # Load dataset
        ds = datasets.load_dataset(**load_kwargs)
        if self._shard_num_shards is not None:
            ds = ds.shard(num_shards=int(self._shard_num_shards), index=int(self._shard_index))

        # Try to get size info
        if hasattr(ds, "info") and ds.info.splits:
            split_info = ds.info.splits.get(self.split)
            if split_info:
                self._size = split_info.num_examples

        idx = 0
        for example in ds:
            text = example.get(self.text_field, "")
            if not text:
                continue

            # Get or generate doc ID
            if self.id_field and self.id_field in example:
                doc_id = str(example[self.id_field])
            else:
                if self._shard_num_shards is None:
                    doc_id = f"{self.name}:{idx}"
                else:
                    # When sharded via IterableDataset.shard, items are selected by
                    # taking every `num_shards`-th example, starting at `index`.
                    # This recovers a stable global index for deterministic doc_ids.
                    global_idx = idx * int(self._shard_num_shards) + int(self._shard_index)
                    doc_id = f"{self.name}:{global_idx}"

            # Collect metadata (excluding text field)
            metadata = {k: v for k, v in example.items() if k != self.text_field}

            yield Document(doc_id=doc_id, text=text, metadata=metadata if metadata else None)
            idx += 1


class HfFileSystemSource(DataSource):
    """Stream documents directly from HuggingFace Hub using HfFileSystem.

    Bypasses datasets library to avoid pyarrow schema inference issues.
    Use this for datasets with heterogeneous metadata types (e.g., dolma3_mix).

    Example:
        source = HfFileSystemSource(
            repo_id="allenai/dolma3_mix-6T-1025",
            data_files="data/olmocr_science_pdfs-*/*.jsonl.zst",
        )
    """

    def __init__(
        self,
        repo_id: str,
        data_files: str,
        text_field: str = "text",
        id_field: str | None = "id",
    ):
        self.repo_id = repo_id
        self.data_files = data_files
        self.text_field = text_field
        self.id_field = id_field
        self._files: List[str] | None = None

    @property
    def name(self) -> str:
        # Extract source name from data_files pattern
        pattern = self.data_files.replace("data/", "").split("-")[0].split("/")[0]
        return f"{self.repo_id}:{pattern}"

    def estimate_size(self) -> int | None:
        return None

    def _get_files(self) -> List[str]:
        """Get list of files matching the pattern."""
        if self._files is None:
            from huggingface_hub import HfFileSystem
            fs = HfFileSystem()
            base = f"datasets/{self.repo_id}"
            self._files = sorted(fs.glob(f"{base}/{self.data_files}"))
        return self._files

    def __iter__(self) -> Iterator[Document]:
        import io
        import zstandard as zstd
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()
        files = self._get_files()
        dctx = zstd.ZstdDecompressor()

        idx = 0
        for file_path in files:
            try:
                with fs.open(file_path, "rb") as f:
                    with dctx.stream_reader(f) as reader:
                        text_reader = io.TextIOWrapper(reader, encoding="utf-8")
                        for line in text_reader:
                            line = line.strip()
                            if not line:
                                continue

                            obj = json.loads(line)
                            text = obj.get(self.text_field, "")
                            if not text:
                                continue

                            if self.id_field and self.id_field in obj:
                                doc_id = str(obj[self.id_field])
                            else:
                                doc_id = f"{self.name}:{idx}"

                            yield Document(doc_id=doc_id, text=text, metadata=None)
                            idx += 1
            except Exception as e:
                # Log but continue with next file
                import logging
                logging.getLogger(__name__).warning(f"Error reading {file_path}: {e}")
                continue


class JSONLSource(DataSource):
    """Stream documents from JSONL file(s).

    Each line should be a JSON object with at least a text field.
    """

    def __init__(
        self,
        paths: str | Path | List[str | Path],
        text_field: str = "text",
        id_field: str | None = "id",
    ):
        if isinstance(paths, (str, Path)):
            paths = [paths]
        self.paths = [Path(p) for p in paths]
        self.text_field = text_field
        self.id_field = id_field

    @property
    def name(self) -> str:
        if len(self.paths) == 1:
            return self.paths[0].stem
        return f"jsonl:{len(self.paths)}_files"

    def estimate_size(self) -> int | None:
        # Could count lines but expensive for large files
        return None

    def __iter__(self) -> Iterator[Document]:
        idx = 0
        for path in self.paths:
            opener = open
            if str(path).endswith(".gz"):
                import gzip
                opener = gzip.open

            with opener(path, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    obj = json.loads(line)
                    text = obj.get(self.text_field, "")
                    if not text:
                        continue

                    if self.id_field and self.id_field in obj:
                        doc_id = str(obj[self.id_field])
                    else:
                        doc_id = f"{path.stem}:{idx}"

                    metadata = {k: v for k, v in obj.items() if k != self.text_field}
                    yield Document(doc_id=doc_id, text=text, metadata=metadata if metadata else None)
                    idx += 1


class TextFileSource(DataSource):
    """Stream documents from plain text files.

    Each file is treated as a single document, or split by a separator.
    """

    def __init__(
        self,
        paths: str | Path | List[str | Path],
        doc_separator: str | None = None,
    ):
        if isinstance(paths, (str, Path)):
            paths = [paths]
        self.paths = [Path(p) for p in paths]
        self.doc_separator = doc_separator

    @property
    def name(self) -> str:
        if len(self.paths) == 1:
            return self.paths[0].stem
        return f"text:{len(self.paths)}_files"

    def estimate_size(self) -> int | None:
        if self.doc_separator is None:
            return len(self.paths)
        return None

    def __iter__(self) -> Iterator[Document]:
        idx = 0
        for path in self.paths:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            if self.doc_separator:
                # Split into multiple documents
                for i, text in enumerate(content.split(self.doc_separator)):
                    text = text.strip()
                    if text:
                        yield Document(doc_id=f"{path.stem}:{i}", text=text)
                        idx += 1
            else:
                # Whole file is one document
                if content.strip():
                    yield Document(doc_id=str(path.stem), text=content.strip())
                    idx += 1


class ArrowSource(DataSource):
    """Stream documents from Arrow/Parquet files.

    Efficient for large datasets stored in columnar format.
    """

    def __init__(
        self,
        paths: str | Path | List[str | Path],
        text_field: str = "text",
        id_field: str | None = None,
    ):
        if isinstance(paths, (str, Path)):
            paths = [paths]
        self.paths = [Path(p) for p in paths]
        self.text_field = text_field
        self.id_field = id_field

    @property
    def name(self) -> str:
        if len(self.paths) == 1:
            return self.paths[0].stem
        return f"arrow:{len(self.paths)}_files"

    def estimate_size(self) -> int | None:
        return None

    def __iter__(self) -> Iterator[Document]:
        pa = _get_pyarrow()
        import pyarrow.parquet as pq

        idx = 0
        for path in self.paths:
            if str(path).endswith(".parquet"):
                table = pq.read_table(path)
            else:
                # Assume Arrow IPC format
                with pa.ipc.open_file(path) as reader:
                    table = reader.read_all()

            text_col = table.column(self.text_field)
            id_col = table.column(self.id_field) if self.id_field and self.id_field in table.column_names else None

            for i in range(len(table)):
                text = str(text_col[i].as_py())
                if not text:
                    continue

                if id_col is not None:
                    doc_id = str(id_col[i].as_py())
                else:
                    doc_id = f"{path.stem}:{i}"

                yield Document(doc_id=doc_id, text=text)
                idx += 1


def create_source(
    source_type: str,
    **kwargs,
) -> DataSource:
    """Factory function to create data sources.

    Args:
        source_type: One of "huggingface", "jsonl", "text", "arrow"
        **kwargs: Source-specific arguments

    Returns:
        DataSource instance
    """
    sources = {
        "huggingface": HuggingFaceSource,
        "hf": HuggingFaceSource,
        "jsonl": JSONLSource,
        "text": TextFileSource,
        "arrow": ArrowSource,
        "parquet": ArrowSource,
    }

    if source_type not in sources:
        raise ValueError(f"Unknown source type: {source_type}. Valid: {list(sources.keys())}")

    return sources[source_type](**kwargs)
