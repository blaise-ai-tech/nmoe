"""
DocID compose/parse helpers for shard-native training and grading.

Format (relative to data_root):
    {source}//{rel_path}.npy#s={start}:e={end}

Where:
- source: logical source id (e.g., "common_crawl")
- rel_path: path from data_root to the shard .npy (without leading slash)
- start, end: token indices within the shard (end is exclusive)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DocId:
    source: str
    rel_path: str
    start: int
    end: int

    def compose(self) -> str:
        return compose_doc_id(self.source, self.rel_path, self.start, self.end)


def compose_doc_id(source: str, rel_path: str, start: int, end: int) -> str:
    rel = rel_path.replace("\\", "/").lstrip("/")
    return f"{source}//{rel}#s={int(start)}:e={int(end)}"


def parse_doc_id(doc_id: str) -> DocId:
    # Split source
    if "//" not in doc_id:
        raise ValueError(f"invalid doc_id (missing //): {doc_id}")
    src, rest = doc_id.split("//", 1)
    if "#" not in rest:
        raise ValueError(f"invalid doc_id (missing #): {doc_id}")
    rel, rng = rest.split("#", 1)
    if not rng.startswith("s=") or ":e=" not in rng:
        raise ValueError(f"invalid doc_id (missing s=/e=): {doc_id}")
    s_part, e_part = rng.split(":e=", 1)
    start = int(s_part[2:])
    end = int(e_part)
    return DocId(source=src, rel_path=rel, start=start, end=end)


def shard_path(data_root: str | Path, doc: DocId) -> Path:
    return Path(data_root) / doc.rel_path

