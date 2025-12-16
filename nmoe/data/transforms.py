"""
Pure, framework-agnostic data transformations.
These functions are the "kernel" of our data pipeline.
They can be used by local scripts, Beam DoFns, or even on-the-fly loaders.
"""
import hashlib
import re
import unicodedata
from typing import List, Tuple, Optional

import numpy as np
import tiktoken

# Default tokenizer for nmoe
DEFAULT_TOKENIZER = "o200k_harmony"
DEFAULT_VOCAB_SIZE = 201088
DEFAULT_EOS_TOKEN_ID = 199999


def normalize_text(text: str) -> str:
    """
    Standardize text to NFC, strip whitespace, and clean control characters.
    This is idempotent.
    """
    if not text:
        return ""
    # NFC normalization
    text = unicodedata.normalize("NFC", text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def tokenize(text: str, tokenizer_name: str = DEFAULT_TOKENIZER) -> List[int]:
    """
    Tokenize text using tiktoken.

    Default: o200k_harmony (OpenAI's latest, vocab=201088, eos=199999)
    """
    if not text:
        return []
    enc = tiktoken.get_encoding(tokenizer_name)
    return enc.encode_ordinary(text)


def get_shard_id(doc_id: str, num_shards: int) -> int:
    """
    Deterministically map a document ID to a shard index [0, num_shards-1].
    """
    h = hashlib.md5(doc_id.encode("utf-8")).hexdigest()
    v = int(h[:16], 16)
    return v % num_shards


def pack_document(
    tokens: List[int], 
    doc_id: str, 
    eos_id: int, 
    dtype: np.dtype = np.uint32
) -> bytes:
    """
    Pack a single document into a binary buffer.
    Appends EOS.
    """
    full_seq = tokens + [eos_id]
    arr = np.array(full_seq, dtype=dtype)
    return arr.tobytes()


def make_file_name(
    dataset: str, 
    version: str, 
    shard_idx: int, 
    extension: str = "npy"
) -> str:
    """
    Canonical file naming convention.
    """
    return f"{dataset}-{version}-shard-{shard_idx:06d}.{extension}"
