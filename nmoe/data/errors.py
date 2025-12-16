"""
Structured error codes for data preprocessing pipeline.

Error codes follow the pattern E-{CATEGORY}-{SPECIFIC}.
All errors exit with non-zero code and print actionable message to stderr.
"""
from __future__ import annotations

import sys
from enum import Enum
from typing import NoReturn


class ErrorCode(str, Enum):
    """Pipeline error codes."""

    # Manifest errors
    MANIFEST_COMMIT = "E-MANIFEST-COMMIT"
    INDEX_MISSING = "E-INDEX-MISSING"
    SHARD_CHECKSUM = "E-SHARD-CHECKSUM"

    # Source errors
    SOURCE_NOTFOUND = "E-SOURCE-NOTFOUND"

    # Tokenizer errors
    TOKENIZER_DRIFT = "E-TOKENIZER-DRIFT"

    # Resume errors
    RESUME_CURSOR_MISMATCH = "E-RESUME-CURSOR-MISMATCH"

    def __str__(self) -> str:
        return self.value


# Error descriptions for help messages
ERROR_DESCRIPTIONS = {
    ErrorCode.MANIFEST_COMMIT: "Atomic rename/write failure; check disk space/permissions",
    ErrorCode.INDEX_MISSING: "Missing .idx for shard; regenerate index or re-run stage",
    ErrorCode.SHARD_CHECKSUM: "Checksum mismatch on verification; shard corrupt, re-run stage",
    ErrorCode.SOURCE_NOTFOUND: "No input files or HF split invalid; check source path/config",
    ErrorCode.TOKENIZER_DRIFT: "Tokenizer-derived values differ from manifest; use consistent tokenizer",
    ErrorCode.RESUME_CURSOR_MISMATCH: "Last processed record checksum mismatch; state file stale or data corrupted",
}


class PipelineError(Exception):
    """Base exception for pipeline errors with structured codes."""

    def __init__(self, code: ErrorCode, message: str, details: str | None = None):
        self.code = code
        self.message = message
        self.details = details
        super().__init__(f"[{code}]: {message}")

    def exit(self) -> NoReturn:
        """Print error to stderr and exit with non-zero code."""
        print(f"ERROR [{self.code}]: {self.message}", file=sys.stderr)
        if self.details:
            print(f"  Details: {self.details}", file=sys.stderr)
        print(f"  Action: {ERROR_DESCRIPTIONS.get(self.code, 'Unknown error')}", file=sys.stderr)
        sys.exit(1)


class ManifestCommitError(PipelineError):
    """Failed to atomically commit manifest."""

    def __init__(self, path: str, cause: str):
        super().__init__(
            ErrorCode.MANIFEST_COMMIT,
            f"Failed to commit manifest to {path}",
            cause,
        )


class IndexMissingError(PipelineError):
    """Index file missing for shard."""

    def __init__(self, shard_path: str, index_path: str):
        super().__init__(
            ErrorCode.INDEX_MISSING,
            f"Missing index file for shard {shard_path}",
            f"Expected index at: {index_path}",
        )


class ShardChecksumError(PipelineError):
    """Shard checksum mismatch."""

    def __init__(self, shard_path: str, expected: str, actual: str):
        super().__init__(
            ErrorCode.SHARD_CHECKSUM,
            f"Checksum mismatch for {shard_path}",
            f"Expected: {expected}, Got: {actual}",
        )


class SourceNotFoundError(PipelineError):
    """Source data not found."""

    def __init__(self, source: str, details: str | None = None):
        super().__init__(
            ErrorCode.SOURCE_NOTFOUND,
            f"Source not found: {source}",
            details,
        )


class TokenizerDriftError(PipelineError):
    """Tokenizer parameters don't match manifest."""

    def __init__(
        self,
        field: str,
        expected: int | str,
        actual: int | str,
        tokenizer_name: str,
    ):
        super().__init__(
            ErrorCode.TOKENIZER_DRIFT,
            f"Tokenizer {field} mismatch: manifest has {expected}, tokenizer {tokenizer_name} has {actual}",
            "Use the same tokenizer as the original run or start a new pipeline version",
        )


class ResumeCursorMismatchError(PipelineError):
    """Resume cursor doesn't match last processed record."""

    def __init__(self, expected_digest: str, actual_digest: str):
        super().__init__(
            ErrorCode.RESUME_CURSOR_MISMATCH,
            "Last processed record checksum mismatch on resume",
            f"State expects {expected_digest}, found {actual_digest}",
        )
