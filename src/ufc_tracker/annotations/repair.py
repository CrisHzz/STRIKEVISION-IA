"""Atomic, backed-up refresh of machine-derived annotation fields only."""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from ufc_tracker.annotations.dataset import sha256
from ufc_tracker.annotations.windows import read_jsonl, write_jsonl

DERIVED_FIELDS = frozenset({"quality", "suggested_label"})


def protected_rows(rows: list[dict]) -> list[dict]:
    return [{k: v for k, v in row.items() if k not in DERIVED_FIELDS} for row in rows]


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(raw)
    try:
        write_jsonl(temporary, rows)
        if read_jsonl(temporary) != rows:
            raise RuntimeError("JSONL serialization validation failed")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def refresh_quality_file(
    path: Path, rows: list[dict], *, expected_sha256: str, backup_dir: Path,
) -> int:
    before = read_jsonl(path)
    if sha256(path) != expected_sha256:
        raise RuntimeError(f"Concurrent annotation modification: {path.name}")
    if protected_rows(before) != protected_rows(rows):
        raise ValueError("Refusing to modify human fields, intervals or row order")
    changed = sum(a != b for a, b in zip(before, rows))
    if not changed:
        return 0
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / path.name
    if not backup.exists():
        shutil.copy2(path, backup)
    # Check again immediately before the atomic replacement.
    if sha256(path) != expected_sha256:
        raise RuntimeError(f"Concurrent annotation modification: {path.name}")
    atomic_jsonl(path, rows)
    return changed
