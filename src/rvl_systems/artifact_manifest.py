from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(paths: list[str | Path]) -> dict:
    entries = []
    for path in sorted(Path(p) for p in paths):
        entries.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {"schema_version": 1, "artifacts": entries}


def verify_manifest(manifest: dict) -> list[str]:
    failures: list[str] = []
    for entry in manifest.get("artifacts", []):
        path = Path(entry["path"])
        if not path.exists():
            failures.append(f"missing artifact: {path}")
            continue
        actual = sha256_file(path)
        if actual != entry["sha256"]:
            failures.append(f"checksum mismatch: {path}")
        if path.stat().st_size != entry["bytes"]:
            failures.append(f"size mismatch: {path}")
    return failures


def write_manifest(paths: list[str | Path], output: str | Path) -> None:
    Path(output).write_text(
        json.dumps(build_manifest(paths), indent=2, sort_keys=True),
        encoding="utf-8",
    )
