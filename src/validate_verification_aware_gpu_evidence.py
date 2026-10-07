"""Validate completeness and provenance of verification-aware GPU phase evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _require(cond: bool, message: str) -> None:
    if not cond:
        raise AssertionError(message)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_cell(plan: dict[str, Any], cell: dict[str, Any], root: Path) -> dict[str, Any]:
    cell_root = root / cell["cell_id"]
    record_path = cell_root / "run_record.json"
    _require(record_path.exists(), f"{cell['cell_id']}: missing run_record.json")
    record = load_json(record_path)
    _require(record.get("schema_version") == 1, f"{cell['cell_id']}: bad record schema")
    _require(record.get("cell_id") == cell["cell_id"], f"{cell['cell_id']}: cell identity mismatch")
    _require(record.get("protocol_sha256") == plan["protocol_sha256"], f"{cell['cell_id']}: protocol hash mismatch")
    _require(record.get("source_sha") == plan["source_sha"], f"{cell['cell_id']}: source SHA mismatch")
    _require(record.get("phase") == cell["phase"], f"{cell['cell_id']}: phase mismatch")
    _require(record.get("arm") == cell["arm"], f"{cell['cell_id']}: arm mismatch")
    _require(int(record.get("seed")) == int(cell["seed"]), f"{cell['cell_id']}: seed mismatch")
    _require(record.get("effective_system") == cell["effective_system"], f"{cell['cell_id']}: system config mismatch")
    _require(record.get("status") in {"completed", "failed"}, f"{cell['cell_id']}: invalid terminal status")
    _require(bool(record.get("hardware")), f"{cell['cell_id']}: hardware provenance missing")
    _require(bool(record.get("command")), f"{cell['cell_id']}: command provenance missing")

    manifest_path = cell_root / "manifest.json"
    _require(manifest_path.exists(), f"{cell['cell_id']}: missing manifest")
    manifest = load_json(manifest_path)
    _require(manifest.get("cell_id") == cell["cell_id"], f"{cell['cell_id']}: manifest cell mismatch")
    files = dict(manifest.get("files") or {})
    _require(files, f"{cell['cell_id']}: empty manifest")
    for name, expected in files.items():
        path = cell_root / name
        _require(path.exists(), f"{cell['cell_id']}: manifest file missing: {name}")
        _require(sha256_file(path) == expected, f"{cell['cell_id']}: manifest hash mismatch: {name}")

    if record["status"] == "failed":
        failure_path = cell_root / "failure.json"
        _require(failure_path.exists(), f"{cell['cell_id']}: failed run missing failure.json")
        failure = load_json(failure_path)
        _require(bool(failure.get("message")), f"{cell['cell_id']}: failure message missing")
        return {"cell_id": cell["cell_id"], "status": "failed", "valid": True}

    ts_path = cell_root / "timeseries.jsonl"
    summary_path = cell_root / "summary.json"
    _require(ts_path.exists(), f"{cell['cell_id']}: missing timeseries")
    _require(summary_path.exists(), f"{cell['cell_id']}: missing summary")
    rows = _load_jsonl(ts_path)
    _require(len(rows) >= 2, f"{cell['cell_id']}: timeseries must have >=2 rows")
    required_ts = set(plan["required_timeseries"])
    for i, row in enumerate(rows):
        _require("t_s" in row, f"{cell['cell_id']}: timeseries row {i} missing t_s")
        missing = required_ts - set(row)
        _require(not missing, f"{cell['cell_id']}: timeseries row {i} missing {sorted(missing)}")

    summary = load_json(summary_path)
    metrics = dict(summary.get("terminal_metrics") or {})
    missing_metrics = set(plan["required_terminal_metrics"]) - set(metrics)
    _require(not missing_metrics, f"{cell['cell_id']}: terminal metrics missing {sorted(missing_metrics)}")
    invariant_rows = list(summary.get("hard_invariants") or [])
    observed = {row.get("statement"): row for row in invariant_rows}
    for statement in plan["hard_invariants"]:
        _require(statement in observed, f"{cell['cell_id']}: invariant not evaluated: {statement}")
        _require(observed[statement].get("passed") is True, f"{cell['cell_id']}: invariant failed: {statement}")
    return {"cell_id": cell["cell_id"], "status": "completed", "valid": True}


def validate_bundle(plan_path: Path, root: Path, *, allow_partial: bool = False) -> dict[str, Any]:
    plan = load_json(plan_path)
    _require(plan.get("status") == "prospective_phase_plan", "prospective phase plan required")
    rows = []
    missing = []
    for cell in plan["cells"]:
        if not (root / cell["cell_id"]).exists():
            missing.append(cell["cell_id"])
            continue
        rows.append(validate_cell(plan, cell, root))
    if not allow_partial:
        _require(not missing, f"missing evidence cells: {missing[:8]}{'...' if len(missing) > 8 else ''}")
    completed = sum(r["status"] == "completed" for r in rows)
    failed = sum(r["status"] == "failed" for r in rows)
    return {
        "valid": True,
        "expected_cells": int(plan["cell_count"]),
        "present_cells": len(rows),
        "completed_cells": completed,
        "failed_cells": failed,
        "missing_cells": missing,
        "all_present": not missing,
        "claim_boundary": plan["claim_boundary"],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--allow-partial", action="store_true")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    result = validate_bundle(args.plan, args.root, allow_partial=args.allow_partial)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
