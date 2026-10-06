"""Independent validator for recovered-Qwen mathematical RSI exploratory replay."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from scripts.run_recovered_qwen_mathematical_rsi import METHODS, load_json, sha256, summarize


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _close(a: Any, b: Any, tol: float = 1e-12) -> bool:
    if a is None or b is None:
        return a is b
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    if isinstance(a, (str, int)) or isinstance(b, (str, int)):
        return a == b
    return abs(float(a) - float(b)) <= tol


def _rows_equal(first: list[dict[str, Any]], second: list[dict[str, Any]]) -> bool:
    if len(first) != len(second):
        return False
    for a, b in zip(first, second):
        if set(a) != set(b):
            return False
        for key in a:
            if not _close(a[key], b[key]):
                return False
    return True


def validate(root: Path, protocol_path: Path, bank_path: Path) -> dict[str, Any]:
    lock = load_json(protocol_path)
    manifest = load_json(root / "manifest.json")
    summary = load_json(root / "summary.json")
    trials = _load_jsonl(root / "trials.jsonl")

    _require(manifest.get("status") == "completed", "manifest not completed")
    _require(manifest.get("protocol_sha256") == sha256(protocol_path), "protocol SHA mismatch")
    _require(manifest.get("bank_sha256") == sha256(bank_path), "bank SHA mismatch")
    _require(manifest.get("bank_sha256") == lock["input"]["bank_sha256"], "locked bank SHA mismatch")
    _require(
        manifest.get("files") == {
            "trials.jsonl": sha256(root / "trials.jsonl"),
            "summary.json": sha256(root / "summary.json"),
        },
        "output hash mismatch",
    )
    _require({row["method"] for row in trials} == set(METHODS), "method set mismatch")

    task_count = int(lock["input"]["expected_tasks"])
    allowed = math.floor(
        float(lock["coded_verification"]["max_corrupt_fraction"]) * task_count
    )
    required = math.ceil(
        float(lock["information_budget"]["multiplier"])
        * task_count
        * math.log(1.0 / float(lock["information_budget"]["target_error"]))
    )
    for row in trials:
        method = row["method"]
        if method != "proxy_only":
            _require(
                int(row["trusted_label_calls"]) <= int(row["budget"]),
                "trusted label calls exceed requested budget",
            )
            if row["action"] == "allow":
                _require(row["lower"] is not None and float(row["lower"]) >= -1e-12,
                         "non-proxy controller allowed without nonnegative certificate")
                _require(not bool(row["harmful"]), "certificate allowed harmful finite-bank update")
        if method in {"coded_fixed_probe", "coded_fixed_probe_info"}:
            _require(abs(float(row["task_coverage"]) - 1.0) <= 1e-12,
                     "fixed probe did not cover every task")
            _require(int(row["allowed_local_violations"]) == allowed,
                     "coded local corruption budget mismatch")
        if method == "coded_fixed_probe_info":
            _require(int(row["information_budget_required"]) == required,
                     "information budget mismatch")
            expected = int(row["budget"]) >= required
            _require(bool(row["information_budget_passed"]) == expected,
                     "information budget pass flag mismatch")
            if not expected:
                _require(row["action"] == "refresh",
                         "underbudget information gate must refresh")

    recomputed = summarize(trials)
    _require(_rows_equal(summary["summary_rows"], recomputed),
             "summary does not recompute from trial rows")
    primary_budget = int(lock["primary_budget"])
    primary = [r for r in recomputed if r["budget"] in {0, primary_budget}]
    _require(_rows_equal(summary["primary_rows"], primary), "primary rows mismatch")

    cells = summary.get("cells", [])
    _require(len(cells) == int(summary["counts"]["proposal_cells"]), "cell count mismatch")
    _require(
        sum(bool(row["harmful"]) for row in cells) == int(summary["counts"]["harmful_cells"]),
        "harmful cell count mismatch",
    )
    _require(
        sum(bool(row["false_progress"]) for row in cells)
        == int(summary["counts"]["false_progress_cells"]),
        "false-progress count mismatch",
    )
    return {
        "valid": True,
        "execution_status": "completed",
        "evidence_status": summary["evidence_status"],
        "proposal_cells": len(cells),
        "harmful_cells": int(summary["counts"]["harmful_cells"]),
        "false_progress_cells": int(summary["counts"]["false_progress_cells"]),
        "primary_budget": primary_budget,
        "information_budget_required": required,
        "claim_boundary": summary["claim_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(args.root, args.protocol, args.bank)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
