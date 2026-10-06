"""Independent semantic validation for the linear-hitting Qwen follow-up."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.run_linear_hitting_qwen import load_json, sha256


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate(root: Path, protocol: Path, bank: Path) -> dict[str, Any]:
    lock = load_json(protocol)
    manifest = load_json(root / "manifest.json")
    summary = load_json(root / "summary.json")
    rows = [
        json.loads(line)
        for line in (root / "rows.jsonl").read_text().splitlines()
        if line.strip()
    ]
    _require(manifest.get("status") == "completed", "manifest not completed")
    _require(manifest["protocol_sha256"] == sha256(protocol), "protocol hash mismatch")
    _require(manifest["bank_sha256"] == sha256(bank), "bank hash mismatch")
    _require(manifest["bank_sha256"] == lock["input"]["bank_sha256"], "locked bank mismatch")
    _require(
        manifest["files"] == {
            "rows.jsonl": sha256(root / "rows.jsonl"),
            "summary.json": sha256(root / "summary.json"),
        },
        "output hashes mismatch",
    )
    expected_reps = set(lock["representations"])
    _require(set(summary["representations"]) == expected_reps, "representation set mismatch")
    proposal_cells = int(summary["proposal_cells"])
    _require(proposal_cells == len(lock["etas"]) * int(lock["rounds"]), "proposal cell mismatch")

    for representation, meta in summary["representations"].items():
        _require(int(meta["probe_count"]) == int(meta["feature_rank"]),
                 f"{representation}: probe count must equal feature rank")
        _require(len(meta["probe_ids"]) == int(meta["probe_count"]),
                 f"{representation}: probe ID count mismatch")
        _require(len(set(meta["probe_ids"])) == len(meta["probe_ids"]),
                 f"{representation}: duplicate probe IDs")

    sufficient_rows = [
        row for row in rows
        if row["residual_class_sufficient_by_full_label_witness"]
    ]
    _require(sufficient_rows, "no evaluator-witnessed valid residual classes")
    for row in sufficient_rows:
        actual = float(row["true_gain"])
        _require(float(row["lower"]) - 1e-9 <= actual <= float(row["upper"]) + 1e-9,
                 "valid residual class failed to cover true gain")
        _require(not bool(row["wrong_sign_decisive"]),
                 "valid residual class produced wrong-sign decisive action")

    for representation in expected_reps:
        oracle = [
            row for row in rows
            if row["representation"] == representation
            and row["radius_source"] == "evaluator_sufficient"
        ]
        _require(len(oracle) == proposal_cells,
                 f"{representation}: evaluator-sufficient row count mismatch")
        _require(all(row["residual_class_sufficient_by_full_label_witness"] for row in oracle),
                 f"{representation}: evaluator-sufficient radius not sufficient")

    return {
        "valid": True,
        "status": summary["status"],
        "proposal_cells": proposal_cells,
        "representations": {
            name: {
                "feature_rank": int(meta["feature_rank"]),
                "probe_count": int(meta["probe_count"]),
                "information_budget_proxy": int(meta["information_budget_proxy"]),
                "lstsq_residual_inf": float(meta["lstsq_residual_inf"]),
                "lstsq_residual_rmse": float(meta["lstsq_residual_rmse"]),
            }
            for name, meta in summary["representations"].items()
        },
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
