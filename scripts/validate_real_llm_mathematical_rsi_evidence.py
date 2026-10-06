"""Independent semantic validation for real-LLM mathematical RSI replay."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

from scripts.run_real_llm_mathematical_rsi import (
    METHODS,
    load_json,
    reconstruction_diagnostics,
    sha256,
    summarize_rows,
)
from scripts.run_real_llm_active_audit import load_real_candidate_cells, validate_input_manifest


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _load_trials(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in gzip.decompress(path.read_bytes()).decode("utf-8").splitlines()
        if line.strip()
    ]


def _close(a: Any, b: Any, tol: float = 1e-12) -> bool:
    if a is None or b is None:
        return a is b
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, str)) or isinstance(b, (int, str)):
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


def _scientific_result(summary: dict[str, Any], lock: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    primary = int(lock["primary_budget"])
    rows = [
        r for r in summary["arm_rows"]
        if int(r["budget"]) == primary and r["arm"] in {"stale", "shuffled"}
    ]
    by = {(r["arm"], r["method"]): r for r in rows}
    if not all((arm, "passive_fixed") in by for arm in ("stale", "shuffled")):
        return "underpowered", {"reason": "missing passive stale/shuffled primary cells"}
    target = "coded_fixed_probe_info"
    if not all((arm, target) in by for arm in ("stale", "shuffled")):
        return "underpowered", {"reason": "missing coded stale/shuffled primary cells"}
    harmful_cells = [
        r for r in rows
        if r["method"] in {"passive_fixed", target} and int(r["harmful_trials"]) > 0
    ]
    if len(harmful_cells) < 2:
        return "underpowered", {"reason": "insufficient harmful primary trials"}

    comparisons = {}
    passed = True
    for arm in ("stale", "shuffled"):
        baseline = by[(arm, "passive_fixed")]
        coded = by[(arm, target)]
        comparisons[arm] = {
            "passive_harmful_allow_rate": baseline["harmful_update_allow_rate"],
            "coded_harmful_allow_rate": coded["harmful_update_allow_rate"],
            "passive_wrong_sign_decisive_rate": baseline["wrong_sign_decisive_rate"],
            "coded_wrong_sign_decisive_rate": coded["wrong_sign_decisive_rate"],
            "coded_inconclusive_rate": coded["inconclusive_rate"],
        }
        if (
            baseline["harmful_update_allow_rate"] is None
            or coded["harmful_update_allow_rate"] is None
        ):
            return "underpowered", {"reason": f"no harmful trials for {arm}"}
        passed = passed and (
            float(coded["harmful_update_allow_rate"])
            <= float(baseline["harmful_update_allow_rate"])
        )
        if (
            baseline["wrong_sign_decisive_rate"] is not None
            and coded["wrong_sign_decisive_rate"] is not None
        ):
            passed = passed and (
                float(coded["wrong_sign_decisive_rate"])
                <= float(baseline["wrong_sign_decisive_rate"]) + 1e-12
            )
    return ("direction_passed" if passed else "direction_failed"), comparisons


def validate_evidence(
    root: Path,
    protocol_path: Path,
    input_root: Path,
) -> dict[str, Any]:
    lock = load_json(protocol_path)
    manifest = load_json(root / "manifest.json")
    _require(manifest.get("status") == "completed", "output manifest not completed")
    _require(manifest.get("protocol_sha256") == sha256(protocol_path), "protocol hash mismatch")
    _require(
        manifest.get("input_manifest_sha256") == sha256(input_root / "manifest.json"),
        "input manifest hash mismatch",
    )
    validate_input_manifest(input_root, lock["input_requirements"])
    files = manifest.get("files", {})
    _require(
        files == {
            "trials.jsonl.gz": sha256(root / "trials.jsonl.gz"),
            "summary.json": sha256(root / "summary.json"),
        },
        "output file hashes mismatch",
    )
    trials = _load_trials(root / "trials.jsonl.gz")
    cells = load_real_candidate_cells(input_root, lock["input_requirements"])
    budgets = [int(x) for x in lock["trusted_label_budgets"]]
    replicates = int(lock["audit_replicates"])
    expected = len(cells) * len(budgets) * len(METHODS) * replicates
    _require(len(trials) == expected == int(manifest["trials"]), "trial count mismatch")
    _require(int(manifest["cells"]) == len(cells), "cell count mismatch")
    _require({r["method"] for r in trials} == set(METHODS), "method set mismatch")
    _require({int(r["budget"]) for r in trials} == set(budgets), "budget set mismatch")

    strata = int(lock["coded_verification"]["probe_strata"])
    target_error = float(lock["information_budget"]["target_error"])
    multiplier = float(lock["information_budget"]["multiplier"])
    required = int(__import__("math").ceil(multiplier * strata * __import__("math").log(1.0 / target_error)))
    for row in trials:
        _require(int(row["trusted_label_calls"]) == int(row["budget"]), "charged budget mismatch")
        if row["method"] in {"coded_fixed_probe", "coded_fixed_probe_info"}:
            _require(_close(row["probe_coverage"], 1.0), "fixed probe coverage must be complete")
            _require(
                int(row["allowed_corrupt_local_obligations"])
                == int(__import__("math").floor(float(lock["coded_verification"]["max_corrupt_fraction"]) * strata)),
                "coded corruption allowance mismatch",
            )
        if row["method"] == "coded_fixed_probe_info":
            _require(int(row["information_budget_required"]) == required, "information requirement mismatch")
            expected_pass = int(row["budget"]) >= required
            _require(bool(row["information_budget_passed"]) == expected_pass, "information gate mismatch")
            if not expected_pass:
                _require(row["decision"] == "inconclusive", "underbudget information gate must refresh")

    summary = load_json(root / "summary.json")
    recomputed = summarize_rows(trials)
    recomputed_arms = summarize_rows(trials, keys=("arm", "method", "budget"))
    _require(_rows_equal(summary["summary_rows"], recomputed), "summary rows do not recompute")
    _require(_rows_equal(summary["arm_rows"], recomputed_arms), "arm rows do not recompute")
    primary = [r for r in recomputed if int(r["budget"]) == int(lock["primary_budget"])]
    _require(_rows_equal(summary["primary_rows"], primary), "primary rows mismatch")
    diag = reconstruction_diagnostics(
        cells, [int(x) for x in lock["reconstruction_diagnostic"]["branching_factors"]]
    )
    _require(_rows_equal(summary["reconstruction_diagnostics"], diag), "reconstruction diagnostics mismatch")
    _require(
        all(not bool(row["assumptions_met"]) for row in diag),
        "reconstruction diagnostic must not claim tree assumptions",
    )
    result, comparison = _scientific_result(summary, lock)
    return {
        "valid": True,
        "execution_status": "completed",
        "scientific_result": result,
        "trials": len(trials),
        "cells": len(cells),
        "primary_budget": int(lock["primary_budget"]),
        "primary_comparison": comparison,
        "research_source_sha": manifest.get("input_research_source_sha"),
        "claim_boundary": (
            "real-model downstream controller replay; mathematical results motivate "
            "the architecture but do not prove LLM RSI safety or convergence"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate_evidence(args.root, args.protocol, args.input_root)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
