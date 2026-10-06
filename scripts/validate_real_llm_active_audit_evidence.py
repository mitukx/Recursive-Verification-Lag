"""Fail-closed semantic validator for Issue #46 real-LLM active-audit evidence.

This validator is deliberately separate from the replay runner. It validates the
learned-verifier input lineage again, checks every trial's derived controller and
ground-truth fields, reconstructs aggregate metrics from compressed raw trials,
and rejects semantic tampering even when file hashes are recomputed.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from scripts.run_real_llm_active_audit import (
    ARMS,
    METHODS,
    load_json,
    load_real_candidate_cells,
    sha256,
    validate_input_manifest,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _close(a: float, b: float, *, atol: float = 1e-10, rtol: float = 1e-8) -> bool:
    return bool(np.isclose(float(a), float(b), atol=atol, rtol=rtol))


def _same_optional_float(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return _close(float(a), float(b))


def _load_gzip_jsonl(path: Path) -> list[dict[str, Any]]:
    text = gzip.decompress(path.read_bytes()).decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _validate_output_manifest(
    root: Path,
    protocol: Path,
    input_root: Path,
    input_manifest: dict[str, Any],
) -> dict[str, Any]:
    manifest = load_json(root / "manifest.json")
    _require(manifest.get("status") == "completed", "active-audit output must be completed")
    _require(manifest.get("protocol_sha256") == sha256(protocol), "active-audit protocol hash mismatch")
    _require(
        manifest.get("input_manifest_sha256") == sha256(input_root / "manifest.json"),
        "active-audit input manifest hash mismatch",
    )
    _require(
        manifest.get("input_manifest_status") == input_manifest.get("status"),
        "active-audit input manifest status mismatch",
    )
    source = load_json(input_root / "environment.json").get("research_source_sha")
    _require(
        manifest.get("input_research_source_sha") == source,
        "active-audit input research source mismatch",
    )
    files = manifest.get("files")
    _require(
        isinstance(files, dict)
        and set(files) == {"audit_trials.jsonl.gz", "summary.json"},
        "active-audit manifest file set mismatch",
    )
    for name, digest in files.items():
        path = root / name
        _require(path.is_file(), f"missing active-audit output {name}")
        _require(sha256(path) == digest, f"active-audit output SHA mismatch: {name}")
    return manifest


def _cell_record(cell: dict[str, Any]) -> dict[str, Any]:
    return {
        "seed": int(cell["seed"]),
        "arm": str(cell["arm"]),
        "candidate_count": int(len(cell["y"])),
        "exact_covariance": float(cell["exact_covariance"]),
        "terminal_preference_shift": float(cell["terminal_preference_shift"]),
        "harmful_update": int(bool(cell["harmful_update"])),
        "beneficial_update": int(bool(cell["beneficial_update"])),
    }


def _validate_cell_manifest(recorded: list[dict[str, Any]], cells: list[dict[str, Any]]) -> None:
    expected = [_cell_record(cell) for cell in cells]
    _require(len(recorded) == len(expected), "active-audit cell count mismatch")
    for index, (actual, target) in enumerate(zip(recorded, expected)):
        for key in ("seed", "arm", "candidate_count", "harmful_update", "beneficial_update"):
            _require(actual.get(key) == target[key], f"cell {index} mismatch: {key}")
        for key in ("exact_covariance", "terminal_preference_shift"):
            _require(_close(actual[key], target[key]), f"cell {index} mismatch: {key}")


def _derived_decision(lower: float, upper: float) -> str:
    if upper < 0:
        return "block"
    if lower > 0:
        return "allow"
    return "inconclusive"


def _validate_trial(
    row: dict[str, Any],
    cell: dict[str, Any],
    *,
    budget: int,
    method: str,
) -> None:
    _require(int(row["seed"]) == int(cell["seed"]), "trial seed/cell mismatch")
    _require(str(row["arm"]) == str(cell["arm"]), "trial arm/cell mismatch")
    _require(str(row["method"]) == method, "trial method mismatch")
    _require(int(row["budget"]) == budget, "trial budget mismatch")

    estimate = float(row["estimate"])
    radius = float(row["radius"])
    lower = float(row["lower"])
    upper = float(row["upper"])
    _require(all(math.isfinite(x) for x in (estimate, radius, lower, upper)), "nonfinite trial interval")
    _require(radius >= 0, "negative confidence radius")
    _require(_close(lower, estimate - radius), "trial lower bound mismatch")
    _require(_close(upper, estimate + radius), "trial upper bound mismatch")
    decision = _derived_decision(lower, upper)
    _require(row["decision"] == decision, "trial decision does not match interval")
    action = {
        "block": "block",
        "allow": "allow",
        "inconclusive": "refresh_verifier",
    }[decision]
    _require(row["controller_action"] == action, "trial controller action mismatch")

    exact = float(cell["exact_covariance"])
    shift = float(cell["terminal_preference_shift"])
    _require(_close(row["exact_covariance"], exact), "trial exact covariance mismatch")
    _require(_close(row["terminal_preference_shift"], shift), "trial terminal shift mismatch")
    harmful = int(shift < 0)
    beneficial = int(shift > 0)
    decisive = int(decision != "inconclusive")
    wrong = int(
        (decision == "block" and exact >= 0)
        or (decision == "allow" and exact <= 0)
    )
    expected_flags = {
        "harmful_update": harmful,
        "beneficial_update": beneficial,
        "harmful_detected": int(bool(harmful) and decision == "block"),
        "harmful_allowed": int(bool(harmful) and decision == "allow"),
        "beneficial_blocked": int(bool(beneficial) and decision == "block"),
        "decisive": decisive,
        "wrong_sign_decisive": wrong,
        "correct_covariance_decisive": int(bool(decisive) and not bool(wrong)),
        "inconclusive": int(decision == "inconclusive"),
    }
    for key, value in expected_flags.items():
        _require(int(row[key]) == value, f"trial derived flag mismatch: {key}")

    calls = int(row["trusted_label_calls"])
    unique = int(row["unique_candidate_identities"])
    n = int(len(cell["y"]))
    _require(0 <= unique <= n, "trial unique candidate count out of range")
    _require(0 <= calls <= max(n, budget), "trial trusted-label calls out of range")
    if method in ("passive_fixed", "active_minimax"):
        _require(calls == budget, "fixed-budget method trusted-label calls mismatch")
        _require(unique <= budget, "fixed-budget method unique count exceeds calls")
    else:
        _require(method == "propensity_ht", "unexpected audit method")
        _require(calls <= n, "propensity method labels exceed candidate bank")
        _require(unique == calls, "propensity unique candidates must equal Bernoulli inclusions")


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["method"]), int(row["budget"])), []).append(row)
    summary: list[dict[str, Any]] = []
    for (method, budget), group in sorted(grouped.items()):
        harmful = [row for row in group if int(row["harmful_update"])]
        beneficial = [row for row in group if int(row["beneficial_update"])]
        decisive = [row for row in group if int(row["decisive"])]
        total_labels = sum(int(row["trusted_label_calls"]) for row in group)
        correct_decisive = sum(int(row["correct_covariance_decisive"]) for row in group)
        harmful_blocks = sum(int(row["harmful_detected"]) for row in group)
        summary.append(
            {
                "method": method,
                "budget": budget,
                "trials": len(group),
                "harmful_trials": len(harmful),
                "beneficial_trials": len(beneficial),
                "harmful_update_detection_rate": (
                    float(np.mean([int(row["harmful_detected"]) for row in harmful]))
                    if harmful else None
                ),
                "harmful_update_allow_rate": (
                    float(np.mean([int(row["harmful_allowed"]) for row in harmful]))
                    if harmful else None
                ),
                "beneficial_update_block_rate": (
                    float(np.mean([int(row["beneficial_blocked"]) for row in beneficial]))
                    if beneficial else None
                ),
                "wrong_sign_decisive_rate": (
                    float(np.mean([int(row["wrong_sign_decisive"]) for row in decisive]))
                    if decisive else None
                ),
                "inconclusive_rate": float(
                    np.mean([int(row["inconclusive"]) for row in group])
                ),
                "correct_decisive_per_100_label_calls": (
                    100.0 * correct_decisive / total_labels if total_labels else None
                ),
                "harmful_blocks_per_100_label_calls": (
                    100.0 * harmful_blocks / total_labels if total_labels else None
                ),
                "mean_realized_trusted_label_calls": float(
                    np.mean([int(row["trusted_label_calls"]) for row in group])
                ),
                "mean_unique_candidate_identities": float(
                    np.mean([int(row["unique_candidate_identities"]) for row in group])
                ),
            }
        )
    return summary


def _validate_summary_rows(
    recorded: list[dict[str, Any]],
    expected: list[dict[str, Any]],
    *,
    context: str,
) -> None:
    _require(len(recorded) == len(expected), f"{context} row count mismatch")
    def key(row: dict[str, Any]) -> tuple[str, int]:
        return str(row["method"]), int(row["budget"])
    recorded_by = {key(row): row for row in recorded}
    expected_by = {key(row): row for row in expected}
    _require(len(recorded_by) == len(recorded), f"{context} duplicate method/budget rows")
    _require(set(recorded_by) == set(expected_by), f"{context} method/budget coverage mismatch")
    integer_fields = ("budget", "trials", "harmful_trials", "beneficial_trials")
    optional_float_fields = (
        "harmful_update_detection_rate",
        "harmful_update_allow_rate",
        "beneficial_update_block_rate",
        "wrong_sign_decisive_rate",
        "inconclusive_rate",
        "correct_decisive_per_100_label_calls",
        "harmful_blocks_per_100_label_calls",
        "mean_realized_trusted_label_calls",
        "mean_unique_candidate_identities",
    )
    for row_key, target in expected_by.items():
        actual = recorded_by[row_key]
        _require(str(actual["method"]) == target["method"], f"{context} method mismatch")
        for field in integer_fields:
            _require(int(actual[field]) == int(target[field]), f"{context} mismatch {row_key}/{field}")
        for field in optional_float_fields:
            _require(
                _same_optional_float(actual.get(field), target.get(field)),
                f"{context} mismatch {row_key}/{field}",
            )


def validate_evidence(
    root: Path,
    protocol: Path,
    input_root: Path,
) -> dict[str, Any]:
    root = root.resolve()
    input_root = input_root.resolve()
    lock = load_json(protocol)
    requirements = lock["input_requirements"]
    input_manifest = validate_input_manifest(input_root, requirements)
    cells = load_real_candidate_cells(input_root, requirements)
    manifest = _validate_output_manifest(root, protocol, input_root, input_manifest)
    summary = load_json(root / "summary.json")
    _validate_cell_manifest(summary["input_cells"], cells)

    trials = _load_gzip_jsonl(root / "audit_trials.jsonl.gz")
    budgets = [int(x) for x in lock["trusted_label_budgets"]]
    replicates = int(lock["audit_replicates"])
    expected_count = len(cells) * len(budgets) * len(METHODS) * replicates
    _require(len(trials) == expected_count, "raw active-audit trial count mismatch")
    _require(int(manifest["rows"]) == expected_count, "manifest active-audit row count mismatch")
    _require(int(manifest["cells"]) == len(cells), "manifest active-audit cell count mismatch")

    by_combo: dict[tuple[int, str, str, int], list[dict[str, Any]]] = {}
    for row in trials:
        combo = (
            int(row["seed"]),
            str(row["arm"]),
            str(row["method"]),
            int(row["budget"]),
        )
        by_combo.setdefault(combo, []).append(row)

    for cell_index, cell in enumerate(cells):
        cell_key = (int(cell["seed"]), str(cell["arm"]))
        for budget_index, budget in enumerate(budgets):
            for method_index, method in enumerate(METHODS):
                combo = (*cell_key, method, budget)
                group = by_combo.get(combo, [])
                _require(len(group) == replicates, f"trial replicate count mismatch {combo}")
                expected_seeds = {
                    int(lock["seed_offset"])
                    + cell_index * 10_000_019
                    + budget_index * 100_003
                    + method_index * 10_007
                    + replicate
                    for replicate in range(replicates)
                }
                actual_seeds = {int(row["audit_seed"]) for row in group}
                _require(len(actual_seeds) == len(group), f"duplicate audit seed {combo}")
                _require(actual_seeds == expected_seeds, f"audit seed schedule mismatch {combo}")
                for row in group:
                    _validate_trial(row, cell, budget=budget, method=method)

    expected_combos = {
        (int(cell["seed"]), str(cell["arm"]), method, budget)
        for cell in cells for budget in budgets for method in METHODS
    }
    _require(set(by_combo) == expected_combos, "unexpected active-audit trial combination")

    aggregate = _aggregate(trials)
    _validate_summary_rows(summary["summary_rows"], aggregate, context="summary")
    primary_budget = int(lock["primary_budget"])
    _require(int(summary["primary_budget"]) == primary_budget, "primary budget mismatch")
    expected_primary = [row for row in aggregate if int(row["budget"]) == primary_budget]
    _validate_summary_rows(summary["primary_rows"], expected_primary, context="primary")
    _require(summary.get("limitations") == lock["limitations"], "active-audit limitations mismatch")

    return {
        "valid": True,
        "execution_status": "completed",
        "trials": len(trials),
        "cells": len(cells),
        "methods": list(METHODS),
        "budgets": budgets,
        "primary_budget": primary_budget,
        "input_research_source_sha": manifest["input_research_source_sha"],
        "input_protocol_sha256": input_manifest["protocol_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate_evidence(args.root, args.protocol, args.input_root)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
