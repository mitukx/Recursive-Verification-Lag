"""Offline trusted-label audit replay on a real learned-verifier LLM candidate bank.

Input is a completed qwen_learned_verifier_bridge_v1 artifact. The estimator
implementations are imported from the existing synthetic/prospective audit code;
this file only adapts them to retained real LLM candidate occurrences and joins
their decisions to the independently measured terminal update outcome.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from src.active_alignment_audit import active_proposal, covariance_audit
from src.propensity_alignment_audit import (
    allocate_propensities,
    propensity_covariance_audit,
)


ARMS = ("oracle", "fresh", "stale", "shuffled")
METHODS = ("passive_fixed", "propensity_ht", "active_minimax")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_input_manifest(root: Path) -> dict[str, Any]:
    manifest = load_json(root / "manifest.json")
    if manifest.get("status") != "completed":
        raise ValueError("real-LLM audit requires a completed learned-verifier artifact")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("input manifest lacks file hashes")
    actual = {
        str(path.relative_to(root)): sha256(path)
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if set(actual) != set(files):
        raise ValueError("input artifact file set differs from manifest")
    for rel, digest in files.items():
        if actual[rel] != digest:
            raise ValueError(f"input artifact SHA mismatch: {rel}")
    return manifest


def load_real_candidate_cells(root: Path) -> list[dict[str, Any]]:
    cells = []
    for seed_root in sorted(root.glob("seed-*")):
        summary_path = seed_root / "seed_summary.json"
        scores_path = seed_root / "effect_verifier_scores.jsonl"
        if not summary_path.exists() or not scores_path.exists():
            continue
        summary = load_json(summary_path)
        if "arms" not in summary or "mean_preference_shift" not in summary:
            continue
        score_rows = load_jsonl(scores_path)
        ids = [str(row["candidate_id"]) for row in score_rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate candidate identity in {seed_root.name}")
        if not score_rows:
            raise ValueError(f"empty candidate bank in {seed_root.name}")
        y = np.asarray([float(row["trusted_reward"]) for row in score_rows], float)
        if not np.isfinite(y).all() or np.any((y < 0) | (y > 1)):
            raise ValueError(f"invalid trusted rewards in {seed_root.name}")
        p = np.full(len(y), 1.0 / len(y), dtype=float)
        recorded_geometry = summary.get("pre_update_geometry", {})
        for arm in ARMS:
            key = f"{arm}_score"
            v = np.asarray([float(row[key]) for row in score_rows], float)
            if not np.isfinite(v).all() or np.any((v < 0) | (v > 1)):
                raise ValueError(f"invalid verifier scores for {seed_root.name}/{arm}")
            mean_v = float(p @ v)
            exact_cov = float(p @ (y * (v - mean_v)))
            if arm in recorded_geometry:
                recorded = float(recorded_geometry[arm]["cov_y_v"])
                if not np.isclose(exact_cov, recorded, atol=1e-10, rtol=1e-8):
                    raise ValueError(
                        f"recorded geometry mismatch for {seed_root.name}/{arm}"
                    )
            progress = float(summary["mean_preference_shift"][arm])
            cells.append(
                {
                    "seed": int(summary["seed"]),
                    "arm": arm,
                    "candidate_ids": ids,
                    "p": p,
                    "y": y,
                    "v": v,
                    "exact_covariance": exact_cov,
                    "terminal_preference_shift": progress,
                    "harmful_update": int(progress < 0),
                    "beneficial_update": int(progress > 0),
                }
            )
    if not cells:
        raise ValueError("no completed learned-verifier seed cells found")
    return cells


def _fixed_unique_count(p: np.ndarray, v: np.ndarray, *, budget: int, seed: int, active: bool) -> int:
    proposal = active_proposal(p, v)["proposal"] if active else p
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(p), size=budget, replace=True, p=proposal)
    return int(len(np.unique(indices)))


def run_one_audit(
    cell: dict[str, Any],
    *,
    method: str,
    budget: int,
    delta: float,
    propensity_minimum: float,
    seed: int,
) -> dict[str, Any]:
    p, y, v = cell["p"], cell["y"], cell["v"]
    if method == "passive_fixed":
        audit = covariance_audit(
            p, y, v, n=budget, delta=delta, seed=seed, mode="passive"
        )
        trusted_calls = budget
        unique = _fixed_unique_count(p, v, budget=budget, seed=seed, active=False)
    elif method == "active_minimax":
        audit = covariance_audit(
            p, y, v, n=budget, delta=delta, seed=seed, mode="active"
        )
        trusted_calls = budget
        unique = _fixed_unique_count(p, v, budget=budget, seed=seed, active=True)
    elif method == "propensity_ht":
        propensities = allocate_propensities(
            p,
            v,
            expected_budget=float(budget),
            method="proxy_leverage",
            minimum_propensity=propensity_minimum,
        )
        audit = propensity_covariance_audit(
            p, y, v, propensities, delta=delta, seed=seed
        )
        trusted_calls = int(audit["realized_labels"])
        unique = len(audit["selected_indices"])
    else:
        raise ValueError(f"unknown audit method: {method}")

    decision = audit["decision"]
    action = {
        "block": "block",
        "allow": "allow",
        "inconclusive": "refresh_verifier",
    }[decision]
    harmful = bool(cell["harmful_update"])
    beneficial = bool(cell["beneficial_update"])
    decisive = decision != "inconclusive"
    wrong_sign = int(audit["wrong_sign_decision"])
    return {
        "seed": int(cell["seed"]),
        "arm": cell["arm"],
        "method": method,
        "budget": int(budget),
        "audit_seed": int(seed),
        "decision": decision,
        "controller_action": action,
        "estimate": float(audit["estimate"]),
        "radius": float(audit["radius"]),
        "lower": float(audit["lower"]),
        "upper": float(audit["upper"]),
        "exact_covariance": float(cell["exact_covariance"]),
        "terminal_preference_shift": float(cell["terminal_preference_shift"]),
        "harmful_update": int(harmful),
        "beneficial_update": int(beneficial),
        "harmful_detected": int(harmful and decision == "block"),
        "harmful_allowed": int(harmful and decision == "allow"),
        "beneficial_blocked": int(beneficial and decision == "block"),
        "decisive": int(decisive),
        "wrong_sign_decisive": int(wrong_sign),
        "correct_covariance_decisive": int(decisive and not wrong_sign),
        "inconclusive": int(decision == "inconclusive"),
        "trusted_label_calls": int(trusted_calls),
        "unique_candidate_identities": int(unique),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["method"], int(row["budget"])), []).append(row)
    summary = []
    for (method, budget), group in sorted(grouped.items()):
        harmful = [row for row in group if row["harmful_update"]]
        beneficial = [row for row in group if row["beneficial_update"]]
        decisive = [row for row in group if row["decisive"]]
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
                    float(np.mean([row["harmful_detected"] for row in harmful]))
                    if harmful
                    else None
                ),
                "harmful_update_allow_rate": (
                    float(np.mean([row["harmful_allowed"] for row in harmful]))
                    if harmful
                    else None
                ),
                "beneficial_update_block_rate": (
                    float(np.mean([row["beneficial_blocked"] for row in beneficial]))
                    if beneficial
                    else None
                ),
                "wrong_sign_decisive_rate": (
                    float(np.mean([row["wrong_sign_decisive"] for row in decisive]))
                    if decisive
                    else None
                ),
                "inconclusive_rate": float(
                    np.mean([row["inconclusive"] for row in group])
                ),
                "correct_decisive_per_100_label_calls": (
                    100.0 * correct_decisive / total_labels if total_labels else None
                ),
                "harmful_blocks_per_100_label_calls": (
                    100.0 * harmful_blocks / total_labels if total_labels else None
                ),
                "mean_realized_trusted_label_calls": float(
                    np.mean([row["trusted_label_calls"] for row in group])
                ),
                "mean_unique_candidate_identities": float(
                    np.mean([row["unique_candidate_identities"] for row in group])
                ),
            }
        )
    return summary


def _deterministic_gzip_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_bytes(gzip.compress(text.encode("utf-8"), compresslevel=9, mtime=0))


def run(protocol_path: Path, input_root: Path, output: Path) -> dict[str, Any]:
    lock = load_json(protocol_path)
    if not lock["status"].startswith(
        "prospective real-LLM active-audit replay locked before learned-verifier GPU outcomes"
    ):
        raise ValueError("locked real-LLM active-audit protocol required")
    input_manifest = validate_input_manifest(input_root)
    cells = load_real_candidate_cells(input_root)
    budgets = [int(x) for x in lock["trusted_label_budgets"]]
    replicates = int(lock["audit_replicates"])
    delta = float(lock["confidence_delta"])
    floor = float(lock["propensity_minimum"])
    if replicates <= 0 or any(b <= 0 for b in budgets):
        raise ValueError("positive budgets/replicates required")
    min_bank = min(len(cell["y"]) for cell in cells)
    if any(b > min_bank for b in budgets):
        raise ValueError("audit budget exceeds smallest real candidate bank")

    output.mkdir(parents=True, exist_ok=False)
    cell_manifest = []
    rows = []
    for cell_index, cell in enumerate(cells):
        cell_manifest.append(
            {
                "seed": cell["seed"],
                "arm": cell["arm"],
                "candidate_count": len(cell["y"]),
                "exact_covariance": cell["exact_covariance"],
                "terminal_preference_shift": cell["terminal_preference_shift"],
                "harmful_update": cell["harmful_update"],
                "beneficial_update": cell["beneficial_update"],
            }
        )
        for budget_index, budget in enumerate(budgets):
            for method_index, method in enumerate(METHODS):
                for replicate in range(replicates):
                    audit_seed = (
                        int(lock["seed_offset"])
                        + cell_index * 10_000_019
                        + budget_index * 100_003
                        + method_index * 10_007
                        + replicate
                    )
                    rows.append(
                        run_one_audit(
                            cell,
                            method=method,
                            budget=budget,
                            delta=delta,
                            propensity_minimum=floor,
                            seed=audit_seed,
                        )
                    )

    summary_rows = summarize_rows(rows)
    primary_budget = int(lock["primary_budget"])
    primary = [
        row for row in summary_rows if int(row["budget"]) == primary_budget
    ]
    _deterministic_gzip_jsonl(output / "audit_trials.jsonl.gz", rows)
    (output / "summary.json").write_text(
        json.dumps(
            {
                "input_cells": cell_manifest,
                "summary_rows": summary_rows,
                "primary_budget": primary_budget,
                "primary_rows": primary,
                "interpretation_rule": (
                    "report harmful-update detection and terminal controller errors "
                    "separately from estimator wrong-sign rate relative to exact Cov(y,v)"
                ),
                "limitations": lock["limitations"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    manifest = {
        "status": "completed",
        "protocol_sha256": sha256(protocol_path),
        "input_manifest_sha256": sha256(input_root / "manifest.json"),
        "input_research_source_sha": load_json(input_root / "environment.json").get(
            "research_source_sha"
        ),
        "input_manifest_status": input_manifest["status"],
        "rows": len(rows),
        "cells": len(cells),
        "files": {
            "audit_trials.jsonl.gz": sha256(output / "audit_trials.jsonl.gz"),
            "summary.json": sha256(output / "summary.json"),
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return load_json(output / "summary.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/real_llm_active_audit_v1.json"),
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/real_llm_active_audit_v1"),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    run(args.protocol, args.input_root, args.output)


if __name__ == "__main__":
    main()
