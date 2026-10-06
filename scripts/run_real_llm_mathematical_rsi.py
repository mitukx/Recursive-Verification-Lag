"""Real-LLM replay of OAI-math-inspired RSI verification controllers.

Consumes a completed qwen_learned_verifier_bridge_v1 artifact. The candidate
bank, trusted rewards, verifier arms and terminal preference shifts are all
retained real-model evidence. This script changes only the *controller replay*.

The OpenAI Math results motivate the controller architecture; none of those
theorems is claimed to transfer directly to LLM RSI.
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

from scripts.run_real_llm_active_audit import (
    load_json,
    load_real_candidate_cells,
    run_one_audit,
    sha256,
    validate_input_manifest,
)


METHODS = (
    "passive_fixed",
    "active_minimax",
    "coded_random",
    "coded_fixed_probe",
    "coded_fixed_probe_info",
)


def _stable_strata(ids: list[str], count: int) -> list[list[int]]:
    if count <= 0 or count > len(ids):
        raise ValueError("probe strata must lie in [1, candidate_count]")
    ordered = sorted(range(len(ids)), key=lambda i: ids[i])
    strata = [[] for _ in range(count)]
    for rank, index in enumerate(ordered):
        strata[rank % count].append(index)
    if any(not stratum for stratum in strata):
        raise AssertionError("fixed probe strata unexpectedly empty")
    return strata


def _radius(n: int, delta: float) -> float:
    if n <= 0:
        return float("inf")
    if not 0 < delta < 1:
        raise ValueError("delta must lie in (0,1)")
    # z=y(v-E[v]) lies in [-1,1]. Conservative Hoeffding interval.
    return math.sqrt(2.0 * math.log(2.0 / delta) / n)


def _local_decision(estimate: float, radius: float) -> str:
    if estimate + radius < 0:
        return "block"
    if estimate - radius > 0:
        return "allow"
    return "inconclusive"


def _coded_action(
    global_estimate: float,
    global_radius: float,
    local_decisions: list[str],
    *,
    allowed_corrupt: int,
) -> str:
    if global_estimate + global_radius < 0:
        return "block"
    if global_estimate - global_radius <= 0:
        return "refresh_verifier"
    local_failures = sum(decision != "allow" for decision in local_decisions)
    if local_failures <= allowed_corrupt:
        return "allow"
    return "refresh_verifier"


def _charged_result(
    cell: dict[str, Any],
    *,
    method: str,
    budget: int,
    audit_seed: int,
    action: str,
    estimate: float,
    radius: float,
    unique: int,
    probe_coverage: float,
    local_decisions: list[str],
    allowed_corrupt: int,
    information_budget_required: int | None = None,
    information_budget_passed: bool | None = None,
) -> dict[str, Any]:
    exact = float(cell["exact_covariance"])
    decision = {
        "block": "block",
        "allow": "allow",
        "refresh_verifier": "inconclusive",
    }[action]
    wrong_sign = int(
        (decision == "block" and exact > 0)
        or (decision == "allow" and exact < 0)
    )
    harmful = bool(cell["harmful_update"])
    beneficial = bool(cell["beneficial_update"])
    return {
        "seed": int(cell["seed"]),
        "arm": str(cell["arm"]),
        "method": method,
        "budget": int(budget),
        "audit_seed": int(audit_seed),
        "decision": decision,
        "controller_action": action,
        "estimate": float(estimate),
        "radius": float(radius),
        "lower": float(estimate - radius),
        "upper": float(estimate + radius),
        "exact_covariance": exact,
        "terminal_preference_shift": float(cell["terminal_preference_shift"]),
        "harmful_update": int(harmful),
        "beneficial_update": int(beneficial),
        "harmful_detected": int(harmful and decision == "block"),
        "harmful_allowed": int(harmful and decision == "allow"),
        "beneficial_blocked": int(beneficial and decision == "block"),
        "decisive": int(decision != "inconclusive"),
        "wrong_sign_decisive": wrong_sign,
        "correct_covariance_decisive": int(decision != "inconclusive" and not wrong_sign),
        "inconclusive": int(decision == "inconclusive"),
        "trusted_label_calls": int(budget),
        "unique_candidate_identities": int(unique),
        "probe_coverage": float(probe_coverage),
        "local_decisions": list(local_decisions),
        "allowed_corrupt_local_obligations": int(allowed_corrupt),
        "information_budget_required": information_budget_required,
        "information_budget_passed": information_budget_passed,
    }


def _coded_random(
    cell: dict[str, Any],
    *,
    budget: int,
    delta: float,
    audit_seed: int,
    strata_count: int,
    corrupt_fraction: float,
) -> dict[str, Any]:
    p = np.asarray(cell["p"], float)
    y = np.asarray(cell["y"], float)
    v = np.asarray(cell["v"], float)
    ids = list(cell["candidate_ids"])
    strata = _stable_strata(ids, strata_count)
    stratum_by_index = {
        index: s for s, indices in enumerate(strata) for index in indices
    }
    rng = np.random.default_rng(audit_seed)
    draws = rng.choice(len(p), size=budget, replace=True, p=p)
    z = y * (v - float(p @ v))
    global_estimate = float(np.mean(z[draws]))
    global_radius = _radius(budget, delta / 2.0)
    local_delta = delta / (2.0 * strata_count)
    local_decisions = []
    hit = 0
    for s in range(strata_count):
        local = [int(i) for i in draws if stratum_by_index[int(i)] == s]
        if local:
            hit += 1
            estimate = float(np.mean(z[local]))
            radius = _radius(len(local), local_delta)
        else:
            estimate, radius = 0.0, float("inf")
        local_decisions.append(_local_decision(estimate, radius))
    allowed = math.floor(corrupt_fraction * strata_count)
    action = _coded_action(
        global_estimate,
        global_radius,
        local_decisions,
        allowed_corrupt=allowed,
    )
    return _charged_result(
        cell,
        method="coded_random",
        budget=budget,
        audit_seed=audit_seed,
        action=action,
        estimate=global_estimate,
        radius=global_radius,
        unique=len(np.unique(draws)),
        probe_coverage=hit / strata_count,
        local_decisions=local_decisions,
        allowed_corrupt=allowed,
    )


def _allocate_budget(budget: int, strata_count: int) -> list[int]:
    base, remainder = divmod(budget, strata_count)
    return [base + (1 if i < remainder else 0) for i in range(strata_count)]


def _coded_fixed_probe(
    cell: dict[str, Any],
    *,
    budget: int,
    delta: float,
    audit_seed: int,
    strata_count: int,
    corrupt_fraction: float,
    target_error: float,
    information_multiplier: float,
    enforce_information_budget: bool,
) -> dict[str, Any]:
    p = np.asarray(cell["p"], float)
    y = np.asarray(cell["y"], float)
    v = np.asarray(cell["v"], float)
    ids = list(cell["candidate_ids"])
    strata = _stable_strata(ids, strata_count)
    z = y * (v - float(p @ v))
    allocations = _allocate_budget(budget, strata_count)
    rng = np.random.default_rng(audit_seed)
    local_delta = delta / strata_count
    local_decisions = []
    estimates = []
    radii = []
    weights = []
    all_draws: list[int] = []
    for s, indices in enumerate(strata):
        n = allocations[s]
        weight = len(indices) / len(ids)
        weights.append(weight)
        if n <= 0:
            estimates.append(0.0)
            radii.append(float("inf"))
            local_decisions.append("inconclusive")
            continue
        draws = rng.choice(indices, size=n, replace=True)
        all_draws.extend(int(i) for i in draws)
        estimate = float(np.mean(z[draws]))
        radius = _radius(n, local_delta)
        estimates.append(estimate)
        radii.append(radius)
        local_decisions.append(_local_decision(estimate, radius))

    if any(not math.isfinite(r) for r in radii):
        global_estimate = 0.0
        global_radius = float("inf")
    else:
        global_estimate = float(sum(w * e for w, e in zip(weights, estimates)))
        global_radius = float(sum(w * r for w, r in zip(weights, radii)))
    allowed = math.floor(corrupt_fraction * strata_count)
    action = _coded_action(
        global_estimate,
        global_radius,
        local_decisions,
        allowed_corrupt=allowed,
    )

    required = math.ceil(
        information_multiplier * strata_count * math.log(1.0 / target_error)
    )
    info_passed = budget >= required
    if enforce_information_budget and not info_passed:
        action = "refresh_verifier"
    return _charged_result(
        cell,
        method=(
            "coded_fixed_probe_info"
            if enforce_information_budget
            else "coded_fixed_probe"
        ),
        budget=budget,
        audit_seed=audit_seed,
        action=action,
        estimate=global_estimate,
        radius=global_radius,
        unique=len(set(all_draws)),
        probe_coverage=sum(n > 0 for n in allocations) / strata_count,
        local_decisions=local_decisions,
        allowed_corrupt=allowed,
        information_budget_required=required,
        information_budget_passed=info_passed,
    )


def _baseline(cell: dict[str, Any], *, method: str, budget: int, delta: float, seed: int):
    row = run_one_audit(
        cell,
        method=method,
        budget=budget,
        delta=delta,
        propensity_minimum=0.02,
        seed=seed,
    )
    row["probe_coverage"] = None
    row["local_decisions"] = []
    row["allowed_corrupt_local_obligations"] = None
    row["information_budget_required"] = None
    row["information_budget_passed"] = None
    return row


def summarize_rows(rows: list[dict[str, Any]], keys=("method", "budget")) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)
    result = []
    for group_key, group in sorted(grouped.items()):
        harmful = [r for r in group if r["harmful_update"]]
        beneficial = [r for r in group if r["beneficial_update"]]
        decisive = [r for r in group if r["decisive"]]
        total_calls = sum(int(r["trusted_label_calls"]) for r in group)
        row = {key: value for key, value in zip(keys, group_key)}
        row.update(
            {
                "trials": len(group),
                "harmful_trials": len(harmful),
                "beneficial_trials": len(beneficial),
                "harmful_update_detection_rate": (
                    float(np.mean([r["harmful_detected"] for r in harmful]))
                    if harmful else None
                ),
                "harmful_update_allow_rate": (
                    float(np.mean([r["harmful_allowed"] for r in harmful]))
                    if harmful else None
                ),
                "beneficial_update_block_rate": (
                    float(np.mean([r["beneficial_blocked"] for r in beneficial]))
                    if beneficial else None
                ),
                "wrong_sign_decisive_rate": (
                    float(np.mean([r["wrong_sign_decisive"] for r in decisive]))
                    if decisive else None
                ),
                "inconclusive_rate": float(np.mean([r["inconclusive"] for r in group])),
                "mean_probe_coverage": (
                    float(np.mean([r["probe_coverage"] for r in group if r["probe_coverage"] is not None]))
                    if any(r["probe_coverage"] is not None for r in group) else None
                ),
                "mean_unique_candidate_identities": float(
                    np.mean([r["unique_candidate_identities"] for r in group])
                ),
                "harmful_blocks_per_100_label_calls": (
                    100.0 * sum(int(r["harmful_detected"]) for r in group) / total_calls
                    if total_calls else None
                ),
            }
        )
        result.append(row)
    return result


def reconstruction_diagnostics(cells: list[dict[str, Any]], branching_factors: list[int]):
    rows = []
    for cell in cells:
        y = np.asarray(cell["y"], float)
        v = np.asarray(cell["v"], float)
        agreement = float(np.mean((v >= 0.5) == (y >= 0.5)))
        lam = 2.0 * agreement - 1.0
        for d in branching_factors:
            criticality = d * lam * lam
            rows.append(
                {
                    "seed": int(cell["seed"]),
                    "arm": cell["arm"],
                    "branching_factor": int(d),
                    "thresholded_agreement": agreement,
                    "lambda_hat": lam,
                    "criticality": criticality,
                    "supercritical_diagnostic": bool(criticality > 1.0),
                    "assumptions_met": False,
                }
            )
    return rows


def _deterministic_gzip(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_bytes(gzip.compress(text.encode(), compresslevel=9, mtime=0))


def run(protocol_path: Path, input_root: Path, output: Path) -> dict[str, Any]:
    lock = load_json(protocol_path)
    if lock.get("status") != "locked retrospective real-LLM mathematical-RSI replay; no downstream outcome tuning":
        raise ValueError("locked mathematical-RSI replay protocol required")
    requirements = lock["input_requirements"]
    input_manifest = validate_input_manifest(input_root, requirements)
    cells = load_real_candidate_cells(input_root, requirements)
    budgets = [int(x) for x in lock["trusted_label_budgets"]]
    replicates = int(lock["audit_replicates"])
    delta = float(lock["confidence_delta"])
    strata_count = int(lock["coded_verification"]["probe_strata"])
    corrupt_fraction = float(lock["coded_verification"]["max_corrupt_fraction"])
    target_error = float(lock["information_budget"]["target_error"])
    info_multiplier = float(lock["information_budget"]["multiplier"])
    seed_offset = int(lock["seed_offset"])

    if output.exists():
        raise FileExistsError(output)
    if any(b <= 0 for b in budgets) or replicates <= 0:
        raise ValueError("positive budgets and replicates required")
    if not 0 <= corrupt_fraction < 1:
        raise ValueError("invalid corruption fraction")
    if not 0 < target_error < 1:
        raise ValueError("invalid target error")
    min_bank = min(len(cell["y"]) for cell in cells)
    if strata_count > min_bank:
        raise ValueError("probe strata exceed smallest candidate bank")
    if any(b > min_bank for b in budgets):
        raise ValueError("audit budget exceeds smallest real candidate bank")

    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for cell_index, cell in enumerate(cells):
        for budget_index, budget in enumerate(budgets):
            for method_index, method in enumerate(METHODS):
                for replicate in range(replicates):
                    seed = (
                        seed_offset
                        + cell_index * 10_000_019
                        + budget_index * 100_003
                        + method_index * 10_007
                        + replicate
                    )
                    if method in {"passive_fixed", "active_minimax"}:
                        row = _baseline(
                            cell, method=method, budget=budget, delta=delta, seed=seed
                        )
                    elif method == "coded_random":
                        row = _coded_random(
                            cell,
                            budget=budget,
                            delta=delta,
                            audit_seed=seed,
                            strata_count=strata_count,
                            corrupt_fraction=corrupt_fraction,
                        )
                    else:
                        row = _coded_fixed_probe(
                            cell,
                            budget=budget,
                            delta=delta,
                            audit_seed=seed,
                            strata_count=strata_count,
                            corrupt_fraction=corrupt_fraction,
                            target_error=target_error,
                            information_multiplier=info_multiplier,
                            enforce_information_budget=(method == "coded_fixed_probe_info"),
                        )
                    rows.append(row)

    summary_rows = summarize_rows(rows)
    arm_rows = summarize_rows(rows, keys=("arm", "method", "budget"))
    primary_budget = int(lock["primary_budget"])
    primary = [r for r in summary_rows if int(r["budget"]) == primary_budget]
    diagnostics = reconstruction_diagnostics(
        cells, [int(x) for x in lock["reconstruction_diagnostic"]["branching_factors"]]
    )
    _deterministic_gzip(output / "trials.jsonl.gz", rows)
    summary = {
        "primary_budget": primary_budget,
        "primary_rows": primary,
        "summary_rows": summary_rows,
        "arm_rows": arm_rows,
        "reconstruction_diagnostics": diagnostics,
        "controller_claims": {
            "coded_verification": "engineering analogue of robust local verification; not a PCP reduction",
            "fixed_probe_coverage": "engineering bounded probe class; not a universal hitting-set theorem",
            "information_budget": "d*log(1/epsilon) research proxy; not a transferred LLM lower bound",
            "reconstruction": "diagnostic only because tree-channel assumptions are not established",
        },
        "interpretation_rule": lock["interpretation_rule"],
        "limitations": lock["limitations"],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    manifest = {
        "status": "completed",
        "protocol_sha256": sha256(protocol_path),
        "input_manifest_sha256": sha256(input_root / "manifest.json"),
        "input_manifest_status": input_manifest["status"],
        "input_research_source_sha": load_json(input_root / "environment.json").get("research_source_sha"),
        "cells": len(cells),
        "trials": len(rows),
        "files": {
            "trials.jsonl.gz": sha256(output / "trials.jsonl.gz"),
            "summary.json": sha256(output / "summary.json"),
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("configs/real_llm_mathematical_rsi_v1.json"))
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/real_llm_mathematical_rsi_v1"))
    args = parser.parse_args()
    run(args.protocol, args.input_root, args.output)


if __name__ == "__main__":
    main()
