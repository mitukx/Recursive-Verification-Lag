"""Exploratory OAI-math-inspired RSI replay on the retained real-Qwen code bank.

This is CPU-only mechanism evidence on an already-inspected development bank.
The bank contains actual Qwen generations. Public scores/features are observable;
trusted scores are evaluator-only and enter a controller only when that candidate
is explicitly audited. Full trusted scores are used after each decision only to
measure the outcome.

The OpenAI Math results motivate the architecture. They do not imply the
certificates below, nor do these experiments prove RSI safety.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from src.candidate_bank_experiment import Config, load_jsonl, propose, task_policy


METHODS = (
    "proxy_only",
    "global_impact",
    "global_random",
    "coded_random",
    "coded_fixed_probe",
    "coded_fixed_probe_info",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def build_cells(df, *, etas: list[float], rounds: int, tolerance: float) -> list[dict[str, Any]]:
    df = df.reset_index(drop=True)
    public = df["f::public_score"].to_numpy(float)
    trusted = df.trusted_score.to_numpy(float)
    baseline = task_policy(df, df.base_logprob.to_numpy(float))
    cells: list[dict[str, Any]] = []
    for eta in etas:
        cfg = Config(eta=float(eta), rounds=rounds, representation="public")
        p = baseline.copy()
        for round_index in range(1, rounds + 1):
            q = propose(df, p, public, cfg)
            delta = q - p
            proxy_gain = float(delta @ public)
            true_gain = float(delta @ trusted)
            task_true = {}
            task_proxy = {}
            for task_id, indices in df.groupby("task_id", sort=True).indices.items():
                ii = np.asarray(indices, int)
                task_true[str(task_id)] = float(delta[ii] @ trusted[ii])
                task_proxy[str(task_id)] = float(delta[ii] @ public[ii])
            cells.append(
                {
                    "cell_id": f"eta={eta:g}:round={round_index}",
                    "eta": float(eta),
                    "round": round_index,
                    "p": p.copy(),
                    "q": q.copy(),
                    "delta": delta,
                    "proxy_gain": proxy_gain,
                    "true_gain": true_gain,
                    "task_true_gain": task_true,
                    "task_proxy_gain": task_proxy,
                    "harmful": bool(true_gain < -tolerance),
                    "beneficial": bool(true_gain > tolerance),
                    "false_progress": bool(proxy_gain > tolerance and true_gain < -tolerance),
                }
            )
            p = q
    return cells


def box_bounds(delta: np.ndarray, trusted: np.ndarray, revealed: set[int]) -> tuple[float, float]:
    known = np.fromiter(sorted(revealed), dtype=int) if revealed else np.asarray([], int)
    mask = np.ones(len(delta), dtype=bool)
    if len(known):
        mask[known] = False
    fixed = float(delta[known] @ trusted[known]) if len(known) else 0.0
    unknown = delta[mask]
    return (
        fixed + float(np.minimum(unknown, 0.0).sum()),
        fixed + float(np.maximum(unknown, 0.0).sum()),
    )


def task_bounds(df, delta: np.ndarray, trusted: np.ndarray, revealed: set[int]):
    out = {}
    for task_id, indices in df.groupby("task_id", sort=True).indices.items():
        ii = np.asarray(indices, int)
        local_revealed = set(int(i) for i in ii if int(i) in revealed)
        local_index = {int(global_i): local_i for local_i, global_i in enumerate(ii)}
        rel = {local_index[i] for i in local_revealed}
        lo, hi = box_bounds(delta[ii], trusted[ii], rel)
        out[str(task_id)] = {"lower": lo, "upper": hi}
    return out


def allocate_fixed_probes(df, delta: np.ndarray, budget: int) -> list[int]:
    groups = [(str(task_id), np.asarray(indices, int))
              for task_id, indices in df.groupby("task_id", sort=True).indices.items()]
    if budget < len(groups):
        raise ValueError("fixed probe budget must cover every task at least once")
    selected: list[int] = []
    remaining: dict[str, list[int]] = {}
    for task_id, ii in groups:
        order = sorted(ii.tolist(), key=lambda i: (-abs(float(delta[i])), str(df.iloc[i].candidate_id)))
        selected.append(order[0])
        remaining[task_id] = order[1:]
    cursor = 0
    while len(selected) < budget:
        task_id = groups[cursor % len(groups)][0]
        cursor += 1
        if remaining[task_id]:
            selected.append(remaining[task_id].pop(0))
        elif not any(remaining.values()):
            break
    return selected


def allocate_global_impact(df, delta: np.ndarray, budget: int) -> list[int]:
    return sorted(
        range(len(df)),
        key=lambda i: (-abs(float(delta[i])), str(df.iloc[i].candidate_id)),
    )[:budget]


def allocate_random(n: int, budget: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=min(budget, n), replace=False).astype(int).tolist()


def task_coverage(df, revealed: set[int]) -> float:
    tasks = list(df.groupby("task_id", sort=True).indices.values())
    return float(np.mean([any(int(i) in revealed for i in ii) for ii in tasks]))


def decision_from_bounds(
    df,
    delta: np.ndarray,
    trusted: np.ndarray,
    revealed: set[int],
    *,
    coded: bool,
    max_corrupt_fraction: float,
    local_slack: float,
    force_refresh: bool = False,
) -> dict[str, Any]:
    lo, hi = box_bounds(delta, trusted, revealed)
    local = task_bounds(df, delta, trusted, revealed)
    violated = tuple(sorted(
        task_id for task_id, bounds in local.items()
        if float(bounds["upper"]) < -local_slack
    ))
    allowed = math.floor(max_corrupt_fraction * len(local))
    if force_refresh:
        action = "refresh"
    elif hi < 0.0:
        action = "block"
    elif lo < 0.0:
        action = "refresh"
    elif coded and len(violated) > allowed:
        action = "refresh"
    else:
        action = "allow"
    return {
        "action": action,
        "lower": lo,
        "upper": hi,
        "certified_local_violations": violated,
        "allowed_local_violations": allowed,
        "task_coverage": task_coverage(df, revealed),
    }


def run_controller(
    df,
    cell: dict[str, Any],
    trusted: np.ndarray,
    *,
    method: str,
    budget: int,
    seed: int,
    max_corrupt_fraction: float,
    local_slack: float,
    info_target_error: float,
    info_multiplier: float,
) -> dict[str, Any]:
    delta = np.asarray(cell["delta"], float)
    task_count = int(df.task_id.nunique())
    required = math.ceil(info_multiplier * task_count * math.log(1.0 / info_target_error))
    if method == "proxy_only":
        revealed: set[int] = set()
        result = {
            "action": "allow" if float(cell["proxy_gain"]) > 0 else "block",
            "lower": None,
            "upper": None,
            "certified_local_violations": (),
            "allowed_local_violations": math.floor(max_corrupt_fraction * task_count),
            "task_coverage": 0.0,
        }
    elif method == "global_impact":
        revealed = set(allocate_global_impact(df, delta, budget))
        result = decision_from_bounds(
            df, delta, trusted, revealed, coded=False,
            max_corrupt_fraction=max_corrupt_fraction, local_slack=local_slack,
        )
    elif method in {"global_random", "coded_random"}:
        revealed = set(allocate_random(len(df), budget, seed))
        result = decision_from_bounds(
            df, delta, trusted, revealed, coded=(method == "coded_random"),
            max_corrupt_fraction=max_corrupt_fraction, local_slack=local_slack,
        )
        if method == "coded_random" and result["task_coverage"] < 1.0:
            result["action"] = "refresh"
    elif method in {"coded_fixed_probe", "coded_fixed_probe_info"}:
        revealed = set(allocate_fixed_probes(df, delta, budget))
        force = method == "coded_fixed_probe_info" and budget < required
        result = decision_from_bounds(
            df, delta, trusted, revealed, coded=True,
            max_corrupt_fraction=max_corrupt_fraction, local_slack=local_slack,
            force_refresh=force,
        )
    else:
        raise ValueError(f"unknown method: {method}")

    harmful_tasks = sum(float(v) < -local_slack for v in cell["task_true_gain"].values())
    severe_local_regression = harmful_tasks > math.floor(max_corrupt_fraction * task_count)
    action = str(result["action"])
    return {
        "cell_id": cell["cell_id"],
        "eta": float(cell["eta"]),
        "round": int(cell["round"]),
        "method": method,
        "budget": int(0 if method == "proxy_only" else budget),
        "audit_seed": int(seed),
        "action": action,
        "proxy_gain": float(cell["proxy_gain"]),
        "true_gain": float(cell["true_gain"]),
        "harmful": int(cell["harmful"]),
        "beneficial": int(cell["beneficial"]),
        "false_progress": int(cell["false_progress"]),
        "harmful_allowed": int(cell["harmful"] and action == "allow"),
        "beneficial_blocked": int(cell["beneficial"] and action == "block"),
        "wrong_sign_decisive": int(
            (action == "allow" and cell["harmful"])
            or (action == "block" and cell["beneficial"])
        ),
        "inconclusive": int(action == "refresh"),
        "lower": result["lower"],
        "upper": result["upper"],
        "task_coverage": float(result["task_coverage"]),
        "certified_local_violations": list(result["certified_local_violations"]),
        "allowed_local_violations": int(result["allowed_local_violations"]),
        "true_harmful_tasks": int(harmful_tasks),
        "severe_local_regression": int(severe_local_regression),
        "severe_local_regression_allowed": int(severe_local_regression and action == "allow"),
        "trusted_label_calls": int(len(revealed)),
        "information_budget_required": int(required),
        "information_budget_passed": bool(method != "coded_fixed_probe_info" or budget >= required),
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["method"]), int(row["budget"])), []).append(row)
    out = []
    for (method, budget), group in sorted(grouped.items()):
        harmful = [r for r in group if r["harmful"]]
        beneficial = [r for r in group if r["beneficial"]]
        severe = [r for r in group if r["severe_local_regression"]]
        decisive = [r for r in group if not r["inconclusive"]]
        allowed = [r for r in group if r["action"] == "allow"]
        out.append({
            "method": method,
            "budget": budget,
            "trials": len(group),
            "harmful_trials": len(harmful),
            "beneficial_trials": len(beneficial),
            "severe_local_regression_trials": len(severe),
            "allow_rate": float(np.mean([r["action"] == "allow" for r in group])),
            "refresh_rate": float(np.mean([r["inconclusive"] for r in group])),
            "harmful_allow_rate": (
                float(np.mean([r["harmful_allowed"] for r in harmful])) if harmful else None
            ),
            "beneficial_block_rate": (
                float(np.mean([r["beneficial_blocked"] for r in beneficial])) if beneficial else None
            ),
            "wrong_sign_decisive_rate": (
                float(np.mean([r["wrong_sign_decisive"] for r in decisive])) if decisive else None
            ),
            "severe_local_regression_allow_rate": (
                float(np.mean([r["severe_local_regression_allowed"] for r in severe])) if severe else None
            ),
            "mean_true_gain_when_allowed": (
                float(np.mean([r["true_gain"] for r in allowed])) if allowed else None
            ),
            "mean_task_coverage": float(np.mean([r["task_coverage"] for r in group])),
            "mean_trusted_label_calls": float(np.mean([r["trusted_label_calls"] for r in group])),
        })
    return out


def run(protocol_path: Path, bank_path: Path, output: Path) -> dict[str, Any]:
    lock = load_json(protocol_path)
    if lock.get("status") != "locked exploratory recovered-Qwen mathematical-RSI replay":
        raise ValueError("locked exploratory protocol required")
    expected_sha = str(lock["input"]["bank_sha256"])
    if sha256(bank_path) != expected_sha:
        raise ValueError("candidate bank SHA mismatch")
    df = load_jsonl(bank_path)
    if int(df.task_id.nunique()) != int(lock["input"]["expected_tasks"]):
        raise ValueError("unexpected task count")
    trusted = df.trusted_score.to_numpy(float)
    cells = build_cells(
        df,
        etas=[float(x) for x in lock["etas"]],
        rounds=int(lock["rounds"]),
        tolerance=float(lock["outcome_tolerance"]),
    )
    budgets = [int(x) for x in lock["trusted_label_budgets"]]
    reps = int(lock["audit_replicates"])
    rows: list[dict[str, Any]] = []
    for cell_index, cell in enumerate(cells):
        for budget_index, budget in enumerate(budgets):
            for method_index, method in enumerate(METHODS):
                method_reps = 1 if method in {"proxy_only", "global_impact", "coded_fixed_probe", "coded_fixed_probe_info"} else reps
                for rep in range(method_reps):
                    seed = (
                        int(lock["seed_offset"])
                        + cell_index * 100003
                        + budget_index * 1009
                        + method_index * 97
                        + rep
                    )
                    rows.append(run_controller(
                        df, cell, trusted,
                        method=method, budget=budget, seed=seed,
                        max_corrupt_fraction=float(lock["coded_verification"]["max_corrupt_fraction"]),
                        local_slack=float(lock["coded_verification"]["local_slack"]),
                        info_target_error=float(lock["information_budget"]["target_error"]),
                        info_multiplier=float(lock["information_budget"]["multiplier"]),
                    ))
    summary_rows = summarize(rows)
    primary_budget = int(lock["primary_budget"])
    primary_rows = [r for r in summary_rows if r["budget"] in {0, primary_budget}]
    cell_summary = [{
        "cell_id": c["cell_id"],
        "eta": c["eta"],
        "round": c["round"],
        "proxy_gain": c["proxy_gain"],
        "true_gain": c["true_gain"],
        "harmful": c["harmful"],
        "beneficial": c["beneficial"],
        "false_progress": c["false_progress"],
        "true_harmful_tasks": int(sum(float(v) < -float(lock["coded_verification"]["local_slack"]) for v in c["task_true_gain"].values())),
    } for c in cells]
    result = {
        "evidence_status": "exploratory_real_Qwen_development_bank",
        "primary_budget": primary_budget,
        "primary_rows": primary_rows,
        "summary_rows": summary_rows,
        "cells": cell_summary,
        "counts": {
            "candidate_occurrences": len(df),
            "tasks": int(df.task_id.nunique()),
            "proposal_cells": len(cells),
            "harmful_cells": int(sum(c["harmful"] for c in cells)),
            "false_progress_cells": int(sum(c["false_progress"] for c in cells)),
        },
        "claim_boundary": (
            "Actual retained Qwen generations, but an already-inspected development bank; "
            "controller replay is exploratory and does not establish frontier-model RSI."
        ),
        "limitations": lock["limitations"],
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "trials.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows),
        encoding="utf-8",
    )
    (output / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    manifest = {
        "status": "completed",
        "protocol_sha256": sha256(protocol_path),
        "bank_sha256": sha256(bank_path),
        "files": {
            "trials.jsonl": sha256(output / "trials.jsonl"),
            "summary.json": sha256(output / "summary.json"),
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("configs/recovered_qwen_mathematical_rsi_v1.json"))
    parser.add_argument("--bank", type=Path, default=Path("data/recovered_qwen05b_bank.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/recovered_qwen_mathematical_rsi_v1"))
    args = parser.parse_args()
    result = run(args.protocol, args.bank, args.output)
    print(json.dumps(result["counts"], indent=2, sort_keys=True))
    print(json.dumps(result["primary_rows"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
