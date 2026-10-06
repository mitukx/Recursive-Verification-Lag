"""Evaluate reusable fixed hitting probes on the retained real-Qwen bank."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from scripts.run_recovered_qwen_mathematical_rsi import build_cells, load_json, sha256
from src.candidate_bank_experiment import load_jsonl
from src.rsi_controller.hitting_verification import (
    LinearErrorHittingVerifier,
    evaluator_residual_diagnostic,
    select_hitting_rows,
)


def feature_matrix(df, representation: str) -> tuple[np.ndarray, list[str]]:
    base_cols = sorted(c for c in df.columns if c.startswith("f::"))
    raw = df[base_cols].to_numpy(float)
    if representation == "global_features":
        x = np.column_stack([np.ones(len(df)), raw])
        names = ["intercept", *base_cols]
    elif representation == "taskwise_features":
        tasks = sorted(df.task_id.unique().tolist())
        blocks = []
        names = []
        local = np.column_stack([np.ones(len(df)), raw])
        local_names = ["intercept", *base_cols]
        for task in tasks:
            mask = (df.task_id.to_numpy() == task).astype(float)[:, None]
            blocks.append(local * mask)
            names.extend([f"{task}::{name}" for name in local_names])
        x = np.column_stack(blocks)
    else:
        raise ValueError("unknown representation")
    keep = np.max(np.abs(x), axis=0) > 0
    return x[:, keep], [name for name, retain in zip(names, keep) if retain]


def run(protocol_path: Path, bank_path: Path, output: Path) -> dict[str, Any]:
    lock = load_json(protocol_path)
    if lock.get("status") != "locked post-hoc linear-hitting follow-up on recovered Qwen bank":
        raise ValueError("locked linear-hitting protocol required")
    if sha256(bank_path) != lock["input"]["bank_sha256"]:
        raise ValueError("candidate bank SHA mismatch")
    df = load_jsonl(bank_path).reset_index(drop=True)
    ids = [f"{row.task_id}:{row.candidate_id}" for row in df.itertuples()]
    proxy = df["f::public_score"].to_numpy(float)
    truth = df.trusted_score.to_numpy(float)
    cells = build_cells(
        df,
        etas=[float(x) for x in lock["etas"]],
        rounds=int(lock["rounds"]),
        tolerance=float(lock["outcome_tolerance"]),
    )
    rows = []
    representations = {}
    for representation in lock["representations"]:
        x, feature_names = feature_matrix(df, representation)
        probes = select_hitting_rows(x, ids)
        verifier = LinearErrorHittingVerifier(x, proxy, probes)
        probe_truth = truth[list(probes)]
        diagnostic = evaluator_residual_diagnostic(x, proxy, truth)
        rhos = [float(x) for x in lock["residual_radius_grid"]]
        # Evaluation-only sufficient radius: the full-label least-squares beta
        # witnesses existence of a residual vector no larger than this value.
        oracle_rho = float(diagnostic["lstsq_residual_inf"])
        radius_specs = [("locked_grid", rho) for rho in rhos]
        radius_specs.append(("evaluator_sufficient", oracle_rho))
        rank = verifier.feature_rank
        info_required = math.ceil(
            float(lock["information_budget_multiplier"])
            * rank
            * math.log(1.0 / float(lock["target_error"]))
        )
        representations[representation] = {
            "feature_columns": feature_names,
            "feature_rank": rank,
            "probe_count": len(probes),
            "probe_ids": [ids[i] for i in probes],
            "information_budget_proxy": info_required,
            **diagnostic,
        }
        for source, rho in radius_specs:
            assumption_sufficient = rho + 1e-12 >= oracle_rho
            for cell in cells:
                interval = verifier.interval(
                    np.asarray(cell["delta"], float),
                    probe_truth,
                    residual_radius=rho,
                )
                actual = float(cell["true_gain"])
                if assumption_sufficient and not (
                    interval.lower - 1e-9 <= actual <= interval.upper + 1e-9
                ):
                    raise AssertionError("valid residual class failed to cover true gain")
                rows.append({
                    "representation": representation,
                    "radius_source": source,
                    "residual_radius": rho,
                    "residual_class_sufficient_by_full_label_witness": assumption_sufficient,
                    "cell_id": cell["cell_id"],
                    "eta": cell["eta"],
                    "round": cell["round"],
                    "decision": interval.decision,
                    "estimate": interval.estimate,
                    "radius": interval.radius,
                    "lower": interval.lower,
                    "upper": interval.upper,
                    "proxy_gain": interval.proxy_gain,
                    "true_gain": actual,
                    "harmful": int(cell["harmful"]),
                    "beneficial": int(cell["beneficial"]),
                    "false_progress": int(cell["false_progress"]),
                    "wrong_sign_decisive": int(
                        (interval.decision == "allow" and cell["harmful"])
                        or (interval.decision == "block" and cell["beneficial"])
                    ),
                    "harmful_allowed": int(interval.decision == "allow" and cell["harmful"]),
                    "probe_count": len(probes),
                    "feature_rank": rank,
                    "contrast_l1": interval.contrast_l1,
                    "transport_l1": interval.transport_l1,
                })
    summary_rows = []
    keys = sorted(set(
        (r["representation"], r["radius_source"], r["residual_radius"])
        for r in rows
    ))
    for representation, source, rho in keys:
        group = [
            r for r in rows
            if r["representation"] == representation
            and r["radius_source"] == source
            and r["residual_radius"] == rho
        ]
        decisive = [r for r in group if r["decision"] != "refresh"]
        harmful = [r for r in group if r["harmful"]]
        summary_rows.append({
            "representation": representation,
            "radius_source": source,
            "residual_radius": rho,
            "residual_class_sufficient_by_full_label_witness": bool(
                group[0]["residual_class_sufficient_by_full_label_witness"]
            ),
            "feature_rank": int(group[0]["feature_rank"]),
            "probe_count": int(group[0]["probe_count"]),
            "amortized_probe_labels_per_proposal": float(group[0]["probe_count"] / len(cells)),
            "decision_rate": float(len(decisive) / len(group)),
            "allow_rate": float(np.mean([r["decision"] == "allow" for r in group])),
            "block_rate": float(np.mean([r["decision"] == "block" for r in group])),
            "harmful_allow_rate": (
                float(np.mean([r["harmful_allowed"] for r in harmful])) if harmful else None
            ),
            "wrong_sign_decisive_rate": (
                float(np.mean([r["wrong_sign_decisive"] for r in decisive])) if decisive else None
            ),
        })
    result = {
        "status": "exploratory_post_hoc_follow_up",
        "bank_sha256": sha256(bank_path),
        "proposal_cells": len(cells),
        "representations": representations,
        "summary_rows": summary_rows,
        "claim_boundary": (
            "Fixed-probe linear error-class study on an already-inspected development bank. "
            "The finite-dimensional bound is valid under its declared residual-class assumption; "
            "the experiment measures whether that assumption is useful for this Qwen bank."
        ),
        "limitations": lock["limitations"],
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "rows.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows),
        encoding="utf-8",
    )
    (output / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (output / "manifest.json").write_text(json.dumps({
        "status": "completed",
        "protocol_sha256": sha256(protocol_path),
        "bank_sha256": sha256(bank_path),
        "files": {
            "rows.jsonl": sha256(output / "rows.jsonl"),
            "summary.json": sha256(output / "summary.json"),
        },
    }, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("configs/linear_hitting_qwen_v1.json"))
    parser.add_argument("--bank", type=Path, default=Path("data/recovered_qwen05b_bank.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/linear_hitting_qwen_v1"))
    args = parser.parse_args()
    result = run(args.protocol, args.bank, args.output)
    print(json.dumps({
        "proposal_cells": result["proposal_cells"],
        "representations": result["representations"],
        "summary_rows": result["summary_rows"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
