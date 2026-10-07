"""Deterministically expand the locked verification-aware GPU experiment protocol."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
from typing import Any


LOCKED_STATUS = "prospective GPU systems protocol locked before verification-aware async GPU outcomes"


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text())
    if protocol.get("status") != LOCKED_STATUS:
        raise ValueError("locked verification-aware GPU protocol required")
    if int(protocol.get("issue", -1)) != 66:
        raise ValueError("protocol must remain bound to Issue #66")
    if not protocol.get("phase_sweeps") or not protocol.get("generation", {}).get("seeds"):
        raise ValueError("protocol missing phase sweeps or seeds")
    return protocol


def _sweep_overrides(sweep: dict[str, Any]) -> list[dict[str, Any]]:
    if "override" in sweep:
        override = dict(sweep["override"])
        return [override]

    ignored = {"name", "arms", "cartesian", "pairing"}
    parameter_keys = [k for k in sweep if k not in ignored]
    if not parameter_keys:
        return [{}]

    if sweep.get("pairing") is not None:
        pairs = list(sweep["pairing"])
        if len(parameter_keys) != 2:
            raise ValueError("paired sweep currently requires exactly two parameter keys")
        out = []
        for pair in pairs:
            if len(pair) != 2:
                raise ValueError("paired sweep row must have two values")
            out.append({parameter_keys[0]: pair[0], parameter_keys[1]: pair[1]})
        return out

    values = [list(sweep[k]) for k in parameter_keys]
    if len(parameter_keys) == 1:
        return [{parameter_keys[0]: v} for v in values[0]]

    if not sweep.get("cartesian", True):
        lengths = {len(v) for v in values}
        if len(lengths) != 1:
            raise ValueError("non-cartesian sweep lists must have equal length")
        return [
            {key: value for key, value in zip(parameter_keys, row)}
            for row in zip(*values)
        ]

    return [
        {key: value for key, value in zip(parameter_keys, row)}
        for row in itertools.product(*values)
    ]


def build_plan(protocol: dict[str, Any], *, protocol_sha256: str, source_sha: str) -> dict[str, Any]:
    default_system = dict(protocol["default_system"])
    seeds = [int(s) for s in protocol["generation"]["seeds"]]
    cells: list[dict[str, Any]] = []
    seen: set[str] = set()

    for sweep in protocol["phase_sweeps"]:
        phase_name = str(sweep["name"])
        arms = [str(a) for a in sweep["arms"]]
        for overrides in _sweep_overrides(sweep):
            for arm in arms:
                if arm not in protocol["arms"]:
                    raise ValueError(f"unknown arm in sweep: {arm}")
                effective = dict(default_system)
                effective.update(overrides)
                for seed in seeds:
                    identity = {
                        "phase": phase_name,
                        "arm": arm,
                        "seed": seed,
                        "overrides": overrides,
                        "effective_system": effective,
                    }
                    digest = hashlib.sha256(canonical(identity).encode()).hexdigest()[:12]
                    cell_id = f"{phase_name}--{arm}--s{seed}--{digest}"
                    if cell_id in seen:
                        raise AssertionError("duplicate experiment cell ID")
                    seen.add(cell_id)
                    cells.append({
                        "cell_id": cell_id,
                        **identity,
                        "model": dict(protocol["model"]),
                        "dataset": dict(protocol["dataset"]),
                        "generation": dict(protocol["generation"]),
                    })

    return {
        "schema_version": 1,
        "status": "prospective_phase_plan",
        "issue": int(protocol["issue"]),
        "protocol_sha256": protocol_sha256,
        "source_sha": source_sha,
        "cell_count": len(cells),
        "required_timeseries": list(protocol["required_timeseries"]),
        "required_terminal_metrics": list(protocol["required_terminal_metrics"]),
        "hard_invariants": list(protocol["hard_invariants"]),
        "claim_boundary": protocol["claim_boundary"],
        "cells": cells,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path, default=Path("configs/verification_aware_async_gpu_v1.json"))
    p.add_argument("--source-sha", default=os.environ.get("GITHUB_SHA", "unknown"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    protocol = load_protocol(args.protocol)
    plan = build_plan(
        protocol,
        protocol_sha256=sha256_file(args.protocol),
        source_sha=str(args.source_sha),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"cell_count": plan["cell_count"], "protocol_sha256": plan["protocol_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
