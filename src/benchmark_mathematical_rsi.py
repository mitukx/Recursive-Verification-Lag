from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from src.rsi_controller.math_rsi import reconstruction_phase_grid


def trusted_budget_grid(
    dimensions: tuple[int, ...] = (1, 2, 4, 8, 16),
    errors: tuple[float, ...] = (0.2, 0.1, 0.05, 0.01),
    multiplier: float = 1.0,
):
    rows = []
    for d in dimensions:
        for eps in errors:
            proxy = d * math.log(1.0 / eps)
            rows.append({
                "effective_dimension": d,
                "target_error": eps,
                "d_log_inv_eps": proxy,
                "required_samples_proxy": math.ceil(multiplier * proxy),
            })
    return rows


def coded_tolerance_grid(
    obligation_counts: tuple[int, ...] = (6, 12, 24, 48),
    corrupt_fractions: tuple[float, ...] = (0.0, 0.1, 0.2, 0.25),
):
    return [
        {
            "noncritical_obligations": n,
            "corrupt_fraction": delta,
            "max_adversarial_failures": math.floor(delta * n),
        }
        for n in obligation_counts
        for delta in corrupt_fractions
    ]


def benchmark() -> dict:
    agreements = (0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 1.00)
    reconstruction = reconstruction_phase_grid((2, 3, 4, 8, 16), agreements)
    lookup = {
        (row["branching_factor"], row["agreement"]): row
        for row in reconstruction
    }
    checks = {
        "critical_equality_is_not_supercritical":
            lookup[(4, 0.75)]["criticality"] == 1.0
            and not lookup[(4, 0.75)]["supercritical"],
        "above_threshold_is_supercritical":
            lookup[(4, 0.80)]["criticality"] > 1.0
            and lookup[(4, 0.80)]["supercritical"],
        "trusted_budget_increases_with_dimension":
            trusted_budget_grid(dimensions=(1, 8), errors=(0.1,))[0]["required_samples_proxy"]
            < trusted_budget_grid(dimensions=(1, 8), errors=(0.1,))[1]["required_samples_proxy"],
        "trusted_budget_increases_with_precision":
            trusted_budget_grid(dimensions=(4,), errors=(0.1, 0.01))[0]["required_samples_proxy"]
            < trusted_budget_grid(dimensions=(4,), errors=(0.1, 0.01))[1]["required_samples_proxy"],
    }
    return {
        "status": "synthetic_contract_only",
        "theorem_transfer_warning": (
            "These tables instantiate design proxies inspired by OpenAI Math "
            "families 136/140/229. They are not transferred theorem guarantees "
            "for LLM recursive self-improvement."
        ),
        "coded_verification": coded_tolerance_grid(),
        "trusted_information_budget": trusted_budget_grid(),
        "reconstruction_phase": reconstruction,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = benchmark()
    if not all(report["checks"].values()):
        raise SystemExit(f"contract failure: {report['checks']}")
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
