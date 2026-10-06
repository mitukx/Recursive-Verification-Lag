from __future__ import annotations

import argparse
import json
import os

from src.rvl_systems.benchmark_report import BenchmarkReport
from src.rvl_systems.precision_parity import evaluate_logprob_parity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/precision-parity.json")
    args = parser.parse_args()

    values = [
        (new / 8.0, old / 8.0)
        for new in range(-32, 33, 3)
        for old in range(-32, 33, 5)
        if abs(new - old) <= 24
    ]
    fp16 = evaluate_logprob_parity(values, precision="fp16")
    bf16 = evaluate_logprob_parity(values, precision="bf16")

    metrics = {
        "cases": len(values),
        "fp16_max_abs_logprob_error": fp16.max_abs_value_error,
        "fp16_max_relative_ratio_error": fp16.max_relative_ratio_error,
        "fp16_mean_relative_ratio_error": fp16.mean_relative_ratio_error,
        "bf16_max_abs_logprob_error": bf16.max_abs_value_error,
        "bf16_max_relative_ratio_error": bf16.max_relative_ratio_error,
        "bf16_mean_relative_ratio_error": bf16.mean_relative_ratio_error,
    }
    report = BenchmarkReport(
        name="precision-parity-emulated",
        metrics=metrics,
        config={"mode": "cpu-ieee-emulation", "max_abs_log_ratio": 8},
        git_sha=os.environ.get("GITHUB_SHA", "unknown"),
    )
    report.write_json(args.output)
    print(json.dumps(report.payload(), indent=2))


if __name__ == "__main__":
    main()
