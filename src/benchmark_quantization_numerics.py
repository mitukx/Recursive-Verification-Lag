from __future__ import annotations

import argparse
import json
import math
import os

from src.rvl_systems.benchmark_report import BenchmarkReport
from src.rvl_systems.quantization import evaluate_int8_linear_parity


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reference per-row INT8 weight-only numerical parity benchmark."
    )
    parser.add_argument("--rows",type=int,default=64)
    parser.add_argument("--cols",type=int,default=128)
    parser.add_argument("--inputs",type=int,default=32)
    parser.add_argument("--output",default="artifacts/int8-quantization-parity.json")
    args = parser.parse_args()
    if min(args.rows,args.cols,args.inputs) <= 0:
        raise ValueError("dimensions must be positive")

    weights = [
        [
            math.sin(i*0.17+j*0.11) * (0.25 + (j % 17)/8)
            for j in range(args.cols)
        ]
        for i in range(args.rows)
    ]
    inputs = [
        [math.cos(seed*0.31+j*0.07) for j in range(args.cols)]
        for seed in range(args.inputs)
    ]
    diagnostics = evaluate_int8_linear_parity(weights,inputs)
    metrics = {
        "max_abs_output_error":diagnostics.max_abs_output_error,
        "mean_abs_output_error":diagnostics.mean_abs_output_error,
        "max_relative_output_error":diagnostics.max_relative_output_error,
        "mean_relative_output_error":diagnostics.mean_relative_output_error,
        "theoretical_storage_ratio":diagnostics.theoretical_storage_ratio,
    }
    report = BenchmarkReport(
        name="int8-weight-only-numerical-parity",
        metrics=metrics,
        config={
            "scheme":"per-row-symmetric-int8-reference",
            "rows":args.rows,
            "cols":args.cols,
            "inputs":args.inputs,
            "optimized_kernel":0,
        },
        git_sha=os.environ.get("GITHUB_SHA","unknown"),
    )
    report.write_json(args.output)
    print(json.dumps(report.payload(),indent=2,sort_keys=True))


if __name__ == "__main__":
    main()
