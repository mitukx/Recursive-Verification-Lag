from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize 1-GPU versus multi-GPU distributed RL scaling."
    )
    parser.add_argument("--single", required=True)
    parser.add_argument("--multi", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    single = json.loads(Path(args.single).read_text(encoding="utf-8"))
    multi = json.loads(Path(args.multi).read_text(encoding="utf-8"))
    single_world = int(single["world_size"])
    multi_world = int(multi["world_size"])
    if single_world <= 0 or multi_world <= single_world:
        raise ValueError("multi report must use a larger world size")
    single_tps = float(single["tokens_per_s"])
    multi_tps = float(multi["tokens_per_s"])
    if single_tps <= 0 or multi_tps <= 0:
        raise ValueError("tokens_per_s must be positive")
    speedup = multi_tps / single_tps
    ideal_scale = multi_world / single_world
    efficiency = speedup / ideal_scale
    summary = {
        "single_world_size": single_world,
        "multi_world_size": multi_world,
        "single_tokens_per_s": single_tps,
        "multi_tokens_per_s": multi_tps,
        "speedup": speedup,
        "scaling_efficiency": efficiency,
        "single_gpu_peak_memory_bytes": single.get("gpu_peak_memory_bytes"),
        "multi_gpu_peak_memory_bytes": multi.get("gpu_peak_memory_bytes"),
    }
    Path(args.output).write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
