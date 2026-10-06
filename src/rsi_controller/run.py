from __future__ import annotations

import argparse
import json
from dataclasses import replace

from .config import load_config
from .controller import RSIController


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bounded recursive self-improvement experiments")
    parser.add_argument("--config", required=True)
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.output is not None:
        cfg = replace(cfg, output_dir=args.output)
    controller = RSIController(cfg)
    try:
        summary = controller.run(args.generations)
    finally:
        controller.close()
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
