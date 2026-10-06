from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.rvl_systems.event_log import ControlPlaneEventLog


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    log = ControlPlaneEventLog.read_jsonl(args.input)
    summary = log.summary()
    if args.output:
        Path(args.output).write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
