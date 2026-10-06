from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.rvl_systems.artifact_manifest import verify_manifest, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    parser.add_argument("--verify")
    parser.add_argument("artifacts", nargs="*")
    args = parser.parse_args()

    if args.verify:
        manifest = json.loads(
            Path(args.verify).read_text(encoding="utf-8")
        )
        failures = verify_manifest(manifest)
        if failures:
            for failure in failures:
                print(f"MANIFEST ERROR: {failure}")
            raise SystemExit(1)
        print("artifact manifest verified")
        return

    if not args.output or not args.artifacts:
        raise SystemExit("--output and artifacts are required")
    write_manifest(args.artifacts, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
