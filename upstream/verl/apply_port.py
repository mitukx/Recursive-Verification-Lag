#!/usr/bin/env python3
"""Apply the staged workload-aware routing port to a verl checkout."""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

BASE_SHA = "8718ca30a3f002f93b7c4fd99b9b2506718681bc"
FILES = (
    "verl/workers/rollout/router.py",
    "verl/workers/rollout/llm_server.py",
    "tests/workers/rollout/test_router_on_cpu.py",
)


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True
    ).strip()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("verl_checkout", type=Path)
    p.add_argument(
        "--staged-root",
        type=Path,
        default=Path(__file__).resolve().parent / "port",
    )
    p.add_argument("--allow-nonbase", action="store_true")
    args = p.parse_args()

    repo = args.verl_checkout.resolve()
    if not (repo / ".git").exists():
        raise SystemExit(f"not a git checkout: {repo}")
    head = git(repo, "rev-parse", "HEAD")
    if head != BASE_SHA and not args.allow_nonbase:
        raise SystemExit(
            f"verl HEAD mismatch: expected {BASE_SHA}, got {head}; "
            "rebase the staged port before applying or pass --allow-nonbase "
            "only for manual conflict review"
        )
    if git(repo, "status", "--porcelain"):
        raise SystemExit("target verl checkout must be clean")

    for rel in FILES:
        src = args.staged_root / rel
        dst = repo / rel
        if not src.is_file():
            raise SystemExit(f"missing staged file: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    subprocess.check_call(["git", "-C", str(repo), "diff", "--check"])
    print(git(repo, "diff", "--stat"))
    print("\nPort applied. Review with: git diff")


if __name__ == "__main__":
    main()
