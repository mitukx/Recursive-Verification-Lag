from __future__ import annotations

import argparse

from src.rvl_systems.regression_gate import check_reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("reports", nargs="+")
    args = parser.parse_args()

    failures = check_reports(args.reports, args.policy)
    if failures:
        for failure in failures:
            print(f"REGRESSION: {failure}")
        raise SystemExit(1)
    print("benchmark regression gate passed")


if __name__ == "__main__":
    main()
