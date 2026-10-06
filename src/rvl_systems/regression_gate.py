from __future__ import annotations

import json
from pathlib import Path


def evaluate_report(report: dict, rules: list[dict]) -> list[str]:
    metrics = report.get("metrics", {})
    failures: list[str] = []
    for rule in rules:
        metric = rule["metric"]
        if metric not in metrics:
            failures.append(f"missing metric: {metric}")
            continue
        value = float(metrics[metric])
        op = rule["op"]
        if op in {"ge", "le"}:
            target = float(rule["value"])
        else:
            other = rule["other"]
            if other not in metrics:
                failures.append(f"missing comparison metric: {other}")
                continue
            target = float(metrics[other]) * float(rule.get("factor", 1.0))
        passed = value >= target if op in {"ge", "ge_metric"} else value <= target
        if not passed:
            failures.append(
                f"{metric}={value} violates {op} target={target}"
            )
    return failures


def check_reports(
    report_paths: list[str | Path],
    policy_path: str | Path,
) -> list[str]:
    policy = json.loads(Path(policy_path).read_text(encoding="utf-8"))
    reports: dict[str, dict] = {}
    for path in report_paths:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        name = report.get("name")
        if name:
            reports[str(name)] = report

    failures: list[str] = []
    for name, rules in policy.items():
        report = reports.get(name)
        if report is None:
            failures.append(f"missing required report: {name}")
            continue
        for failure in evaluate_report(report, rules):
            failures.append(f"{name}: {failure}")
    return failures
