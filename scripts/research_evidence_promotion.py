"""Create and validate fail-closed research evidence promotion receipts.

Receipts bind an immutable upstream Actions artifact and its semantic validation
report to the exact downstream workflow run, research source, and protocol. They
are process evidence only: negative/null/underpowered scientific outcomes remain
promotable when execution completed and semantic validation passed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PROMOTABLE_RESULTS = {
    "direction_passed",
    "direction_failed",
    "underpowered",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validated_summary(report: dict[str, Any]) -> dict[str, Any]:
    _require(report.get("valid") is True, "upstream semantic validation did not pass")
    _require(
        report.get("execution_status") == "completed",
        "upstream execution did not complete; failed/not-evaluable evidence cannot promote",
    )
    scientific_result = str(report.get("scientific_result"))
    _require(
        scientific_result in PROMOTABLE_RESULTS,
        f"upstream scientific result is not promotable: {scientific_result}",
    )
    summary: dict[str, Any] = {
        "valid": True,
        "execution_status": "completed",
        "scientific_result": scientific_result,
    }
    for key in (
        "eligible_seeds",
        "research_source_sha",
        "systems_source_sha",
    ):
        if key in report:
            summary[key] = report[key]
    return summary


def create_receipt(
    *,
    stage: str,
    upstream_run_id: int,
    upstream_artifact_name: str,
    upstream_manifest: Path,
    upstream_validation: Path,
    downstream_run_id: int,
    downstream_source_sha: str,
    downstream_protocol: Path,
) -> dict[str, Any]:
    _require(bool(stage.strip()), "promotion stage is required")
    _require(upstream_run_id > 0, "positive upstream workflow run id required")
    _require(downstream_run_id > 0, "positive downstream workflow run id required")
    _require(bool(upstream_artifact_name.strip()), "upstream artifact name required")
    _require(upstream_manifest.is_file(), "upstream manifest missing")
    _require(upstream_validation.is_file(), "upstream validation report missing")
    _require(downstream_protocol.is_file(), "downstream protocol missing")
    _require(
        bool(SHA256_RE.fullmatch(downstream_source_sha)),
        "downstream source must be an immutable 64-hex SHA",
    )

    validation = load_json(upstream_validation)
    validation_summary = _validated_summary(validation)
    receipt = {
        "schema_version": 1,
        "stage": stage,
        "upstream": {
            "workflow_run_id": int(upstream_run_id),
            "artifact_name": upstream_artifact_name,
            "manifest_sha256": sha256(upstream_manifest),
            "validation_report_sha256": sha256(upstream_validation),
            "validation": validation_summary,
        },
        "downstream": {
            "workflow_run_id": int(downstream_run_id),
            "research_source_sha": downstream_source_sha,
            "protocol_sha256": sha256(downstream_protocol),
        },
        "policy": {
            "negative_null_underpowered_retained": True,
            "failed_execution_promotable": False,
            "semantic_validation_required": True,
        },
    }
    return receipt


def validate_receipt(
    receipt: dict[str, Any],
    *,
    expected_stage: str,
    expected_downstream_run_id: int,
    expected_downstream_source_sha: str,
    expected_downstream_protocol: Path,
) -> dict[str, Any]:
    _require(receipt.get("schema_version") == 1, "unsupported promotion receipt schema")
    _require(receipt.get("stage") == expected_stage, "promotion stage mismatch")
    downstream = receipt.get("downstream")
    upstream = receipt.get("upstream")
    policy = receipt.get("policy")
    _require(isinstance(downstream, dict), "downstream receipt block missing")
    _require(isinstance(upstream, dict), "upstream receipt block missing")
    _require(isinstance(policy, dict), "promotion policy block missing")

    _require(
        int(downstream.get("workflow_run_id", -1)) == int(expected_downstream_run_id),
        "promotion receipt is bound to a different downstream workflow run",
    )
    _require(
        downstream.get("research_source_sha") == expected_downstream_source_sha,
        "promotion receipt downstream source mismatch",
    )
    _require(
        downstream.get("protocol_sha256") == sha256(expected_downstream_protocol),
        "promotion receipt downstream protocol mismatch",
    )
    _require(
        bool(SHA256_RE.fullmatch(str(upstream.get("manifest_sha256", "")))),
        "invalid upstream manifest digest in receipt",
    )
    _require(
        bool(SHA256_RE.fullmatch(str(upstream.get("validation_report_sha256", "")))),
        "invalid upstream validation digest in receipt",
    )
    _require(int(upstream.get("workflow_run_id", 0)) > 0, "invalid upstream run id in receipt")
    _require(bool(str(upstream.get("artifact_name", "")).strip()), "invalid upstream artifact name")

    validation = upstream.get("validation")
    _require(isinstance(validation, dict), "receipt validation summary missing")
    _validated_summary(validation)

    _require(policy.get("negative_null_underpowered_retained") is True, "receipt policy mismatch")
    _require(policy.get("failed_execution_promotable") is False, "receipt failure policy mismatch")
    _require(policy.get("semantic_validation_required") is True, "receipt semantic policy mismatch")
    return receipt


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser("create")
    create.add_argument("--stage", required=True)
    create.add_argument("--upstream-run-id", type=int, required=True)
    create.add_argument("--upstream-artifact-name", required=True)
    create.add_argument("--upstream-manifest", type=Path, required=True)
    create.add_argument("--upstream-validation", type=Path, required=True)
    create.add_argument("--downstream-run-id", type=int, required=True)
    create.add_argument("--downstream-source-sha", required=True)
    create.add_argument("--downstream-protocol", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)

    validate = sub.add_parser("validate")
    validate.add_argument("receipt", type=Path)
    validate.add_argument("--stage", required=True)
    validate.add_argument("--downstream-run-id", type=int, required=True)
    validate.add_argument("--downstream-source-sha", required=True)
    validate.add_argument("--downstream-protocol", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "create":
        receipt = create_receipt(
            stage=args.stage,
            upstream_run_id=args.upstream_run_id,
            upstream_artifact_name=args.upstream_artifact_name,
            upstream_manifest=args.upstream_manifest,
            upstream_validation=args.upstream_validation,
            downstream_run_id=args.downstream_run_id,
            downstream_source_sha=args.downstream_source_sha,
            downstream_protocol=args.downstream_protocol,
        )
        _write(args.output, receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return

    receipt = validate_receipt(
        load_json(args.receipt),
        expected_stage=args.stage,
        expected_downstream_run_id=args.downstream_run_id,
        expected_downstream_source_sha=args.downstream_source_sha,
        expected_downstream_protocol=args.downstream_protocol,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
