from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from src.rvl_systems.benchmark_report import BenchmarkReport
from src.rvl_systems.policy_deployment import PolicyDeploymentCoordinator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="artifacts/policy-deployment-recovery.json",
    )
    args = parser.parse_args()

    metrics: dict[str, int] = {
        "partial_ack_recovered": 0,
        "active_state_recovered": 0,
        "version_monotonic_after_restart": 0,
        "stale_coordinator_fenced": 0,
        "corrupt_pending_rejected": 0,
        "unexpected_failures": 0,
    }

    try:
        with tempfile.TemporaryDirectory() as tmp:
            first = PolicyDeploymentCoordinator(tmp, ["w0", "w1"])
            pending = first.publish(b"weights-v1")
            first.acknowledge("w0", pending.version)

            recovered = PolicyDeploymentCoordinator(tmp, ["w0", "w1"])
            status = recovered.status()
            metrics["partial_ack_recovered"] = int(
                status.pending_version == 1 and status.workers_behind == {"w1": 1}
            )

            try:
                first.publish(b"stale-writer")
            except RuntimeError as exc:
                metrics["stale_coordinator_fenced"] = int(
                    "stale policy deployment coordinator" in str(exc)
                )

            recovered.acknowledge("w1", 1)
            recovered.activate()
            restarted = PolicyDeploymentCoordinator(tmp, ["w0", "w1"])
            metrics["active_state_recovered"] = int(
                restarted.status().active_version == 1
                and restarted.status().pending_version is None
            )
            next_manifest = restarted.publish(b"weights-v2")
            metrics["version_monotonic_after_restart"] = int(next_manifest.version == 2)

        with tempfile.TemporaryDirectory() as tmp:
            coordinator = PolicyDeploymentCoordinator(tmp, ["w0"])
            pending = coordinator.publish(b"valid")
            (Path(tmp) / pending.artifact).write_bytes(b"corrupt")
            try:
                PolicyDeploymentCoordinator(tmp, ["w0"])
            except RuntimeError as exc:
                metrics["corrupt_pending_rejected"] = int("mismatch" in str(exc))
    except Exception:
        metrics["unexpected_failures"] = 1
        raise
    finally:
        required = [
            "partial_ack_recovered",
            "active_state_recovered",
            "version_monotonic_after_restart",
            "stale_coordinator_fenced",
            "corrupt_pending_rejected",
        ]
        if not all(metrics[key] == 1 for key in required):
            metrics["unexpected_failures"] = 1

        report = BenchmarkReport(
            name="policy-deployment-recovery",
            metrics=metrics,
            config={
                "durability": "atomic-json+immutable-manifests",
                "fencing": "monotonic-coordinator-epoch",
                "synthetic": "true",
            },
            git_sha=os.environ.get("GITHUB_SHA", "unknown"),
        )
        report.write_json(args.output)
        print(json.dumps(report.payload(), indent=2, sort_keys=True))

    if metrics["unexpected_failures"]:
        raise RuntimeError(f"policy deployment recovery invariants failed: {metrics}")


if __name__ == "__main__":
    main()
