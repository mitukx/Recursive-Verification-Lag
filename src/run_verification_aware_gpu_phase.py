"""Execute locked verification-aware GPU phase cells with crash-retained provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def hardware_snapshot(*, require_gpu: bool) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version,
        "hostname": platform.node(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    nvidia_smi = shutil.which("nvidia-smi")
    snapshot["nvidia_smi_path"] = nvidia_smi
    if nvidia_smi:
        proc = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=index,name,uuid,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            timeout=30,
        )
        snapshot["nvidia_smi_returncode"] = proc.returncode
        snapshot["gpus"] = [
            line.strip() for line in proc.stdout.splitlines() if line.strip()
        ]
        snapshot["nvidia_smi_stderr"] = proc.stderr.strip()
    else:
        snapshot["gpus"] = []
    if require_gpu and not snapshot["gpus"]:
        raise RuntimeError("GPU evidence run requires a visible NVIDIA GPU")
    return snapshot


def start_gpu_sampler(path: Path):
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None, None
    handle = path.open("w")
    proc = subprocess.Popen(
        [
            exe,
            "--query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
            "--loop-ms=500",
        ],
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc, handle


def stop_gpu_sampler(proc, handle) -> None:
    if proc is not None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    if handle is not None:
        handle.close()


def build_manifest(root: Path, cell_id: str) -> dict[str, Any]:
    files = {}
    for path in sorted(root.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            files[path.name] = sha256_file(path)
    return {
        "schema_version": 1,
        "cell_id": cell_id,
        "files": files,
    }


def execute_cell(
    plan: dict[str, Any],
    cell: dict[str, Any],
    root: Path,
    command: list[str],
    *,
    require_gpu: bool,
) -> dict[str, Any]:
    cell_root = root / cell["cell_id"]
    if cell_root.exists():
        raise FileExistsError(f"evidence cell already exists: {cell_root}")
    cell_root.mkdir(parents=True)
    (cell_root / "cell.json").write_text(json.dumps(cell, indent=2, sort_keys=True) + "\n")
    started_wall = time.time()
    started_perf = time.perf_counter()
    status = "failed"
    returncode: int | None = None
    failure: dict[str, Any] | None = None
    sampler = sampler_handle = None
    hardware: dict[str, Any] = {}
    try:
        hardware = hardware_snapshot(require_gpu=require_gpu)
        (cell_root / "hardware.json").write_text(
            json.dumps(hardware, indent=2, sort_keys=True) + "\n"
        )
        env = dict(os.environ)
        env.update(
            {
                "RVL_PHASE_CELL_ID": cell["cell_id"],
                "RVL_PHASE_NAME": cell["phase"],
                "RVL_PHASE_ARM": cell["arm"],
                "RVL_PHASE_SEED": str(cell["seed"]),
                "RVL_PHASE_PROTOCOL_SHA256": plan["protocol_sha256"],
                "RVL_PHASE_SOURCE_SHA": plan["source_sha"],
                "RVL_PHASE_EFFECTIVE_SYSTEM_JSON": json.dumps(
                    cell["effective_system"], sort_keys=True
                ),
                "RVL_EVIDENCE_DIR": str(cell_root.resolve()),
            }
        )
        sampler, sampler_handle = start_gpu_sampler(cell_root / "gpu-telemetry.csv")
        with (cell_root / "driver.stdout.log").open("w") as stdout, (
            cell_root / "driver.stderr.log"
        ).open("w") as stderr:
            proc = subprocess.run(
                command,
                env=env,
                stdout=stdout,
                stderr=stderr,
                text=True,
            )
        returncode = int(proc.returncode)
        if returncode != 0:
            raise RuntimeError(f"driver exited with status {returncode}")
        for required in ("timeseries.jsonl", "summary.json"):
            if not (cell_root / required).exists():
                raise RuntimeError(f"driver did not produce required {required}")
        status = "completed"
    except BaseException as exc:
        failure = {
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "driver_returncode": returncode,
        }
        (cell_root / "failure.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n"
        )
    finally:
        stop_gpu_sampler(sampler, sampler_handle)

    record = {
        "schema_version": 1,
        "cell_id": cell["cell_id"],
        "protocol_sha256": plan["protocol_sha256"],
        "source_sha": plan["source_sha"],
        "phase": cell["phase"],
        "arm": cell["arm"],
        "seed": int(cell["seed"]),
        "effective_system": cell["effective_system"],
        "status": status,
        "started_unix_s": started_wall,
        "finished_unix_s": time.time(),
        "elapsed_s": time.perf_counter() - started_perf,
        "command": command,
        "driver_returncode": returncode,
        "hardware": hardware,
    }
    (cell_root / "run_record.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    manifest = build_manifest(cell_root, cell["cell_id"])
    (cell_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return record


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--cell-id", action="append", default=[])
    p.add_argument("--allow-no-gpu", action="store_true")
    p.add_argument("--continue-on-failure", action="store_true")
    p.add_argument("driver", nargs=argparse.REMAINDER)
    args = p.parse_args()
    command = list(args.driver)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("driver command required after --")

    plan = load_json(args.plan)
    if plan.get("status") != "prospective_phase_plan":
        raise ValueError("prospective phase plan required")
    selected = set(args.cell_id)
    cells = [
        cell for cell in plan["cells"]
        if not selected or cell["cell_id"] in selected
    ]
    if selected - {cell["cell_id"] for cell in cells}:
        raise ValueError("unknown requested cell ID")
    if not cells:
        raise ValueError("no cells selected")

    args.root.mkdir(parents=True, exist_ok=True)
    records = []
    failed = 0
    for cell in cells:
        record = execute_cell(
            plan,
            cell,
            args.root,
            command,
            require_gpu=not args.allow_no_gpu,
        )
        records.append(record)
        if record["status"] != "completed":
            failed += 1
            if not args.continue_on_failure:
                break

    phase_record = {
        "schema_version": 1,
        "plan": str(args.plan),
        "protocol_sha256": plan["protocol_sha256"],
        "source_sha": plan["source_sha"],
        "selected_cells": len(cells),
        "executed_cells": len(records),
        "completed_cells": sum(r["status"] == "completed" for r in records),
        "failed_cells": failed,
        "driver": command,
    }
    (args.root / "phase_execution.json").write_text(
        json.dumps(phase_record, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(phase_record, indent=2, sort_keys=True))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
