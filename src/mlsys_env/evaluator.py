"""Correctness-first, evaluator-timed MLSys development evaluation.

Default candidate execution requires a locally available Docker image ID/digest.
Trusted-local execution is only for reviewing code and CI smoke checks; it is
explicitly ineligible for isolated performance evidence.
"""
from __future__ import annotations

import argparse
import ast
import concurrent.futures
import hashlib
import importlib.util
import json
import math
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import torch

from .worker import evaluate, metadata, read_frame, write_frame


ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / "tasks/mlsys/grpo-surrogate-v1"
WORKER = Path(__file__).with_name("worker.py")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_task(task_dir=TASK):
    spec = json.loads((task_dir / "task.json").read_text())
    baseline = (task_dir / "baseline.py").read_bytes()
    if spec["schema_version"] != 1 or spec["task_id"] != "grpo-surrogate-v1":
        raise ValueError("unsupported task")
    if sha(baseline) != spec["baseline_sha256"]:
        raise ValueError("baseline SHA-256 mismatch")
    tree = ast.parse(baseline.decode())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == spec["source_function"])
    if sha(ast.get_source_segment(baseline.decode(), fn).encode()) != spec["function_sha256"]:
        raise ValueError("source function SHA-256 mismatch")
    return spec, baseline


def prepare(output, task_dir=TASK):
    spec, baseline = load_task(task_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "solution.py").write_bytes(baseline)
    (output / "task.json").write_text(json.dumps(spec, indent=2) + "\n")
    (output / "README.md").write_text((task_dir / "README.md").read_text())
    return {"task_id": spec["task_id"], "workspace": str(output.resolve()),
            "baseline_sha256": spec["baseline_sha256"]}


class NumericWorker:
    def __init__(self, candidate, budget, *, trusted_local=False, image=None, stderr_path=None):
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError("candidate must be a regular Python file")
        if not trusted_local and (not image or not re.fullmatch(
            r"(?:sha256:[0-9a-f]{64}|[^\s]+@sha256:[0-9a-f]{64})", image
        )):
            raise ValueError("Docker requires an immutable local image ID/digest")
        self.container = None
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.stderr = open(stderr_path, "wb") if stderr_path else tempfile.TemporaryFile()
        if trusted_local:
            command = [sys.executable, "-u", str(WORKER), str(candidate.resolve())]
        else:
            self.container = "rvl-mlsys-" + uuid.uuid4().hex
            command = ["docker", "run", "--rm", "--interactive", "--pull=never",
                       "--name", self.container, "--network=none", "--read-only",
                       "--cap-drop=ALL", "--security-opt=no-new-privileges",
                       "--user=65534:65534", "--cpus", str(budget["cpu_threads"]),
                       "--memory", str(budget["memory_mb"]) + "m", "--pids-limit=128",
                       "--tmpfs=/tmp:rw,noexec,nosuid,size=256m",
                       "--mount", f"type=bind,src={WORKER},dst=/worker.py,readonly",
                       "--mount", f"type=bind,src={candidate.resolve()},dst=/solution.py,readonly",
                       image, "python", "-u", "/worker.py", "/solution.py"]
        self.timeout = budget["request_timeout_s"]
        self.process = None
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE,
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            def drain_stderr():
                retained = 0
                while True:
                    chunk = self.process.stderr.read(4096)
                    if not chunk:
                        return
                    allowed = min(len(chunk), max(0, 65536 - retained))
                    self.stderr.write(chunk[:allowed])
                    retained += allowed
            self.stderr_thread = threading.Thread(target=drain_stderr, daemon=True)
            self.stderr_thread.start()
            result = self.pool.submit(read_frame, self.process.stdout).result(
                timeout=budget["startup_timeout_s"])
            if json.loads(result["meta"].tobytes()) != {"ready": True}:
                raise ValueError("worker did not become ready")
        except BaseException:
            self.close()
            raise

    def call(self, arrays):
        def exchange():
            write_frame(self.process.stdin, arrays)
            return read_frame(self.process.stdout)
        started = time.perf_counter()
        try:
            result = self.pool.submit(exchange).result(timeout=self.timeout)
        except BaseException:
            self.close()
            raise
        return result, time.perf_counter() - started

    def close(self):
        # Stop the container itself; killing only the Docker CLI leaks workers.
        if self.container:
            subprocess.run(["docker", "rm", "--force", self.container],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
            self.container = None
        if self.process is not None:
            self.process.kill()
            self.process.wait(timeout=10)
            self.stderr_thread.join(timeout=10)
            self.process.stdin.close()
            self.process.stdout.close()
            self.process.stderr.close()
            self.process = None
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.stderr.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def case(seed, size, *, dtype="float32", strided=False, clip_eps=0.2,
         max_abs_log_ratio=20.0, invalid=None):
    rng = np.random.default_rng(seed)
    old = rng.normal(-5, 1, size).astype(dtype)
    delta = rng.normal(0, 2, size).astype(dtype)
    if size >= 9:
        delta[:9] = [0, 20, -20, 21, -21, math.log(1 - clip_eps),
                     math.log(1 + clip_eps), 1e-7, -1e-7]
    current = old + delta
    advantage = rng.normal(0, 1, size).astype(dtype)
    if size >= 3:
        advantage[:3] = [0, 1, -1]
    if invalid == "shape":
        old = old[:-1]
    return {"current": current, "old": old, "advantage": advantage,
            "upstream": rng.normal(0, 1, size).astype(dtype),
            "meta": metadata({"clip_eps": clip_eps, "max_abs_log_ratio": max_abs_log_ratio,
                              "strided": strided})}


def same_result(expected, actual, evaluation):
    em = json.loads(expected["meta"].tobytes())
    am = json.loads(actual["meta"].tobytes())
    if em != am:
        return False
    if not em["ok"]:
        return True
    if set(expected) != set(actual):
        return False
    for key in expected.keys() - {"meta"}:
        left, right = expected[key], actual[key]
        if left.shape != right.shape or left.dtype != right.dtype:
            return False
        if not np.isfinite(right).all():
            return False
        if key == "output_1":
            if not np.array_equal(left, right):
                return False
        elif not np.allclose(left, right, rtol=evaluation["rtol"], atol=evaluation["atol"]):
            return False
    return True


def oracle(fn, arrays):
    try:
        return evaluate(fn, arrays)
    except Exception as exc:
        return {"meta": metadata({"ok": False, "error_type": type(exc).__name__})}


def validate_evaluation(evaluation):
    if evaluation["scope"] not in {"public_development", "private_terminal"}:
        raise ValueError("evaluation scope required")
    if not evaluation["seeds"] or any(type(seed) is not int for seed in evaluation["seeds"]):
        raise ValueError("integer seeds required")
    if not evaluation["sizes"] or any(type(n) is not int or n < 0 or n > 65536
                                      for n in evaluation["sizes"]):
        raise ValueError("bounded case sizes required")
    if not 1 <= evaluation["benchmark_size"] <= 65536:
        raise ValueError("bounded benchmark size required")
    if not 3 <= evaluation["benchmark_repeats"] <= 100 or not 0 <= evaluation["warmups"] <= 10:
        raise ValueError("bounded repeat counts required")
    if evaluation["minimum_speedup"] < 1.05:
        raise ValueError("minimum_speedup must be >= 1.05")
    if not 0 < evaluation["rtol"] <= 1e-5 or not 0 < evaluation["atol"] <= 1e-6:
        raise ValueError("tolerances cannot be relaxed")


def summarize(correctness, paired, *, isolated, evaluation):
    eligible = bool(correctness and correctness["failed"] == 0 and isolated and paired)
    speedups = [row["baseline_s"] / row["candidate_s"] for row in paired]
    median_speedup = statistics.median(speedups) if speedups else None
    # Every retained pair must meet the locked threshold; keep negative results.
    gate = bool(eligible and min(speedups) >= evaluation["minimum_speedup"])
    performance_reward = min(1.0, max(0.0, math.log2(median_speedup))) if gate else 0.0
    return {"correctness_passed": bool(correctness and correctness["failed"] == 0),
            "isolated_performance_eligible": eligible, "performance_gate_passed": gate,
            "median_roundtrip_speedup": median_speedup,
            "minimum_roundtrip_speedup": min(speedups) if speedups else None,
            "performance_reward": performance_reward,
            "claim": "operator_development_only"}


def run(candidate, output, *, trusted_local=False, image=None, evaluation_path=None,
        correctness_only=False):
    spec, baseline = load_task()
    evaluation_bytes = (evaluation_path.read_bytes() if evaluation_path else
                        json.dumps(spec["evaluation"], sort_keys=True).encode())
    evaluation = json.loads(evaluation_bytes)
    validate_evaluation(evaluation)
    output.mkdir(parents=True, exist_ok=False)
    candidate_bytes = candidate.read_bytes()
    # Evaluate immutable copies, avoiding edits during a run.
    with tempfile.TemporaryDirectory(prefix="rvl-mlsys-") as tmp:
        tmp = Path(tmp)
        base_path, candidate_path = tmp / "baseline.py", tmp / "solution.py"
        base_path.write_bytes(baseline)
        candidate_path.write_bytes(candidate_bytes)
        module_spec = importlib.util.spec_from_file_location("trusted_baseline", base_path)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        fn = getattr(module, spec["entrypoint"])
        torch.set_num_threads(spec["budget"]["cpu_threads"])
        report = {"schema_version": 1, "task_id": spec["task_id"],
                  "source_commit": spec["source_commit"],
                  "baseline_sha256": sha(baseline), "candidate_sha256": sha(candidate_bytes),
                  "evaluation_sha256": sha(evaluation_bytes), "evaluation_scope": evaluation["scope"],
                  "isolation": "trusted_local_unisolated" if trusted_local else "docker",
                  "image": image, "budget": spec["budget"],
                  "hardware": {"platform": platform.platform(), "machine": platform.machine(),
                               "processor": platform.processor()},
                  "dependencies": {"python": platform.python_version(), "torch_evaluator": torch.__version__,
                                   "numpy_evaluator": np.__version__},
                  "timing_scope": "evaluator_wallclock_numeric_rpc_including_serialization_not_kernel_time",
                  "claim_boundary": spec["claim_boundary"],
                  "correctness": {"passed": 0, "failed": 0}, "paired_timings": [],
                  "status": "failed"}
        try:
            with NumericWorker(candidate_path, spec["budget"], trusted_local=trusted_local,
                               image=image, stderr_path=output / "candidate.stderr") as worker:
                probes = [case(seed, size, dtype=dtype, strided=strided)
                          for seed in evaluation["seeds"] for size in evaluation["sizes"]
                          for dtype in ("float32", "float64") for strided in (False, True)]
                probes += [case(997, 17, clip_eps=0), case(998, 17, max_abs_log_ratio=0),
                           case(999, 17, invalid="shape"), case(1000, 257, clip_eps=0.1,
                                                                 max_abs_log_ratio=3.0)]
                for probe in probes:
                    actual, _ = worker.call(probe)
                    match = same_result(oracle(fn, probe), actual, evaluation)
                    report["correctness"]["passed" if match else "failed"] += 1
                if report["correctness"]["failed"] == 0 and not correctness_only:
                    with NumericWorker(base_path, spec["budget"], trusted_local=trusted_local,
                                       image=image, stderr_path=output / "baseline.stderr") as reference:
                        for i in range(evaluation["warmups"] + evaluation["benchmark_repeats"]):
                            probe = case(2000 + i, evaluation["benchmark_size"])
                            expected = oracle(fn, probe)
                            order = (("baseline", reference), ("candidate", worker)) if i % 2 == 0 else (
                                ("candidate", worker), ("baseline", reference))
                            timings = {}
                            for label, target in order:
                                actual, elapsed = target.call(probe)
                                if not same_result(expected, actual, evaluation):
                                    raise ValueError("benchmark correctness regression")
                                timings[label + "_s"] = elapsed
                            if i >= evaluation["warmups"]:
                                report["paired_timings"].append(timings)
                report["status"] = "completed"
        except Exception as exc:
            report["error"] = {"type": type(exc).__name__, "detail": str(exc)[:500]}
        report["summary"] = summarize(
            report["correctness"] if report["status"] == "completed" else None,
            report["paired_timings"], isolated=not trusted_local, evaluation=evaluation)
        report["artifact_sha256"] = {}
        for name, data in (("candidate.py", candidate_bytes), ("baseline.py", baseline)):
            (output / name).write_bytes(data)
            report["artifact_sha256"][name] = sha(data)
        if evaluation["scope"] == "public_development":
            (output / "evaluation.json").write_bytes(evaluation_bytes)
            report["artifact_sha256"]["evaluation.json"] = sha(evaluation_bytes)
        for log in output.glob("*.stderr"):
            report["artifact_sha256"][log.name] = sha(log.read_bytes())
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--output", type=Path, required=True)
    grade = commands.add_parser("evaluate")
    grade.add_argument("--candidate", type=Path, required=True)
    grade.add_argument("--output", type=Path, required=True)
    isolation = grade.add_mutually_exclusive_group(required=True)
    isolation.add_argument("--image", help="local Docker image ID or name@sha256:digest")
    isolation.add_argument("--trusted-local", action="store_true", help="trusted code only; no isolation evidence")
    grade.add_argument("--evaluation-spec", type=Path, help="evaluator-owned private spec, kept outside candidate mounts")
    grade.add_argument("--correctness-only", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.output)
    else:
        report = run(args.candidate, args.output, trusted_local=args.trusted_local,
                     image=args.image, evaluation_path=args.evaluation_spec,
                     correctness_only=args.correctness_only)
        result = report["summary"] | {"status": report["status"], "report": str(args.output / "report.json")}
        print(json.dumps(result, indent=2))
        if report["status"] != "completed" or not result["correctness_passed"]:
            raise SystemExit(1)
        return
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
