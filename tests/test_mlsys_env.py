import ast
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    from src.mlsys_env.agent import coding_task
    from src.mlsys_env.evaluator import (
        ROOT, TASK, NumericWorker, case, load_task, prepare, run, same_result,
        summarize, validate_evaluation,
    )
    from src.mlsys_env.worker import metadata, pack, unpack


@unittest.skipUnless(TORCH_AVAILABLE, "MLSys task workflow installs optional PyTorch")
class MLSysEnvironmentTests(unittest.TestCase):
    def test_task_is_exact_real_repo_function(self):
        spec, baseline = load_task()
        source = (ROOT / spec["source_path"]).read_text()
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
                    and n.name == spec["source_function"])
        self.assertEqual(hashlib.sha256(ast.get_source_segment(source, node).encode()).hexdigest(),
                         spec["function_sha256"])
        self.assertIn(ast.get_source_segment(source, node), baseline.decode())
        self.assertEqual(spec["split"], "development")
        self.assertIn(spec["source_function"], coding_task().prompt)

    def test_prepare_refuses_to_overwrite_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "candidate"
            prepare(output)
            self.assertEqual((output / "solution.py").read_bytes(), load_task()[1])
            with self.assertRaises(FileExistsError):
                prepare(output)

    def test_modified_baseline_cannot_become_oracle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "task.json").write_bytes((TASK / "task.json").read_bytes())
            (root / "baseline.py").write_bytes(load_task()[1] + b"\n# changed\n")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_task(root)

    def test_no_pickle_protocol_and_strict_parity(self):
        left = {"meta": metadata({"ok": True}), "output_1": np.zeros(2, dtype=np.float32)}
        self.assertTrue(same_result(left, unpack(pack(left)), load_task()[0]["evaluation"]))
        for wrong in (np.array([0, float("nan")], dtype=np.float32),
                      np.zeros(3, dtype=np.float32), np.zeros(2, dtype=np.float64),
                      np.full(2, 1e-7, dtype=np.float32)):
            self.assertFalse(same_result(left, {"meta": left["meta"], "output_1": wrong},
                                         load_task()[0]["evaluation"]))
        with self.assertRaises(ValueError):
            unpack(pack({"object": np.asarray([{"bad": 1}], dtype=object)}))

    def test_incorrect_unisolated_or_regressed_speedup_gets_no_reward(self):
        evaluation = load_task()[0]["evaluation"]
        fast = [{"baseline_s": 2.0, "candidate_s": 1.0}] * 7
        good = {"passed": 64, "failed": 0}
        self.assertEqual(summarize(good, fast, isolated=True,
                                  evaluation=evaluation, candidate_changed=True)["performance_reward"], 1.0)
        self.assertEqual(summarize(good, fast, isolated=True,
                                  evaluation=evaluation, candidate_changed=False)["performance_reward"], 0.0)
        for checks, pairs, isolated in (({**good, "failed": 1}, fast, True),
                                       (good, fast, False), (None, fast, True),
                                       (good, [], True),
                                       (good, fast + [{"baseline_s": 1.0, "candidate_s": 2.0}], True)):
            self.assertEqual(summarize(checks, pairs, isolated=isolated,
                                      evaluation=evaluation, candidate_changed=True)["performance_reward"], 0.0)

    def test_tolerance_cannot_be_relaxed(self):
        for key, value in (("rtol", 0.1), ("atol", 0.1), ("minimum_speedup", 1.0),
                           ("sizes", [1000000000]), ("benchmark_repeats", 1)):
            spec = {**load_task()[0]["evaluation"], key: value}
            with self.assertRaises(ValueError):
                validate_evaluation(spec)

    def test_no_implicit_host_execution_or_mutable_image(self):
        spec = load_task()[0]
        for image in (None, "python:latest", "rvl-mlsys:test"):
            with self.assertRaisesRegex(ValueError, "immutable"):
                NumericWorker(TASK / "baseline.py", spec["budget"], image=image)

    def test_subprocess_backward_and_timeout_are_enforced(self):
        spec, baseline = load_task()
        # A forward-only shortcut must fail even when all four outputs are right.
        shortcut = baseline.decode().replace(
            "return surrogate, clip_mask, abs_log_ratio, behavior_kl",
            "return surrogate.detach(), clip_mask, abs_log_ratio, behavior_kl")
        with tempfile.TemporaryDirectory() as tmp:
            candidate = Path(tmp) / "solution.py"
            candidate.write_text(shortcut)
            with NumericWorker(candidate, spec["budget"], trusted_local=True) as worker:
                result, _ = worker.call(case(17, 17))
                self.assertFalse(json.loads(result["meta"].tobytes())["ok"])
            candidate.write_text("def torch_grpo_surrogate(*args, **kwargs):\n    while True: pass\n")
            budget = {**spec["budget"], "request_timeout_s": 0.1}
            with NumericWorker(candidate, budget, trusted_local=True) as worker:
                with self.assertRaises(TimeoutError):
                    worker.call(case(17, 17))
                self.assertIsNone(worker.process)

    def test_private_terminal_spec_stays_outside_candidate_and_public_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = {**load_task()[0]["evaluation"], "scope": "private_terminal",
                    "seeds": [101], "sizes": [0, 17], "benchmark_size": 257,
                    "benchmark_repeats": 3, "warmups": 0}
            evaluation = root / "private.json"
            evaluation.write_text(json.dumps(spec))
            output = root / "result"
            result = run(TASK / "baseline.py", output, trusted_local=True,
                         evaluation_path=evaluation, correctness_only=True)
            self.assertEqual(result["status"], "completed")
            self.assertTrue(result["summary"]["correctness_passed"])
            self.assertFalse((output / "evaluation.json").exists())
            self.assertEqual(result["evaluation_sha256"], hashlib.sha256(evaluation.read_bytes()).hexdigest())
            self.assertFalse(result["summary"]["isolated_performance_eligible"])


if __name__ == "__main__":
    unittest.main()
