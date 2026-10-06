import json
import tempfile
import unittest
from pathlib import Path

from src.summarize_gpu_evidence import render_markdown, summarize


class GPUEvidenceSummaryTests(unittest.TestCase):
    def write(self, root, name, payload):
        path = Path(root) / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_complete_bundle_requires_all_measured_contracts(self):
        with tempfile.TemporaryDirectory() as tmp:
            sha = "f" * 40
            qwen = self.write(tmp, "qwen.json", {
                "git_sha": sha,
                "model": "Qwen/test",
                "metrics": {
                    "before_accuracy": 0.5,
                    "after_accuracy": 0.6,
                    "accuracy_delta": 0.1,
                    "transactional_promotion": 1,
                    "promotion_records": 8,
                    "promotion_head_sha256": "a" * 64,
                },
            })
            fsdp = self.write(tmp, "fsdp.json", {
                "git_sha": sha,
                "single_world_size": 1,
                "multi_world_size": 2,
                "single_tokens_per_s": 100,
                "multi_tokens_per_s": 180,
                "speedup": 1.8,
                "scaling_efficiency": 0.9,
            })
            resume = self.write(tmp, "resume.json", {
                "git_sha": sha,
                "world_size": 2,
                "resumed_from_checkpoint": True,
                "tokens_per_s": 175,
            })
            vllm = self.write(tmp, "vllm.json", [
                {"git_sha":sha,"config":{"concurrency":1},"metrics":{"tokens_per_s":10,"requests_per_s":1,"ttft_ms_p95":10,"tbt_ms_p95":2,"latency_ms_p95":20}},
                {"git_sha":sha,"config":{"concurrency":2},"metrics":{"tokens_per_s":18,"requests_per_s":1.8,"ttft_ms_p95":11,"tbt_ms_p95":2.2,"latency_ms_p95":21}},
                {"git_sha":sha,"config":{"concurrency":4},"metrics":{"tokens_per_s":30,"requests_per_s":3,"ttft_ms_p95":15,"tbt_ms_p95":2.5,"latency_ms_p95":25}},
            ])
            failover = self.write(tmp, "failover.json", {
                "git_sha": sha,
                "metrics": {
                    "completion_rate": 1.0,
                    "scheduler.failover_successes": 3,
                    "scheduler.failures": 1,
                    "requests": 64,
                    "wall_s": 4.0,
                }
            })
            inventory = self.write(tmp, "gpu.json", {
                "available": True,
                "gpus": [
                    {"index":"0","name":"Test GPU","driver_version":"1","memory_total_mb":"24576","compute_capability":"9.0"},
                    {"index":"1","name":"Test GPU","driver_version":"1","memory_total_mb":"24576","compute_capability":"9.0"},
                ],
            })
            report = summarize(qwen=qwen,fsdp=fsdp,fsdp_resume=resume,vllm=vllm,failover=failover,gpu_inventory=inventory)
            self.assertTrue(report["complete"])
            self.assertFalse(report["missing_checks"])
            self.assertFalse(report["failed_checks"])
            self.assertEqual(len(report["sources"]["qwen_rlvr"]["sha256"]),64)
            self.assertEqual(report["git_sha"],sha)
            card = render_markdown(report)
            self.assertIn("GPU Evidence Card",card)
            self.assertIn("Test GPU",card)
            self.assertIn("Scaling efficiency",card)

    def test_missing_and_failed_evidence_are_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            qwen = self.write(tmp, "qwen.json", {
                "git_sha": "a" * 40,
                "metrics": {
                    "transactional_promotion": 0,
                    "promotion_records": 0,
                }
            })
            report = summarize(qwen=qwen)
            self.assertFalse(report["complete"])
            self.assertIn("qwen_transactional_rlvr",report["failed_checks"])
            self.assertIn("fsdp_scaling_measured",report["missing_checks"])


if __name__ == "__main__":
    unittest.main()
