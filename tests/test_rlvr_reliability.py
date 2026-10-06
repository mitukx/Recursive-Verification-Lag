import asyncio
import tempfile
import unittest
from pathlib import Path

from src.benchmark_rollout_engine import run_benchmark
from src.rvl_systems.backends import ToyTabularBackend
from src.rvl_systems.checkpoint import load_toy_checkpoint, save_toy_checkpoint


class RLVRReliabilityTest(unittest.TestCase):
    def test_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.json"
            source = ToyTabularBackend(actions=("0", "1", "2"))
            source.logits["p"] = [0.2, -0.4, 1.7]
            save_toy_checkpoint(path, source, step=7)
            restored = ToyTabularBackend(actions=("0", "1", "2"))
            step = load_toy_checkpoint(path, restored)
            self.assertEqual(step, 7)
            self.assertEqual(restored.logits, source.logits)

    def test_checkpoint_rejects_mismatched_action_space(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.json"
            save_toy_checkpoint(path, ToyTabularBackend(actions=("0", "1")), step=1)
            with self.assertRaises(ValueError):
                load_toy_checkpoint(path, ToyTabularBackend(actions=("0", "1", "2")))

    def test_async_benchmark_preserves_work_and_improves_wall_clock(self):
        result = asyncio.run(run_benchmark(
            requests=6,
            samples=3,
            latency_ms=4.0,
            concurrency=6,
        ))
        self.assertEqual(result["samples"], 18)
        self.assertEqual(result["serial_samples"], 18)
        self.assertGreater(result["speedup"], 1.5)


if __name__ == "__main__":
    unittest.main()
