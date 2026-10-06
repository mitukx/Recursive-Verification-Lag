import asyncio
import unittest

from src.rvl_systems.backends import ToyTabularBackend
from src.rvl_systems.grpo import GRPOConfig, compute_group_advantages
from src.rvl_systems.pipeline import PipelineConfig, RLVRPipeline
from src.rvl_systems.refresh import AdaptiveRefreshController
from src.rvl_systems.rollout import AsyncRolloutEngine, RolloutRequest
from src.rvl_systems.telemetry import Telemetry
from src.rvl_systems.types import Generation, VerifiedGeneration
from src.rvl_systems.verifier import ExactMatchVerifier


class RLVRSystemsTest(unittest.TestCase):
    def test_group_advantages_center_within_prompt(self):
        def item(prompt, reward):
            return VerifiedGeneration(
                Generation(prompt, prompt, "x", -1.0, 1, 0.0),
                reward=reward,
                verifier_latency_s=0.0,
                verifier_version=0,
            )
        records = compute_group_advantages(
            [item("a", 0.0), item("a", 1.0), item("b", 1.0), item("b", 1.0)]
        )
        by_prompt = {}
        for row in records:
            by_prompt.setdefault(row.prompt_id, []).append(row.advantage)
        self.assertAlmostEqual(sum(by_prompt["a"]), 0.0, places=6)
        self.assertAlmostEqual(sum(by_prompt["b"]), 0.0, places=6)

    def test_async_rollout_counts_samples(self):
        async def run():
            telemetry = Telemetry()
            engine = AsyncRolloutEngine(
                ToyTabularBackend(actions=("0", "1")),
                max_concurrency=2,
                telemetry=telemetry,
            )
            out = await engine.run(
                [
                    RolloutRequest("a", "a", samples=3, seed=1),
                    RolloutRequest("b", "b", samples=4, seed=2),
                ]
            )
            self.assertEqual(len(out), 7)
            self.assertEqual(telemetry.snapshot()["rollout.samples"], 7)
        asyncio.run(run())

    def test_refresh_controller_uses_movement_and_staleness(self):
        controller = AdaptiveRefreshController(movement_budget=0.5, max_stale_steps=3)
        self.assertEqual(controller.observe(0.1), (False, "none"))
        self.assertEqual(controller.observe(0.4), (True, "movement"))
        self.assertEqual(controller.observe(0.0), (False, "none"))
        self.assertEqual(controller.observe(0.0), (False, "none"))
        self.assertEqual(controller.observe(0.0), (True, "staleness"))

    def test_end_to_end_policy_improves(self):
        async def run():
            prompts = {"a": "answer a", "b": "answer b"}
            answers = {"a": "2", "b": "1"}
            telemetry = Telemetry()
            backend = ToyTabularBackend(actions=("0", "1", "2"))
            verifier = ExactMatchVerifier(answers, telemetry=telemetry)
            pipeline = RLVRPipeline(
                backend=backend,
                verifier=verifier,
                refresh=AdaptiveRefreshController(
                    movement_budget=0.4, max_stale_steps=4
                ),
                grpo=GRPOConfig(learning_rate=0.25),
                telemetry=telemetry,
            )
            before_a = backend.distribution("a")[2]
            before_b = backend.distribution("b")[1]
            history = await pipeline.run(
                prompts,
                PipelineConfig(
                    rounds=12,
                    samples_per_prompt=64,
                    base_seed=17,
                ),
            )
            self.assertGreater(backend.distribution("a")[2], before_a)
            self.assertGreater(backend.distribution("b")[1], before_b)
            self.assertGreater(history[-1]["reward"], 0.45)
            self.assertGreaterEqual(verifier.version, 1)
            self.assertGreater(telemetry.snapshot()["verifier.calls"], 0)
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
