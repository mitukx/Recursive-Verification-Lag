import math
import unittest

from src.rvl_systems.grpo import compute_group_advantages
from src.rvl_systems.objectives import clipped_grpo_objective, mean_clipped_grpo_loss
from src.rvl_systems.types import Generation, VerifiedGeneration


class GRPOObjectiveTest(unittest.TestCase):
    @staticmethod
    def sample(prompt: str, reward: float) -> VerifiedGeneration:
        return VerifiedGeneration(
            Generation(prompt, prompt, "x", -1.0, 1, 0.0),
            reward=reward,
            verifier_latency_s=0.0,
            verifier_version=0,
        )

    def test_positive_advantage_clips_large_ratio(self):
        obj = clipped_grpo_objective(math.log(2.0), 0.0, 1.0, clip_eps=0.2)
        self.assertAlmostEqual(obj, 1.2)

    def test_negative_advantage_clips_small_ratio(self):
        obj = clipped_grpo_objective(math.log(0.5), 0.0, -1.0, clip_eps=0.2)
        self.assertAlmostEqual(obj, -0.8)

    def test_loss_is_negative_mean_objective(self):
        loss = mean_clipped_grpo_loss([(0.0, 0.0, 1.0), (0.0, 0.0, -0.5)])
        self.assertAlmostEqual(loss, -0.25)

    def test_group_advantages_preserve_input_order(self):
        samples = [
            self.sample("a", 0.0),
            self.sample("b", 1.0),
            self.sample("a", 1.0),
            self.sample("b", 0.0),
        ]
        records = compute_group_advantages(samples)
        self.assertEqual([x.prompt_id for x in records], ["a", "b", "a", "b"])
        self.assertLess(records[0].advantage, 0.0)
        self.assertGreater(records[1].advantage, 0.0)


if __name__ == "__main__":
    unittest.main()
