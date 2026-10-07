import json
import tempfile
import unittest
from pathlib import Path

from src.generate_vllm_token_exact_replay import replay_diagnostics
from src.rvl_systems.types import Generation, VerifiedGeneration
from src.validate_vllm_token_exact_bridge import validate


def sample(prompt_id="p", reward=1.0):
    g = Generation(
        prompt_id,
        "prompt",
        "response",
        -0.5,
        2,
        0.01,
        {
            "backend": "token-exact-http",
            "model": "model",
            "prompt_token_ids": [1, 2],
            "response_token_ids": [3, 4],
            "response_token_logprobs": [-0.2, -0.3],
        },
    )
    return VerifiedGeneration(g, reward, 0.001, 0)


class VLLMTokenExactBridgeTests(unittest.TestCase):
    def test_replay_diagnostics_require_exact_token_alignment(self):
        report = replay_diagnostics([sample("a", 0.0), sample("a", 1.0), sample("b", 0.0)])
        self.assertEqual(report["groups"], 2)
        self.assertEqual(report["samples"], 3)
        self.assertEqual(report["generated_tokens"], 6)
        self.assertEqual(report["informative_reward_groups"], 1)

        bad = sample()
        bad = VerifiedGeneration(
            Generation(
                bad.generation.prompt_id,
                bad.generation.prompt,
                bad.generation.response,
                bad.generation.logprob,
                bad.generation.token_count,
                bad.generation.latency_s,
                {**bad.generation.metadata, "response_token_logprobs": [-0.2]},
            ),
            bad.reward,
            bad.verifier_latency_s,
            bad.verifier_version,
        )
        with self.assertRaisesRegex(ValueError, "alignment"):
            replay_diagnostics([bad])

    def test_validator_binds_same_replay_to_serving_and_learner(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            replay = root / "replay.jsonl"
            replay.write_text(json.dumps({"row": 1}) + "\n")
            import hashlib
            digest = hashlib.sha256(replay.read_bytes()).hexdigest()
            metadata = root / "meta.json"
            metadata.write_text(json.dumps({
                "status": "remote_token_exact_verified_replay",
                "model": "model",
                "model_revision": "revision",
                "replay_sha256": digest,
                "samples": 4,
                "groups": 1,
                "generated_tokens": 8,
            }))
            train = root / "train.json"
            train.write_text(json.dumps({
                "status": "completed",
                "replay_sha256": digest,
                "samples": 4,
                "groups": 1,
                "informative_reward_groups": 1,
                "sampled_parameter_values": 16,
                "optimizer_had_nonzero_gradient": True,
                "parameter_probe_l1_change": 0.1,
                "preupdate_parity": {
                    "passed": True,
                    "max_abs_log_ratio": 0.01,
                    "threshold": 0.2,
                },
            }))
            result = validate(
                replay, metadata, train,
                model_repo="model",
                model_revision="revision",
            )
            self.assertTrue(result["valid"])
            self.assertEqual(result["replay_sha256"], digest)

            altered = root / "other.jsonl"
            altered.write_text(json.dumps({"row": 2}) + "\n")
            with self.assertRaisesRegex(AssertionError, "hash"):
                validate(
                    altered, metadata, train,
                    model_repo="model",
                    model_revision="revision",
                )


if __name__ == "__main__":
    unittest.main()
