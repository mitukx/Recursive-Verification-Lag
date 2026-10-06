import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.rvl_systems.openai_math import (
    LeanGenerationGrader,
    LeanTrustedVerifier,
    OpenAIMathManifest,
    OpenAIMathTask,
    hash_chain,
    summarize_rows,
)
from src.rvl_systems.types import Generation


def tiny_manifest(tasks):
    return OpenAIMathManifest(
        1, "openai/math", "a" * 40, "leanprover/lean4:v4.34.1", tuple(tasks)
    )


class ManifestTests(unittest.TestCase):
    def test_family_split_leakage_is_rejected(self):
        a = OpenAIMathTask(
            "a", 116, "a", "lean/ComparatorChallenges/A.lean", "main",
            "development", "coverage", "a" * 40
        )
        b = OpenAIMathTask(
            "b", 116, "b", "lean/ComparatorChallenges/B.lean", "main",
            "heldout", "coverage", "b" * 40
        )
        with self.assertRaisesRegex(ValueError, "crosses"):
            tiny_manifest([a, b]).validate()

    def test_path_escape_is_rejected(self):
        task = OpenAIMathTask(
            "a", 1, "a", "../A.lean", "main",
            "development", "coverage", "a" * 40
        )
        with self.assertRaises(ValueError):
            task.validate()


class LeanVerifierTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        path = root / "lean" / "ComparatorChallenges"
        path.mkdir(parents=True)
        self.source = (
            "import Mathlib\n"
            "namespace OAI\n"
            "theorem target : True := by\n"
            "  sorry\n"
            "end OAI\n"
        )
        (path / "Fixture.lean").write_text(self.source, encoding="utf-8")
        task = OpenAIMathTask(
            "fixture", 116, "fixture",
            "lean/ComparatorChallenges/Fixture.lean", "target",
            "development", "coverage", "a" * 40
        )
        self.verifier = LeanTrustedVerifier(
            root, tiny_manifest([task]),
            allow_unsandboxed=True, verify_provenance=False
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_candidate_injection_and_escape_rejection(self):
        materialized = self.verifier.inject_candidate(
            self.source, "target", "exact True.intro"
        )
        self.assertNotIn("sorry", materialized)
        with self.assertRaisesRegex(ValueError, "forbidden"):
            self.verifier.verify_candidate("fixture", "exact (by sorry)")

    @patch("src.rvl_systems.openai_math.subprocess.run")
    def test_kernel_success_and_async_adapter(self, run):
        run.return_value.returncode = 0
        run.return_value.stdout = b""
        run.return_value.stderr = b""
        result = self.verifier.verify_candidate("fixture", "exact True.intro")
        self.assertTrue(result.passed)
        self.assertEqual(run.call_args.args[0][:3], ("lake", "env", "lean"))

        grader = LeanGenerationGrader(self.verifier)
        generation = Generation(
            prompt_id="fixture", prompt="p", response="exact True.intro",
            logprob=-1.0, token_count=2, latency_s=0.0
        )
        self.assertEqual(asyncio.run(grader(generation)), 1.0)
        self.assertEqual(grader.calls, 1)


class MetricsTests(unittest.TestCase):
    def test_false_progress_is_measured(self):
        rows = [
            {"task_id":"a","proxy_score":0.4,"trusted_pass":True,
             "policy_version":0,"verifier_version":0,"sequence":0,"stale_age":0},
            {"task_id":"a","proxy_score":0.9,"trusted_pass":False,
             "policy_version":4,"verifier_version":0,"sequence":1,"stale_age":4},
        ]
        summary = summarize_rows(rows)
        self.assertEqual(summary["false_progress_transitions"], 1)
        self.assertEqual(summary["false_positive_n"], 1)
        self.assertEqual(summary["trusted_pass_rate_by_stale_age"]["0"], 1.0)
        self.assertEqual(summary["trusted_pass_rate_by_stale_age"]["3-7"], 0.0)

    def test_hash_chain_changes_with_order(self):
        rows = [{"x":1},{"x":2}]
        chained, head = hash_chain(rows)
        self.assertEqual(chained[1]["previous_hash"], chained[0]["row_hash"])
        _, other = hash_chain(list(reversed(rows)))
        self.assertNotEqual(head, other)


if __name__ == "__main__":
    unittest.main()
