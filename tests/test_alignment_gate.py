import unittest

from src.rvl_systems.lab.alignment_gate import (
    AlignmentAuditObservation,
    AlignmentGatePolicy,
    evaluate_alignment_gate,
)


def observations(*, helpful: bool, n: int = 64, version: int = 7):
    rows = []
    for i in range(n):
        proxy = -1.0 if i < n // 2 else 1.0
        if helpful:
            trusted = 0.0 if proxy < 0 else 1.0
        else:
            trusted = 1.0 if proxy < 0 else 0.0
        rows.append(
            AlignmentAuditObservation(
                trajectory_id=f"t-{i}",
                policy_version=version,
                proxy_score=proxy,
                trusted_reward=trusted,
            )
        )
    return rows


class AlignmentGateTests(unittest.TestCase):
    def test_allows_positive_alignment(self):
        decision = evaluate_alignment_gate(
            observations(helpful=True),
            proxy_mean=0.0,
            proxy_min=-1.0,
            proxy_max=1.0,
            sampling_scheme="iid_policy",
            policy=AlignmentGatePolicy(min_samples=64, delta=0.05),
        )
        self.assertTrue(decision.allowed)
        self.assertFalse(decision.blocked)
        self.assertGreater(decision.lower, 0.0)
        self.assertEqual(decision.policy_version, 7)
        self.assertEqual(len(decision.evidence_sha256), 64)

    def test_blocks_negative_alignment(self):
        decision = evaluate_alignment_gate(
            observations(helpful=False),
            proxy_mean=0.0,
            proxy_min=-1.0,
            proxy_max=1.0,
            sampling_scheme="iid_policy",
            policy=AlignmentGatePolicy(min_samples=64, delta=0.05),
        )
        self.assertTrue(decision.blocked)
        self.assertFalse(decision.allowed)
        self.assertLess(decision.upper, 0.0)

    def test_zero_signal_is_inconclusive(self):
        rows = [
            AlignmentAuditObservation(f"z-{i}", 3, (-1.0 if i % 2 else 1.0), 0.0)
            for i in range(64)
        ]
        decision = evaluate_alignment_gate(
            rows,
            proxy_mean=0.0,
            proxy_min=-1.0,
            proxy_max=1.0,
            sampling_scheme="iid_policy",
            policy=AlignmentGatePolicy(min_samples=64),
        )
        self.assertEqual(decision.decision, "inconclusive")
        self.assertLessEqual(decision.lower, 0.0)
        self.assertGreaterEqual(decision.upper, 0.0)

    def test_adaptive_priority_sampling_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "iid_policy"):
            evaluate_alignment_gate(
                observations(helpful=True),
                proxy_mean=0.0,
                proxy_min=-1.0,
                proxy_max=1.0,
                sampling_scheme="adaptive_priority",
                policy=AlignmentGatePolicy(min_samples=64),
            )

    def test_mixed_policy_versions_are_rejected(self):
        rows = observations(helpful=True)
        rows[-1] = AlignmentAuditObservation("mixed", 8, 1.0, 1.0)
        with self.assertRaisesRegex(ValueError, "one policy version"):
            evaluate_alignment_gate(
                rows,
                proxy_mean=0.0,
                proxy_min=-1.0,
                proxy_max=1.0,
                sampling_scheme="iid_policy",
                policy=AlignmentGatePolicy(min_samples=64),
            )

    def test_duplicate_id_and_support_violation_fail_closed(self):
        rows = observations(helpful=True)
        rows[-1] = AlignmentAuditObservation(rows[0].trajectory_id, 7, 1.0, 1.0)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            evaluate_alignment_gate(
                rows,
                proxy_mean=0.0,
                proxy_min=-1.0,
                proxy_max=1.0,
                sampling_scheme="iid_policy",
                policy=AlignmentGatePolicy(min_samples=64),
            )

        bad = observations(helpful=True)
        bad[-1] = AlignmentAuditObservation("outside", 7, 2.0, 1.0)
        with self.assertRaisesRegex(ValueError, "outside"):
            evaluate_alignment_gate(
                bad,
                proxy_mean=0.0,
                proxy_min=-1.0,
                proxy_max=1.0,
                sampling_scheme="iid_policy",
                policy=AlignmentGatePolicy(min_samples=64),
            )


if __name__ == "__main__":
    unittest.main()
