from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace

from src.rsi_controller.config import (
    MathematicalRSIConfig,
    PromotionThresholds,
    RSIConfig,
    VerifierTrustThresholds,
)
from src.rsi_controller.controller import DEFAULT_STATE, RSIController
from src.rsi_controller.evaluation import EvaluationStack
from src.rsi_controller.math_rsi import MathematicalRSIGate, reconstruction_phase_grid
from src.rsi_controller.models import Candidate, Component, ResourceLimits
from src.rsi_controller.promotion import PromotionGate, detect_false_progress
from src.rsi_controller.verifier_lag import RecursiveVerificationLagMonitor


def candidate(diff=None):
    return Candidate(
        candidate_id="candidate",
        proposal_id="proposal",
        parent_champion_id="champion",
        target_component=Component.HARNESS,
        diff=diff or {"reasoning_budget": 3},
        full_state=DEFAULT_STATE,
        seed=17,
        resource_limits=ResourceLimits(),
        dependency_metadata={},
    )


class MathematicalRSIGateTests(unittest.TestCase):
    def setUp(self):
        self.stack = EvaluationStack(17)
        self.base = self.stack.evaluate_state(
            DEFAULT_STATE,
            seed=17,
            policy_version=0,
            verifier_version=0,
            policy_verifier_age=0,
        )
        self.hacking = detect_false_progress(self.base, self.base)
        self.lag = RecursiveVerificationLagMonitor(
            VerifierTrustThresholds()
        ).assess(self.base, self.base)

    def gate(self, **kwargs):
        return MathematicalRSIGate(
            MathematicalRSIConfig(enabled=True, **kwargs),
            PromotionThresholds(),
        )

    def test_bounded_probe_and_information_budget_pass_for_known_mutation(self):
        assessment = self.gate().assess(
            candidate(), self.base, self.base, self.hacking, self.lag
        )
        self.assertTrue(assessment.accepted)
        self.assertTrue(assessment.probes.covered)
        self.assertIn("long-chain-consistency", assessment.probes.probe_ids)
        self.assertTrue(assessment.information_budget.passed)
        self.assertFalse(assessment.reconstruction.enforced)

    def test_coded_verification_tolerates_bounded_noncritical_failure(self):
        degraded = replace(
            self.base,
            promotion=replace(
                self.base.promotion,
                throughput=self.base.promotion.throughput * 0.5,
            ),
        )
        hacking = detect_false_progress(degraded, self.base)
        lag = RecursiveVerificationLagMonitor(
            VerifierTrustThresholds()
        ).assess(degraded, self.base)
        assessment = self.gate(max_corrupt_fraction=0.20).assess(
            candidate(), degraded, self.base, hacking, lag
        )
        self.assertIn("throughput_floor", assessment.coded.failed_constraints)
        self.assertEqual(assessment.coded.allowed_noncritical_failures, 1)
        self.assertTrue(assessment.coded.passed)

    def test_coded_verification_never_tolerates_critical_failure(self):
        degraded = replace(
            self.base,
            promotion=replace(
                self.base.promotion,
                trusted_score=max(0.0, self.base.promotion.trusted_score - 0.1),
            ),
        )
        hacking = detect_false_progress(degraded, self.base)
        lag = RecursiveVerificationLagMonitor(
            VerifierTrustThresholds()
        ).assess(degraded, self.base)
        assessment = self.gate(max_corrupt_fraction=0.99).assess(
            candidate(), degraded, self.base, hacking, lag
        )
        self.assertIn(
            "promotion_trusted_nonregression",
            assessment.coded.critical_failures,
        )
        self.assertFalse(assessment.coded.passed)
        self.assertFalse(assessment.accepted)

    def test_probe_class_fails_closed_outside_declared_surface(self):
        assessment = self.gate().assess(
            candidate({"unknown_knob": 1}),
            self.base,
            self.base,
            self.hacking,
            self.lag,
        )
        self.assertFalse(assessment.probes.covered)
        self.assertIn("H.unknown_knob", assessment.probes.uncovered_mutations)
        self.assertFalse(assessment.accepted)

    def test_information_budget_can_block_promotion(self):
        assessment = self.gate(
            min_trusted_samples=256,
            max_trusted_samples=512,
        ).assess(candidate(), self.base, self.base, self.hacking, self.lag)
        self.assertEqual(assessment.information_budget.available_trusted_samples, 200)
        self.assertEqual(assessment.information_budget.required_trusted_samples, 256)
        self.assertFalse(assessment.information_budget.passed)
        self.assertFalse(assessment.accepted)

    def test_reconstruction_threshold_equality_is_not_supercritical(self):
        rows = reconstruction_phase_grid((4,), (0.75, 0.80))
        at_equality, above = rows
        self.assertAlmostEqual(at_equality["criticality"], 1.0)
        self.assertFalse(at_equality["supercritical"])
        self.assertGreater(above["criticality"], 1.0)
        self.assertTrue(above["supercritical"])

    def test_promotion_gate_enforces_mathematical_certificate(self):
        thresholds = PromotionThresholds(
            min_promotion_gain=0.0,
            min_development_gain=0.0,
            min_trusted_gain=0.0,
            confidence_z=0.0,
        )
        gate = MathematicalRSIGate(
            MathematicalRSIConfig(
                enabled=True,
                min_trusted_samples=256,
                max_trusted_samples=512,
            ),
            thresholds,
        )
        assessment = gate.assess(
            candidate(), self.base, self.base, self.hacking, self.lag
        )
        decision = PromotionGate(thresholds).decide(
            "candidate",
            self.base,
            self.base,
            self.hacking,
            self.lag,
            assessment,
        )
        self.assertFalse(decision.safety_checks["mathematical_rsi"])
        self.assertFalse(decision.accepted)
        self.assertTrue(
            any("trusted-information budget" in reason for reason in decision.reasons)
        )


class MathematicalRSIEndToEndTests(unittest.TestCase):
    def test_controller_records_mathematical_rsi_assessment(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = RSIConfig(
                generations=1,
                output_dir=tmp,
                mathematical_rsi=MathematicalRSIConfig(enabled=True),
                resources=ResourceLimits(
                    wall_time_s=4,
                    cpu_time_s=2,
                    memory_mb=1024,
                    max_output_bytes=100000,
                ),
            )
            controller = RSIController(cfg)
            try:
                summary = controller.run(1)
                events = controller.memory.recent_events(500)
                assessments = [
                    event for event in events
                    if event["kind"] == "mathematical_rsi_assessment"
                ]
                self.assertEqual(len(assessments), 1)
                self.assertTrue(summary["mathematical_rsi_enabled"])
            finally:
                controller.close()


if __name__ == "__main__":
    unittest.main()
