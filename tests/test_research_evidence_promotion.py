import json
import tempfile
import unittest
from pathlib import Path

from scripts.research_evidence_promotion import (
    create_receipt,
    load_json,
    validate_receipt,
)


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


class ResearchEvidencePromotionTest(unittest.TestCase):
    SOURCE = "a" * 64

    def fixture(self, base: Path, scientific_result: str = "underpowered"):
        manifest = base / "manifest.json"
        validation = base / "validation.json"
        protocol = base / "protocol.json"
        write_json(manifest, {"status": "completed", "files": {"x": "y"}})
        write_json(
            validation,
            {
                "valid": True,
                "execution_status": "completed",
                "scientific_result": scientific_result,
                "eligible_seeds": 1,
                "research_source_sha": "b" * 40,
            },
        )
        write_json(protocol, {"status": "locked"})
        return manifest, validation, protocol

    def test_underpowered_completed_evidence_can_promote(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            manifest, validation, protocol = self.fixture(base, "underpowered")
            receipt = create_receipt(
                stage="qwen_alignment_to_learned_verifier",
                upstream_run_id=101,
                upstream_artifact_name="rvl-qwen-alignment-bridge-v1-101",
                upstream_manifest=manifest,
                upstream_validation=validation,
                downstream_run_id=202,
                downstream_source_sha=self.SOURCE,
                downstream_protocol=protocol,
            )
            validated = validate_receipt(
                receipt,
                expected_stage="qwen_alignment_to_learned_verifier",
                expected_downstream_run_id=202,
                expected_downstream_source_sha=self.SOURCE,
                expected_downstream_protocol=protocol,
            )
            self.assertEqual(
                validated["upstream"]["validation"]["scientific_result"],
                "underpowered",
            )
            self.assertTrue(
                validated["policy"]["negative_null_underpowered_retained"]
            )

    def test_negative_direction_can_promote(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            manifest, validation, protocol = self.fixture(base, "direction_failed")
            receipt = create_receipt(
                stage="qwen_alignment_to_learned_verifier",
                upstream_run_id=101,
                upstream_artifact_name="artifact",
                upstream_manifest=manifest,
                upstream_validation=validation,
                downstream_run_id=202,
                downstream_source_sha=self.SOURCE,
                downstream_protocol=protocol,
            )
            self.assertEqual(
                receipt["upstream"]["validation"]["scientific_result"],
                "direction_failed",
            )

    def test_failed_execution_cannot_promote(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            manifest, validation, protocol = self.fixture(base)
            report = load_json(validation)
            report["execution_status"] = "failed"
            report["scientific_result"] = "not_evaluable"
            write_json(validation, report)
            with self.assertRaises(ValueError):
                create_receipt(
                    stage="qwen_alignment_to_learned_verifier",
                    upstream_run_id=101,
                    upstream_artifact_name="artifact",
                    upstream_manifest=manifest,
                    upstream_validation=validation,
                    downstream_run_id=202,
                    downstream_source_sha=self.SOURCE,
                    downstream_protocol=protocol,
                )

    def test_semantically_invalid_validation_cannot_promote(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            manifest, validation, protocol = self.fixture(base)
            report = load_json(validation)
            report["valid"] = False
            write_json(validation, report)
            with self.assertRaises(ValueError):
                create_receipt(
                    stage="qwen_alignment_to_learned_verifier",
                    upstream_run_id=101,
                    upstream_artifact_name="artifact",
                    upstream_manifest=manifest,
                    upstream_validation=validation,
                    downstream_run_id=202,
                    downstream_source_sha=self.SOURCE,
                    downstream_protocol=protocol,
                )

    def test_receipt_is_bound_to_downstream_run_source_and_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            manifest, validation, protocol = self.fixture(base)
            receipt = create_receipt(
                stage="qwen_alignment_to_learned_verifier",
                upstream_run_id=101,
                upstream_artifact_name="artifact",
                upstream_manifest=manifest,
                upstream_validation=validation,
                downstream_run_id=202,
                downstream_source_sha=self.SOURCE,
                downstream_protocol=protocol,
            )
            with self.assertRaises(ValueError):
                validate_receipt(
                    receipt,
                    expected_stage="qwen_alignment_to_learned_verifier",
                    expected_downstream_run_id=203,
                    expected_downstream_source_sha=self.SOURCE,
                    expected_downstream_protocol=protocol,
                )
            with self.assertRaises(ValueError):
                validate_receipt(
                    receipt,
                    expected_stage="qwen_alignment_to_learned_verifier",
                    expected_downstream_run_id=202,
                    expected_downstream_source_sha="c" * 64,
                    expected_downstream_protocol=protocol,
                )
            write_json(protocol, {"status": "changed"})
            with self.assertRaises(ValueError):
                validate_receipt(
                    receipt,
                    expected_stage="qwen_alignment_to_learned_verifier",
                    expected_downstream_run_id=202,
                    expected_downstream_source_sha=self.SOURCE,
                    expected_downstream_protocol=protocol,
                )


if __name__ == "__main__":
    unittest.main()
