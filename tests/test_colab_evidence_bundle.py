"""Integrity checks over retained evidence; no training or new hardware claim."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from scripts.verify_colab_evidence_bundle import DEFAULT, report


class EvidenceBundleTests(unittest.TestCase):
    def test_retained_failure_is_not_promoted_to_success(self):
        result = report()
        self.assertEqual(result, json.loads((DEFAULT / "scorecard.json").read_text()))
        self.assertEqual(result["diagnostic_process_status"], "failed")
        self.assertFalse(result["diagnostic"]["penalty_reconstruction_passed"])
        self.assertTrue(result["diagnostic"]["corrected_parity_passed"])
        self.assertFalse(result["diagnostic"]["capability_gain_claim"])

    def test_modified_raw_artifact_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "evidence"
            shutil.copytree(DEFAULT, root)
            (root / "raw/diagnostic-exit-02.json").write_text('{"returncode":0}')
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                report(root)

    def test_rehashed_forged_pass_flag_still_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "evidence"
            shutil.copytree(DEFAULT, root)
            raw = root / "raw"
            path = raw / "behavior-diagnostic-02/summary.json"
            value = json.loads(path.read_text())
            value["penalty_reconstruction_passed"] = True
            path.write_text(json.dumps(value))
            manifest_path = raw / "behavior-diagnostic-02/manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["files"]["summary.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest))
            bundle_path = raw / "bundle-manifest.json"
            bundle = json.loads(bundle_path.read_text())
            for p in (path, manifest_path):
                bundle[p.relative_to(raw).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
            bundle_path.write_text(json.dumps(bundle))
            with self.assertRaisesRegex(ValueError, "diagnostic gate mismatch"):
                report(root)


if __name__ == "__main__":
    unittest.main()
