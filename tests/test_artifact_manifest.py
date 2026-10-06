import tempfile
import unittest
from pathlib import Path

from src.rvl_systems.artifact_manifest import build_manifest, verify_manifest


class ArtifactManifestTest(unittest.TestCase):
    def test_detects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.json"
            path.write_text('{"ok": true}', encoding="utf-8")
            manifest = build_manifest([path])
            self.assertEqual(verify_manifest(manifest), [])
            path.write_text('{"ok": false}', encoding="utf-8")
            failures = verify_manifest(manifest)
            self.assertTrue(
                any("checksum mismatch" in item for item in failures)
            )


if __name__ == "__main__":
    unittest.main()
