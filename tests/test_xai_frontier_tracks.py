import json
import tempfile
import unittest
from pathlib import Path

from scripts.validate_xai_frontier_tracks import TRACKS, validate


class FrontierTrackContractTests(unittest.TestCase):
    def test_repository_manifest_is_valid(self):
        result = validate(Path("configs/xai_frontier_tracks_v1.json"))
        self.assertTrue(result["valid"])
        self.assertEqual(result["tracks"], 6)

    def test_ready_claim_fails_closed_without_raw_evidence(self):
        source = json.loads(
            Path("configs/xai_frontier_tracks_v1.json").read_text(encoding="utf-8")
        )
        name = next(iter(TRACKS))
        source["tracks"][name]["status"] = "portfolio_ready"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "requires raw evidence"):
                validate(path)


if __name__ == "__main__":
    unittest.main()
