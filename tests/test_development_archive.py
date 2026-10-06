"""Audit immutable Actions evidence, including decompressed source bytes."""
import gzip
import hashlib
import json
from pathlib import Path
import unittest


class DevelopmentArchiveTest(unittest.TestCase):
    def test_complete_archive_and_original_transfer_lineage(self):
        root = Path(__file__).resolve().parents[1]
        archive = root/'results/mbppplus_development_v2_archive'
        manifest = json.loads((archive/'archive_manifest.json').read_text())
        self.assertEqual(len(manifest['entries']), 14)
        for entry in manifest['entries']:
            path = archive/entry['archive_path']
            raw = path.read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), entry['archive_sha256'])
            original = gzip.decompress(raw) if path.suffix == '.gz' else raw
            self.assertEqual(len(original), entry['original_bytes'])
            self.assertEqual(hashlib.sha256(original).hexdigest(), entry['original_sha256'])
        original_root = root/'results/fresh_task_verifier_transfer_v1'
        transfer = json.loads((original_root/'manifest.json').read_text())
        for name, expected in transfer['output_sha256'].items():
            self.assertEqual(hashlib.sha256((original_root/name).read_bytes()).hexdigest(), expected)
        self.assertEqual(hashlib.sha256((root/'src/fresh_task_transfer.py').read_bytes()).hexdigest(),
                         transfer['code_sha256'])
        self.assertEqual(hashlib.sha256((root/'data/mbppplus_qwen15b_development_v2_scored.jsonl').read_bytes()).hexdigest(),
                         transfer['bank_sha256'])
        self.assertEqual(transfer['generator_parameter_updates'], 0)
        self.assertEqual(transfer['decision_time_evaluation_task_trusted_queries'], 0)


if __name__ == '__main__':
    unittest.main()
