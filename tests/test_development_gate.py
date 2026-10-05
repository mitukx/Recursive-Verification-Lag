import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from src.summarize_development_gate import onset_records
from src.score_mbppplus_docker import run_capped, CandidateResourceLimit


class DevelopmentGateTest(unittest.TestCase):
    def test_round_one_failure_is_not_lag_and_constant_zero_task_stays(self):
        records = [
            {'task_id': 'initial', 'design': 'early', 'rewards': [.1, .1, .1]},
            {'task_id': 'later', 'design': 'uniform', 'rewards': [.5, .6, .1]},
            {'task_id': 'zero', 'design': 'late', 'rewards': [0., 0., 0.]},
        ]
        out = onset_records(records, {'initial': .5, 'later': .5, 'zero': 0.})
        self.assertEqual(out.later_onset.tolist(), [0, 1, 0])
        self.assertEqual(out.initial_failure.tolist(), [1, 0, 0])
        self.assertEqual(len(out), 3)

    def test_less_than_six_sources_is_reported_not_replaced(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bank = root/'bank.jsonl'
            rows = [{'task_id': 'small', 'candidate_id': str(i),
                'source_sha256': 'same', 'trusted_score': 0.,
                'public_score': 0., 'features': {}} for i in range(3)]
            bank.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
            output = root/'result'
            subprocess.run([sys.executable, '-m', 'src.source_timing_pilot',
                str(bank), '--output', str(output)], check=True, capture_output=True)
            manifest = json.loads((output/'manifest.json').read_text())
            self.assertEqual(manifest['completed_runs'], 0)
            self.assertEqual(manifest['source_eligibility'][0]['task_id'], 'small')
            self.assertEqual(manifest['source_eligibility'][0]['source_timing_identified'], 0)

    def test_docker_infrastructure_exit_is_not_a_resource_zero_label(self):
        for code, expected in [(125, RuntimeError), (137, CandidateResourceLimit)]:
            process = mock.MagicMock()
            process.wait.return_value = code
            selector = mock.MagicMock()
            selector.get_map.return_value = {}
            with mock.patch('src.score_mbppplus_docker.subprocess.Popen', return_value=process), \
                    mock.patch('src.score_mbppplus_docker.subprocess.run'), \
                    mock.patch('src.score_mbppplus_docker.selectors.DefaultSelector', return_value=selector):
                with self.assertRaises(expected) as error:
                    run_capped(['docker'], {}, 'test')
                if code == 125:
                    self.assertNotIsInstance(error.exception, CandidateResourceLimit)


if __name__ == '__main__':
    unittest.main()
