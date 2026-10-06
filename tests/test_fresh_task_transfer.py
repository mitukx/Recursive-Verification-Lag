import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from src.fresh_task_transfer import transfer_policies, evaluate


class FreshTaskTransferTest(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({'task_id': ['train']*3+['test']*3,
            'candidate_id': list('abcdef'), 'base_logprob': [0.]*6,
            'f::public_score': [0., .5, 1.]*2})

    def test_no_evaluation_labels_accepted_and_normalized_policies(self):
        df = self.frame()
        sources = list('abcdef')
        known = {('train', 'a'): 0., ('train', 'c'): 1.}
        for optimizer in ('soft', 'bon'):
            result = transfer_policies(df, sources, ['train'], ['test'], known,
                                       optimizer=optimizer)
            p = result['policies']['test']['policy']
            self.assertAlmostEqual(sum(p), 1.)
            self.assertGreater(p[2], p[0])
        with self.assertRaisesRegex(ValueError, 'training-task'):
            transfer_policies(df, sources, ['train'], ['test'], {('test', 'd'): 1.})
        df['trusted_score'] = [0., 0., 1., 1., 0., 0.]
        with self.assertRaisesRegex(ValueError, 'unlabeled'):
            transfer_policies(df, sources, ['train'], ['test'], known)

    def test_paid_transcript_only_and_zero_label_fit(self):
        df, sources = self.frame(), list('abcdef')
        a = transfer_policies(df, sources, ['train'], ['test'], {('train', 'a'): 0.})
        b = transfer_policies(df.copy(), sources, ['train'], ['test'], {('train', 'a'): 0.})
        self.assertEqual(a, b)
        np.testing.assert_allclose(a['policies']['test']['policy'], [1/3]*3)
        with self.assertRaises(ValueError):
            transfer_policies(df, sources, ['train'], ['train'], {('train', 'a'): 0.})
        with self.assertRaises(ValueError):
            transfer_policies(df, sources, ['train'], ['test'], {('train', 'z'): 0.})

    def test_complete_synthetic_pipeline_and_provenance_abort(self):
        # Synthetic fixture, never mixed into research evidence.
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            train, test = [str(i) for i in range(16)], [str(i) for i in range(16, 32)]
            records = []
            for task in train+test:
                for i in range(16):
                    x = float(i % 2)
                    records.append({'task_id': task, 'candidate_id': f'MBPPPlus/{task}:{i}',
                        'source_sha256': hashlib.sha256(f'{task}:{i}'.encode()).hexdigest(),
                        'base_logprob': 0., 'public_score': x, 'trusted_score': x,
                        'features': {'public_score': x, 'valid': 1., 'length_scaled': i/16},
                        'model': 'fixture', 'revision': 'fixture'})
            bank = folder/'bank.jsonl'
            bank.write_text(''.join(json.dumps(r)+'\n' for r in records))
            provenance = {'scored_bank_sha256': hashlib.sha256(bank.read_bytes()).hexdigest(),
                          'frozen_split_sha256': 'fixture'}
            bank.with_suffix('.score_manifest.json').write_text(json.dumps(provenance))
            config = folder/'lock.json'
            config.write_text(json.dumps({'training_task_ids': train, 'evaluation_task_ids': test,
                'generator_model': 'fixture', 'generator_revision': 'fixture',
                'split_manifest_sha256': 'fixture', 'samples_per_task': 16, 'source_seed': 0,
                'initial_sources_per_training_task': 2, 'total_sources_per_training_task': 4,
                'ridge': 1e-4, 'secondary': {'representations': ['all'], 'rounds': [1, 4]},
                'primary': {'representation': 'all', 'optimizer': 'soft', 'rounds': 4}}))
            output = folder/'out'
            evaluate(bank, config, output)
            manifest = json.loads((output/'manifest.json').read_text())
            self.assertEqual(manifest['actual_paid_training_sources_both_arms'], 64)
            self.assertEqual(manifest['initial_fit_sources'], 32)
            self.assertEqual(manifest['decision_time_evaluation_task_trusted_queries'], 0)
            self.assertEqual(manifest['evaluator_candidate_occurrences'], 256)
            self.assertEqual(manifest['research_bank_scoring_candidate_occurrences'], 512)
            self.assertEqual(manifest['research_bank_reference_validations'], 32)
            self.assertTrue(manifest['frozen_decisions_before_additional_labels'])
            labels = json.loads((output/'paid_training_transcript.json').read_text())
            self.assertEqual(len(labels), 64)
            self.assertTrue(all(r['task_id'] in train for r in labels))
            contrasts = pd.read_csv(output/'contrasts.csv')
            self.assertEqual(int(contrasts.primary.sum()), 1)
            self.assertTrue((contrasts.evaluation_tasks == 16).all())
            provenance['scored_bank_sha256'] = 'wrong'
            bank.with_suffix('.score_manifest.json').write_text(json.dumps(provenance))
            with self.assertRaisesRegex(ValueError, 'provenance'):
                evaluate(bank, config, folder/'invalid')


if __name__ == '__main__':
    unittest.main()
