import json
import tempfile
import unittest
from pathlib import Path
from src.diagnose_residual_metric import diagnose


class ResidualMetricDiagnosticTest(unittest.TestCase):
    def test_same_observables_distinct_source_rewards_refute_finite_lipschitz(self):
        with tempfile.TemporaryDirectory() as root:
            file=Path(root)/'bank.jsonl'
            records=[]
            for i,reward in enumerate([0.,1.]):
                records.append({'task_id':'t','candidate_id':str(i),
                                'source_sha256':f'different-{i}',
                                'public_score':.5,'trusted_score':reward,
                                'features':{'valid':1}})
            file.write_text('\n'.join(json.dumps(x) for x in records)+'\n')
            result=diagnose(file)
            self.assertEqual(list(result.contradictory_pairs),[1,1])
            self.assertTrue(result.minimum_lipschitz_constant.eq(float('inf')).all())


if __name__=='__main__':unittest.main()
