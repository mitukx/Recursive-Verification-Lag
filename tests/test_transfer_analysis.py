import unittest
import pandas as pd
from src.analyze_transfer_replication import contrasts,summarize


class TransferAnalysisTest(unittest.TestCase):
    def fixture(self):
        return pd.DataFrame([dict(bank=b,task_id=t,optimizer='soft',strength=1,
            representation='public',seed=0,design=d,cost=32,
            failure=int(d=='uniform'),gain=(.1 if d=='uniform' else (.2 if b=='a' else 0)))
            for b in ['a','b'] for t in ['x','y'] for d in ['early','uniform']])

    def test_reversal_is_not_joint_replication(self):
        s=summarize(contrasts(self.fixture())).set_index('design').loc['early']
        self.assertTrue(s.failure_desired_in_every_bank)
        self.assertFalse(s.gain_desired_in_every_bank)
        self.assertFalse(s.joint_desired_in_every_bank)
        self.assertEqual(s.tasks,2)

    def test_unmatched_cost_rejected(self):
        df=self.fixture();df.loc[0,'cost']=16
        with self.assertRaises(ValueError):contrasts(df)
