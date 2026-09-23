import unittest
import pandas as pd
from src.candidate_bank_experiment import Config
from src.certified_refresh import certified_run


class CertifiedRefreshTest(unittest.TestCase):
    def test_adversarial_public_score_abstains_safely(self):
        # Cheap ranking points away from the two truly good candidates. The
        # controller must certify every accepted policy despite this reversal.
        df=pd.DataFrame({'task_id':['t']*4,'candidate_id':[str(i) for i in range(4)],
            'base_logprob':[0.]*4,'trusted_score':[1.,1.,0.,0.],
            'f::public_score':[0.,0.,1.,1.]})
        for seed in range(5):
            h=certified_run(df,['a','a','b','c'],
                Config(eta=4.,rounds=5,representation='public',seed=seed),
                initial_audits=1,budget=2)
            self.assertTrue((h.certified_lower>=-1e-9).all())
            self.assertTrue((h.gain_evaluation_only>=-1e-9).all())
            self.assertTrue((h.paid_source_labels<=2).all())

    def test_duplicate_sources_need_one_label(self):
        df=pd.DataFrame({'task_id':['t']*3,'candidate_id':['0','1','2'],
            'base_logprob':[0.]*3,'trusted_score':[1.,1.,0.],
            'f::public_score':[1.,1.,0.]})
        h=certified_run(df,['a','a','b'],Config(rounds=3,seed=3),
            initial_audits=2,budget=2)
        self.assertTrue((h.certified_lower>=-1e-9).all())
        self.assertEqual(h.paid_source_labels.max(),2)


if __name__=='__main__':unittest.main()
