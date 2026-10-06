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

    def test_interpolation_preserves_initial_safety(self):
        df=pd.DataFrame({'task_id':['t']*4,'candidate_id':[str(i) for i in range(4)],
            'base_logprob':[0.]*4,'trusted_score':[1.,1.,0.,0.],
            'f::public_score':[0.,0.,1.,1.]})
        for seed in range(5):
            h=certified_run(df,['a','a','b','c'],
                Config(eta=4.,rounds=5,representation='public',seed=seed),
                initial_audits=1,budget=2,fallback='interpolate')
            self.assertTrue((h.certified_lower>=-1e-9).all())
            self.assertTrue((h.gain_evaluation_only>=-1e-9).all())
            self.assertTrue(h.mixture_fraction.between(0,1).all())

    def test_projection_is_paid_label_safe_and_can_escape_fixed_direction(self):
        df=pd.DataFrame({'task_id':['t']*3,'candidate_id':['0','1','2'],
            'base_logprob':[0.]*3,'trusted_score':[1.,0.,0.],
            'f::public_score':[1.,1.,0.]})
        h=certified_run(df,['a','b','c'],Config(eta=3.,rounds=4,seed=0),
                        initial_audits=2,budget=2,fallback='project')
        self.assertTrue((h.certified_lower>=-1e-9).all())
        self.assertTrue((h.gain_evaluation_only>=-1e-9).all())
        self.assertTrue((h.paid_source_labels==2).all())
        self.assertTrue((h.fallback=='project').all())


if __name__=='__main__':unittest.main()
