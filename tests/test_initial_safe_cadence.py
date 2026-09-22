import unittest
import pandas as pd
from src.candidate_bank_experiment import Config,run
from src.initial_safe_cadence import compare

class SharedInitialStateTest(unittest.TestCase):
    def fixture(self):
        bank=pd.DataFrame({'task_id':['a','a'],'candidate_id':['0','1'],
            'base_logprob':[0.,0.],'trusted_score':[0.,1.],'f::public_score':[0.,1.]})
        histories=[];summaries=[]
        for cadence in [1,2,4,12]:
            cfg=Config(eta=1,refresh_interval=cadence,rounds=12,audit_per_refresh=8,
                       total_audit_budget=96,representation='public',seed=0)
            h=run(bank,cfg);h['task_id']='a';histories.append(h)
            summaries.append({**cfg.__dict__,'task_id':'a','ever_below_initial':int(h.below_initial.any()),
                              'final_gain':h.iloc[-1].true_reward-h.iloc[-1].initial_true_reward,
                              'audit_labels':int(h.iloc[-1].audit_labels)})
        return pd.DataFrame(summaries),pd.concat(histories,ignore_index=True)
    def test_shared_cohort_and_cost_difference(self):
        runs,rows=self.fixture();pairs,summary=compare(runs,rows)
        self.assertEqual(len(pairs),3)
        self.assertEqual(summary.fresh_draws.tolist(),[96,96,96])
        self.assertEqual(summary.stale_draws.tolist(),[48,24,8])
    def test_mismatched_initial_state_is_rejected(self):
        runs,rows=self.fixture()
        rows.loc[(rows['round']==1)&(rows.refresh_interval==12),'true_reward']-=.01
        with self.assertRaises(ValueError):compare(runs,rows)

if __name__=='__main__':unittest.main()
