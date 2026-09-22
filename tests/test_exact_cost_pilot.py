import unittest
import numpy as np
import pandas as pd
from src.candidate_bank_experiment import Config,run
from src.exact_cost_pilot import DESIGNS

class ExactCostTest(unittest.TestCase):
    def bank(self):
        return pd.DataFrame({'task_id':['x']*3,'candidate_id':['a','b','c'],
            'base_logprob':[0.,0.,0.],'trusted_score':[0.,.4,1.],'f::public_score':[1.,.5,0.]})
    def test_same_initial_state_exact_cost_and_all_schedules(self):
        first=None
        for controller,threshold,schedule in DESIGNS.values():
            cfg=Config(controller=controller,threshold=threshold,rounds=12,audit_per_refresh=8,total_audit_budget=32)
            h,tr=run(self.bank(),cfg,return_trace=True,audit_schedule=schedule,exact_audit_events=4)
            self.assertEqual(h.iloc[-1].audit_labels,32)
            self.assertEqual(int(h.refresh.sum()),4)
            if schedule:self.assertEqual(tuple(h.loc[h.refresh==1,'round']),schedule)
            if first is None:first=tr[0]
            self.assertEqual(first['audit_indices'],tr[0]['audit_indices'])
            np.testing.assert_array_equal(first['policy'],tr[0]['policy'])
    def test_never_trigger_forced_deadline(self):
        cfg=Config(controller='kl',threshold=float('inf'),rounds=12,audit_per_refresh=8,total_audit_budget=32)
        h=run(self.bank(),cfg,exact_audit_events=4)
        self.assertEqual(h.loc[h.refresh==1,'round'].tolist(),[1,10,11,12])
        self.assertEqual(int((h.refresh_reason=='deadline').sum()),3)
    def test_invalid_cost_design(self):
        with self.assertRaises(ValueError):run(self.bank(),Config(),exact_audit_events=4)
