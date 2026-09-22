import itertools
import unittest
from dataclasses import replace
import numpy as np
import pandas as pd
from src.candidate_bank_experiment import Config, run, best_of_n, restricted_geometry
from src.finite_code_tasks import tasks, parse_expression, interpret, score
from src.observable_certificate import exact_box_gain_bounds, error_envelope_lower_bound


class ResearchPipelineTest(unittest.TestCase):
    def bank(self):
        return pd.DataFrame({'task_id':['a']*4,'candidate_id':list('abcd'),
          'base_logprob':[0.]*4,'trusted_score':[0.,0.5,1.,0.],
          'f::public_score':[0.,0.4,0.7,1.]})

    def test_best_of_n_matches_exhaustive_draws_with_ties(self):
        p=np.array([.2,.3,.5]); s=np.array([0.,1.,1.]); expected=np.zeros(3)
        for draws in itertools.product(range(3),repeat=3):
            w=np.prod(p[list(draws)]); best=max(s[list(draws)])
            winners=[i for i in draws if s[i]==best]
            for i in winners: expected[i]+=w/len(winners)
        np.testing.assert_allclose(best_of_n(p,s,3),expected,atol=1e-14)
        np.testing.assert_allclose(best_of_n(p,s,1),p)

    def test_frozen_best_of_n_composition(self):
        p=np.array([.1,.2,.3,.4]); score=np.array([0.,1.,1.,2.])
        composed=best_of_n(best_of_n(p,score,3),score,5)
        np.testing.assert_allclose(composed,best_of_n(p,score,15),atol=1e-14)

    def test_score_rescaling_invariance(self):
        cfg=Config(rounds=8,eta=2.,audit_per_refresh=8,total_audit_budget=32)
        for controller in ['fixed','kl','maxlog','geometry']:
            c=replace(cfg,controller=controller)
            a,ta=run(self.bank(),c,return_trace=True)
            b,tb=run(self.bank(),replace(c,eta=c.eta/10,score_scale=10),return_trace=True)
            for x,y in zip(ta,tb): np.testing.assert_allclose(x['policy'],y['policy'],atol=1e-12)
            np.testing.assert_array_equal(a.audit_labels,b.audit_labels)

    def test_unqueried_labels_cannot_change_policy_or_trigger(self):
        df=self.bank(); cfg=Config(rounds=8,audit_per_refresh=1,total_audit_budget=1,controller='kl',threshold=.01)
        a,ta=run(df,cfg,return_trace=True); queried=set(ta[-1]['audit_indices'])
        changed=df.copy()
        for i in range(len(df)):
            if i not in queried: changed.loc[i,'trusted_score']=1-df.loc[i,'trusted_score']
        b,tb=run(changed,cfg,return_trace=True)
        for x,y in zip(ta,tb): np.testing.assert_array_equal(x['policy'],y['policy'])
        np.testing.assert_array_equal(a.trigger,b.trigger)
        self.assertEqual(int(a.audit_labels.max()),1)

    def test_budget_and_fixed_cadence(self):
        result=run(self.bank(),Config(rounds=9,refresh_interval=3,audit_per_refresh=5,total_audit_budget=12))
        self.assertEqual(result.loc[result.refresh==1,'round'].tolist(),[1,4,7])
        self.assertEqual(result.audit_labels.tolist(),[5,5,5,10,10,10,12,12,12])

    def test_reference_tasks_and_disjoint_hidden(self):
        for task in tasks():
            s=score(task.reference,task)
            self.assertEqual(s['trusted_score'],1.); self.assertEqual(s['public_score'],1.)
        t=tasks()[0]; s=score('x',t)
        self.assertEqual(s['public_score'],1.); self.assertLess(s['trusted_score'],1.)

    def test_no_python_execution_and_limits(self):
        for text in ["__import__('os').system('id')",'x.__class__','2**999999','[x for x in range(5)]','open(1)']:
            with self.assertRaises((ValueError,SyntaxError)): parse_expression(text)
        with self.assertRaises(ValueError): interpret(parse_expression('9999*9999*9999'),1)

    def test_box_certificate_is_sharp_on_all_reward_completions(self):
        p=np.array([.5,.25,.25]); q=np.array([.1,.7,.2]); known=np.array([.4,np.nan,np.nan])
        low,high=exact_box_gain_bounds(p,q,known); gains=[]
        for a,b in itertools.product([0.,1.],repeat=2): gains.append((q-p)@np.array([.4,a,b]))
        self.assertAlmostEqual(low,min(gains)); self.assertAlmostEqual(high,max(gains))
        self.assertEqual(exact_box_gain_bounds(p,p,known),(0.,0.))

    def test_structured_geometry_and_conditional_bound(self):
        p=np.array([.5,.5]); q=np.array([.25,.75]); X=np.eye(2)
        self.assertAlmostEqual(restricted_geometry(p,q,X,p),.25)
        theta=np.array([.1,.8]); A=np.eye(2); beta=.1; eps=.05
        bound=error_envelope_lower_bound(p,q,X,theta,A,beta,eps)
        d=q-p; worst=theta-beta*d/np.linalg.norm(d)-eps*np.sign(d)
        self.assertAlmostEqual(bound,float(d@worst))

if __name__=='__main__': unittest.main()
