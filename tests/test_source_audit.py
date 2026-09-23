import unittest
import numpy as np
from src.source_audit import SourceAudit

class SourceAuditTest(unittest.TestCase):
    def test_no_repeat_and_shared_initial_sources(self):
        ids=['a','a','b','c','d','e','f'];p=np.array([1.,0,0,0,0,0,0])
        initial=[]
        for mode in ['stream','policy']:
            sampler=SourceAudit(ids,3,mode);audit=[]
            for _ in range(3):
                out=sampler(p,2,tuple(audit));audit.extend(out.tolist())
                if len(audit)==2:initial.append(audit.copy())
            self.assertEqual(len(set(np.asarray(ids)[audit])),6)
            with self.assertRaises(ValueError):sampler(p,1,tuple(audit))
        self.assertEqual(*initial)
    def test_exogenous_stream_is_policy_independent(self):
        left=SourceAudit(list('abcdef'),4);right=SourceAudit(list('abcdef'),4)
        a=[];b=[]
        for _ in range(3):
            a.extend(left(np.array([1,0,0,0,0,0]),2,tuple(a)))
            b.extend(right(np.array([0,0,0,0,0,1]),2,tuple(b)))
        self.assertEqual(a,b)
