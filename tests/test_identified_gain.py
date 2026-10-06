"""Test sharp endpoints, including duplicates and sign-changing mass."""
import itertools
import unittest
import numpy as np
from src.identified_gain import identified_gain,maximal_certified_mix,lower_bound_right_slope


class IdentifiedGainTest(unittest.TestCase):
    def test_sharp_against_all_binary_completions(self):
        p=np.array([.10,.40,.10,.40]);q=np.full(4,.25)
        sources=['same','same','up','down']
        known={'same':.4}
        lo,hi=identified_gain(p,q,sources,known)
        feasible=[]
        for up,down in itertools.product([0.,1.],repeat=2):
            rewards=np.array([.4,.4,up,down])
            feasible.append(float((p-q)@rewards))
        self.assertAlmostEqual(lo,min(feasible));self.assertAlmostEqual(hi,max(feasible))
        self.assertLess(lo,0);self.assertGreater(hi,0)

    def test_all_labeled_collapses_and_duplicate_label_is_shared(self):
        p=np.array([.1,.7,.2]);q=np.array([.4,.3,.3]);ids=['a','a','b']
        lo,hi=identified_gain(p,q,ids,{'a':.8,'b':.2})
        self.assertAlmostEqual(lo,hi)
        self.assertAlmostEqual(lo,float((p-q)@np.array([.8,.8,.2])))
        other=identified_gain(p,q,ids,{'a':.8})
        self.assertGreater(other[1]-other[0],0)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            identified_gain([.2,.2],[.5,.5],['a','b'],{})
        with self.assertRaises(ValueError):
            identified_gain([.5,.5],[.5,.5],['a','b'],{'a':1.2})

    def test_maximal_mix_is_sharp_and_safe(self):
        baseline=np.array([.3,.3,.4]);current=np.array([.5,.1,.4])
        proposal=np.array([.2,.6,.2]);ids=['a','b','c'];known={'a':1.,'b':0.}
        mixed,alpha=maximal_certified_mix(current,proposal,baseline,ids,known)
        self.assertAlmostEqual(alpha,.4,places=10)
        self.assertGreaterEqual(identified_gain(mixed,baseline,ids,known)[0],-1e-12)
        for reward_c in [0.,.3,1.]:
            self.assertGreaterEqual((mixed-baseline)@np.array([1.,0.,reward_c]),-1e-12)
        with self.assertRaises(ValueError):
            maximal_certified_mix(proposal,current,baseline,ids,known)
        slope=lower_bound_right_slope(current,proposal,baseline,ids,known)
        step=1e-6
        direct=(identified_gain((1-step)*current+step*proposal,baseline,ids,known)[0]
                -identified_gain(current,baseline,ids,known)[0])/step
        self.assertAlmostEqual(slope,direct,places=8)


if __name__=='__main__': unittest.main()
