"""Test sharp endpoints, including duplicates and sign-changing mass."""
import itertools
import unittest
import numpy as np
from src.identified_gain import identified_gain


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


if __name__=='__main__': unittest.main()
