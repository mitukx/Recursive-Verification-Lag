import unittest
import numpy as np
from src.analyze_candidate_sweep import fit_threshold, balanced_accuracy

class TransferAnalysisTest(unittest.TestCase):
    def test_ties_are_not_split_and_fit_matches_bruteforce(self):
        rng=np.random.default_rng(8)
        for _ in range(30):
            x=rng.integers(0,5,30).astype(float); y=rng.integers(0,2,30)
            t,d=fit_threshold(x,y)
            actual=balanced_accuracy(y,x>=t if d==1 else x<t)
            unique=np.unique(x); thresholds=np.r_[-np.inf,(unique[1:]+unique[:-1])/2,np.inf]
            expected=max(balanced_accuracy(y,p) for th in thresholds for p in [x>=th,x<th])
            self.assertAlmostEqual(actual,expected)
    def test_single_class_is_not_reported_as_perfect_transfer(self):
        self.assertIsNone(fit_threshold([1,2,3],[0,0,0]))
        self.assertTrue(np.isnan(balanced_accuracy([1,1],[1,1])))

if __name__=='__main__': unittest.main()
