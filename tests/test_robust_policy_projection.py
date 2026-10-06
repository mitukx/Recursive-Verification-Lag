import itertools
import unittest
import numpy as np
from src.identified_gain import identified_gain
from src.robust_policy_projection import closest_certified_policy


class ProjectionTest(unittest.TestCase):
    def test_unseen_reward_extremes_and_non_line_direction(self):
        b=np.ones(3)/3
        target=b+np.array([.1,.1,-.2])
        ids=['a','b','c']; known={'a':1.,'b':0.}
        self.assertLess(identified_gain(target,b,ids,known)[0],0.)
        q,d=closest_certified_policy(target,b,ids,known)
        self.assertLess(d,abs(target-b).sum()-1e-6)
        self.assertGreater(np.abs(np.cross(q-b,target-b)).max(),1e-4)
        for unseen in [0.,1.]:
            self.assertGreaterEqual(float((q-b)@np.array([1.,0.,unseen])),-1e-8)

    def test_duplicate_source_labels_cannot_fake_certainty(self):
        b=np.ones(4)/4; target=np.array([.1,.1,.6,.2])
        ids=['a','a','b','c']; known={'a':0.,'b':1.}
        q,d=closest_certified_policy(target,b,ids,known)
        self.assertAlmostEqual(d,0.)
        self.assertGreaterEqual(identified_gain(q,b,ids,known)[0],-1e-8)

    def test_lp_minimum_matches_enumerated_simplex_grid(self):
        b=np.ones(3)/3; target=np.array([.45,.45,.10])
        ids=['a','b','c']; known={'a':1.,'b':0.}
        q,d=closest_certified_policy(target,b,ids,known)
        for i,j in itertools.product(range(101),repeat=2):
            if i+j>100:continue
            grid=np.array([i,j,100-i-j])/100
            if identified_gain(grid,b,ids,known)[0]>=-1e-12:
                self.assertLessEqual(d,np.abs(grid-target).sum()+1e-8)
        self.assertGreaterEqual(identified_gain(q,b,ids,known)[0],-1e-8)


if __name__=='__main__': unittest.main()
