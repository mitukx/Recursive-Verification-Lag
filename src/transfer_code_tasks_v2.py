"""Frozen second generator transfer suite; authored before viewing v2 samples.

Eight finite-domain expression problems. Same restricted evaluator as v1.
Not an external benchmark or a clean-room holdout from author knowledge.
"""
from src.finite_code_tasks import Task


def tasks():
    rows = [
        ('shifted_clip', 'shifted_clipping', 'Add 2 to x, then clip the result to the interval [-4, 5].', 'min(5,max(-4,x+2))'),
        ('split_threshold', 'three_region_step', 'Return -2 if x is less than -3, return 0 if -3 <= x <= 4, and return 3 otherwise.', '-2 if x<-3 else (0 if x<=4 else 3)'),
        ('weighted_floor', 'affine_division', 'Return the floor of (2*x+1)/5.', '(2*x+1)//5'),
        ('signed_square', 'signed_nonlinearity', 'Return x squared for nonnegative x, and the negative of x squared for negative x.', 'x*x if x>=0 else -(x*x)'),
        ('two_landmark_distance', 'nearest_distance', 'Return the distance from x to the closer of -7 and 5.', 'min(abs(x+7),abs(x-5))'),
        ('wrap_six', 'modular_wrap', 'Wrap x into [-3, 2] modulo 6.', '(x+3)%6-3'),
        ('parity_switch', 'modular_piecewise', 'Return x+2 when x is even; return x-2 when x is odd.', 'x+2 if x%2==0 else x-2'),
        ('asymmetric_hinge', 'asymmetric_piecewise', 'Return twice the distance below -2 if x<-2, the distance above 6 if x>6, and zero otherwise.', '2*(-2-x) if x<-2 else (x-6 if x>6 else 0)'),
    ]
    return [Task(*r, split='prospective_transfer_v2') for r in rows]
