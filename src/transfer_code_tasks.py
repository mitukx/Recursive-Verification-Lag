"""Prospective transfer tasks: unseen specifications, shared expression grammar.

Not an independent benchmark or clean-room holdout: tasks were authored after
development findings. Never mutate this suite after viewing generated outcomes.
"""
from src.finite_code_tasks import Task


def tasks():
    rows = [
        ('round4', 'rounding', 'Round x to the nearest multiple of 4; on ties choose the larger multiple.', '4*((x+2)//4)'),
        ('digits', 'digit_arithmetic', 'Return the sum of the decimal digits of the absolute value of x.', 'abs(x)//10+abs(x)%10'),
        ('fold', 'periodic_fold', 'Let r be the nonnegative remainder when x is divided by 8. Return the smaller of r and 8-r.', 'min(x%8,8-x%8)'),
        ('saturating_square', 'nonlinear_saturation', 'Return x squared, capped above at 25.', 'min(x*x,25)'),
        ('bucket', 'threshold_count', 'Return how many of the thresholds -4, 1, and 7 are less than or equal to x.', '(1 if x>=-4 else 0)+(1 if x>=1 else 0)+(1 if x>=7 else 0)'),
        ('divisor_code', 'divisor_encoding', 'Return 2 if x is divisible by 2 plus 3 if x is divisible by 3, adding both contributions when applicable.', '(2 if x%2==0 else 0)+(3 if x%3==0 else 0)'),
        ('nearest', 'nearest_landmark', 'Return whichever of -5 and 6 is closer to x; on a tie return -5.', '-5 if abs(x+5)<=abs(x-6) else 6'),
        ('signed_remainder', 'signed_remainder', 'Return the remainder after dividing x by 4 with quotient truncated toward zero; the remainder has the sign of x.', 'x%4 if x>=0 else -((-x)%4)'),
    ]
    return [Task(*r, split='prospective_transfer_v1') for r in rows]
