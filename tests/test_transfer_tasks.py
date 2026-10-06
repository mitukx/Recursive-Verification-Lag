import unittest
from src.transfer_code_tasks import tasks
from src.finite_code_tasks import tasks as development, score, parse_expression, interpret


class TransferTasksTest(unittest.TestCase):
    def test_disjoint_ids_and_reference_scores(self):
        self.assertFalse(set(t.task_id for t in tasks()) & set(t.task_id for t in development()))
        for t in tasks():
            self.assertEqual(t.split,'prospective_transfer_v1')
            self.assertEqual(score(t.reference,t)['trusted_score'],1)
            self.assertEqual(score(t.reference,t)['public_score'],1)

    def test_independent_reference_checks(self):
        for t in tasks():
            for x in t.domain:
                if t.task_id=='round4': want=min(range(-20,21,4),key=lambda k:(abs(x-k),-k))
                elif t.task_id=='digits': want=sum(int(c) for c in str(abs(x)))
                elif t.task_id=='fold': want=min(x%8,8-x%8)
                elif t.task_id=='saturating_square': want=min(x**2,25)
                elif t.task_id=='bucket': want=sum(x>=k for k in [-4,1,7])
                elif t.task_id=='divisor_code': want=2*(x%2==0)+3*(x%3==0)
                elif t.task_id=='nearest': want=min([-5,6],key=lambda k:(abs(x-k),k))
                else: want=x-4*int(x/4)
                self.assertEqual(interpret(parse_expression(t.reference),x),want)
