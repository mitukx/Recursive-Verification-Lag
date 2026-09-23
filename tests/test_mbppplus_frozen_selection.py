"""Task selection must be fixed before trusted model outcomes are read."""
import json
import unittest
from pathlib import Path
from src.generate_mbppplus_bank import select_eligible_rows


class FrozenSelectionTest(unittest.TestCase):
    def test_rank_slices_are_disjoint_and_input_order_independent(self):
        rows=[{'task_id':str(i),'code':'def solve(x):\n return x',
               'test_list':['assert solve(1)==1']*3,'prompt':'Return input.'}
              for i in range(100)]
        pilot=select_eligible_rows(rows,count=8)
        development=select_eligible_rows(rows,count=32,offset=8)
        held=select_eligible_rows(rows,count=32,offset=40)
        self.assertEqual([x[0] for x in pilot],
                         [x[0] for x in select_eligible_rows(rows[::-1],count=8)])
        self.assertEqual(len({x[0] for x in pilot+development+held}),72)

    def test_default_order_preserves_frozen_pilot_ids(self):
        manifest=json.loads(Path('data/mbppplus_qwen15b_pilot_v1.manifest.json').read_text())
        rows=[]
        for x in manifest['selected_tasks']:
            rows.append({'task_id':x['task_id'],
                         'code':f"def {x['entry_point']}(*args):\n return None",
                         'test_list':x['public_tests'],'prompt':x['prompt']})
        self.assertEqual([x[0] for x in select_eligible_rows(rows)],
                         [x['task_id'] for x in manifest['selected_tasks']])

    def test_bad_offset_is_rejected(self):
        with self.assertRaises(ValueError):select_eligible_rows([],count=0)
        with self.assertRaises(ValueError):select_eligible_rows([],count=1,offset=1)


if __name__=='__main__':unittest.main()
