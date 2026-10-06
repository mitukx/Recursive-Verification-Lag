"""Task selection must be fixed before trusted model outcomes are read."""
import json
import hashlib
import unittest
from pathlib import Path
from src.generate_mbppplus_bank import (select_eligible_rows,validate_frozen_split,
                                        FILE_SHA256,REVISION,SALT,MODEL,MODEL_REVISION)
from src.score_mbppplus_docker import validate_bank_records


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

    def test_freeze_checks_outcomes_blind_metadata_before_generation(self):
        row={'prompt':'What is x?','test_list':['assert solve(1)==1']*3,
             'test':'assert solve(2)==2'}
        task=('a','solve',row)
        record={'rank':8,'task_id':'a','entry_point':'solve',
                'prompt_sha256':hashlib.sha256(row['prompt'].encode()).hexdigest(),
                'public_tests_sha256':hashlib.sha256(json.dumps(row['test_list']).encode()).hexdigest(),
                'additional_tests_sha256':hashlib.sha256(row['test'].encode()).hexdigest()}
        frozen={'dataset_revision':REVISION,'dataset_file_sha256':FILE_SHA256,
                'selection_salt':SALT,'generator_model':MODEL,
                'generator_revision':MODEL_REVISION,
                'sections':{'development_unscored':[record]},
                'candidate_generation':{'development_seed':11,'heldout_seed':12,
                                        'samples_per_task':16,'max_new_tokens':256,
                                        'temperature':.8,'top_p':.95}}
        args={'offset':8,'count':1,'seed':11,'samples':16,'max_new_tokens':256}
        validate_frozen_split([task],frozen,'development_unscored',**args)
        with self.assertRaises(ValueError):
            validate_frozen_split([task],frozen,'development_unscored',**{**args,'seed':12})
        with self.assertRaises(ValueError):
            validate_frozen_split([('a','solve',{**row,'prompt':'Changed'})],
                                  frozen,'development_unscored',**args)
        with self.assertRaises(ValueError):
            validate_frozen_split([task],{**frozen,'sections':{'development_unscored':
                [{**record,'rank':9}]}},'development_unscored',**args)

    def test_scorer_rejects_source_or_occurrence_changes(self):
        task=('a','solve',{'test_list':[]})
        rows=[{'candidate_id':f'MBPPPlus/a:{j}','task_id':'a','entry_point':'solve',
               'source':f'def solve(x): return {j}',
               'source_sha256':hashlib.sha256(f'def solve(x): return {j}'.encode()).hexdigest()}
              for j in range(2)]
        manifest={'generation':{'samples':2}}
        validate_bank_records(rows,manifest,[task])
        with self.assertRaises(ValueError):
            validate_bank_records([{**rows[0],'source':'bad'},rows[1]],manifest,[task])
        with self.assertRaises(ValueError):
            validate_bank_records([rows[0],rows[0]],manifest,[task])
        with self.assertRaises(ValueError):
            validate_bank_records([rows[0],{**rows[1],'candidate_id':'MBPPPlus/a:5'}],
                                  manifest,[task])


if __name__=='__main__':unittest.main()
