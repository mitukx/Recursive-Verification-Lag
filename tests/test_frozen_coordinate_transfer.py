import gzip,json,tempfile,unittest
from pathlib import Path
import pandas as pd
from src.frozen_coordinate_transfer import evaluate


class FrozenCoordinateTest(unittest.TestCase):
    def test_fixed_threshold_and_onset_only_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);bank=root/'bank';bank.mkdir()
            models=root/'models.json'
            models.write_text(json.dumps({'models':[{'optimizer':'soft','representation':'public','metric':'KL','column':'kl_from_refresh','fit':['0.5',1]}]}))
            record={'design':'uniform','task_id':'x','optimizer':'soft','representation':'public',
                'below_initial':[0,1,0],'coordinates':{'kl_from_refresh':[0.,1.,0.],
                'restricted_geometry':[0.,1.,0.],'proxy_margin':[1.,1.,1.]}}
            with gzip.open(bank/'runs.jsonl.gz','wt') as f:f.write(json.dumps(record)+'\n')
            output=root/'result.csv';evaluate(models,[bank],output)
            row=pd.read_csv(output).iloc[0]
            self.assertEqual(row['rows'],2)
            self.assertEqual(row.balanced_accuracy,1.)
            self.assertEqual(row.threshold,.5)
