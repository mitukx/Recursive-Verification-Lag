"""Synthetic integration fixture: never treated as a pretrained-model result."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

class PipelineIntegrationTest(unittest.TestCase):
    def test_generation_free_sweep_and_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); bank=root/'bank.jsonl'; config=root/'config.json'; out=root/'out'
            rows=[{'task_id':t,'candidate_id':str(i),'public_score':s,'trusted_score':r,
                   'features':{'public_score':s,'valid':1.}}
                  for t in ['fixture_a','fixture_b'] for i,(s,r) in enumerate([(0.,0.),(.7,1.),(1.,0.)])]
            bank.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            spec=json.loads(Path('configs/pilot_sweep.json').read_text())
            spec.update(rounds=3,seeds=[0],soft_eta=[1.],best_of_n=[2],fixed_intervals=[1,3],
                        kl_thresholds=[.1],maxlog_thresholds=[],geometry_thresholds=[],
                        representations=['public'],total_audit_budgets=[8])
            config.write_text(json.dumps(spec))
            commands=[['src.run_candidate_sweep',str(bank),'--config',str(config),'--output',str(out)],
                      ['src.analyze_candidate_sweep',str(out)],['src.compare_refresh',str(out)]]
            for command in commands:
                result=subprocess.run([sys.executable,'-m',*command],capture_output=True,text=True,timeout=60)
                self.assertEqual(result.returncode,0,result.stderr)
            manifest=json.loads((out/'manifest.json').read_text())
            self.assertEqual(manifest['runs'],12)
            self.assertTrue((out/'transfer.csv').exists())
            self.assertTrue((out/'crossfit_refresh_summary.csv').exists())

if __name__=='__main__': unittest.main()
