"""Write disjoint, outcome-blind MBPP+ task ranks from the pinned parquet."""
import argparse
import hashlib
import json
from pathlib import Path
from src.generate_mbppplus_bank import (load_selected,DATASET,FILE,FILE_SHA256,
                                        REVISION,SALT,MODEL,MODEL_REVISION)


def frozen_split():
    ordered=load_selected(count=72)
    def section(rows):
        return [{'rank':rank,'task_id':task_id,'entry_point':entry,
                 'prompt_sha256':hashlib.sha256(row['prompt'].encode()).hexdigest(),
                 'public_tests_sha256':hashlib.sha256(json.dumps(row['test_list']).encode()).hexdigest(),
                 'additional_tests_sha256':hashlib.sha256(row['test'].encode()).hexdigest()}
                for rank,(task_id,entry,row) in rows]
    indexed=list(enumerate(ordered))
    sections={'scored_pilot':section(indexed[:8]),
              'development_unscored':section(indexed[8:40]),
              'heldout_unscored':section(indexed[40:72])}
    pilot=json.loads(Path('data/mbppplus_qwen15b_pilot_v1.manifest.json').read_text())
    if [x['task_id'] for x in sections['scored_pilot']]!=[
            x['task_id'] for x in pilot['selected_tasks']]:
        raise ValueError('task selection no longer reproduces committed pilot')
    return {'purpose':'pre-candidate-outcome disjoint extension of scored pilot',
            'dataset':DATASET,'dataset_revision':REVISION,'dataset_file':FILE,
            'dataset_file_sha256':FILE_SHA256,'selection_salt':SALT,
            'ordering':'SHA256(salt+task_id), eligible reference function used in >=3 public tests, prompt<600 chars',
            'generator_model':MODEL,'generator_revision':MODEL_REVISION,
            'candidate_generation':{'samples_per_task':16,'temperature':0.8,
                                    'top_p':0.95,'max_new_tokens':256,
                                    'development_seed':20260928,
                                    'heldout_seed':20260929},
            'sections':sections}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('refuse to overwrite frozen split')
    split=frozen_split();args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(split,indent=2)+'\n')
    print({name:len(rows) for name,rows in split['sections'].items()})


if __name__=='__main__':main()
