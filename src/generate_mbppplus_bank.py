"""Freeze unexecuted Python-program completions on pinned official MBPP+.

Generated Python is DATA here. Never import, exec, or evaluate it on the host.
The scoring stage must run inside an explicitly isolated container.
"""
import argparse
import ast
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import time
import warnings

DATASET='evalplus/mbppplus'
REVISION='b2d74c91837c3f2a20c1299ae98133cbe7cfa077'
FILE='data/test-00000-of-00001-d5781c9c51e02795.parquet'
FILE_SHA256='dc20030b3788fccf617444edcb34138ef13d7e4fafd17bfcb8c1279dbb12399b'
MODEL='Qwen/Qwen2.5-Coder-1.5B-Instruct'
MODEL_REVISION='2e1fd397ee46e1388853d2af2c993145b0f1098a'
SALT='rvl-mbppplus-pilot-v1:'


def select_eligible_rows(rows,count=8,offset=0):
    """Prompt-only deterministic task order; no candidate scores consulted."""
    if not isinstance(count,int) or not isinstance(offset,int) or count<1 or offset<0:
        raise ValueError('count must be positive and offset nonnegative integers')
    eligible=[]
    for row in rows:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',SyntaxWarning)
                names=[n.name for n in ast.parse(row['code']).body
                       if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
            entries=[n for n in names if all(re.search(r'\b'+re.escape(n)+r'\s*\(',t)
                                               for t in row['test_list'])]
        except (SyntaxError,TypeError): continue
        if len(entries)==1 and len(row['test_list'])>=3 and len(row['prompt'])<600:
            eligible.append((str(row['task_id']),entries[0],row))
    eligible.sort(key=lambda v:hashlib.sha256((SALT+v[0]).encode()).hexdigest())
    if len({i for i,_,_ in eligible})!=len(eligible):
        raise ValueError('eligible task identifiers must be unique')
    if len(eligible)<offset+count:
        raise ValueError('insufficient eligible tasks for frozen offset/count')
    return eligible[offset:offset+count]


def load_selected(count=8,offset=0):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    path=Path(hf_hub_download(DATASET,filename=FILE,repo_type='dataset',revision=REVISION))
    if hashlib.sha256(path.read_bytes()).hexdigest()!=FILE_SHA256:
        raise ValueError('pinned MBPP+ data file hash mismatch')
    return select_eligible_rows(pq.read_table(path).to_pylist(),count,offset)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=20260927)
    p.add_argument('--samples',type=int,default=8)
    p.add_argument('--max-new-tokens',type=int,default=256)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--task-offset',type=int,default=0)
    p.add_argument('--task-count',type=int,default=8)
    p.add_argument('--device',choices=['cpu','mps','cuda'],default='cpu')
    a=p.parse_args()
    if a.output.exists() or a.output.with_suffix('.manifest.json').exists():
        p.error('refuse to overwrite a frozen bank or manifest')
    if min(a.samples,a.max_new_tokens,a.batch_size,a.threads)<1:p.error('invalid settings')
    tasks=load_selected(count=a.task_count,offset=a.task_offset)
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,set_seed
    torch.set_num_threads(a.threads);set_seed(a.seed)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_REVISION,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=MODEL_REVISION,
        trust_remote_code=False,torch_dtype=torch.float32 if a.device=='cpu' else torch.float16).to(a.device).eval()
    prompts=[]
    for task_id,entry,row in tasks:
        messages=[{'role':'system','content':'Write a correct standalone Python function. Return code only.'},
          {'role':'user','content':f"Implement {entry}. {row['prompt']}\nPublic examples:\n"+
             '\n'.join(row['test_list'])+'\nReturn complete Python source defining the function.'}]
        prompts.append(tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True))
    manifest={'status':'unscored_pretrained_python_program_bank',
        'evidence':'real_pretrained_lm_generating_standard_MBPPplus_programs; no trusted execution yet',
        'dataset':DATASET,'dataset_revision':REVISION,'dataset_file':FILE,
        'dataset_file_sha256':FILE_SHA256,'selection_salt':SALT,
        'selection_rule':('first 8 SHA256(salt+task_id) among unique reference defs used in >=3 public tests; prompt<600 chars'
                          if a.task_offset==0 and a.task_count==8 else
                          f'hash ranks [{a.task_offset},{a.task_offset+a.task_count}) among eligible tasks; unique reference def, >=3 public tests, prompt<600 chars'),
        'selection_offset':a.task_offset,'selection_count':a.task_count,
        'model':MODEL,'model_revision':MODEL_REVISION,
        'generation':{'seed':a.seed,'samples':a.samples,'max_new_tokens':a.max_new_tokens,
            'temperature':.8,'top_p':.95,'batch_size':a.batch_size,'threads':a.threads,'device':a.device},
        'selected_tasks':[{'task_id':i,'entry_point':entry,'public_tests':r['test_list'],
                           'prompt':r['prompt'],'plus_test_sha256':hashlib.sha256(r['test'].encode()).hexdigest()}
                          for i,entry,r in tasks],
        'versions':{n:importlib.metadata.version(n) for n in ['torch','transformers','pyarrow']},
        'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'started_utc':datetime.now(timezone.utc).isoformat(),'complete':False}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    meta=a.output.with_suffix('.manifest.json');meta.write_text(json.dumps(manifest,indent=2)+'\n')
    start=time.time();count=0
    with a.output.open('x') as out,torch.inference_mode():
        for ti,((task_id,entry,_),prompt) in enumerate(zip(tasks,prompts)):
            inp=tokenizer(prompt,return_tensors='pt').to(a.device)
            for offset in range(0,a.samples,a.batch_size):
                batch_seed=a.seed+ti*10000+offset;set_seed(batch_seed)
                n=min(a.batch_size,a.samples-offset)
                generation=model.generate(**inp,num_return_sequences=n,do_sample=True,
                    return_dict_in_generate=True,temperature=.8,top_p=.95,
                    max_new_tokens=a.max_new_tokens,pad_token_id=tokenizer.eos_token_id)
                toks=generation.sequences[:,inp.input_ids.shape[1]:]
                texts=tokenizer.batch_decode(toks,skip_special_tokens=True)
                for j,source in enumerate(texts):
                    rec={'task_id':task_id,'entry_point':entry,'candidate_id':f'MBPPPlus/{task_id}:{offset+j}',
                         'source':source,'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
                         'generated_token_ids':toks[j].tolist(),'batch_seed':batch_seed,
                         'base_logprob':0.,'base_weight_semantics':'empirical_sample_occurrence',
                         'model':MODEL,'revision':MODEL_REVISION}
                    out.write(json.dumps(rec)+'\n');count+=1
                out.flush()
            print(f'MBPPPlus/{task_id}: {count} unscored candidates; {time.time()-start:.1f}s',flush=True)
    manifest.update(complete=True,rows=count,elapsed_seconds=time.time()-start,
                    bank_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest())
    meta.write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
