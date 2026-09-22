"""Sample a frozen empirical bank from an immutable pretrained model revision."""
import argparse
import hashlib
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import time

from src.finite_code_tasks import tasks, score


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='Qwen/Qwen2.5-Coder-0.5B-Instruct')
    p.add_argument('--revision', required=True, help='Immutable 40-character HF commit')
    p.add_argument('--samples', type=int, default=32)
    p.add_argument('--task-limit', type=int, default=12)
    p.add_argument('--task-suite', choices=['development','transfer_v1'], default='development')
    p.add_argument('--seed', type=int, default=20260922)
    p.add_argument('--temperature', type=float, default=0.8)
    p.add_argument('--top-p', type=float, default=0.95)
    p.add_argument('--max-new-tokens', type=int, default=64)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--device', choices=['cpu','cuda','mps'], default='cpu')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.task_suite=='transfer_v1':
        from src.transfer_code_tasks import tasks as selected_tasks
    else:
        selected_tasks=tasks
    suite=selected_tasks()
    if len(a.revision)!=40 or any(c not in '0123456789abcdef' for c in a.revision):
        p.error('revision must be an immutable hexadecimal commit')
    if a.samples<1 or a.batch_size<1 or not 1<=a.task_limit<=len(suite): p.error('invalid bank size')
    if a.output.exists(): p.error('refusing to overwrite a frozen bank')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    torch.set_num_threads(a.threads)
    set_seed(a.seed)
    tokenizer = AutoTokenizer.from_pretrained(a.model, revision=a.revision, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision,
             trust_remote_code=False, torch_dtype=torch.float32 if a.device=='cpu' else torch.float16).to(a.device).eval()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time(); count=0
    manifest = {'schema_version':2, 'evidence':'pretrained_code_lm_restricted_expression_pilot',
        'configuration':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        'base_measure':'uniform_over_sample_occurrences; duplicates retained',
        'trusted_target':'uniform hidden domain, disjoint from public suite',
        'tasks':[t.__dict__ for t in suite[:a.task_limit]],
        'selected_task_source_sha256':hashlib.sha256(Path(__file__).with_name('transfer_code_tasks.py' if a.task_suite=='transfer_v1' else 'finite_code_tasks.py').read_bytes()).hexdigest(),
        'versions':{k:importlib.metadata.version(k) for k in ['torch','transformers','numpy']},
        'python':platform.python_version(),
        'started_utc':datetime.now(timezone.utc).isoformat(),
        'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'task_source_sha256':hashlib.sha256(Path(__file__).with_name('finite_code_tasks.py').read_bytes()).hexdigest(),
        'sweep_config_sha256':hashlib.sha256(Path('configs/pilot_sweep.json').read_bytes()).hexdigest(),
        'complete':False}
    meta=a.output.with_suffix('.manifest.json')
    meta.write_text(json.dumps(manifest,indent=2)+'\n')
    with a.output.open('x') as out, torch.inference_mode():
        for ti,task in enumerate(suite[:a.task_limit]):
            messages=[{'role':'system','content':'You write correct concise Python expressions.'},
                      {'role':'user','content':task.prompt()}]
            prompt=tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inp=tokenizer(prompt,return_tensors='pt').to(a.device)
            for start in range(0,a.samples,a.batch_size):
                batch_seed=a.seed+ti*10000+start
                set_seed(batch_seed)
                n=min(a.batch_size,a.samples-start)
                generated=model.generate(**inp,num_return_sequences=n,do_sample=True,
                    return_dict_in_generate=True,output_scores=True,
                    temperature=a.temperature,top_p=a.top_p,max_new_tokens=a.max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id)
                ids=generated.sequences
                transition=model.compute_transition_scores(ids,generated.scores,normalize_logits=True)
                token_ids=ids[:,inp.input_ids.shape[1]:]
                # Include the first EOS and exclude padding after it.
                active=torch.ones_like(token_ids,dtype=torch.bool)
                for bi in range(len(token_ids)):
                    eos=model.generation_config.eos_token_id
                    eos=eos if isinstance(eos,list) else [eos]
                    stops=torch.isin(token_ids[bi],torch.tensor(eos,device=token_ids.device)).nonzero()
                    if len(stops): active[bi,int(stops[0])+1:]=False
                sampling_logprob=torch.where(active,transition,torch.zeros_like(transition)).sum(dim=1).tolist()
                texts=tokenizer.batch_decode(ids[:,inp.input_ids.shape[1]:],skip_special_tokens=True)
                for j,text in enumerate(texts):
                    row={'task_id':task.task_id,'family':task.family,'split':task.split,
                         'candidate_id':f'{task.task_id}:{start+j}', 'source':text,
                         'source_sha256':hashlib.sha256(text.encode()).hexdigest(),
                         'prompt':prompt, 'model':a.model,'revision':a.revision,
                         'batch_seed':batch_seed,'base_logprob':0.0,
                         'sampling_logprob':sampling_logprob[j],
                         'generated_token_ids':token_ids[j][active[j]].tolist(),
                         'base_weight_semantics':'empirical_sample_occurrence', **score(text,task)}
                    out.write(json.dumps(row)+'\n'); count+=1
                out.flush()
            print(f'{task.task_id}: {count} candidates, {time.time()-started:.1f}s',flush=True)
    manifest.update(complete=True,rows=count,elapsed_seconds=time.time()-started,
                    bank_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest())
    meta.write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__': main()
