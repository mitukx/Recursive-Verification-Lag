"""Pinned single-GPU real-update pilot; evaluation never chooses updates.

Run from a checkout of this research branch. This script clones the separately
pinned systems source, uses immutable local model/dataset snapshots, and saves
all raw attempts before interpreting terminal evaluation. It never executes
model-generated code. No Google Drive or GitHub credential is required.
"""
from __future__ import annotations
import asyncio
from dataclasses import asdict, replace
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT/'configs/colab_rlvr_pilot_v1.json'
OUT = ROOT/'results/colab_rlvr_pilot_v1'


def save(name, obj):
    path = OUT/name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True)+'\n')


def log_rows(name, rows):
    path = OUT/name
    with path.open('a') as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True)+'\n')


async def run(lock, systems):
    import numpy as np
    import torch
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from src.rvl_systems.hf_backend import HFLocalBackend
    from src.rvl_systems.hf_trainer import HFCausalLMGRPOTrainer, HFTTrainerConfig
    from src.rvl_systems.rlvr_benchmark import build_math_prompt, gsm8k_reference_answer, response_reward
    from src.rvl_systems.types import VerifiedGeneration
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; CPU substitution prohibited')
    hardware = subprocess.check_output(['nvidia-smi'], text=True)
    (OUT/'nvidia-smi.txt').write_text(hardware)
    save('environment.json', {'python': sys.version, 'torch': torch.__version__,
        'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name(0),
        'systems_source_sha': subprocess.check_output(['git','rev-parse','HEAD'], cwd=systems, text=True).strip(),
        'research_source_sha': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()})
    print('GPU:', torch.cuda.get_device_name(0), flush=True)
    model_path = snapshot_download(lock['model'], revision=lock['model_revision'])
    data = load_dataset(lock['dataset'], 'main', revision=lock['dataset_revision'])
    train = [(f'train-{i}', build_math_prompt(row['question']), gsm8k_reference_answer(row['answer']))
             for i, row in enumerate(data['train'].select(range(lock['train_tasks'])))]
    testing = [(f'test-{i}', build_math_prompt(row['question']), gsm8k_reference_answer(row['answer']))
               for i, row in enumerate(data['test'].select(range(lock['evaluation_tasks'])))]
    save('task_manifest.json', {'selection': lock['split_selection'], 'train': train, 'evaluation': testing,
        'model_revision': lock['model_revision'], 'dataset_revision': lock['dataset_revision']})
    answers = {t: answer for t, prompt, answer in train}
    def backend():
        b = HFLocalBackend(model_path, max_new_tokens=lock['max_new_tokens'], precision=lock['precision'])
        # Release generation caches between calls; retain gradients only for one response at a time.
        b.ensure_loaded()
        return b
    async def evaluate(b, seed):
        rows = []
        for i, (task, prompt, answer) in enumerate(testing):
            generation = (await b.generate(task, prompt, n=1, temperature=0., seed=seed+i))[0]
            rows.append({'task_id': task, 'expected': answer, 'response': generation.response,
                'correct': response_reward(generation.response, answer), 'token_count': generation.token_count,
                'latency_s': generation.latency_s})
        return rows
    b = backend()
    baseline = await evaluate(b, 100000)
    repeat = await evaluate(b, 100000)
    save('baseline.json', baseline); save('zero_update_repeat.json', repeat)
    identical = all(a['response'] == z['response'] and a['correct'] == z['correct']
                    for a, z in zip(baseline, repeat)) and len(baseline) == len(repeat)
    save('zero_update_control.json', {'identical': identical, 'tasks': len(baseline)})
    if not identical:
        raise RuntimeError('zero-update greedy reproducibility failed')
    del b; gc.collect(); torch.cuda.empty_cache()
    baseline_by_id = {r['task_id']: r for r in baseline}
    summaries = []
    for seed in lock['seeds']:
        for arm in lock['arms']:
            print('START', seed, arm, flush=True)
            b = backend()
            trainer = HFCausalLMGRPOTrainer(b.model, config=HFTTrainerConfig(learning_rate=lock['learning_rate']))
            rng = random.Random(seed); permutation = list(train); rng.shuffle(permutation)
            history = []; start = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
            for step in range(lock['steps']):
                chosen = permutation[step % len(permutation)]
                task, prompt, answer = chosen
                generations = await b.generate(task, prompt, n=lock['samples_per_prompt'],
                    temperature=lock['train_temperature'], seed=seed+step*10000)
                true_rewards = [response_reward(g.response, answer) for g in generations]
                supplied = list(true_rewards)
                if arm == 'within_prompt_shuffled_reward':
                    random.Random(seed+900000+step).shuffle(supplied)
                verified = [VerifiedGeneration(generation=g, reward=y, verifier_latency_s=0., verifier_version=0,
                    metadata={'true_reward_evaluation_only': original, 'arm': arm})
                    for g,y,original in zip(generations,supplied,true_rewards)]
                log_rows(f'{seed}_{arm}_rollouts.jsonl', [asdict(v) for v in verified])
                # Full FP32 probe of one layer documents actual parameter motion.
                probe = next(b.model.parameters()).detach().flatten()[:4096].clone()
                torch.cuda.synchronize(); tick = time.perf_counter()
                b.model.config.use_cache = False
                metrics = trainer.train_step(verified)
                b.model.config.use_cache = True
                torch.cuda.synchronize(); elapsed = time.perf_counter()-tick
                delta = float((next(b.model.parameters()).detach().flatten()[:4096]-probe).abs().max())
                row = {'step':step, 'seed':seed, 'arm':arm, 'true_reward_mean': float(np.mean(true_rewards)),
                    'nonconstant_reward_group': len(set(true_rewards)) > 1, 'parameter_probe_max_abs_change':delta,
                    'train_wall_s':elapsed, 'response_tokens':sum(g.token_count for g in generations), **metrics}
                history.append(row); save(f'{seed}_{arm}_history.json',history)
                print('STEP', seed, arm, step, row, flush=True)
            terminal = await evaluate(b, 200000)
            save(f'{seed}_{arm}_terminal.json',terminal)
            delta = float(np.mean([r['correct']-baseline_by_id[r['task_id']]['correct'] for r in terminal]))
            summary = {'seed':seed,'arm':arm,'baseline_accuracy':float(np.mean([r['correct'] for r in baseline])),
                'terminal_accuracy':float(np.mean([r['correct'] for r in terminal])), 'accuracy_delta':delta,
                'nonconstant_groups':sum(h['nonconstant_reward_group'] for h in history),
                'max_parameter_probe_change':max(h['parameter_probe_max_abs_change'] for h in history),
                'peak_allocated_gpu_bytes':torch.cuda.max_memory_allocated(), 'wall_s':time.perf_counter()-start}
            summaries.append(summary); save('summary.json',summaries)
            print('RESULT',summary,flush=True)
            del trainer,b; gc.collect(); torch.cuda.empty_cache()
    save('completion.json',{'status':'completed','all_seed_arm_runs':len(summaries),
        'claim':'raw small single-GPU learning pilot; terminal evaluation does not choose updates'})


def main():
    lock = json.loads(LOCK.read_text()); OUT.mkdir(parents=True, exist_ok=False)
    save('protocol.json',lock)
    systems = ROOT.parent/'rvl-pinned-systems'
    if not systems.exists():
        subprocess.run(['git','clone','https://github.com/mitukx/Recursive-Verification-Lag.git',str(systems)],check=True)
    subprocess.run(['git','checkout','--detach',lock['systems_source_sha']],cwd=systems,check=True)
    sys.path.insert(0,str(systems))
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    try:
        asyncio.run(run(lock, systems))
    except BaseException:
        save('failure.json', {'status':'failed','traceback':traceback.format_exc()})
        raise
    finally:
        entries={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in OUT.rglob('*') if p.is_file() and p.name != 'manifest.json'}
        save('manifest.json',{'files':entries,'protocol_sha256':hashlib.sha256(LOCK.read_bytes()).hexdigest(),
            'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})


if __name__ == '__main__':
    main()
