"""Fail-closed MBPP+ scoring in disposable pinned Docker containers.

Requires an immutable local image ID (sha256:digest) or pinned registry image
(name@sha256:digest) with Python and NumPy. No generated Python is executed here.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import selectors
import subprocess
import time
import uuid
from src.generate_mbppplus_bank import load_selected,FILE_SHA256,REVISION

MAX_OUTPUT=65536


def sanitize(source):
    source=source.strip()
    fence=re.fullmatch(r'```(?:python)?\s*\n(.*?)\n```\s*',source,re.S)
    return fence.group(1) if fence else source


def syntax_valid(source):
    try:ast.parse(source)
    except (SyntaxError,ValueError,RecursionError):return False
    return True


def docker_command(image,name,worker,platform='linux/arm64'):
    if not (re.fullmatch(r'sha256:[a-f0-9]{64}',image) or
            re.fullmatch(r'[a-zA-Z0-9./:_-]+@sha256:[a-f0-9]{64}',image)):
        raise ValueError('Docker image must be pinned by SHA256 ID or manifest digest')
    if platform not in ('linux/arm64','linux/amd64'):raise ValueError('unsupported platform')
    return ['docker','run','--rm','--interactive','--name',name,
        '--platform',platform,
        '--network','none','--read-only','--tmpfs','/tmp:rw,nosuid,nodev,size=16m',
        '--user','65534:65534','--pids-limit','32','--memory','512m','--cpus','1',
        '--cap-drop','ALL','--security-opt','no-new-privileges',
        '--pull','never',image,'python','-I','-B','-c',worker]


def run_capped(command,payload,container_name,timeout=20):
    try:
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,start_new_session=True)
    except OSError as exc:
        raise RuntimeError('Docker is unavailable; no model code was executed') from exc
    out=bytearray();err=bytearray();deadline=time.monotonic()+timeout
    try:
        process.stdin.write(json.dumps(payload).encode());process.stdin.close()
        selector=selectors.DefaultSelector()
        selector.register(process.stdout,selectors.EVENT_READ,out)
        selector.register(process.stderr,selectors.EVENT_READ,err)
        while selector.get_map():
            remaining=deadline-time.monotonic()
            if remaining<=0:raise TimeoutError('Docker candidate exceeded wall timeout')
            for key,_ in selector.select(min(remaining,.5)):
                chunk=key.fileobj.read1(4096)
                if not chunk:
                    selector.unregister(key.fileobj);continue
                key.data.extend(chunk)
                if len(out)+len(err)>MAX_OUTPUT:
                    raise RuntimeError('Docker candidate exceeded output cap')
        if process.wait(timeout=1)!=0:
            raise RuntimeError('Docker evaluation failed: '+err.decode(errors='replace')[-300:])
        return json.loads(out.decode())
    except BaseException:
        process.kill()
        try:process.wait(timeout=2)
        except subprocess.TimeoutExpired:pass
        # A killed CLI can leave the container alive; force removal by its
        # randomized name. This cleanup does not execute candidate code.
        subprocess.run(['docker','rm','-f',container_name],capture_output=True,timeout=5)
        raise


def main():
    p=argparse.ArgumentParser()
    p.add_argument('bank',type=Path)
    p.add_argument('--image',required=True,help='immutable sha256:<64 hex> image ID or name@sha256:<64 hex>')
    p.add_argument('--platform',choices=['linux/arm64','linux/amd64'],default='linux/arm64')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--timeout',type=float,default=20.)
    a=p.parse_args()
    if a.output.exists():p.error('refuse to overwrite scored bank')
    if a.timeout<=0:p.error('timeout must be positive')
    worker=Path(__file__).with_name('mbppplus_docker_worker.py').read_text()
    manifest=json.loads(a.bank.with_suffix('.manifest.json').read_text())
    if not manifest['complete'] or manifest['dataset_revision']!=REVISION or manifest['dataset_file_sha256']!=FILE_SHA256:
        raise ValueError('incomplete bank or dataset mismatch')
    if hashlib.sha256(a.bank.read_bytes()).hexdigest()!=manifest['bank_sha256']:
        raise ValueError('candidate bank integrity mismatch')
    tasks={tid:(entry,row) for tid,entry,row in load_selected()}
    rows=[json.loads(s) for s in a.bank.read_text().splitlines()]
    # Reference programs must pass every test before a model score is emitted.
    # The reference is never placed in generation prompts or in the optimizer.
    for tid,(_,task) in tasks.items():
        name='rvl_ref_'+uuid.uuid4().hex
        cmd=docker_command(a.image,name,worker,a.platform)
        result=run_capped(cmd,{'source':task['code'],'public_tests':task['test_list'],
            'plus_test':task['test']},name,a.timeout)
        if result.get('public_passes')!=[1]*len(task['test_list']) or result.get('trusted_pass')!=1:
            raise ValueError(f'benchmark reference failed isolated tests: task {tid}')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=a.output.with_name(a.output.name+'.incomplete')
    if temporary.exists():raise ValueError('incomplete previous score file exists')
    with temporary.open('x') as out:
        for i,record in enumerate(rows):
            tid=record['task_id'];_,task=tasks[tid]
            source=sanitize(record['source'])
            name='rvl_candidate_'+uuid.uuid4().hex
            command=docker_command(a.image,name,worker,a.platform)
            try:
                result=run_capped(command,{'source':source,'public_tests':task['test_list'],
                    'plus_test':task['test']},name,a.timeout)
                passes=result['public_passes'];trusted=result['trusted_pass']
                if len(passes)!=len(task['test_list']) or any(v not in (0,1) for v in passes) or trusted not in (0,1):
                    raise ValueError('malformed test result')
                status='scored'
            except (RuntimeError,TimeoutError,ValueError,KeyError) as exc:
                passes=[0]*len(task['test_list']);trusted=0;status=type(exc).__name__
            record.update({'public_score':sum(passes)/len(passes),
                'trusted_score':float(trusted),
                'features':{'public_score':sum(passes)/len(passes),
                    'valid':int(syntax_valid(source)),'length_scaled':min(len(source),4096)/4096},
                'scoring_status':status,'scored_source_sha256':hashlib.sha256(source.encode()).hexdigest()})
            out.write(json.dumps(record)+'\n');out.flush()
            print(f'scored {i+1}/{len(rows)} {tid} {status}',flush=True)
    temporary.rename(a.output)
    a.output.with_suffix('.score_manifest.json').write_text(json.dumps({
        'input_bank_sha256':manifest['bank_sha256'],
        'scored_bank_sha256':hashlib.sha256(a.output.read_bytes()).hexdigest(),
        'docker_image_digest':a.image,'docker_platform':a.platform,
        'per_candidate_wall_seconds':a.timeout,
        'reference_validation':'all 8 references passed original assertions and released plus tests',
        'scoring_worker_sha256':hashlib.sha256(worker.encode()).hexdigest(),
        'scorer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')


if __name__=='__main__':main()
