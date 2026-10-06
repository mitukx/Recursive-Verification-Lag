"""Static MBPP+ generation diagnostics; never execute model completions."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import pandas as pd
from src.score_mbppplus_docker import sanitize


def diagnostics(bank,output,eos_id):
    meta=json.loads(bank.with_suffix('.manifest.json').read_text())
    if not meta['complete'] or hashlib.sha256(bank.read_bytes()).hexdigest()!=meta['bank_sha256']:
        raise ValueError('bank incomplete or modified')
    rows=[json.loads(line) for line in bank.read_text().splitlines()]
    expected=len(meta['selected_tasks'])*meta['generation']['samples']
    if len(rows)!=expected or len({r['candidate_id'] for r in rows})!=expected:
        raise ValueError('missing or duplicate candidate occurrences')
    records=[]
    for r in rows:
        source=sanitize(r['source']);functions=[];valid=0
        try:
            tree=ast.parse(source);valid=1
            functions=[n.name for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
        except (SyntaxError,ValueError,RecursionError):pass
        tokens=r['generated_token_ids']
        records.append({'task_id':r['task_id'],'candidate_id':r['candidate_id'],
            'raw_hash':r['source_sha256'],'sanitized_hash':hashlib.sha256(source.encode()).hexdigest(),
            'syntax_valid':valid,'has_entry_function':int(r['entry_point'] in functions),
            'terminated_by_eos':int(eos_id in tokens),
            'hit_token_cap':int(eos_id not in tokens and len(tokens)>=meta['generation']['max_new_tokens']),
            'fenced':int(r['source'].strip().startswith('```'))})
    detail=pd.DataFrame(records)
    out=detail.groupby('task_id').agg(candidates=('candidate_id','size'),
        distinct_sources=('raw_hash','nunique'),distinct_sanitized=('sanitized_hash','nunique'),
        syntax_valid=('syntax_valid','sum'),has_entry_function=('has_entry_function','sum'),
        hit_token_cap=('hit_token_cap','sum'),fenced=('fenced','sum')).reset_index()
    output.parent.mkdir(parents=True,exist_ok=True);out.to_csv(output,index=False)
    print(out.to_string(index=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('bank',type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--eos-id',type=int,required=True,
        help='EOS token id of the immutable model; never infer truncation from padded sequence length')
    a=p.parse_args();diagnostics(a.bank,a.output,a.eos_id)


if __name__=='__main__':main()
