from __future__ import annotations
import csv, json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable
from .models import GenerationRecord

FIELDS=["generation","champion_id","candidate_id","development_score","promotion_score","reward","trusted_score","verification_gap","verifier_version","policy_version","policy_verifier_age","latency_p50","latency_p95","throughput","failure_rate","compute_cost","promotion_decision"]

class MetricsWriter:
    def __init__(self,root):
        self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True); self.jsonl=self.root/"generations.jsonl"; self.csv_path=self.root/"generations.csv"
    def append(self,record):
        raw=asdict(record)
        with self.jsonl.open("a",encoding="utf-8") as f: f.write(json.dumps(raw,sort_keys=True,allow_nan=False)+"\n")
        exists=self.csv_path.exists()
        with self.csv_path.open("a",encoding="utf-8",newline="") as f:
            w=csv.DictWriter(f,fieldnames=FIELDS)
            if not exists: w.writeheader()
            w.writerow(raw)
    def read(self):
        if not self.jsonl.exists(): return []
        return [GenerationRecord(**json.loads(line)) for line in self.jsonl.read_text().splitlines() if line.strip()]
    def write_summary(self,payload): (self.root/"summary.json").write_text(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
    def render_plots(self,records):
        rows=list(records)
        specs=[
          ("capability_vs_generation.svg",[("development_score","development"),("promotion_score","promotion")],"Capability vs generation"),
          ("reward_vs_trusted.svg",[("reward","reward"),("trusted_score","trusted")],"Reward vs trusted score"),
          ("verification_gap_vs_generation.svg",[("verification_gap","verification gap")],"Verification gap vs generation"),
          ("verifier_age_vs_failure_rate.svg",[("policy_verifier_age","verifier age"),("failure_rate","failure rate")],"Verifier age and failure rate"),
          ("champion_progression.svg",[("promotion_score","promotion score")],"Champion/candidate progression"),
          ("development_vs_promotion.svg",[("development_score","development"),("promotion_score","promotion")],"Development vs promotion performance")]
        out=[]
        for filename,series,title in specs:
            path=self.root/filename; _svg_plot(path,rows,series,title); out.append(path)
        return out

def _svg_plot(path,rows,series,title):
    width,height=760,360; left,right,top,bottom=58,20,42,52
    xs=[r.generation for r in rows] or [0]; vals=[float(getattr(r,f)) for r in rows for f,_ in series] or [0.0]
    ymin,ymax=min(vals),max(vals)
    if ymin==ymax: ymin-=.5; ymax+=.5
    pad=(ymax-ymin)*.08; ymin-=pad; ymax+=pad; xmin,xmax=min(xs),max(xs)
    if xmin==xmax: xmax=xmin+1
    sx=lambda x:left+(x-xmin)/(xmax-xmin)*(width-left-right)
    sy=lambda y:top+(ymax-y)/(ymax-ymin)*(height-top-bottom)
    palette=["#2563eb","#dc2626","#059669","#7c3aed"]
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">','<rect width="100%" height="100%" fill="white"/>',f'<text x="{width/2}" y="24" text-anchor="middle">{title}</text>',f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#555"/>',f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#555"/>']
    for idx,(field,label) in enumerate(series):
        color=palette[idx%len(palette)]; points=" ".join(f"{sx(r.generation):.1f},{sy(float(getattr(r,field))):.1f}" for r in rows)
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
        for row in rows:
            x,y=sx(row.generation),sy(float(getattr(row,field)))
            if row.promotion_decision=="REJECTED": parts.append(f'<rect x="{x-3:.1f}" y="{y-3:.1f}" width="6" height="6" fill="{color}"/>')
            else: parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}"/>')
        parts.append(f'<text x="{left+130*idx}" y="{height-16}" fill="{color}">{label}</text>')
    parts.append("</svg>"); path.write_text("\n".join(parts)+"\n",encoding="utf-8")
