"""Task-clustered source-timing replication, keeping acquisition arms separate."""
import argparse,json
from pathlib import Path
import pandas as pd
from src.analyze_transfer_replication import contrasts,summarize


def main():
    p=argparse.ArgumentParser();p.add_argument('results',type=Path,nargs='+');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    frames=[]
    for root in a.results:
        df=pd.read_json(root/'runs.jsonl.gz',lines=True);df['bank']=root.name;frames.append(df)
    rows=pd.concat(frames,ignore_index=True);outputs=[]
    for mode,g in rows.groupby('acquisition'):
        out=summarize(contrasts(g));out['acquisition']=mode;outputs.append(out)
    pd.concat(outputs).to_csv(a.output,index=False)
    print(pd.concat(outputs).to_string(index=False))


if __name__=='__main__':main()
