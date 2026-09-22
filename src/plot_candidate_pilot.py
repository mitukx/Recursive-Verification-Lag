"""Static research figure; no universal-law claim is inferred from this plot."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('results',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    runs=pd.read_csv(a.results/'runs.csv');transfer=pd.read_csv(a.results/'transfer.csv')
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
    for ax,op,strength in zip(axes[:2],['soft','bon'],['eta','best_of_n']):
        g=runs[(runs.controller=='fixed')&(runs.optimizer==op)&(runs.representation=='public')&(runs.total_audit_budget==32)]
        grid=g.pivot_table(index=strength,columns='refresh_interval',values='ever_below_initial',aggfunc='mean')
        im=ax.imshow(grid.to_numpy(),vmin=0,vmax=1,cmap='magma',aspect='auto')
        ax.set_xticks(range(len(grid.columns)),grid.columns);ax.set_yticks(range(len(grid.index)),grid.index)
        ax.set_xlabel('Fixed refresh interval');ax.set_ylabel('Soft eta' if op=='soft' else 'Best-of-N')
        ax.set_title(f'{op}: baseline-crossing fraction')
        for i in range(len(grid.index)):
            for j in range(len(grid.columns)):
                value=grid.iloc[i,j];ax.text(j,i,f'{value:.2f}',ha='center',va='center',color='white' if value<.6 else 'black',fontsize=9)
    fig.colorbar(im,ax=axes[:2],shrink=.8,label='Fraction of task/audit-seed runs')
    g=transfer[(transfer.axis=='task_id')&(transfer.status=='ok')].groupby('metric').balanced_accuracy.agg(['mean','count']).sort_values('mean')
    labels=[f'{m}\n({int(n)} identifiable folds)' for m,n in zip(g.index,g['count'])]
    if len(g):
        axes[2].barh(range(len(g)),g['mean'],color='#397B8E');axes[2].set_yticks(range(len(g)),labels,fontsize=8)
    else: axes[2].text(.5,.5,'No identifiable folds',ha='center',va='center',transform=axes[2].transAxes)
    axes[2].set_xlim(0,1);axes[2].axvline(.5,color='gray',ls='--',lw=1)
    axes[2].set_xlabel('Held-task balanced accuracy');axes[2].set_title('Exploratory first-crossing prediction')
    fig.suptitle('Pretrained expression pilot — one frozen bank; development evidence only',fontsize=12)
    a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=170);plt.close(fig)

if __name__=='__main__': main()
