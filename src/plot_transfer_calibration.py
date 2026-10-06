"""Generate a descriptive research figure from archived transfer contrasts."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def plot(original, diagnostic, output):
    raw = pd.read_csv(original/'contrasts.csv')
    matched = pd.read_csv(diagnostic/'contrasts.csv')
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    colors = {'public': '#007f80', 'all': '#b35428'}
    for rep in ('public', 'all'):
        row = raw[(raw.representation == rep) & (raw.optimizer == 'soft') & (raw['rounds'] == 4)].iloc[0]
        mean, lo, hi = [100*row[k] for k in ['refreshed_minus_frozen',
            'descriptive_task_bootstrap_lo', 'descriptive_task_bootstrap_hi']]
        y = 0 if rep == 'public' else 1
        axes[0].errorbar(mean, y, xerr=[[mean-lo], [hi-mean]], fmt='o',
                         color=colors[rep], capsize=4, linewidth=2)
        frame = matched[(matched.arm == rep+'_refreshed') & (matched.control == rep+'_frozen')]
        x = frame.requested_kl.to_numpy(); m = 100*frame.matched_kl_gain_difference.to_numpy()
        l = 100*frame.descriptive_task_bootstrap_lo.to_numpy()
        h = 100*frame.descriptive_task_bootstrap_hi.to_numpy()
        axes[1].plot(x, m, 'o-', color=colors[rep], label=rep+' features')
        axes[1].fill_between(x, l, h, color=colors[rep], alpha=.12)
    axes[0].set_yticks([0, 1], ['Public feature', 'All features'])
    axes[0].axvline(0, color='#707070', linewidth=.8)
    axes[0].set_xlabel('Refresh effect (percentage points)')
    axes[0].set_title('Original fixed-strength study\n4 soft updates, eta = 1')
    axes[1].axhline(0, color='#707070', linewidth=.8)
    axes[1].set_xlabel('Requested KL from common baseline')
    axes[1].set_ylabel('Refresh effect (percentage points)')
    axes[1].set_title('Exploratory equal-KL diagnostic\nPublic-feature effect vanishes')
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(alpha=.15)
    fig.text(.5, .035, '16 development evaluation tasks; descriptive task-bootstrap intervals; no heldout confirmation.\n'
             'KL is capped per task; 2 constant-score tasks retain zero radius and remain included.',
             ha='center', fontsize=9, color='#555555')
    fig.tight_layout(rect=[0, .12, 1, 1])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original', type=Path, default=Path('results/fresh_task_verifier_transfer_v1'))
    p.add_argument('--diagnostic', type=Path, default=Path('results/transfer_calibration_v1'))
    p.add_argument('--output', type=Path, default=Path('figures/transfer_calibration_v1.png'))
    a = p.parse_args(); plot(a.original, a.diagnostic, a.output)
