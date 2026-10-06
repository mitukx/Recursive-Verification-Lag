"""Scientific descriptive figure; no population or matched-cost claim."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def plot(summary, output):
    data = pd.read_csv(summary)
    data = data[data.planning_prior == .5].set_index('strategy')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'svg.hashsalt': 'rvl-endogenous-audit-v1'})
    fig, ax = plt.subplots(figsize=(8.4, 4.8), layout='constrained')
    labels = {'myopic': ('Current comparison only', (-4, -24)),
        'escape': ('One exploratory query', (-135, 10)),
        'lookahead': ('Two-update lookahead', (12, -21)),
        'one_update': ('One-update horizon', (-105, 25)),
        'stream': ('All stream queries first', (-100, 12)),
        'direct': ('Select paid known positive', (12, 2))}
    for strategy, (label, offset) in labels.items():
        row = data.loc[strategy]
        color = '#D97706' if strategy == 'direct' else '#0F766E' if strategy == 'lookahead' else '#475569'
        ax.scatter(row.paid_total_labels, row.gain_evaluation_only, s=65, c=color, zorder=3)
        ax.annotate(label, (row.paid_total_labels, row.gain_evaluation_only),
                    xytext=offset, textcoords='offset points', fontsize=9, color=color,
                    arrowprops={'arrowstyle': '-', 'color': color, 'lw': .6})
    ax.set(xlim=(2.65, 4.12), ylim=(.025, .19),
           xlabel='Mean paid source labels, including two initial labels',
           ylabel='Mean trusted-score gain against initial policy')
    ax.grid(alpha=.16)
    ax.set_title('Finite-bank progress: strong baselines challenge lookahead', loc='left', pad=30, weight='bold')
    ax.text(0, 1.035, '8 retrospective tasks | 160 settings per controller | same four-label cap',
            transform=ax.transAxes, fontsize=9, color='#475569')
    output.mkdir(parents=True, exist_ok=True)
    files = []
    for suffix in ('png', 'svg'):
        path = output/f'endogenous_audit_cost_gain.{suffix}'
        fig.savefig(path, dpi=180, metadata={'Date': None} if suffix == 'svg' else None)
        if suffix == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
        files.append(path)
    plt.close(fig)
    manifest = {'scope': 'descriptive correlated-setting means; see paired task-bootstrap tables',
        'summary_sha256': hashlib.sha256(summary.read_bytes()).hexdigest(),
        'plot_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'figures_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (output/'endogenous_audit_cost_gain.manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary', type=Path)
    parser.add_argument('--output', type=Path, default=Path('figures'))
    args = parser.parse_args()
    plot(args.summary, args.output)
