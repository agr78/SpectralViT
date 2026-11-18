"""Data-efficiency curves for every subsampled experiment, matching the simulated figure.

usage: python figures/make_fig_data_efficiency.py [out.png] [--row]"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repository root: shared modules and paths.py
from paths import DATA
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

ROW = '--row' in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith('--')]
OUT = args[0] if args else ('paper/cvpr/fig/data_efficiency_row.png' if ROW
                            else 'paper/cvpr/fig/data_efficiency.png')
def _sweep_dir():
    env = os.environ.get('IXI_SWEEP_DIR')
    if env:
        return env
    mirror = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'ixi_curve')
    if os.path.isdir(mirror) and any(f.startswith('oof_') for f in os.listdir(mirror)):
        return mirror
    return f'{DATA}/ixi_nested'


IXI_DIR = _sweep_dir()

ARMS = [('Spectral ViT (proposed)',  'SpectralViT rank (Block-DCT)',   'tab:blue',   '-',  2.6, 5),
        ('Spatial ViT',              'SpatialViT (Micro)',             'tab:purple', '-',  1.3, 2),
        ('Swin ViT',                 'Swin (Micro)',                   'tab:pink',   '-',  1.3, 2),
        ('U-net with attention',     'AttnUNet (Micro)',               'tab:orange', '-',  1.3, 2),
        ('Logistic regression',      'LogisticRegression (Block-DCT)', 'tab:green',  '-',  1.3, 2),
        ('Multilayer perceptron',    'MLP (Block-DCT)',                'tab:red',    '-',  1.3, 2)]
SIZES = [25, 50, 100, 200, 1000]
IXI_SIZES = [25, 50, 100, 200, 566]
IXI_ARMS = {'spec': 'Spectral ViT (proposed)', 'unet': 'U-net with attention',
            'spat': 'Spatial ViT', 'swin': 'Swin ViT',
            'logreg': 'Logistic regression', 'mlp': 'Multilayer perceptron'}

STYLE = {d: (c, ls, lw, z) for d, _, c, ls, lw, z in ARMS}
ORDER = [d for d, *_ in ARMS]


def hca(patt, col):
    out = {}
    for disp, key, *_ in ARMS:
        xs, mu, sd = [], [], []
        for n in SIZES:
            f = patt % n
            if not os.path.exists(f):
                continue
            d = pd.read_csv(f)
            v = d[d.Model == key][col].values
            if not len(v):
                continue
            xs.append(n); mu.append(v.mean()); sd.append(v.std(ddof=1))
        if xs:
            out[disp] = (np.array(xs), np.array(mu), np.array(sd))
    return out


def ixi():
    from sklearn.metrics import roc_auc_score
    out = {}
    for tag, disp in IXI_ARMS.items():
        xs, mu, sd = [], [], []
        for n in IXI_SIZES:
            per_seed = []
            for s in range(5):
                f = os.path.join(IXI_DIR, f'oof_{tag}_n{n}_s{s}.npz')
                if not os.path.exists(f):
                    continue
                z = np.load(f)
                cuts = np.cumsum(z['fold_sizes'])[:-1]
                aucs = [roc_auc_score(a, b) for a, b in
                        zip(np.split(z['y'], cuts), np.split(z['probs'], cuts))
                        if len(np.unique(a)) > 1]
                if aucs:
                    per_seed.append(np.mean(aucs))
            if len(per_seed) >= 2:
                xs.append(n); mu.append(np.mean(per_seed))
                sd.append(np.std(per_seed, ddof=1))
        if xs:
            out[disp] = (np.array(xs), np.array(mu), np.array(sd))
    return out


PANELS = [('Brain age, HCP-Aging', 'Mean absolute error (years)',
           hca('results/regress/hca_regperseed_n%d.csv', 'MAE')),
          ('Sex classification, IXI', 'AUC', ixi())]
PANELS = [p for p in PANELS if p[2]]

if ROW:
    fig, axes = plt.subplots(1, len(PANELS), figsize=(6.9, 2.0))
else:
    fig, axes = plt.subplots(len(PANELS), 1, figsize=(3.3, 2.35 * len(PANELS)))
axes = np.atleast_1d(axes)

for ax, (title, ylab, data) in zip(axes, PANELS):
    for disp in ORDER:
        if disp not in data:
            continue
        xs, mu, sd = data[disp]
        c, ls, lw, z = STYLE[disp]
        hero = z >= 5
        ax.plot(xs, mu, color=c, ls=ls, lw=lw, zorder=z)
        ax.fill_between(xs, mu - sd, mu + sd, color=c,
                        alpha=0.22 if hero else 0.10, lw=0, zorder=z - 1)
    ax.set_xscale('log')
    ax.set_xticks(sorted({x for _, (xs, _, _) in data.items() for x in xs}))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_title(title, fontsize=9, pad=4)
    ax.set_ylabel(ylab, fontsize=8)
    ax.tick_params(labelsize=7.5)
    ax.grid(True, alpha=0.2)

for ax in (axes if ROW else axes[-1:]):
    ax.set_xlabel('Training subjects $n$', fontsize=8)

present = [d for d in ORDER if any(d in data for _, _, data in PANELS)]
handles = [mlines.Line2D([], [], color=STYLE[d][0], ls=STYLE[d][1], lw=2, label=d)
           for d in present]
leg_ax = axes[-1] if len(axes) > 1 else axes[0]
solo = len(axes) == 1
if solo and ROW:
    leg_ax.legend(handles=handles, fontsize=6.6, ncol=1, loc='center left',
                  bbox_to_anchor=(1.01, 0.5), frameon=False, handlelength=1.9)
else:
    leg_ax.legend(handles=handles, fontsize=6.2, ncol=2 if solo else 1,
                  loc='upper right' if solo else 'lower right',
                  frameon=True, framealpha=0.92, handlelength=1.9,
                  columnspacing=1.0)

plt.tight_layout(pad=0.4)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
tmp = OUT + '.tmp.png'
plt.savefig(tmp, dpi=300)
os.replace(tmp, OUT)
print(f'wrote {OUT} with {len(PANELS)} panel(s): ' + ', '.join(t for t, _, _ in PANELS))
