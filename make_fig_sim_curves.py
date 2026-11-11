"""Simulated pattern-classification curves, redrawn from the persisted sweep.

The original figure encoded SNR in color and the model in line style. Here color carries
the model, matching the data-efficiency figure, and line style carries SNR -- so one
convention holds across the paper: a color is a model wherever it appears.

Means over 100 seeds with a +/-1 sd band. Sized for a single supplement column rather
than the full text width, since the figure now lives in the appendix.

usage: python make_fig_sim_curves.py [out.png]
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from util import smooth_curve

OUT = sys.argv[1] if len(sys.argv) > 1 else 'paper/cvpr/fig/sim_snr_all.png'
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'sim_sweep')
SIZES = np.array([10, 100, 1000, 10000, 100000])
# noise sigma -> (SNR label, line style); SNR = 1/sigma^2
SNRS = [(1.0, 'SNR $=1$', '-'), (2.0, r'SNR $=\frac{1}{4}$', '--'),
        (3.0, r'SNR $=\frac{1}{9}$', ':')]
COLOR = {'spec': 'tab:blue', 'spat': 'tab:purple'}

fig, ax = plt.subplots(figsize=(3.3, 2.45))
for nl, lab, ls in SNRS:
    fs = sorted(glob.glob(f'{SRC}/sim_nl{nl:g}_b*.npz'))
    if not fs:
        continue
    for key in ('spec', 'spat'):
        a = np.concatenate([np.load(f)[key] for f in fs])
        mu, sd = a.mean(0), a.std(0, ddof=1)
        # Smooth the mean as the original did; the band stays at measured sizes, since
        # interpolating a spread would imply resolution in N that the grid does not have.
        xs, ys = smooth_curve(SIZES, mu, 1000)
        ax.plot(xs, ys, color=COLOR[key], ls=ls, lw=1.6 if key == 'spec' else 1.2,
                zorder=3 if key == 'spec' else 2)
        ax.fill_between(SIZES, mu - sd, mu + sd, color=COLOR[key], alpha=0.10, lw=0,
                        zorder=1)

ax.set_xscale('log')
ax.set_ylim(0.45, 1.03)
ax.set_xlabel('Training samples $N$', fontsize=8)
ax.set_ylabel('AUC', fontsize=8)
ax.tick_params(labelsize=7.5)
ax.grid(True, alpha=0.2)

handles = [mlines.Line2D([], [], color=COLOR['spec'], lw=2, label='Spectral ViT'),
           mlines.Line2D([], [], color=COLOR['spat'], lw=2, label='Spatial ViT')]
handles += [mlines.Line2D([], [], color='0.35', ls=ls, lw=1.4, label=lab)
            for _, lab, ls in SNRS]
ax.legend(handles=handles, fontsize=6.2, loc='lower right', frameon=True,
          framealpha=0.92, handlelength=2.0, ncol=2, columnspacing=1.0)

plt.tight_layout(pad=0.3)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
tmp = OUT + '.tmp.png'
plt.savefig(tmp, dpi=300)
os.replace(tmp, OUT)
print(f'wrote {OUT}')
