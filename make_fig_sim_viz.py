"""Regenerate the pattern-classification sample figure as a vertical stack.

The committed sim_viz.png is a single row of four panels (aspect 3.5), which has to span
both CVPR columns. A 2x2 block is square, fits a single column and frees the full-width
slot. Same generator and preprocessing as the notebook version.

usage: python make_fig_sim_viz.py [out.png]
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

sys.path.insert(0, '/workspace')
from train import generate_checkerboard_data

OUT = sys.argv[1] if len(sys.argv) > 1 else 'paper/cvpr/fig/sim_viz_vertical.png'
np.random.seed(0)
IMG = 28
# noise_level is the simulated sigma; SNR = 1/sigma^2 as used in the sweep
CONFIGS = [('Class 0: noise', 1.0, True),
           ('Class 1: SNR $=1$', 1.0, False),
           (r'Class 1: SNR $=\frac{1}{4}$', 2.0, False),
           (r'Class 1: SNR $=\frac{1}{9}$', 3.0, False)]

# 2x2 rather than 4x1: a single column of four squares would be ~15 in tall at CVPR's
# 3.25 in column width. A square block fits one column, which is the point.
fig, axes = plt.subplots(2, 2, figsize=(3.2, 3.45),
                         gridspec_kw={'hspace': 0.22, 'wspace': 0.04})
for ax, (label, sigma, pure) in zip(axes.ravel(), CONFIGS):
    X, y = generate_checkerboard_data(n_samples=200, noise_level=sigma)
    img = X[y == (0 if pure else 1)][0].reshape(IMG, IMG)
    sm = gaussian_filter(img, sigma=0.8)
    sm = (sm - sm.min()) / (sm.max() - sm.min())
    ax.imshow(sm, cmap='magma', aspect='equal')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(label, fontsize=9, pad=4)
plt.subplots_adjust(left=0.03, right=0.97, top=0.955, bottom=0.015)
fig.savefig(OUT, dpi=300, bbox_inches='tight')
print(f'wrote {OUT}')
