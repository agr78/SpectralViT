"""Cross-site consistent spectral modes, as a responder minus non-responder difference map.

usage: python make_fig_dbs.py [out.png]"""
import os, sys
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

OUT = (sys.argv[1] if len(sys.argv) > 1 else
       os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paper', 'cvpr', 'fig',
                    'dbs_modes.png'))
_FD = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paper', 'cvpr', 'fig', 'data')
MAPS = os.environ.get('DBS_MAPS', os.path.join(_FD, 'padded_maps.npz'))
ATLAS = os.environ.get('DBS_ATLAS', os.path.join(_FD, 'atlas.npz'))
CENTROIDS = os.environ.get('DBS_CENTROIDS', os.path.join(_FD, 'atlas_centroids.npy'))
H, DZ = 10, 8

D = np.load(MAPS)
_R = [float(stats.pearsonr(D[f'int_{k}'].ravel(), D[f'ext_{k}'].ravel())[0]) for k in ('hi', 'lo')]
assert all(abs(a - b) < 0.005 for a, b in zip(_R, (0.67, 0.66))), \
    f'cross-site r is {_R[0]:.3f}, {_R[1]:.3f}, not the published 0.67, 0.66'
print(f'cross-site r: {_R[0]:+.3f} (higher-susceptibility), {_R[1]:+.3f} (lower) -- matches')
VOL = np.load(ATLAS)['mean']
LOC, HIC = np.load(CENTROIDS)


def window(center):
    return tuple(int(min(max(0, round(center[d]) - (H if d < 2 else DZ)),
                         VOL.shape[d] - (2 * H if d < 2 else 2 * DZ))) for d in range(3))


BH, BL = window(HIC), window(LOC)
YC = BH[1] + H
cmap = plt.get_cmap('RdBu_r')
fig, axes = plt.subplots(1, 2, figsize=(11, 2.9))
for i, (site, label) in enumerate((('int', 'Internal ($n$=66)'),
                                   ('ext', 'External ($n$=37)'))):
    field = np.zeros(VOL.shape)
    cover = np.zeros(VOL.shape)
    for box, nucleus in ((BH, 'hi'), (BL, 'lo')):
        field[box[0]:box[0] + 20, box[1]:box[1] + 20, box[2]:box[2] + 16] = D[f'{site}_{nucleus}']
        cover[box[0]:box[0] + 20, box[1]:box[1] + 20, box[2]:box[2] + 16] = 1.0
    field = gaussian_filter(field, 1.6)
    field = field / (np.abs(field).max() + 1e-12)
    mask = np.clip(gaussian_filter(cover, 2.2), 0, 1)
    plane, alpha = field[:, YC, :].T, mask[:, YC, :].T
    rgba = cmap((plane + 1) / 2)
    rgba[..., 3] = np.clip(np.abs(plane) / 0.55, 0, 1) ** 0.8 * alpha
    ax = axes[i]
    ax.imshow(VOL[:, YC, :].T, cmap='gray', origin='lower')
    ax.imshow(rgba, origin='lower')
    PAD = 10
    z0 = min(BH[2], BL[2]) - PAD
    z1 = max(BH[2], BL[2]) + 2 * DZ + PAD
    ax.set_ylim(max(0, z0), min(VOL.shape[2], z1))
    ax.set_title(label, fontsize=10)
    ax.text(0.02, 0.97, 'LOWER\nsusceptibility', transform=ax.transAxes, color='w',
            fontsize=8, va='top')
    ax.text(0.98, 0.97, 'HIGHER\nsusceptibility', transform=ax.transAxes, color='w',
            fontsize=8, va='top', ha='right')
    ax.set_xticks([]); ax.set_yticks([])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-1, 1))
fig.colorbar(sm, ax=axes, fraction=0.035,
             label='Responder $-$ non-responder (normalized)')
os.makedirs(os.path.dirname(OUT), exist_ok=True)
tmp = OUT + '.tmp.png'
plt.savefig(tmp, dpi=145, bbox_inches='tight')
os.replace(tmp, OUT)
print(f'wrote {OUT}')
