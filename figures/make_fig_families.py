"""The three basis families, each shown with its own basis vectors and what they rebuild.

usage: python figures/make_fig_families.py [out.png]"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repository root: shared modules and paths.py
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'paper', 'cvpr', 'fig',
                                                         'families.png')
_src = open(os.path.join(HERE, 'figures', 'make_fig_bases.py')).read()
_g = {'__name__': 'bases', '__file__': os.path.join(HERE, 'figures', 'make_fig_bases.py')}
_argv, sys.argv = sys.argv, [sys.argv[0]]
exec(compile(_src[:_src.index('fig, axes = plt.subplots')], 'bases', 'exec'), _g)
sys.argv = _argv
S, K, ROWS, BASES = _g['S'], _g['K'], _g['ROWS'], _g['BASES']
psnr, pca, EVEC, VL = _g['psnr'], _g['pca'], _g['EVEC'], _g['VL']
ORD_BLK, _keep = _g['ORD_BLK'], _g['_keep']
from scipy.fft import idctn

DARK = os.environ.get('FIG_DARK') == '1'
INK, FRAME = ('#E6EBF0', '0.55') if DARK else ('0.15', '0.35')
BGC = '#171C22' if DARK else 'white'
MUTE = '0.6' if DARK else '0.35'
FS = 7.0
W, H = 6.9, 2.52
SQ = 0.50
N_W = 6

fig = plt.figure(figsize=(W, H), facecolor=BGC)
bg = fig.add_axes([0, 0, 1, 1], facecolor=BGC)
bg.set_xlim(0, W); bg.set_ylim(0, H); bg.axis('off')


def image_axes(x, y, w, h):
    ax = fig.add_axes([x / W, y / H, w / W, h / H]); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(FRAME); s.set_linewidth(0.5)
    return ax


def arrow(x1, y1, x2, y2, lw=0.7, ls='-'):
    bg.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=6,
                                 lw=lw, ls=ls, color=INK, zorder=5, shrinkA=0, shrinkB=1.5))


def text(x, y, s, **kw):
    kw.setdefault('fontsize', FS); kw.setdefault('color', INK)
    kw.setdefault('ha', 'center'); kw.setdefault('va', 'center')
    return bg.text(x, y, s, zorder=6, **kw)


def signed(ax, v):
    m = np.abs(v).max() + 1e-12
    ax.imshow(v, cmap='RdBu_r', vmin=-m, vmax=m)


def harmonic_vectors(k):
    cand = sorted([(a, b) for a in range(6) for b in range(6)],
                  key=lambda t: (t[0] ** 2 + t[1] ** 2, max(t)))
    out = []
    for a, b in cand[1:k + 1]:
        c = np.zeros((S, S)); c[a, b] = 1.0
        out.append(idctn(c, norm='ortho'))
    return out


VECS = {'Empirical (PCA)':   [pca.components_[i].reshape(S, S) for i in range(N_W)],
        'Harmonic (Fourier)': harmonic_vectors(N_W),
        'Graphical (Laplacian)': [EVEC[:, i + 1].cpu().numpy().reshape(S, S)
                                  for i in range(N_W)]}

YS = [H - 0.50, H - 1.30, H - 2.10]
x_data, x_b0 = 0.26, 1.98
x_b1 = x_b0 + N_W * SQ + (N_W - 1) * 0.035
x_in = W - 0.06 - 2 * SQ - 0.05
x_rc = x_in + SQ + 0.05

ymid = sum(YS) / len(YS)
DX = 0.035
for d in (2, 1):
    bg.add_patch(Rectangle((x_data + d * DX, ymid - SQ / 2 + d * DX), SQ, SQ,
                           fc=BGC, ec=FRAME, lw=0.5, zorder=3 - d))
ax = image_axes(x_data, ymid - SQ / 2, SQ, SQ)
ax.imshow(_g['base'], cmap='bone')
text(x_data + SQ / 2, ymid - SQ / 2 - 0.11, 'Input', fontsize=FS - 1.0, color=MUTE)
_d = 0.085
text(x_data + SQ + 2 * DX + _d, ymid + SQ / 2 + 2 * DX + _d, r'$\cdots$',
     fontsize=FS + 2, color=MUTE, rotation=45)
bus = x_data + SQ + 0.34
bg.add_line(Line2D([x_data + SQ + 2 * DX, bus], [ymid, ymid], lw=0.7, color=INK, zorder=5))
bg.add_line(Line2D([bus, bus], [min(YS), max(YS)], lw=0.7, color=INK, zorder=5))

for (name, fn), (prop, gt, noise), yc in zip(BASES, ROWS, YS):
    head = yc + SQ / 2 + 0.115
    text(x_b0, head, name.split(' (')[0], ha='left', fontsize=FS + 0.3, weight='bold')
    text(x_b0 + 0.78, head, '(' + name.split('(')[1], ha='left', fontsize=FS - 1.0, color=MUTE)
    arrow(bus, yc, x_b0, yc)
    for i, v in enumerate(VECS[name]):
        ax = image_axes(x_b0 + i * (SQ + 0.035), yc - SQ / 2, SQ, SQ)
        signed(ax, v)
        ax.text(0.05, 0.95, r'$\mathbf{w}_{%d}$' % (i + 1), transform=ax.transAxes,
                fontsize=FS - 2, ha='left', va='top', color=INK,
                bbox=dict(boxstyle='square,pad=0.08', fc=BGC, ec='none', alpha=0.8))
    np.random.seed(1)
    noisy = gt + np.random.normal(0, noise, gt.shape)
    rec = [f(noisy) for _, f in BASES]
    val = [psnr(gt, r) for r in rec]
    j = [n for n, _ in BASES].index(name)
    text(W - 0.06, head, prop, ha='right', fontsize=FS - 1.0, color=MUTE)
    ax = image_axes(x_in, yc - SQ / 2, SQ, SQ); ax.imshow(gt, cmap='bone')
    ax = image_axes(x_rc, yc - SQ / 2, SQ, SQ)
    ax.imshow(rec[j], cmap='bone', vmin=gt.min(), vmax=gt.max())
    assert val[j] == max(val), (name, prop, val)

assert x_rc + SQ + 0.05 <= W, f'layout overflows: {x_rc + SQ:.2f}in of {W}'

os.makedirs(os.path.dirname(OUT) or '.', exist_ok=True)
tmp = OUT + '.tmp.png'
plt.savefig(tmp, dpi=400, facecolor=BGC)
os.replace(tmp, OUT)
print(f'wrote {OUT}')
