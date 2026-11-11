"""The three basis families, each shown with its own basis vectors and what they rebuild.

One row per family. Reading left to right: the cohort it is handed and the operator it
diagonalizes, its leading basis vectors as images, then the condition that family is the
right choice for and its reconstruction of that condition, scored against the other two.
That merges what used to be two figures -- a families diagram and a reconstruction grid --
into one, and makes the same point in half the page: a family is not better in general, it
is better for the structure its ordering matches.

Everything is computed on the same Shepp-Logan phantom at 64 x 64 with 128 coefficients,
so the basis images and the reconstructions are the same objects; make_fig_bases.py's
prefix is executed to reuse its bases, its PCA fit and its graph, rather than rebuilding
them here and risking a different construction.

usage: python make_fig_families.py [out.png]
env:   FIG_DARK=1 for the dark variant used in the web write-up
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'paper', 'cvpr', 'fig',
                                                         'families.png')
# Reuse make_fig_bases.py up to its figure block: same phantom, same PCA fit, same Voronoi
# graph, same K. Importing would run its plotting, so the prefix is exec'd instead.
_src = open(os.path.join(HERE, 'make_fig_bases.py')).read()
_g = {'__name__': 'bases', '__file__': os.path.join(HERE, 'make_fig_bases.py')}
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
# H was 2.78 to clear the strap line below the bottom row. With that gone the canvas
# trims to the panels plus a small margin, which is ~9% less height on the page.
W, H = 6.9, 2.52
SQ = 0.50                           # every square panel, input included, is this size
N_W = 6                             # basis vectors shown per family

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


# --- the basis vectors of each family, as images ------------------------------------------
def harmonic_vectors(k):
    """The k lowest block-cosine modes, drawn on one block so the shape is visible."""
    cand = sorted([(a, b) for a in range(6) for b in range(6)],
                  key=lambda t: (t[0] ** 2 + t[1] ** 2, max(t)))
    out = []
    for a, b in cand[1:k + 1]:                    # skip DC, which is a constant image
        c = np.zeros((S, S)); c[a, b] = 1.0
        out.append(idctn(c, norm='ortho'))
    return out


VECS = {'Empirical (PCA)':   [pca.components_[i].reshape(S, S) for i in range(N_W)],
        'Harmonic (Fourier)': harmonic_vectors(N_W),
        'Graphical (Laplacian)': [EVEC[:, i + 1].cpu().numpy().reshape(S, S)
                                  for i in range(N_W)]}
# Two of the three diagonalize a Laplacian, so naming only the operator makes the
# harmonic and graphical rows look like the same thing -- which, taken as operators, they
# nearly are: the harmonic basis is the graphical one with the sampling lattice as its
# graph. What actually separates the families is where the operator comes from, so each
# arrow carries that underneath.
# 2026-11-06: the arrow labels are commented out, not deleted. The provenance framing
# they carried is still in the caption, which is where it can be argued properly; in the
# figure it asserted a distinction the drawing cannot support on its own. Uncomment both
# this and the two text() calls below to restore them.
# OPERATOR = {'Empirical (PCA)': ('Sample covariance', 'estimated from data'),
#             'Harmonic (Fourier)': ('Lattice Laplacian', 'fixed by the lattice'),
#             'Graphical (Laplacian)': ('Graph Laplacian', 'supplied alongside')}

# --- geometry -----------------------------------------------------------------------------
YS = [H - 0.50, H - 1.30, H - 2.10]
# the operator label sits on the arrow, so the gap has to hold 'Sample covariance'
x_data, x_b0 = 0.26, 1.98
x_b1 = x_b0 + N_W * SQ + (N_W - 1) * 0.035
# the figure is set at \linewidth, so a narrower canvas is scaled UP and costs more
# page height, not less; the pair goes to the right margin rather than the width
# being trimmed to it
x_in = W - 0.06 - 2 * SQ - 0.05                                # the condition, then the reconstruction
x_rc = x_in + SQ + 0.05

# one cohort, branching to all three: every family is offered the same data and differs in
# what it does with it, which is the operator named on its arrow
# One image, at the same size as every other panel, followed by an ellipsis: the empirical
# basis is fitted to a sample (512 jittered copies of this phantom), the other two take
# nothing from it, and the ellipsis says 'more of these' without drawing a cohort of
# distinct subjects that does not exist.
ymid = sum(YS) / len(YS)
# Cards drawn on the background axes, which is created first and so renders beneath the
# panel; they recede up and to the right, and the wire leaves from the far card's edge so
# nothing crosses the stack. The ellipsis continues that recession instead of sitting on
# the wire, where it read as a further stage rather than as more of the same data.
DX = 0.035
for d in (2, 1):
    bg.add_patch(Rectangle((x_data + d * DX, ymid - SQ / 2 + d * DX), SQ, SQ,
                           fc=BGC, ec=FRAME, lw=0.5, zorder=3 - d))
ax = image_axes(x_data, ymid - SQ / 2, SQ, SQ)
ax.imshow(_g['base'], cmap='bone')
text(x_data + SQ / 2, ymid - SQ / 2 - 0.11, 'Input', fontsize=FS - 1.0, color=MUTE)
# On the diagonal through the three panel corners, which all lie on y - x = const, so
# equal offsets in both keep it on that line; rotated to run along it. Background axes
# span the figure in inches, so 45 degrees in data coordinates is 45 degrees on the page.
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
    # 2026-11-06: one line style for all three. The dashing encoded which operators are
    # estimated from the cohort, an argument the caption makes and the drawing cannot.
    arrow(bus, yc, x_b0, yc)
    # op, src = OPERATOR[name]
    # text((bus + x_b0) / 2, yc + 0.155, op, fontsize=FS - 1.2)
    # text((bus + x_b0) / 2, yc + 0.062, src, fontsize=FS - 2.2, color=MUTE)
    for i, v in enumerate(VECS[name]):
        ax = image_axes(x_b0 + i * (SQ + 0.035), yc - SQ / 2, SQ, SQ)
        signed(ax, v)
        ax.text(0.05, 0.95, r'$\mathbf{w}_{%d}$' % (i + 1), transform=ax.transAxes,
                fontsize=FS - 2, ha='left', va='top', color=INK,
                bbox=dict(boxstyle='square,pad=0.08', fc=BGC, ec='none', alpha=0.8))
    # the condition this family is the right choice for, and what it rebuilds from it
    np.random.seed(1)
    noisy = gt + np.random.normal(0, noise, gt.shape)
    rec = [f(noisy) for _, f in BASES]
    val = [psnr(gt, r) for r in rec]
    j = [n for n, _ in BASES].index(name)
    text(W - 0.06, head, prop, ha='right', fontsize=FS - 1.0, color=MUTE)
    ax = image_axes(x_in, yc - SQ / 2, SQ, SQ); ax.imshow(gt, cmap='bone')
    ax = image_axes(x_rc, yc - SQ / 2, SQ, SQ)
    ax.imshow(rec[j], cmap='bone', vmin=gt.min(), vmax=gt.max())
    # PSNR and the comparison against the other two families are the appendix
    # figure's job; here the pair only has to show that each family recovers the
    # condition its ordering matches. The scores are still computed so the row
    # can be asserted to be won by its own family.
    assert val[j] == max(val), (name, prop, val)

# 2026-11-09: the strap line under the panels is gone at the author's request. The
# caption already carries both facts (the coefficient count and the pointer to the
# supplementary reconstructions), so it was duplication that cost page height.
assert x_rc + SQ + 0.05 <= W, f'layout overflows: {x_rc + SQ:.2f}in of {W}'

os.makedirs(os.path.dirname(OUT) or '.', exist_ok=True)
tmp = OUT + '.tmp.png'
plt.savefig(tmp, dpi=400, facecolor=BGC)
os.replace(tmp, OUT)
print(f'wrote {OUT}')
