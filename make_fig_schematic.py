"""Figure 1: the spectral ViT against a spatial ViT, differing only in the projection E.

Two rows share the input v on the left and a single transformer encoder on the right, so
the only thing that differs between them is the middle: the projection E. The top row cuts
v into a 4x4 patch grid, flattens the patches and projects each with a learned matrix; the
bottom row projects the whole of v onto the basis vectors w_i (the leading principal
components), drawn as images of exactly the same size as v, giving spectral tokens
s_i = v . w_i with a rank weight initialized at 1/i and trained thereafter, and a
positional embedding that indexes mode rather than location.

Pooling, the classifier and the encoder internals are not drawn: they are identical in the
two models and so carry none of the figure's argument.

The basis images, the input and the variance fractions come from a cache; rebuild it with
make_fig1_cache.py (the published figure uses FIG1_SET=mnist, whose basis is the n=1,000
seed-0 draw of the MNIST experiment, and whose input is the first test digit). The cache
lives next to the figure, not under results/, which .gitignore excludes -- the inputs have
to survive a clean clone.

usage: python make_fig_schematic.py [out.png]
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
                                                         'spectralViT.png')
CACHE = os.environ.get('FIG1_CACHE', os.path.join(HERE, 'paper', 'cvpr', 'fig', 'data',
                                                  'fig1_cache_mnist.npz'))
z = np.load(CACHE)
EIG, SUBJ = z['eig_sag'], z['brain_sag']

# Restrained palette: one ink for every stroke and label, light tints only on the token
# cells (dark text on the tint, never white text on a saturated fill), RdBu for the signed
# basis images and gray for the digit. A tint means the cell carries learned weights: the
# spatial projection E and both positional embeddings are learned, the rank weight is
# learned from its 1/i start, and the spectral projection is a fixed basis, so the s_i
# cells stay white. Without this the extra orange column reads as extra parameters, when
# the spatial model has the larger learned input side.
# Dark variant for the web write-up: same geometry, inverted ink and deeper token tints so
# the figure sits on a dark panel instead of a white page. FIG_DARK=1 selects it; the paper
# always uses the light one.
DARK = os.environ.get('FIG_DARK') == '1'
INK, FRAME = '0.15', '0.35'
C_TOK, C_RANK, C_POS, C_FIX = '#d3e2f1', '#f8d9c0', '#d8e8d1', 'white'
if DARK:
    INK, FRAME = '#E6EBF0', '0.55'
    C_TOK, C_RANK, C_POS, C_FIX = '#24384d', '#4a3121', '#24382a', '#171C22'
BGC = '#171C22' if DARK else 'white'
FS = 7.0
W, H = 6.9, 2.0                     # inches; every coordinate below is in inches

S = 0.60                            # side of v and of every w_i: the same square
N_W = 4                             # basis images shown (and token rows before the vdots)
YT, YB = 1.50, 0.58                 # centerlines of the spatial and spectral rows
CY = (YT + YB) / 2                  # the shared input sits between them
Y_LAB_T, Y_LAB_B = YT + S / 2 + 0.09, YB + S / 2 + 0.09      # row-label lines
Y_SUB_T, Y_SUB_B = YT - S / 2 - 0.085, YB - S / 2 - 0.085    # small labels under images

fig = plt.figure(figsize=(W, H), facecolor=BGC)
bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, W); bg.set_ylim(0, H); bg.axis('off')


def image_axes(x, y, w, h, frame=True):
    ax = fig.add_axes([x / W, y / H, w / W, h / H])
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor(BGC)
    for sp in ax.spines.values():
        sp.set_edgecolor(FRAME); sp.set_linewidth(0.5); sp.set_visible(frame)
    return ax


def arrow(x1, y1, x2, y2, lw=0.7, gapA=1.5, gapB=1.5):
    bg.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=6,
                                 lw=lw, color=INK, zorder=5, shrinkA=gapA, shrinkB=gapB))


def text(x, y, s, **kw):
    kw.setdefault('fontsize', FS); kw.setdefault('color', INK)
    kw.setdefault('ha', 'center'); kw.setdefault('va', 'center')
    return bg.text(x, y, s, zorder=6, **kw)


NP = 7                              # patch 4 on 28 px, as the digit experiment runs it: 7 x 7 = 49
PS = SUBJ.shape[0] // NP
NPP = NP * NP


def patch(r, c):
    return SUBJ[PS * r:PS * (r + 1), PS * c:PS * (c + 1)]


# --- input v, shared by both rows ----------------------------------------------------------
x_in = 0.06
ax = image_axes(x_in, CY - S / 2, S, S)
ax.imshow(SUBJ, cmap='gray', vmin=0, vmax=1)
text(x_in + S / 2, CY - S / 2 - 0.085, r'Input $\mathbf{v}$', fontsize=FS - 0.5)

x_bus, x_g = x_in + S + 0.12, x_in + S + 0.50        # the branch to the two rows
bg.add_line(Line2D([x_in + S, x_bus, x_bus], [CY, CY, YT], lw=0.7, color=INK, zorder=5))
bg.add_line(Line2D([x_bus, x_bus], [CY, YB], lw=0.7, color=INK, zorder=5))
arrow(x_bus, YT, x_g, YT, gapA=0, gapB=1.0)
arrow(x_bus, YB, x_g, YB, gapA=0, gapB=1.0)

x_b1 = x_g + N_W * S + (N_W - 1) * 0.04              # right edge of both decompositions
x_t0 = x_b1 + 0.44                                   # left edge of both token columns
w_pos, w_rank, w_tok = 0.36, 0.22, 0.60
x_t1 = x_t0 + w_tok + w_rank + w_pos                 # right edge of the wider column
rh = S / (N_W + 1)                                  # five token rows span the image side

# --- top row: spatial ViT ------------------------------------------------------------------
text(x_g, Y_LAB_T, 'Spatial ViT (conventional)', ha='left', fontsize=FS + 0.5, weight='bold')
# the same 7 at the same size, cut into a 4x4 grid of 7x7 patches
gg = 0.02                                           # width of the cuts between patches
ps = (S - (NP - 1) * gg) / NP
for r in range(NP):
    for c in range(NP):
        axp = image_axes(x_g + c * (ps + gg), YT + S / 2 - (r + 1) * ps - r * gg, ps, ps,
                         frame=False)
        axp.imshow(patch(r, c), cmap='gray', vmin=0, vmax=1)
bg.add_patch(Rectangle((x_g, YT - S / 2), S, S, fc='none', ec=FRAME, lw=0.5, zorder=4))
text(x_g + S / 2 + 0.06, Y_SUB_T, r'$%d\times%d$ local patches' % (NP, NP), fontsize=FS - 0.5)

# the patches flattened into a sequence, raster order, at nearly the size they have in
# the grid, ending just before the token column
# Forty-nine would not fit at a legible size, and the leading patches are blank margin, so
# the strip shows the first, an ellipsis, the contiguous run of NSHOW patches carrying the
# most ink (where the digit is), an ellipsis, and the last.
x_s0, x_s1 = x_g + S + 0.17, x_b1                 # the strip ends where the basis row ends
NSHOW, sg, ell = 12, 0.012, 0.14
ink = np.array([patch(*divmod(j, NP)).sum() for j in range(NPP)])
j0 = int(np.argmax([ink[a:a + NSHOW].sum() for a in range(NPP - NSHOW + 1)]))
shown = [0] + list(range(j0, j0 + NSHOW)) + [NPP - 1]
ts = (x_s1 - x_s0 - 2 * ell - (len(shown) - 1) * sg) / len(shown)
arrow(x_g + S, YT, x_s0, YT, gapB=0.5)
x_run = x_s0 + ts + sg + ell                      # the run starts after the first ellipsis
xs = [x_s0] + [x_run + m * (ts + sg) for m in range(NSHOW)] + [x_s1 - ts]
for x, j in zip(xs, shown):
    axp = image_axes(x, YT - ts / 2, ts, ts)
    axp.imshow(patch(*divmod(j, NP)), cmap='gray', vmin=0, vmax=1)
for xe in (x_s0 + ts + sg + ell / 2, x_s1 - ts - sg - ell / 2):
    text(xe, YT, r'$\cdots$', fontsize=FS)
text((x_s0 + x_s1) / 2, Y_SUB_T, r'Flattened patches $\mathbf{p}_1\,\cdots\,\mathbf{p}_{%d}$' % NPP,
     fontsize=FS - 0.5)

# --- bottom row: spectral ViT --------------------------------------------------------------
text(x_g, Y_LAB_B, 'Spectral ViT (proposed)', ha='left', fontsize=FS + 0.5, weight='bold')
for i in range(N_W):
    x = x_g + i * (S + 0.04)
    ax = image_axes(x, YB - S / 2, S, S)
    v = EIG[i]
    ax.imshow(v, cmap='RdBu_r', vmin=-np.abs(v).max(), vmax=np.abs(v).max())
    # the corners of the leading components are blank, so the name needs no backing box
    ax.text(0.06, 0.94, r'$\mathbf{w}_%d$' % (i + 1), transform=ax.transAxes, fontsize=FS - 1,
            ha='left', va='top', color=INK)
# Where the basis comes from: not from v, but from the training set. A small deck of the
# draw's own digits, then 'PCA', then E, in the caption line under the basis images.
TRX = z['train_examples']
# The deck of training digits lies under the arrow that feeds the basis row: the w_i are
# a decomposition of the data, not of v. The basis is not named -- it need not be PCA.
ic, io, nic = 0.12, 0.02, 3                          # icon card side, offset, count
tot = ic + (nic - 1) * io                            # the deck's full height and width
x_deck = x_bus + 0.32                                # along the arrow, clear of the branch
x_i0 = x_deck - tot / 2
y_i0 = YB - 0.06 - tot                               # just below the arrow line
for m in range(nic):                                 # back card top-left, front bottom-right
    ax = image_axes(x_i0 + m * io, y_i0 + (nic - 1 - m) * io, ic, ic)
    ax.imshow(TRX[(m * 3) % len(TRX)], cmap='gray', vmin=0, vmax=1)
text(x_deck, Y_SUB_B, 'Training set', fontsize=FS - 0.8)
text((x_g + x_b1) / 2, Y_SUB_B, r'$E=[\mathbf{w}_1\,\cdots\,\mathbf{w}_n]^{\top}$, global basis',
     fontsize=FS - 0.5)


# --- token columns: one per row, right-aligned so both feed the encoder alike -------------
def cell(x, y, w, color, lab, fs):
    bg.add_patch(Rectangle((x, y), w, rh, fc=color, ec=FRAME, lw=0.5, zorder=3))
    text(x + w / 2, y + rh / 2, lab, fontsize=fs)


def column(yc, cols, rows):
    """cols: (width, tint) from the left; rows: one label per column, top row first.

    Both columns start at x_t0, so the two projection steps line up; returns the right edge.
    """
    for k, labs in enumerate(rows):
        y = yc + S / 2 - (k + 1) * rh
        x = x_t0
        for (w, col), lab in zip(cols, labs):
            cell(x, y, w, col, lab, fs=FS - 1.0 if k < N_W else FS)
            x += w
    return x


idx = [str(i + 1) for i in range(N_W)]
# spatial: patch tokens x_j = E p_j with a positional embedding by patch location
rows_sp = [(r'$\mathbf{x}_{%s}=E\,\mathbf{p}_{%s}$' % (j, j), r'$\mathbf{e}_{\mathrm{pos},%s}$' % j)
           for j in idx] + [(r'$\vdots$', r'$\vdots$')]
# the same footprint as the spectral column: the token cell absorbs the width of the rank
# column it does not have, so both tables span x_t0 to x_t1 and e_pos sits in one column
x_r_sp = column(YT, [(w_tok + w_rank, C_TOK), (w_pos, C_POS)], rows_sp)
arrow(x_s1, YT, x_t0, YT)
text((x_s1 + x_t0) / 2, YT + 0.09, r'$E$', fontsize=FS + 0.5)
text((x_s1 + x_t0) / 2, YT - 0.10, 'Learned', fontsize=FS - 1.2, color='0.6' if DARK else '0.3')
text(x_t0, Y_SUB_T, r'Patch tokens, $\mathbf{e}_{\mathrm{pos}}$ by location', ha='left',
     fontsize=FS - 0.5)

# spectral: s_i = v . w_i, initial rank weight 1/i (trained), positional embedding by mode
# omega_i, the symbol of sec:methods, rather than its 1/i initial value: the weight is
# trained, and a number in the cell reads as a constant
rows_sc = [(r'$s_{%s}=\mathbf{v}\cdot\mathbf{w}_{%s}$' % (i, i), r'$\omega_{%s}$' % i,
            r'$\mathbf{e}_{\mathrm{pos},%s}$' % i) for i in idx] + [(r'$\vdots$',) * 3]
x_r_sc = column(YB, [(w_tok, C_FIX), (w_rank, C_RANK), (w_pos, C_POS)], rows_sc)
arrow(x_b1, YB, x_t0, YB)
text((x_b1 + x_t0) / 2, YB + 0.09, r'$E$', fontsize=FS + 0.5)
text((x_b1 + x_t0) / 2, YB - 0.10, 'Fixed', fontsize=FS - 1.2, color='0.6' if DARK else '0.3')
# left-aligned with the table, so it does not reach back under the arrow where the deck is
# two lines, the first at the caption line and the second below it, so the top line
# keeps the same clearance from the table that the other captions have
text(x_t0, Y_SUB_B, r'Spectral tokens, rank $\omega_i$,', ha='left', fontsize=FS - 0.5)
text(x_t0, Y_SUB_B - 0.11, r'$\mathbf{e}_{\mathrm{pos}}$ by mode', ha='left', fontsize=FS - 0.5)

# --- the proposed row, boxed -------------------------------------------------------------
# A dashed frame around everything that is ours: the row label, the training deck, the
# basis, the fixed projection and the spectral tokens. The shared input and the shared
# encoder stay outside, which is the point -- they are the same in both models.
# the left edge clears the 'Training set' label, which is centered on the input arrow and
# so reaches back past the branch line; the bottom edge clears the caption line
bx0, bx1 = x_bus + 0.04, x_r_sc + 0.07
by0, by1 = Y_SUB_B - 0.11 - 0.07, Y_LAB_B + 0.075     # below the caption's second line
bg.add_patch(Rectangle((bx0, by0), bx1 - bx0, by1 - by0, fc='none', ec=INK, lw=0.8,
                       ls=(0, (3.5, 2.2)), zorder=1))

# --- the encoder, drawn once ----------------------------------------------------------------
x_e, w_e = x_t1 + 0.30, 0.74
y_e0, y_e1 = YB - 0.14, YT + 0.14
bg.add_patch(Rectangle((x_e, y_e0), w_e, y_e1 - y_e0, fc='none', ec=INK, lw=0.7, zorder=3))
text(x_e + w_e / 2, CY, 'Transformer\nencoder', linespacing=1.2)
arrow(x_r_sp, YT, x_e, YT)
arrow(x_r_sc, YB, x_e, YB)
arrow(x_e + w_e, CY, x_e + w_e + 0.26, CY)
text(x_e + w_e + 0.30, CY, r'$\hat{y}$', ha='left', fontsize=FS + 1.5)
assert x_e + w_e + 0.48 <= W, f'layout overflows the page width: {x_e + w_e + 0.48:.2f} in'
# under the encoder box, between its bottom edge and the caption line
text(x_e + w_e / 2, YB - 0.14 - 0.10, 'Tinted: learned weights', fontsize=FS - 0.8,
     color='0.6' if DARK else '0.3')

os.makedirs(os.path.dirname(OUT) or '.', exist_ok=True)
tmp = OUT + '.tmp.png'
plt.savefig(tmp, dpi=400, facecolor=BGC)
os.replace(tmp, OUT)
print(f'wrote {OUT}')
