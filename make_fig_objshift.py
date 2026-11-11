"""Object-classification distribution shift, laid out for a single column.

util.plot_obj_detect draws 2 rows x 4 columns at 10x5 inches, which has to span both
columns and leaves each panel under an inch wide when set at CVPR column width. The same
content is transposed here to 4 rows x 2 columns: the two classes side by side, with the
image and its Fourier magnitude stacked within each distribution, so the panels stay
legible in one column.

usage: python make_fig_objshift.py [out.png]
"""
import os, sys
import numpy as np, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from util import seed_everything, generate_distance_data

OUT = sys.argv[1] if len(sys.argv) > 1 else 'paper/cvpr/fig/infer_shift.png'
VOL = 32
seed_everything(0)
Xtr, Ytr = generate_distance_data(10, VOL_SIZE=VOL, swapped=False)
Xte, Yte = generate_distance_data(10, VOL_SIZE=VOL, swapped=True)


def pick(X, Y, c):
    return X[(Y == c).nonzero()[0].item()]


def proj(vol):
    """Max projection, so both cubes are visible wherever they sit in depth."""
    return torch.max(vol[0], dim=1)[0].numpy()


def mag(p):
    return torch.fft.fftshift(torch.abs(torch.fft.fft2(torch.from_numpy(p)))).numpy()


def nrmse(a, b):
    return np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-9)


# (group label, class 0 volume, class 1 volume, per-class position)
GROUPS = [('Training distribution', pick(Xtr, Ytr, 0), pick(Xtr, Ytr, 1), ('left', 'right')),
          ('Testing distribution', pick(Xte, Yte, 0), pick(Xte, Yte, 1), ('right', 'left'))]

# A thin spacer row separates the two blocks, so the lower block's panel titles do not
# land on the Fourier panels above them.
fig = plt.figure(figsize=(3.3, 4.6))
gs = fig.add_gridspec(5, 2, height_ratios=[1, 1, 0.52, 1, 1], hspace=0.16, wspace=0.06,
                      left=0.13, right=0.99, top=0.91, bottom=0.05)
for g, (gl, v0, v1, pos) in enumerate(GROUPS):
    r_img, r_fft = (0, 1) if g == 0 else (3, 4)
    for c, vol in enumerate((v0, v1)):
        # Max projection rather than one slice, so both cubes are visible wherever
        # they sit in depth.
        pr = proj(vol)
        ax = fig.add_subplot(gs[r_img, c])
        ax.imshow(pr, cmap='hot', vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'Class {c} ({"close" if c == 0 else "far"}/{pos[c]})',
                     fontsize=6.5, pad=2.5)
        if c == 0:
            ax.set_ylabel('Image\nspace', fontsize=6.5)

        ax2 = fig.add_subplot(gs[r_fft, c])
        ax2.imshow(np.log(mag(pr) + 1e-3), cmap='viridis')
        ax2.set_xticks([]); ax2.set_yticks([])
        if c == 0:
            ax2.set_ylabel('Fourier\nspace', fontsize=6.5)
        if g == 1:
            # The visual difference across the shift is easy to miss in the Fourier
            # panels, so state it: the shift is a pure translation, so the magnitude is
            # unchanged to numerical precision while the image support is disjoint.
            ref = proj(GROUPS[0][1 + c])
            ax.set_xlabel(rf'$\Delta={nrmse(ref, pr):.2f}$', fontsize=6.5, labelpad=1.5)
            ax2.set_xlabel(rf'$\Delta={nrmse(mag(ref), mag(pr)):.2f}$', fontsize=6.5,
                           labelpad=1.5)

    # Block header, centered over both columns and above the panel titles.
    y = 0.962 if g == 0 else 0.478
    fig.text(0.56, y, gl, ha='center', va='center', fontsize=7.5, fontweight='bold')

os.makedirs(os.path.dirname(OUT), exist_ok=True)
tmp = OUT + '.tmp.png'
plt.savefig(tmp, dpi=400)
os.replace(tmp, OUT)
print(f'wrote {OUT}')
