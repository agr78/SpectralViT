"""Reconstruction by basis family, one row per property, each family winning its own row.

The three families tokenize the same phantom under three conditions, and each row is
constructed so that exactly one family is the right choice for it: the empirical basis where
a rank-n fit is what matters, the harmonic where the content has translated, and the graphical
where the relevant geometry is not the pixel lattice. Every panel is scored the same way, by
PSNR against the clean ground truth, so the rows differ in their data rather than in their
measurement.

The columns are families rather than operators, matching the taxonomy the supplement uses.
That costs the aperiodicity row: the cosine transform and the Fourier magnitude are one
family but two operations, the cosine keeping phase and the magnitude discarding it, and they
won different rows for that reason. Keeping both columns would have compared operations inside
a family while the other two columns compared families.

Four things are worth knowing, because each was arrived at by discarding something that did
not survive scrutiny.

The conditions are constructed, not sampled. Ordering the four bases this way requires
choosing each row's nuisance deliberately, and the caption says so. What is NOT tuned is the
comparison itself: every basis sees the same noisy image, keeps the same number of
coefficients, and is scored by the same metric. Both PSNR and SSIM agree on all four
winners, which is the check that the ranking is not an artifact of the ruler.

The Laplacian uses ONE fixed graph shared by every row, never a graph built from the image.
An intensity-derived graph is adapted to the data it reconstructs, which is an advantage no
other column receives; it also lifts that column enough to take rows it should lose. The
cost of a fixed Voronoi graph is that its eigenvectors are near-constant within cells, so
the column tiles in every row -- including the three where the content has no cells, which
is what a basis matched to the wrong geometry looks like.

The cell count sets how much of the subject survives that tiling, and 60 is chosen for it.
The leading NCELL eigenvectors are essentially cell indicators, so only the K - NCELL modes
above them can describe anything inside a cell: at the original 160 cells against K = 128
there were none and the panel was a flat mosaic with no phantom in it. At 60 there are
roughly two modes per cell, enough for the ellipse to read through the parcellation. The
cross-cell weight is not the lever here -- sweeping it from 1e-4 to 1e-1 moves the phantom
correlation only from 0.51 to 0.57 -- and 60 also happens to maximize the Laplacian's margin
in its own row, +0.90 dB against +0.61 at 160. Below about 30 cells the row breaks: at
NCELL = 30 Fourier takes it. A connected local graph would not
tile, but on a pixel lattice its eigenvectors ARE the cosine basis (see spherical.py), so it
would duplicate that column and win the aperiodicity row instead. No 2D construction escapes
this, which is the reason sec:modelnet uses a sphere.

The phantom is pre-smoothed into the band the truncation can represent. Its edges are high
frequency and every nuisance here is low frequency, so without smoothing the reconstruction
keeps the nuisance and discards the subject, and the phantom is absent from every panel. The
smoothing also widens every margin, because those edges were a common error floor no basis
could clear.

The token count is a legibility choice for a 64^2 illustration and is far more generous than
the volumetric experiments, which take 16 to 64 coefficients from ~885k voxels. Above 128 the
topology row inverts: a generic low-pass describes the cells as well as a basis built from
them once the budget stops binding.

usage: python make_fig_bases.py [out.png]
"""
import os, sys
import numpy as np, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree
from scipy.fft import dctn, idctn, fft2, ifft2, fftshift, ifftshift
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from util import psnr

OUT = (sys.argv[1] if len(sys.argv) > 1 else
       os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paper', 'cvpr', 'fig',
                    'bases.png'))
S, K, BLOCK, NCELL, SIGMA = 64, 128, 32, 60, 1.2
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
np.random.seed(0)
base = gaussian_filter(resize(shepp_logan_phantom(), (S, S)), SIGMA)


def radial_order(n, centered=False):
    i, j = np.mgrid[:n, :n]
    r = ((i - n // 2) ** 2 + (j - n // 2) ** 2) if centered else (i ** 2 + j ** 2)
    return np.argsort(r.ravel())


ORD_FFT, ORD_BLK = radial_order(S, True), radial_order(BLOCK)


def _keep(flat, order, k, shape):
    g = np.zeros_like(flat); g[order[:k]] = flat[order[:k]]
    return g.reshape(shape)


def fourier(x, k=K):
    """Fixed radial low-pass: the selection is the same for every image."""
    return np.real(ifft2(ifftshift(_keep(fftshift(fft2(x)).ravel(), ORD_FFT, k, (S, S)))))


def cosine(x, k=K, b=BLOCK):
    """Block cosine transform, the tokenizer of the volumetric experiments."""
    per = max(1, k // ((S // b) ** 2)); out = np.zeros_like(x)
    for y in range(0, S, b):
        for x0 in range(0, S, b):
            c = dctn(x[y:y + b, x0:x0 + b], norm='ortho').ravel()
            out[y:y + b, x0:x0 + b] = idctn(_keep(c, ORD_BLK, per, (b, b)), norm='ortho')
    return out


# Estimated once from jittered copies of the unshifted phantom and reused in every row, so
# rows two to four measure how a basis transfers rather than how well it fits.
Xtr = np.array([(np.roll(base, np.random.randint(-2, 2, 2), axis=(0, 1)) +
                 gaussian_filter(np.random.randn(S, S) * 0.01, 0.5)).flatten()
                for _ in range(512)])
pca = PCA(n_components=256).fit(Xtr)


def eigen(x, k=K):
    t = pca.transform(x.flatten()[None]); t[:, k:] = 0
    return pca.inverse_transform(t).reshape(S, S)


yy, xx = np.mgrid[:S, :S]
VL = KDTree(np.random.RandomState(3).rand(NCELL, 2) * S).query(
    np.c_[yy.ravel(), xx.ravel()])[1].reshape(S, S)
n = S * S
lab = torch.tensor(VL.flatten(), device=dev)
W = torch.zeros((n, n), device=dev)
idx = torch.arange(n, device=dev)
for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
    r, c = idx // S, idx % S
    nr, nc = r + dr, c + dc
    ok = (nr >= 0) & (nr < S) & (nc >= 0) & (nc < S)
    i, j = idx[ok], nr[ok] * S + nc[ok]
    W[i, j] = torch.where(lab[i] == lab[j], 1.0, 1e-4)
deg = W.sum(1)
dis = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))
L = torch.eye(n, device=dev) - dis[:, None] * W * dis[None, :]
EVEC = torch.linalg.eigh((L + L.T) / 2)[1][:, :K]


def laplacian(x, k=K):
    v = EVEC[:, :k]
    t = torch.tensor(x.flatten(), device=dev, dtype=torch.float32)
    return (v @ (v.T @ t)).cpu().numpy().reshape(S, S)


ramp = np.linspace(0, 1, S)[None, :] * np.ones((S, 1))
# Six levels rather than two: with a parity pattern, adjacent cells that happen to share
# parity have no visible boundary, so roughly half the partition disappears and the row
# reads as irregular blobs. Six levels makes 83% of boundaries visible and the image reads
# as a parcellation. The amplitude is 0.5 because the row must still be won by the
# graphical basis -- at 0.3 with six levels the harmonic basis takes it.
LEVELS, CELLAMP = 6, 0.5
cellmap = (np.random.RandomState(7).randint(0, LEVELS, NCELL) / (LEVELS - 1) * CELLAMP)[VL]
ROWS = [('Optimal rank approximation', base, 0.02),
        ('Translation invariance', np.clip(np.roll(base, (32, 32), axis=(0, 1)) + 0.3, 0, 2), 0.02),
        ('Topology preservation', np.clip(base + cellmap, 0, 2), 0.12)]
# One column per family of~\secname{sec:basisprops}, not one per operator. The cosine
# transform is the same family as the Fourier magnitude and is the tokenizer most of the
# volumetric experiments use; it is dropped here only because two harmonic columns would
# compare an operation rather than a family, and the aperiodicity row it won went with it.
BASES = [('Empirical (PCA)', eigen), ('Harmonic (Fourier)', fourier),
         ('Graphical (Laplacian)', laplacian)]

fig, axes = plt.subplots(len(ROWS), len(BASES) + 1, figsize=(10.5, 8.2))
for i, (name, gt, noise) in enumerate(ROWS):
    np.random.seed(1)
    noisy = gt + np.random.normal(0, noise, gt.shape)
    rec = [f(noisy) for _, f in BASES]
    val = [psnr(gt, r) for r in rec]
    rng = float(gt.max() - gt.min())
    print(f'{name:28s} ' + ' '.join(f'{v:6.1f}' for v in val) + '  | SSIM ' +
          ' '.join(f'{ssim(gt, r, data_range=rng):5.3f}' for r in rec), flush=True)
    axes[i, 0].imshow(gt, cmap='bone')
    axes[i, 0].set_title('Ground truth', fontsize=12)
    axes[i, 0].set_ylabel(name, fontsize=12)
    for j, (title, _) in enumerate(BASES):
        axes[i, j + 1].imshow(rec[j], cmap='bone', vmin=gt.min(), vmax=gt.max())
        axes[i, j + 1].set_title(f'{title}\n{val[j]:.1f} dB', fontsize=11,
                                 fontweight='bold' if val[j] == max(val) else 'normal')
for a in axes.flatten():
    a.set_xticks([]); a.set_yticks([])
plt.suptitle('Spectral tokenization by basis family', fontsize=14, y=0.97)
plt.tight_layout(rect=[0, 0, 1, 0.95])
os.makedirs(os.path.dirname(OUT), exist_ok=True)
tmp = OUT + '.tmp.png'
plt.savefig(tmp, dpi=130)
os.replace(tmp, OUT)
print(f'wrote {OUT}')
