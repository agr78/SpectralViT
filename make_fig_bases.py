"""Reconstruction by basis family, one row per property, each family winning its own row.

usage: python make_fig_bases.py [out.png]"""
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
    return np.real(ifft2(ifftshift(_keep(fftshift(fft2(x)).ravel(), ORD_FFT, k, (S, S)))))


def cosine(x, k=K, b=BLOCK):
    per = max(1, k // ((S // b) ** 2)); out = np.zeros_like(x)
    for y in range(0, S, b):
        for x0 in range(0, S, b):
            c = dctn(x[y:y + b, x0:x0 + b], norm='ortho').ravel()
            out[y:y + b, x0:x0 + b] = idctn(_keep(c, ORD_BLK, per, (b, b)), norm='ortho')
    return out


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
LEVELS, CELLAMP = 6, 0.5
cellmap = (np.random.RandomState(7).randint(0, LEVELS, NCELL) / (LEVELS - 1) * CELLAMP)[VL]
ROWS = [('Optimal rank approximation', base, 0.02),
        ('Translation invariance', np.clip(np.roll(base, (32, 32), axis=(0, 1)) + 0.3, 0, 2), 0.02),
        ('Topology preservation', np.clip(base + cellmap, 0, 2), 0.12)]
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
