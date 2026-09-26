"""Spectral ViT for DBS outcome: crops, log-magnitude Fourier tokens, attention.

Basis is Fourier magnitude for translation invariance; magnitudes are log
compressed because their variance scales with their mean. Modes are ordered by
true Fourier frequency. No rank weight: the clinical covariates already carry the
structure it would encode.
"""
import numpy as np
import sys
import os

import torch
import torch.nn as nn
from scipy import ndimage
from scipy.fft import fftn
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from networks import SpectralViT

H, DZ = 10, 8                     # crop half-extents, 20 x 20 x 16
DEFAULT_K = 16


def crops(volume, mask):
    """Both nuclei ordered by mean susceptibility, or None if the mask is not bilateral.

    The window is clamped to the volume rather than clipped, so a nucleus near
    the edge yields a full-size crop instead of being discarded.
    """
    lb, n = ndimage.label(mask)
    if n < 2 or mask.sum() > 0.5 * mask.size:
        return None
    sz = ndimage.sum(mask, lb, range(1, n + 1))
    big = np.argsort(sz)[::-1][:2] + 1
    order = sorted([(float(volume[lb == b].mean()), b) for b in big], key=lambda t: t[0])
    out = []
    for _, b in order:
        cx, cy, cz = [int(round(v)) for v in ndimage.center_of_mass(mask, lb, [b])[0]]
        x0 = min(max(0, cx - H), volume.shape[0] - 2 * H)
        y0 = min(max(0, cy - H), volume.shape[1] - 2 * H)
        z0 = min(max(0, cz - DZ), volume.shape[2] - 2 * DZ)
        out.append(volume[x0:x0 + 2 * H, y0:y0 + 2 * H, z0:z0 + 2 * DZ])
    if any(c.shape != (2 * H, 2 * H, 2 * DZ) for c in out):
        return None
    return out[1], out[0]


def low_modes(K, shape):
    """The K lowest Fourier frequencies, one per conjugate pair.

    Index k and N - k are the same frequency and, for a real volume, the same
    magnitude, so only the representative with k <= N / 2 is kept.
    """
    freq = lambda t: sum((min(i, shape[d] - i) / shape[d]) ** 2 for d, i in enumerate(t))
    half = [range(n // 2 + 1) for n in shape]
    cand = [(a, b, c) for a in half[0] for b in half[1] for c in half[2]]
    return sorted(cand, key=freq)[:K]


def features(higher, lower, K=DEFAULT_K):
    """log1p Fourier magnitude of both nuclei, concatenated."""
    idx = low_modes(K, (2 * H, 2 * H, 2 * DZ))
    f = lambda V: np.stack([np.log1p(np.abs(fftn(V, axes=(1, 2, 3))))[:, a, b, c]
                            for a, b, c in idx], 1)
    return np.hstack([f(higher), f(lower)])


def variance_exponent(X):
    """Slope of log-variance on log-mean across features; 2 implies log compression.

    Uses no labels, so the compression can be chosen before any modeling.
    """
    mu, va = X.mean(0), X.var(0)
    ok = (mu > 0) & (va > 0)
    return float(np.polyfit(np.log(mu[ok]), np.log(va[ok]), 1)[0])


def spectral_vit(n_spectral, n_clinical, d=16):
    """The repo SpectralViT configured for DBS: one layer, no rank weight, linear residual.

    Spectral coefficients and clinical covariates form one token sequence. The
    residual on the raw vector lets the model fall back to plain concatenation.
    """
    return SpectralViT(
        n_inputs=n_spectral + n_clinical,
        n_heads=2, embed_dim=d, n_layers=1, pooling='mean',
        use_rank_weights=False, use_layer_norm=False,
        use_linear_residual=True, pos_embed_init='zeros', batch_first=True,
    )


def fit(Xtr, Ctr, ytr, steps=200, lr=3e-3, weight_decay=0.1, seed=0, device='cuda'):
    """Train on one fold.

    K is set by cross-validation on the training set. Width and step count are
    fixed across all models rather than tuned, so no model gets a capacity
    advantage; sensitivity to both is reported separately.
    """
    torch.manual_seed(seed)
    m = spectral_vit(Xtr.shape[1], Ctr.shape[1]).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr, weight_decay=weight_decay)
    loss = nn.BCEWithLogitsLoss()
    v = torch.tensor(np.hstack([Xtr, Ctr]), dtype=torch.float32, device=device)
    y = torch.tensor(ytr, dtype=torch.float32, device=device)
    for _ in range(steps):
        opt.zero_grad(); loss(m(v), y).backward(); opt.step()
    return m.eval()


def predict(m, X, C, device='cuda'):
    """Responder probability."""
    v = np.hstack([X, C])
    with torch.no_grad():
        z = m(torch.tensor(v, dtype=torch.float32, device=device))
    return 1.0 / (1.0 + np.exp(-z.cpu().numpy()))


def operating_quantile(y_train, p_train):
    """Quantile of the threshold that maximizes balanced accuracy on the training set."""
    best, thr = -1.0, float(np.median(p_train))
    for t in np.unique(np.concatenate([p_train, np.linspace(p_train.min(), p_train.max(), 300)])):
        b = balanced_accuracy_score(y_train, (p_train >= t).astype(int))
        if b > best:
            best, thr = b, float(t)
    return float((p_train < thr).mean())


def apply_quantile(p_test, q):
    """Test threshold at the training operating quantile.

    Transfers the quantile rather than the threshold value, which assumes only
    that the cohorts have similar class prevalence.
    """
    return float(np.quantile(p_test, q))
