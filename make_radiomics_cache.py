"""Extract a radiomics feature table from the IXI volume cache.

usage: python make_radiomics_cache.py"""
import os
import sys

import numpy as np
from scipy import ndimage, stats
from skimage.feature import graycomatrix, graycoprops

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.environ.get('IXI_CACHE', os.path.join(HERE, 'data', 'ixi_vols_96'))
OUT = os.environ.get('RADIOMICS_OUT', os.path.join(HERE, 'results', 'radiomics_ixi.npz'))
LEVELS = 32
PROPS = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
OFFSETS = [1, 2]
ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
SIGMAS = [1.0, 2.0]


def first_order(v):
    f = v.ravel()
    q = np.percentile(f, [10, 25, 50, 75, 90])
    h = np.histogram(f, bins=LEVELS)[0].astype(float)
    h /= max(h.sum(), 1)
    return [f.mean(), f.std(), stats.skew(f), stats.kurtosis(f), np.square(f).mean(),
            stats.entropy(h + 1e-12), f.min(), f.max(), f.max() - f.min(),
            *q, q[3] - q[1], np.abs(f - f.mean()).mean(), np.square(h).sum()]


def glcm(v):
    lo, hi = v.min(), v.max()
    q = np.clip(((v - lo) / max(hi - lo, 1e-9) * (LEVELS - 1)).round(), 0, LEVELS - 1)
    out = []
    mid = [s // 2 for s in v.shape]
    for plane in (q[mid[0], :, :], q[:, mid[1], :], q[:, :, mid[2]]):
        g = graycomatrix(plane.astype(np.uint8), OFFSETS, ANGLES,
                         levels=LEVELS, symmetric=True, normed=True)
        out += [float(graycoprops(g, p).mean()) for p in PROPS]
    return out


def features(v):
    f = first_order(v) + glcm(v)
    for s in SIGMAS:
        f += first_order(ndimage.gaussian_laplace(v, sigma=s))
    return f


X = np.load(CACHE + '.npy', mmap_mode='r')
Y = np.load(CACHE + '_meta.npz')['y']
print(f'{CACHE}.npy: {X.shape}', flush=True)
rows = []
for i in range(len(X)):
    rows.append(features(np.asarray(X[i], dtype=np.float32)))
    if (i + 1) % 50 == 0:
        print(f'  {i + 1}/{len(X)}', flush=True)
F = np.asarray(rows, dtype=np.float32)
bad = ~np.isfinite(F)
if bad.any():
    print(f'replacing {int(bad.sum())} non-finite entries with the column mean')
    F[bad] = np.take(np.nanmean(np.where(bad, np.nan, F), 0), np.where(bad)[1])
os.makedirs(os.path.dirname(OUT), exist_ok=True)
np.savez(OUT, F=F, y=Y)
print(f'\nwrote {OUT}: {F.shape[0]} volumes x {F.shape[1]} features')
