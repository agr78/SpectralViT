"""Isotropic low-frequency mode selection for the 3D bases.

The original basis3 truncation `[:k,:k,:k].reshape(...)[:, :K]` takes a row-major
prefix, which keeps only modes with fx == 0 -- the x axis is collapsed to its DC
component. This selects the K modes of lowest total frequency instead
(equivalently, lowest Neumann-Laplacian eigenvalue).
"""
import numpy as np
from scipy.fft import fftn, dctn


def low_modes(K, shape, order='l1'):
    """Indices of the K lowest-frequency 3D modes, isotropically chosen."""
    n = [min(s, K) for s in shape]
    cand = [(a, b, c) for a in range(n[0]) for b in range(n[1]) for c in range(n[2])]
    if order == 'l1':
        key = lambda t: (t[0] + t[1] + t[2], max(t))
    else:
        key = lambda t: sum(2 * (1 - np.cos(np.pi * i / shape[d])) for d, i in enumerate(t))
    return sorted(cand, key=key)[:K]


def eigvals(idx, shape):
    """Neumann-Laplacian eigenvalue of each selected mode."""
    return np.array([sum(2 * (1 - np.cos(np.pi * i / shape[d])) for d, i in enumerate(t))
                     for t in idx])


def basis3_iso(V, name, K, bb, order='l1'):
    cb = V[:, bb[0], bb[1], bb[2]]
    shape = cb.shape[1:]
    idx = low_modes(K, shape, order)
    if name == '3DDCT':
        T = dctn(cb, axes=(1, 2, 3), norm='ortho')
    elif name == '3D|FFT|':
        T = np.abs(fftn(cb, axes=(1, 2, 3)))
    elif name == '3DlogFFT':
        T = np.log1p(np.abs(fftn(cb, axes=(1, 2, 3))))
    else:
        return None
    return np.stack([T[:, a, b, c] for a, b, c in idx], 1)
