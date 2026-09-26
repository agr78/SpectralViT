"""Undo the CHH-only unsharp mask in the DCT domain.

fast_resample_sharp applies, to CHH only:

    out = img + amount * (img - gaussian_filter(img, sigma))

which is linear and shift-invariant, so in the frequency domain it is a
per-mode gain

    H(f) = 1 + amount * (1 - exp(-2 pi^2 sigma^2 f^2)),   f in cycles/voxel

gaussian_filter uses reflect boundaries, the same boundary condition the DCT
assumes, so dividing each CHH coefficient by H(f_i) inverts the filter without
needing the raw volumes. Uses no outcome labels.
"""
import numpy as np

SIGMA, AMOUNT = 0.8, 0.5


def mode_freq(idx, shape):
    """radial frequency of each mode in cycles/voxel (Nyquist = 0.5)"""
    return np.array([
        0.5 * np.sqrt(sum((t[d] / (shape[d] / 2.0)) ** 2 for d in range(3)) / 3.0)
        for t in idx])


def unsharp_gain(f_cyc, sigma=SIGMA, amount=AMOUNT):
    return 1.0 + amount * (1.0 - np.exp(-2 * np.pi ** 2 * sigma ** 2 * f_cyc ** 2))


def desharpen(XB, idx, shape, sigma=SIGMA, amount=AMOUNT):
    """divide CHH coefficients by the unsharp gain of their mode"""
    g = unsharp_gain(mode_freq(idx, shape), sigma, amount)
    return XB / g
