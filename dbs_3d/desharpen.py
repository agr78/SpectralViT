"""Undo the CHH-only unsharp mask in the DCT domain."""
import numpy as np

SIGMA, AMOUNT = 0.8, 0.5


def mode_freq(idx, shape):
    return np.array([
        0.5 * np.sqrt(sum((t[d] / (shape[d] / 2.0)) ** 2 for d in range(3)) / 3.0)
        for t in idx])


def unsharp_gain(f_cyc, sigma=SIGMA, amount=AMOUNT):
    return 1.0 + amount * (1.0 - np.exp(-2 * np.pi ** 2 * sigma ** 2 * f_cyc ** 2))


def desharpen(XB, idx, shape, sigma=SIGMA, amount=AMOUNT):
    g = unsharp_gain(mode_freq(idx, shape), sigma, amount)
    return XB / g
