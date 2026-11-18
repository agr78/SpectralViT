"""Profile cohort difficulty across MedMNIST3D."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repository root: shared modules and paths.py
from paths import DATA
import os
import numpy as np
from scipy import fft

SETS = {'organmnist3d': f'{DATA}/medmnist/organmnist3d.npz'}
for d in ('nodule', 'adrenal', 'fracture', 'vessel', 'synapse'):
    SETS[d + 'mnist3d'] = f'{DATA}/medmnist/{d}mnist3d.npz'
ORDER = ['organmnist3d', 'nodulemnist3d', 'adrenalmnist3d',
         'fracturemnist3d', 'vesselmnist3d', 'synapsemnist3d']
NB = 12


def pr(v):
    v = np.asarray(v, float).ravel()
    s = (v ** 2).sum()
    return float(s ** 2 / ((v ** 4).sum() * len(v) + 1e-30))


def bands(side):
    k = np.arange(side)
    kz, ky, kx = np.meshgrid(k, k, k, indexing='ij')
    kr = np.sqrt(kz ** 2 + ky ** 2 + kx ** 2).ravel()
    e = np.linspace(0, kr.max() + 1e-6, NB + 1)
    return [np.digitize(kr, e) - 1 == b for b in range(NB)]


rows = {}
for name in ORDER:
    z = np.load(SETS[name])
    X = z['train_images'].astype(np.float32)
    y = z['train_labels'].ravel().astype(int)
    side = X.shape[1]
    M = bands(side)
    E = np.zeros((len(X), NB), dtype=np.float64)
    for i in range(len(X)):
        c2 = fft.dctn(X[i], type=2, norm='ortho').ravel() ** 2
        for b, m in enumerate(M):
            E[i, b] = c2[m].mean()
    cls, cnt = np.unique(y, return_counts=True)
    a, b_ = cls[np.argsort(-cnt)[:2]]
    ia, ib = y == a, y == b_
    mu = X[ia].mean(0) - X[ib].mean(0)
    sd = np.sqrt((X[ia].var(0) + X[ib].var(0)) / 2) + 1e-6
    dmap = np.abs(mu) / sd
    fb = ((E[ia].mean(0) - E[ib].mean(0)) ** 2 /
          (E[ia].var(0) + E[ib].var(0) + 1e-30))
    rows[name] = dict(
        n=len(X), ncls=len(cls),
        uniq=int(len(np.unique(X[:50]))),
        frac_extreme=float(np.mean((X[:200] == X.min()) | (X[:200] == X.max()))),
        d_max=float(dmap.max()), d_mean=float(dmap.mean()),
        conc=pr(dmap),
        band_peak=int(np.argmax(fb)),
        band_conc=pr(fb),
        hi_frac=float(fb[NB // 2:].sum() / (fb.sum() + 1e-30)),
        spec_slope=float(np.polyfit(np.log(np.arange(1, NB + 1)),
                                    np.log(E.mean(0) + 1e-30), 1)[0]))

print('%-16s %5s %4s %6s %7s %7s %7s %7s %6s %6s %7s'
      % ('dataset', 'n', 'cls', 'uniq', 'd_max', 'd_mean', 'conc', 'bandpk',
         'bandc', 'hi', 'slope'))
for k in ORDER:
    r = rows[k]
    print('%-16s %5d %4d %6d %7.2f %7.3f %7.3f %7d %6.3f %6.2f %7.2f'
          % (k, r['n'], r['ncls'], r['uniq'], r['d_max'], r['d_mean'],
             r['conc'], r['band_peak'], r['band_conc'], r['hi_frac'], r['spec_slope']))
print()
print('conc   = participation ratio of the voxelwise effect-size map (1.0 = class signal')
print('         spread over every voxel, small = focal)')
print('bandpk = radial band carrying the most class signal (0 = lowest frequency)')
print('bandc  = participation ratio over bands; hi = share of class signal above mid-band')
print('slope  = log-log power spectrum slope (steeper = smoother volumes)')
np.save(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'medmnist_profile.npy'), rows, allow_pickle=True)
