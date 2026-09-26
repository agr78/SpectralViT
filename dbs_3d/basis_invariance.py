"""Why does log|FFT| transfer when DCT and PCA do not?

Two candidate mechanisms, both testable without outcome labels:

  translation   the crop is centred on a computed centroid, and CHH centroids
                come from masks on coarser data, so alignment error is a
                site-dependent nuisance. |FFT| discards phase and is invariant
                to it; DCT is not.

  scaling       a site difference in intensity scale is multiplicative in |FFT|,
                which per-feature standardisation cannot remove. In log space it
                is additive, which standardisation removes exactly.

Each basis is perturbed and the relative feature change measured.
"""
import sys, os
SPDIR = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(SPDIR)
OUT = os.path.join(REPO, 'docs', 'dbs_3d')
DATA = os.environ.get('DBS_CLINICAL_DIR', os.path.join(REPO, 'docs'))
CACHE = os.environ.get('DBS_CACHE_DIR', '/checkpoints')
sys.path.insert(0, REPO); sys.path.insert(0, SPDIR); os.chdir(REPO)
import numpy as np, warnings
warnings.filterwarnings('ignore')
from scipy.fft import dctn, fftn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from basis_fix import low_modes

src = open(os.path.join(SPDIR, 'roi_x_models.py')).read().split('IDX = low_modes')[0]
g = {'__file__': os.path.join(SPDIR, 'roi_x_models.py')}
exec(compile(src, 'x', 'exec'), g)
HA, HB = g['HA'], g['HB']
H, DZ, K = 10, 8, 16
IDX = low_modes(K, (2*H, 2*H, 2*DZ), 'lap')


def feat(V, name, pca=None):
    if name == 'PCA':
        return pca.transform(V.reshape(len(V), -1))
    Tm = {'DCT': lambda x: dctn(x, axes=(1, 2, 3), norm='ortho'),
          '|FFT|': lambda x: np.abs(fftn(x, axes=(1, 2, 3))),
          'logFFT': lambda x: np.log1p(np.abs(fftn(x, axes=(1, 2, 3))))}[name](V)
    return np.stack([Tm[:, a, b, c] for a, b, c in IDX], 1)


pca = PCA(K, random_state=0).fit(HA.reshape(len(HA), -1))
print("relative feature change under a nuisance perturbation")
print("(mean |delta| / mean |feature|, after the standardisation each model uses)\n")
print(f"{'basis':8s} {'1-voxel shift':>14} {'3-voxel shift':>14} {'x1.1 intensity':>15} {'+5% noise':>11}")
for name in ['DCT', '|FFT|', 'logFFT', 'PCA']:
    base = feat(HA, name, pca)
    s = StandardScaler().fit(base); B = s.transform(base)
    out = []
    for pert in ['shift1', 'shift3', 'scale', 'noise']:
        if pert.startswith('shift'):
            k = int(pert[-1]); V = np.roll(HA, k, axis=1)
        elif pert == 'scale':
            V = HA * 1.1
        else:
            V = HA + np.random.RandomState(0).normal(0, 0.05*HA.std(), HA.shape).astype(np.float32)
        P = s.transform(feat(V, name, pca))
        out.append(np.abs(P - B).mean() / (np.abs(B).mean() + 1e-12))
    print(f"{name:8s} {out[0]:14.3f} {out[1]:14.3f} {out[2]:15.3f} {out[3]:11.3f}")

print("\ncross-site scale difference actually present in the data:")
for name in ['DCT', '|FFT|', 'logFFT']:
    a, b = feat(HA, name), feat(HB, name)
    print(f"  {name:8s} MSW mean |coef| {np.abs(a).mean():9.4f}   CHH {np.abs(b).mean():9.4f}   "
          f"ratio {np.abs(b).mean()/np.abs(a).mean():.3f}")
