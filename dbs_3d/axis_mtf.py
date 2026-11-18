"""Per-axis spectral falloff, to test whether the acquisition is anisotropic."""
import sys, os
SPDIR = os.path.dirname(os.path.abspath(__file__))
REPO  = os.path.dirname(SPDIR)
OUT   = os.path.join(REPO, 'docs', 'dbs_3d')
DATA  = os.environ.get('DBS_CLINICAL_DIR', os.path.join(REPO, 'docs'))
CACHE = os.environ.get('DBS_CACHE_DIR', '/checkpoints')
sys.path.insert(0, REPO); sys.path.insert(0, SPDIR); os.chdir(REPO)
import numpy as np, warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
from scipy.fft import fft
from reader import prepare_qsm_dataset
from util import mask_crop

mcf = lambda d, m, p: mask_crop(d, m, p)
ds = {}
ds['MSW'] = prepare_qsm_dataset('MSW', '/x', '/y', os.path.join(DATA, 'dbs_03292024.csv'),
                                os.path.join(CACHE, 'msw_cache_6d_cv.pt'), load_cache=True,
                                mask_crop_fn=mcf, cv_pad=False)
ds['CHH'] = prepare_qsm_dataset('CHH', '/x', '/y', os.path.join(DATA, 'chh_subjects_table1_20240729.csv'),
                                os.path.join(CACHE, 'chh_cache_6d_cv.pt'), load_cache=True,
                                mask_crop_fn=mcf, cv_pad=False)


def axis_spectrum(V, axis, nsub=20):
    V = np.moveaxis(V, axis, -1)
    n = V.shape[-1]
    s = [slice(d // 4, 3 * d // 4) for d in V.shape[:-1]]
    lines = V[tuple(s)].reshape(-1, n)
    lines = lines[np.abs(lines).sum(1) > 0]
    if len(lines) > 4000:
        lines = lines[np.random.RandomState(0).choice(len(lines), 4000, replace=False)]
    lines = lines - lines.mean(1, keepdims=True)
    P = np.abs(fft(lines, axis=1)) ** 2
    P = P[:, :n // 2].mean(0)
    return P / (P.sum() + 1e-30)


print("per-axis normalised power spectra (fraction of power above half-Nyquist)")
print("a coarser axis loses high-frequency power sooner\n")
for site in ('MSW', 'CHH'):
    subs = sorted(ds[site].volumes)[:25]
    V = [np.asarray(ds[site].volumes[s], np.float32) for s in subs]
    V = [v for v in V if v.shape == V[0].shape]
    print(f"{site}  ({len(V)} subjects, volume {V[0].shape})")
    for ax, nm in enumerate('xyz'):
        Ps = np.mean([axis_spectrum(v, ax) for v in V], 0)
        f = np.arange(len(Ps)) / (2.0 * len(Ps))
        hi = Ps[len(Ps) // 2:].sum()
        c = np.cumsum(Ps); f90 = f[np.searchsorted(c, 0.90)]
        print(f"   {nm}: power>half-Nyq {hi:.4f}   f90 {f90:.3f} cyc/vox   "
              f"(isotropic would match across axes)")
    print()
