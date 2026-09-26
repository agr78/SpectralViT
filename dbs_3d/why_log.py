"""Why does log|FFT| beat |FFT|? Two untested mechanisms.

Already falsified: normality/conditioning (rank-normal and Yeo-Johnson are more
Gaussian and perform worse) and spectral ratios (sum-to-zero constraint hurts).

  A  variance stabilisation in the mean-variance sense. If a coefficient's
     across-subject variance scales as var ~ mean^p, the stabilising transform
     is x^(1-p/2), i.e. sqrt for p=1 and log for p=2. Estimated by regressing
     log(var) on log(mean) across coefficients.

  B  a genuinely logarithmic dose-response. If outcome depends on log energy,
     the per-coefficient correlation with outcome should be largest under log,
     with no model involved.
"""
import sys, os
SPDIR = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(SPDIR)
DATA = os.environ.get('DBS_CLINICAL_DIR', os.path.join(REPO, 'docs'))
CACHE = os.environ.get('DBS_CACHE_DIR', '/checkpoints')
sys.path.insert(0, REPO); sys.path.insert(0, SPDIR); os.chdir(REPO)
import numpy as np, warnings
warnings.filterwarnings('ignore')
from scipy.fft import fftn
from scipy.stats import pearsonr, linregress
from basis_fix import low_modes

src = open(os.path.join(SPDIR, 'roi_x_models.py')).read().split('IDX = low_modes')[0]
g = {'__file__': os.path.join(SPDIR, 'roi_x_models.py')}
exec(compile(src, 'x', 'exec'), g)
HA, LA, HB, LB, bA, bB = g['HA'], g['LA'], g['HB'], g['LB'], g['bA'], g['bB']
H, DZ, K = 10, 8, 16
IDX = low_modes(K, (2*H, 2*H, 2*DZ), 'lap')
mag = lambda V: np.stack([np.abs(fftn(V, axes=(1, 2, 3)))[:, a, b, c] for a, b, c in IDX], 1)
MA = np.hstack([mag(HA), mag(LA)])

# ---- A: mean-variance scaling ----
mu = MA.mean(0); va = MA.var(0)
ok = (mu > 0) & (va > 0)
sl = linregress(np.log(mu[ok]), np.log(va[ok]))
print("A. mean-variance scaling across coefficients")
print(f"   log(var) = {sl.slope:.2f} * log(mean) + c     R2 = {sl.rvalue**2:.3f}")
print(f"   slope 1 -> sqrt is the stabilising transform")
print(f"   slope 2 -> log is the stabilising transform")
print(f"   estimated stabilising exponent 1 - p/2 = {1 - sl.slope/2:+.2f}"
      f"   ({'log' if abs(sl.slope-2) < abs(sl.slope-1) else 'sqrt'} is closer)\n")

# within-coefficient: does each feature's spread scale with its own level?
sds, mns = [], []
for j in range(MA.shape[1]):
    x = MA[:, j]
    lo, hi = x < np.median(x), x >= np.median(x)
    sds.append([x[lo].std(), x[hi].std()]); mns.append([x[lo].mean(), x[hi].mean()])
sds, mns = np.array(sds), np.array(mns)
r = (sds[:, 1]/(sds[:, 0]+1e-12)) / (mns[:, 1]/(mns[:, 0]+1e-12))
print(f"   within-coefficient: spread ratio / mean ratio (high vs low half) = {np.median(r):.2f}")
print(f"   ~1 means spread grows in proportion to level, which is what log removes\n")

# ---- B: per-coefficient association with outcome under each transform ----
print("B. per-coefficient association with outcome (no model)")
print(f"   {'transform':10s} {'mean |r|':>9} {'max |r|':>9} {'n with p<0.05':>14}")
for name, f in [('raw', lambda x: x), ('sqrt', np.sqrt), ('cbrt', np.cbrt), ('log', np.log1p)]:
    A = f(MA)
    rs = [pearsonr(A[:, j], bA) for j in range(A.shape[1])]
    r_ = np.array([abs(a) for a, _ in rs]); p_ = np.array([b for _, b in rs])
    print(f"   {name:10s} {r_.mean():9.3f} {r_.max():9.3f} {int((p_ < 0.05).sum()):14d}")
print("\n   if the dose-response is genuinely logarithmic, log maximises these")
