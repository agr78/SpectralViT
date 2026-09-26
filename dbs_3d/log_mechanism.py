"""Is log|FFT|'s advantage over |FFT| a conditioning effect?

Both are translation-invariant, so the shift mechanism cannot explain why
log|FFT| (LOO 0.800 / ext 0.892) beats |FFT| (0.613 / 0.804). The proposed
explanation is conditioning: magnitude spectra are heavy-tailed with DC
dominating, and log1p makes the feature distribution closer to Gaussian for a
linear readout.

That predicts other variance-stabilising transforms of the same coefficients
should behave like log. Each transform below is a pointwise function of |FFT|,
so all share its invariances and differ only in distribution shape:

  raw        |F|
  sqrt       sqrt(|F|)
  log        log1p(|F|)          (what we use)
  cbrt       |F|^(1/3)
  ranknorm   rank -> normal quantiles, per feature
  boxcox     Yeo-Johnson, per feature

Reported alongside skewness, kurtosis and the design-matrix condition number,
all computed after the standardisation the models actually use.
"""
import sys, os
SPDIR = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(SPDIR)
OUT = os.path.join(REPO, 'docs', 'dbs_3d')
DATA = os.environ.get('DBS_CLINICAL_DIR', os.path.join(REPO, 'docs'))
CACHE = os.environ.get('DBS_CACHE_DIR', '/checkpoints')
sys.path.insert(0, REPO); sys.path.insert(0, SPDIR); os.chdir(REPO)
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from scipy.fft import fftn
from scipy.stats import skew, kurtosis, norm, rankdata
from sklearn.preprocessing import StandardScaler, PowerTransformer, QuantileTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from basis_fix import low_modes

src = open(os.path.join(SPDIR, 'roi_x_models.py')).read().split('IDX = low_modes')[0]
g = {'__file__': os.path.join(SPDIR, 'roi_x_models.py')}
exec(compile(src, 'x', 'exec'), g)
HA, LA, HB, LB, bA, bB, Ca, Cb = g['HA'], g['LA'], g['HB'], g['LB'], g['bA'], g['bB'], g['Ca'], g['Cb']
H, DZ, K = 10, 8, 16
IDX = low_modes(K, (2*H, 2*H, 2*DZ), 'lap')
mag = lambda V: np.stack([np.abs(fftn(V, axes=(1, 2, 3)))[:, a, b, c] for a, b, c in IDX], 1)
MA = np.hstack([mag(HA), mag(LA)]); MB = np.hstack([mag(HB), mag(LB)])   # 'both' ROI


def tf(name, A, B):
    if name == 'raw':   return A, B
    if name == 'sqrt':  return np.sqrt(A), np.sqrt(B)
    if name == 'log':   return np.log1p(A), np.log1p(B)
    if name == 'cbrt':  return np.cbrt(A), np.cbrt(B)
    if name == 'ranknorm':
        f = lambda X, R: np.stack([norm.ppf((np.searchsorted(np.sort(R[:, j]), X[:, j], 'left') + 0.5)/(len(R)+1))
                                   for j in range(X.shape[1])], 1)
        return f(A, A), f(B, A)
    if name == 'boxcox':
        p = PowerTransformer(method='yeo-johnson').fit(A)
        return p.transform(A), p.transform(B)


def loo(Z, y):
    p = np.empty(len(y))
    for i in range(len(y)):
        tr = np.delete(np.arange(len(y)), i)
        m = LogisticRegression(C=1.0, max_iter=3000, class_weight='balanced').fit(Z[tr], y[tr])
        p[i] = m.decision_function(Z[i:i+1])[0]
    return roc_auc_score(y, p)


print(f"'both' ROI, |FFT| coefficients under pointwise transforms (all equally shift-invariant)")
print(f"MSW {MA.shape}  CHH {MB.shape}\n")
print(f"{'transform':10s} {'LOO int':>8} {'ext':>7} {'|skew|':>8} {'kurtosis':>9} {'cond':>9}")
rows = []
for name in ['raw', 'sqrt', 'cbrt', 'log', 'ranknorm', 'boxcox']:
    A, B = tf(name, MA, MB)
    s = StandardScaler().fit(A); ZA, ZB = s.transform(A), s.transform(B)
    sk = float(np.abs(skew(ZA, axis=0)).mean()); ku = float(kurtosis(ZA, axis=0).mean())
    cond = float(np.linalg.cond(ZA))
    Zc = np.hstack([ZA, StandardScaler().fit_transform(Ca)])
    ia = loo(Zc, bA)
    m = LogisticRegression(C=1.0, max_iter=3000, class_weight='balanced').fit(Zc, bA)
    sC = StandardScaler().fit(Ca)
    ea = roc_auc_score(bB, m.decision_function(np.hstack([ZB, sC.transform(Cb)])))
    print(f"{name:10s} {ia:8.3f} {ea:7.3f} {sk:8.2f} {ku:9.2f} {cond:9.1f}")
    rows.append(dict(transform=name, loo_int=ia, ext=ea, skew=sk, kurt=ku, cond=cond))
d = pd.DataFrame(rows); d.to_csv(os.path.join(OUT, 'log_mechanism.csv'), index=False)
from scipy.stats import pearsonr
print(f"\ncorr(|skew|, LOO internal) = {pearsonr(d.skew, d.loo_int)[0]:+.3f}")
print(f"corr(kurtosis, LOO internal) = {pearsonr(d.kurt, d.loo_int)[0]:+.3f}")
print("if conditioning is the mechanism, the variance-stabilising transforms")
print("(sqrt, cbrt, log, ranknorm, boxcox) should cluster together and beat raw.")
