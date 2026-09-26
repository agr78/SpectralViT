"""Does log|FFT| work because the model is computing spectral RATIOS?

A weighted sum of log-coefficients is the log of a product of powers:
    sum_i w_i log x_i = log( prod_i x_i^{w_i} )
If sum_i w_i = 0 that quantity is invariant to any overall rescaling of the
spectrum -- it depends only on the RATIOS between modes, i.e. spectral shape.

So if the ratio hypothesis is right, a model fitted on log features should place
weights that roughly cancel. Compared against the same model on raw |FFT| and on
sqrt, where no such interpretation applies.

Also fits a model restricted to sum-to-zero weights: if that matches the
unrestricted fit, the model was only ever using shape.
"""
import sys, os
SPDIR = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(SPDIR)
DATA = os.environ.get('DBS_CLINICAL_DIR', os.path.join(REPO, 'docs'))
CACHE = os.environ.get('DBS_CACHE_DIR', '/checkpoints')
sys.path.insert(0, REPO); sys.path.insert(0, SPDIR); os.chdir(REPO)
import numpy as np, warnings
warnings.filterwarnings('ignore')
from scipy.fft import fftn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from basis_fix import low_modes

src = open(os.path.join(SPDIR, 'roi_x_models.py')).read().split('IDX = low_modes')[0]
g = {'__file__': os.path.join(SPDIR, 'roi_x_models.py')}
exec(compile(src, 'x', 'exec'), g)
HA, LA, HB, LB, bA, bB = g['HA'], g['LA'], g['HB'], g['LB'], g['bA'], g['bB']
H, DZ, K = 10, 8, 16
IDX = low_modes(K, (2*H, 2*H, 2*DZ), 'lap')
mag = lambda V: np.stack([np.abs(fftn(V, axes=(1, 2, 3)))[:, a, b, c] for a, b, c in IDX], 1)
MA = np.hstack([mag(HA), mag(LA)]); MB = np.hstack([mag(HB), mag(LB)])


def loo(Z, y):
    p = np.empty(len(y))
    for i in range(len(y)):
        tr = np.delete(np.arange(len(y)), i)
        m = LogisticRegression(C=1.0, max_iter=3000, class_weight='balanced').fit(Z[tr], y[tr])
        p[i] = m.decision_function(Z[i:i+1])[0]
    return roc_auc_score(y, p)


print("weights fitted on standardised features; 'raw-scale' weights are the")
print("coefficients that act on the untransformed log values\n")
print(f"{'transform':10s} {'LOO int':>8} {'ext':>7} {'sum w':>9} {'sum|w|':>8} {'|sum w|/sum|w|':>15}")
for name, f in [('raw', lambda X: X), ('sqrt', np.sqrt), ('log', np.log1p)]:
    A, B = f(MA), f(MB)
    s = StandardScaler().fit(A); ZA, ZB = s.transform(A), s.transform(B)
    m = LogisticRegression(C=1.0, max_iter=3000, class_weight='balanced').fit(ZA, bA)
    w = m.coef_[0] / s.scale_                       # weights acting on the transformed values
    ia = loo(ZA, bA); ea = roc_auc_score(bB, m.decision_function(ZB))
    print(f"{name:10s} {ia:8.3f} {ea:7.3f} {w.sum():9.4f} {np.abs(w).sum():8.4f} "
          f"{abs(w.sum())/np.abs(w).sum():15.4f}")

# a model constrained to sum-to-zero weights on the log features:
# project the design matrix onto the subspace orthogonal to the all-ones vector
A, B = np.log1p(MA), np.log1p(MB)
s = StandardScaler().fit(A); ZA, ZB = s.transform(A), s.transform(B)
one = np.ones(ZA.shape[1]) / np.sqrt(ZA.shape[1])
PA_ = ZA - np.outer(ZA @ one, one); PB_ = ZB - np.outer(ZB @ one, one)
m = LogisticRegression(C=1.0, max_iter=3000, class_weight='balanced').fit(PA_, bA)
print(f"\nlog features, weights CONSTRAINED to sum to zero (pure spectral shape):")
print(f"   LOO int {loo(PA_, bA):.3f}   ext {roc_auc_score(bB, m.decision_function(PB_)):.3f}")
print("   if this matches the unconstrained log fit, the model was only using ratios")
# and the complementary part: the overall spectral level alone
LA_ = (ZA @ one).reshape(-1, 1); LB_ = (ZB @ one).reshape(-1, 1)
m = LogisticRegression(C=1.0, max_iter=3000, class_weight='balanced').fit(LA_, bA)
print(f"\nthe discarded component alone (overall spectral level, 1 feature):")
print(f"   LOO int {loo(LA_, bA):.3f}   ext {roc_auc_score(bB, m.decision_function(LB_)):.3f}")
