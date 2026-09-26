"""ROI construction x basis, LOO internal on the corrected 66-patient cohort.

Restoring the 7 edge-clipped patients dropped internal LOO from 0.821 to 0.578
for the higher-side DCT model. This asks whether any ROI construction or basis
handles those cases better -- i.e. whether the internal/external correspondence
survives anywhere.

ROI:    higher, both, mean, diff, and the gradient magnitude of each
bases:  3D DCT, |FFT|, log|FFT|, PCA
LOO throughout, since 4-fold is unusable at 5 non-responders.
"""
import sys, os
SPDIR = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(SPDIR)
OUT = os.path.join(REPO, 'docs', 'dbs_3d')
DATA = os.environ.get('DBS_CLINICAL_DIR', os.path.join(REPO, 'docs'))
CACHE = os.environ.get('DBS_CACHE_DIR', '/checkpoints')
sys.path.insert(0, REPO); sys.path.insert(0, SPDIR); os.chdir(REPO)
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from scipy.fft import dctn, fftn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

src = open(os.path.join(SPDIR, 'roi_x_models.py')).read().split('IDX = low_modes')[0]
g = {'__file__': os.path.join(SPDIR, 'roi_x_models.py')}
exec(compile(src, 'x', 'exec'), g)
HA, LA, HB, LB = g['HA'], g['LA'], g['HB'], g['LB']
bA, bB, Ca, Cb = g['bA'], g['bB'], g['Ca'], g['Cb']
from basis_fix import low_modes
H, DZ, K = 10, 8, 16
IDX = low_modes(K, (2*H, 2*H, 2*DZ), 'lap')
print(f"corrected cohort: MSW {len(bA)} ({bA.sum()}/{(bA==0).sum()})  CHH {len(bB)} ({bB.sum()}/{(bB==0).sum()})\n")


def grad(V):
    gx, gy, gz = np.gradient(V, axis=1), np.gradient(V, axis=2), np.gradient(V, axis=3)
    return np.sqrt(gx**2 + gy**2 + gz**2).astype(np.float32)


ROI = {
    'higher': (HA, HB), 'both': (None, None), 'mean': ((HA+LA)/2, (HB+LB)/2), 'diff': (HA-LA, HB-LB),
    'higher_grad': (grad(HA), grad(HB)), 'diff_grad': (grad(HA)-grad(LA), grad(HB)-grad(LB)),
}


def basis(V, name, fitset=None):
    if name == 'PCA':
        return None
    T = {'DCT': lambda x: dctn(x, axes=(1, 2, 3), norm='ortho'),
         '|FFT|': lambda x: np.abs(fftn(x, axes=(1, 2, 3))),
         'logFFT': lambda x: np.log1p(np.abs(fftn(x, axes=(1, 2, 3))))}[name](V)
    return np.stack([T[:, a, b, c] for a, b, c in IDX], 1)


def loo_auc(Z, y):
    p = np.empty(len(y))
    for i in range(len(y)):
        tr = np.delete(np.arange(len(y)), i)
        m = LogisticRegression(C=1.0, max_iter=3000, class_weight='balanced').fit(Z[tr], y[tr])
        p[i] = m.decision_function(Z[i:i+1])[0]
    return roc_auc_score(y, p), p


def site_auc(A, B):
    X = np.vstack([A, B]); y = np.r_[np.zeros(len(A)), np.ones(len(B))]
    s = StandardScaler().fit_transform(X); o = []
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=0).split(s, y):
        m = LogisticRegression(C=1.0, max_iter=3000).fit(s[tr], y[tr])
        o.append(roc_auc_score(y[va], m.decision_function(s[va])))
    return float(np.mean(o))


rows = []
print(f"{'ROI':12s} {'basis':7s} {'LOO int':>8} {'ext':>7} {'site':>7}  (+clinical covariates)")
for rname, (VA_, VB_) in ROI.items():
    for bname in ['DCT', '|FFT|', 'logFFT', 'PCA']:
        if rname == 'both':
            if bname == 'PCA': continue
            XA = np.hstack([basis(HA, bname), basis(LA, bname)])
            XB = np.hstack([basis(HB, bname), basis(LB, bname)])
        elif bname == 'PCA':
            fa = VA_.reshape(len(VA_), -1); fb = VB_.reshape(len(VB_), -1)
            s1 = StandardScaler().fit(fa); t = PCA(K, random_state=0, whiten=True).fit(s1.transform(fa))
            XA, XB = t.transform(s1.transform(fa)), t.transform(s1.transform(fb))
        else:
            XA, XB = basis(VA_, bname), basis(VB_, bname)
        sa = site_auc(StandardScaler().fit_transform(XA), StandardScaler().fit(XA).transform(XB))
        Zc = np.hstack([XA, Ca]); s = StandardScaler().fit(Zc)
        ia, _ = loo_auc(s.transform(Zc), bA)
        m = LogisticRegression(C=1.0, max_iter=3000, class_weight='balanced').fit(s.transform(Zc), bA)
        ea = roc_auc_score(bB, m.decision_function(s.transform(np.hstack([XB, Cb]))))
        print(f"{rname:12s} {bname:7s} {ia:8.3f} {ea:7.3f} {sa:7.3f}", flush=True)
        rows.append(dict(roi=rname, basis=bname, loo_int=ia, ext=ea, site=sa))
d = pd.DataFrame(rows); d.to_csv(os.path.join(OUT, 'roi_basis_loo.csv'), index=False)
print(f"\nclinical-only reference: LOO int {loo_auc(StandardScaler().fit_transform(Ca), bA)[0]:.3f}")
print("\nbest by LOO internal:"); print(d.nlargest(5, 'loo_int').round(3).to_string(index=False))
