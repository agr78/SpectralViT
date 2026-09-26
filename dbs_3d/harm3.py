"""Harmonisation estimated from only the three held-out CHH cases.

Subjects 12, 15 and 17 have no usable segmentation, so they are excluded from
every evaluation. That makes them the one part of CHH that can be used to
estimate a site correction without leaking anything into the test set.

Compared against:
  none          no correction
  harm3-mean    location shift estimated from the 3 held-out cases only  (leak-free)
  harm3-ms      location + scale from the 3                              (leak-free)
  all37-mean    location shift from the 37 test patients                 (transductive)
  all37-coral   CORAL from the 37 test patients                          (transductive)

The transductive rows are an upper bound on what harmonisation could buy, not a
usable result. Three samples give a poor covariance estimate, so only location
and per-feature scale are attempted from them.
"""
import sys, os
SPDIR = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(SPDIR)
OUT = os.path.join(REPO, 'docs', 'dbs_3d')
DATA = os.environ.get('DBS_CLINICAL_DIR', os.path.join(REPO, 'docs'))
CACHE = os.environ.get('DBS_CACHE_DIR', '/checkpoints')
sys.path.insert(0, REPO); sys.path.insert(0, SPDIR); os.chdir(REPO)
import numpy as np, pandas as pd, warnings, pickle
warnings.filterwarnings('ignore')
from scipy import ndimage
from scipy.fft import dctn
from scipy.linalg import sqrtm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from reader import prepare_qsm_dataset
from util import mask_crop
from basis_fix import low_modes
from _clin import clin_tables, FULL

H, DZ, K = 10, 8, 16
IDX = low_modes(K, (2*H, 2*H, 2*DZ), 'lap')
mcf = lambda d, m, p: mask_crop(d, m, p)
src = open(os.path.join(SPDIR, 'roi_x_models.py')).read().split('IDX = low_modes')[0]
g = {'__file__': os.path.join(SPDIR, 'roi_x_models.py')}
exec(compile(src, 'x', 'exec'), g)
HA, LA, HB, LB, bA, bB, Ca, Cb = g['HA'], g['LA'], g['HB'], g['LB'], g['bA'], g['bB'], g['Ca'], g['Cb']

# the three held-out CHH cases, cropped at the correct side via the two cluster centres
dsB = prepare_qsm_dataset('CHH', '/x', '/y', os.path.join(DATA, 'chh_subjects_table1_20240729.csv'),
                          os.path.join(CACHE, 'chh_cache_6d_cv.pt'), load_cache=True, mask_crop_fn=mcf, cv_pad=False)
cents = []
for s in sorted(dsB.volumes):
    s = int(s); v = np.asarray(dsB.volumes[s], np.float32); m = np.asarray(dsB.seg_masks[s]) > 0
    lb, n = ndimage.label(m)
    if n < 2 or m.sum() > 0.5*m.size: continue
    sz = ndimage.sum(m, lb, range(1, n+1)); big = np.argsort(sz)[::-1][:2] + 1
    info = sorted([(float(v[lb == b].mean()), b) for b in big], key=lambda t: t[0])
    cents.append([int(round(x)) for x in ndimage.center_of_mass(m, lb, [info[1][1]])[0]])
cents = np.array(cents); xs = cents[:, 0]
gap = np.argmax(np.diff(np.sort(xs))); thr = np.sort(xs)[gap]
cA = np.round(cents[xs <= thr].mean(0)).astype(int)
cB = np.round(cents[xs > thr].mean(0)).astype(int)


def crop(v, c):
    x0 = min(max(0, c[0]-H), v.shape[0]-2*H); y0 = min(max(0, c[1]-H), v.shape[1]-2*H)
    z0 = min(max(0, c[2]-DZ), v.shape[2]-2*DZ)
    return v[x0:x0+2*H, y0:y0+2*H, z0:z0+2*DZ]


held = []
for s in [12, 15, 17]:

    v = np.asarray(dsB.volumes[s], np.float32)
    hi = max([crop(v, cA), crop(v, cB)], key=lambda c: float(c.mean()))
    lo = min([crop(v, cA), crop(v, cB)], key=lambda c: float(c.mean()))
    held.append((hi, lo))
HH = np.array([h for h, _ in held], np.float32); HL = np.array([l for _, l in held], np.float32)
print(f"held-out CHH cases recovered: {HH.shape[0]}  (subjects 12, 15, 17)")

f = lambda V: np.stack([dctn(V, axes=(1, 2, 3), norm='ortho')[:, a, b, c] for a, b, c in IDX], 1)
XA = np.hstack([f(HA), f(LA)])            # 'both' ROI, the best construction
XB = np.hstack([f(HB), f(LB)])
XH = np.hstack([f(HH), f(HL)])            # the three held-out cases
print(f"MSW {XA.shape}  CHH-test {XB.shape}  CHH-heldout {XH.shape}\n")


def apply_harm(A, B, ref, mode):
    if mode == 'none': return A, B
    if mode.endswith('mean'):
        return A, B - ref.mean(0) + A.mean(0)
    if mode.endswith('ms'):
        return A, (B - ref.mean(0)) / (ref.std(0) + 1e-8) * A.std(0) + A.mean(0)
    if mode.endswith('coral'):
        ca = np.cov(A, rowvar=False) + 1e-3*np.eye(A.shape[1])
        cb = np.cov(ref, rowvar=False) + 1e-3*np.eye(A.shape[1])
        W = np.real(sqrtm(ca)) @ np.linalg.inv(np.real(sqrtm(cb)))
        return A, (B - ref.mean(0)) @ W.T + A.mean(0)
    raise ValueError(mode)


def site_auc(A, B):
    X = np.vstack([A, B]); y = np.r_[np.zeros(len(A)), np.ones(len(B))]
    s = StandardScaler().fit_transform(X); o = []
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=0).split(s, y):
        m = LogisticRegression(C=1.0, max_iter=3000).fit(s[tr], y[tr])
        o.append(roc_auc_score(y[va], m.decision_function(s[va])))
    return float(np.mean(o))


def loo(Z, y):
    p = np.empty(len(y))
    for i in range(len(y)):
        tr = np.delete(np.arange(len(y)), i)
        m = LogisticRegression(C=1.0, max_iter=3000, class_weight='balanced').fit(Z[tr], y[tr])
        p[i] = m.decision_function(Z[i:i+1])[0]
    return p


print(f"{'harmonisation':16s} {'source':12s} {'LOO int':>8} {'ext':>7} {'site':>7}")
rows = []
for mode, ref, lab in [('none', None, '-'),
                       ('harm3-mean', XH, '3 held-out'), ('harm3-ms', XH, '3 held-out'),
                       ('all37-mean', XB, '37 test'), ('all37-coral', XB, '37 test')]:
    A, B = apply_harm(XA, XB, ref if ref is not None else XB, mode)
    sa = site_auc(StandardScaler().fit_transform(A), StandardScaler().fit(A).transform(B))
    Zc = np.hstack([A, Ca]); s = StandardScaler().fit(Zc)
    ia = roc_auc_score(bA, loo(s.transform(Zc), bA))
    m = LogisticRegression(C=1.0, max_iter=3000, class_weight='balanced').fit(s.transform(Zc), bA)
    ea = roc_auc_score(bB, m.decision_function(s.transform(np.hstack([B, Cb]))))
    print(f"{mode:16s} {lab:12s} {ia:8.3f} {ea:7.3f} {sa:7.3f}", flush=True)
    rows.append(dict(mode=mode, source=lab, loo_int=ia, ext=ea, site=sa))
pd.DataFrame(rows).to_csv(os.path.join(OUT, 'harm3.csv'), index=False)
print("\nthe all37 rows use the test patients to fit the correction and are an upper")
print("bound, not a usable result. the harm3 rows are leak-free.")
