"""Harmonization that acts on the joint structure, not just the marginals.

The imaging features identify the site at AUC 0.940 while predicting outcome
externally at 0.729 -- site is the dominant axis. Per-feature corrections
(z-score, ComBat, quantile) leave the correlation structure untouched, which is
where site can hide. These act on second-order structure instead:

  coral      whiten CHH by its own covariance, recolour with MSW's
  sa         subspace alignment: rotate CHH's principal subspace onto MSW's
  coral_reg  CORAL with shrinkage, for the n < p regime
  whiten     whiten both sites independently (destroys shared scale too)

Each is scored on two axes:
  site-AUC   how separable the sites remain  (target 0.5)
  ext-AUC    outcome performance on CHH      (target: beat clinical 0.725)

No outcome labels are used by any harmonization. CHH inputs are used, which is
transductive but standard for domain adaptation, and is stated as such.
"""
import sys, os
SPDIR = os.path.dirname(os.path.abspath(__file__))
REPO  = os.path.dirname(SPDIR)
OUT   = os.path.join(REPO, 'docs', 'dbs_3d')
DATA  = os.environ.get('DBS_CLINICAL_DIR', os.path.join(REPO, 'docs'))
CACHE = os.environ.get('DBS_CACHE_DIR', '/checkpoints')
sys.path.insert(0, REPO); sys.path.insert(0, SPDIR); os.chdir(REPO)
import numpy as np, pandas as pd, torch, torch.nn as nn, pickle, warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.linalg import sqrtm

NS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
exec(open(os.path.join(SPDIR, 'concat_attn.py')).read().split("rows=[]")[0])
from basis_fix import basis3_iso, low_modes, eigvals
from rank_model import RankAttn

dev = torch.device('cuda'); T = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
SHAPE = (BB[0].stop-BB[0].start, BB[1].stop-BB[1].start, BB[2].stop-BB[2].start)
sig = lambda z: 1.0/(1.0+np.exp(-z))
K = 16
IDX = low_modes(K, SHAPE, 'lap'); LAM = eigvals(IDX, SHAPE)
A0 = basis3_iso(VA, '3DDCT', K, BB, 'lap')
B0 = basis3_iso(VB, '3DDCT', K, BB, 'lap')


def _cov(X, reg):
    C = np.cov(X, rowvar=False)
    return C + reg * np.trace(C) / len(C) * np.eye(len(C))


def harm(A, B, mode):
    A = A.copy(); B = B.copy()
    if mode == 'none':
        return A, B
    if mode == 'zscore':
        return ((A - A.mean(0)) / (A.std(0) + 1e-8), (B - B.mean(0)) / (B.std(0) + 1e-8))
    if mode in ('coral', 'coral_reg'):
        reg = 1e-3 if mode == 'coral' else 0.2
        ma, mb = A.mean(0), B.mean(0)
        Ca_, Cb_ = _cov(A, reg), _cov(B, reg)
        W = np.real(sqrtm(np.linalg.inv(np.real(sqrtm(Cb_)))) if False else
                    np.real(sqrtm(Ca_)) @ np.linalg.inv(np.real(sqrtm(Cb_))))
        return A, (B - mb) @ W.T + ma
    if mode == 'sa':                       # subspace alignment
        from sklearn.decomposition import PCA
        d = min(8, A.shape[1] - 1)
        pa = PCA(d).fit(A - A.mean(0)); pb = PCA(d).fit(B - B.mean(0))
        M = pb.components_ @ pa.components_.T
        return (A - A.mean(0)) @ pa.components_.T, (B - B.mean(0)) @ pb.components_.T @ M
    if mode == 'whiten':
        wa = np.linalg.inv(np.real(sqrtm(_cov(A, 1e-2))))
        wb = np.linalg.inv(np.real(sqrtm(_cov(B, 1e-2))))
        return (A - A.mean(0)) @ wa.T, (B - B.mean(0)) @ wb.T
    raise ValueError(mode)


def site_auc(A, B):
    X = np.vstack([A, B]); y = np.r_[np.zeros(len(A)), np.ones(len(B))]
    s = StandardScaler().fit_transform(X)
    out = []
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=0).split(s, y):
        m = LogisticRegression(C=1.0, max_iter=3000).fit(s[tr], y[tr])
        out.append(roc_auc_score(y[va], m.decision_function(s[va])))
    return float(np.mean(out))


def gfit(Xtr, Ctr, ytr, Xte, Cte, seed, ep=200):
    torch.manual_seed(seed)
    m = RankAttn(Xtr.shape[1], Ctr.shape[1], LAM[:Xtr.shape[1]], weight='lap').to(dev)
    opt = torch.optim.AdamW(m.parameters(), 3e-3, weight_decay=0.1)
    lf = nn.BCEWithLogitsLoss()
    xt, ct, yt = T(Xtr), T(Ctr), T(ytr)
    for _ in range(ep):
        opt.zero_grad(); lf(m(xt, ct), yt).backward(); opt.step()
    m.eval()
    with torch.no_grad(): return sig(m(T(Xte), T(Cte)).cpu().numpy())


sC = StandardScaler().fit(Ca); DA, DB = sC.transform(Ca), sC.transform(Cb)
rows = []
for mode in ['none', 'zscore', 'coral', 'coral_reg', 'sa', 'whiten']:
    try:
        A, B = harm(A0, B0, mode)
    except Exception as e:
        print(f"{mode}: failed ({e})"); continue
    sA = StandardScaler().fit(A); ZA, ZB = sA.transform(A), sA.transform(B)
    sauc = site_auc(ZA, ZB)
    ii, ee = [], []
    for sd in range(NS):
        pr = np.full(len(bA), np.nan)
        for tr, va in StratifiedKFold(4, shuffle=True, random_state=0).split(ZA, bA):
            s2 = StandardScaler().fit(ZA[tr]); s3 = StandardScaler().fit(DA[tr])
            pr[va] = gfit(s2.transform(ZA[tr]), s3.transform(DA[tr]), bA[tr].astype(float),
                          s2.transform(ZA[va]), s3.transform(DA[va]), sd)
        s2 = StandardScaler().fit(ZA); s3 = StandardScaler().fit(DA)
        pb = gfit(s2.transform(ZA), s3.transform(DA), bA.astype(float),
                  s2.transform(ZB), s3.transform(DB), sd)
        ii.append(roc_auc_score(bA, pr)); ee.append(roc_auc_score(bB, pb))
    rows.append(dict(harm=mode, site_auc=sauc, int_auc=np.mean(ii), int_sd=np.std(ii),
                     ext_auc=np.mean(ee), ext_sd=np.std(ee)))
    print(f"  {mode:10s} site-AUC {sauc:.3f}   internal {np.mean(ii):.3f}+-{np.std(ii):.3f}   "
          f"external {np.mean(ee):.3f}+-{np.std(ee):.3f}", flush=True)

df = pd.DataFrame(rows)
print("\n=== joint-structure harmonization ===")
print(df.round(3).to_string(index=False))
print("\nreference: clinical-only external AUC 0.725;  unharmonised imaging site-AUC 0.940")
df.to_csv(os.path.join(OUT, 'harmonize2.csv'), index=False)
