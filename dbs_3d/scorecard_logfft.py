"""Five-test scorecard for `both` + log|FFT| on the corrected 66-patient cohort.

Pre-specified space, fixed before looking at any external number:
  ROI    both (two nuclei concatenated) -- best on the corrected cohort
  basis  log|FFT|, lambda-sorted modes
  K      {8, 16, 32}
  omega  none -- excluded on principle: the clinical covariates already supply
         the structure a rank prior would encode
  -> 3 cells; the cell is chosen on seed-averaged LOO internal AUC only.

Tests
  1 family-wise null over the 6 cells
  2 pre-specified cell, external read once
  3 seed stability over independent initialisations
  4 internal permutation, full LOO refit per permutation
  5 imaging ablation at test time

usage: scorecard_logfft.py [nseeds] [nperm]
"""
import sys, os
SPDIR = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(SPDIR)
OUT = os.path.join(REPO, 'docs', 'dbs_3d')
DATA = os.environ.get('DBS_CLINICAL_DIR', os.path.join(REPO, 'docs'))
CACHE = os.environ.get('DBS_CACHE_DIR', '/checkpoints')
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, REPO); sys.path.insert(0, SPDIR); os.chdir(REPO)
import numpy as np, pandas as pd, torch, torch.nn as nn, itertools, time, pickle, warnings
warnings.filterwarnings('ignore')
from scipy.fft import fftn
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from basis_fix import low_modes, eigvals
from rank_model import RankAttn

NS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
NPERM = int(sys.argv[2]) if len(sys.argv) > 2 else 100
src = open(os.path.join(SPDIR, 'roi_x_models.py')).read().split('IDX = low_modes')[0]
g = {'__file__': os.path.join(SPDIR, 'roi_x_models.py')}
exec(compile(src, 'x', 'exec'), g)
HA, LA, HB, LB = g['HA'], g['LA'], g['HB'], g['LB']
bA, bB, Ca, Cb = g['bA'], g['bB'], g['Ca'], g['Cb']
H, DZ = 10, 8
dev = torch.device('cuda'); T = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
sig = lambda z: 1/(1+np.exp(-z))
t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
print(f"cohort: MSW {len(bA)} ({bA.sum()} resp / {(bA==0).sum()} non-resp)   "
      f"CHH {len(bB)} ({bB.sum()} / {(bB==0).sum()})\n")


def feats(K):
    idx = low_modes(K, (2*H, 2*H, 2*DZ), 'lap')
    lam = eigvals(idx, (2*H, 2*H, 2*DZ))
    lg = lambda V: np.stack([np.log1p(np.abs(fftn(V, axes=(1, 2, 3))))[:, a, b, c] for a, b, c in idx], 1)
    return (np.hstack([lg(HA), lg(LA)]), np.hstack([lg(HB), lg(LB)]), np.r_[lam, lam])


def fit(Xtr, Ctr, ytr, Xte, Cte, lam, w, seed, ep=200):
    torch.manual_seed(seed)
    m = RankAttn(Xtr.shape[1], Ctr.shape[1], lam, weight=w).to(dev)
    opt = torch.optim.AdamW(m.parameters(), 3e-3, weight_decay=0.1); lf = nn.BCEWithLogitsLoss()
    xt, ct = T(Xtr), T(Ctr); yt = torch.tensor(ytr, dtype=torch.float32, device=dev)
    for _ in range(ep):
        opt.zero_grad(); lf(m(xt, ct), yt).backward(); opt.step()
    m.eval()
    with torch.no_grad(): return sig(m(T(Xte), T(Cte)).cpu().numpy()), m


def loo_ext(ZA, ZB, DA, DB, y, lam, w, seed):
    pr = np.empty(len(y))
    for i in range(len(y)):
        tr = np.delete(np.arange(len(y)), i)
        pr[i] = fit(ZA[tr], DA[tr], y[tr].astype(float), ZA[i:i+1], DA[i:i+1], lam, w, seed)[0][0]
    pb, m = fit(ZA, DA, y.astype(float), ZB, DB, lam, w, seed)
    return roc_auc_score(y, pr), pb, m


# omega fixed to none: the clinical covariates supply the structure a rank
# prior would encode, so it is excluded from the space rather than searched
CELLS = list(itertools.product([8, 16, 32], ['none']))
res = {}
print(f"{'K':>3} {'omega':6s} {'LOO int':>9} {'ext':>9}")
for K, w in CELLS:
    XA, XB, lam = feats(K)
    sX = StandardScaler().fit(XA); sC = StandardScaler().fit(Ca)
    ZA, ZB, DA, DB = sX.transform(XA), sX.transform(XB), sC.transform(Ca), sC.transform(Cb)
    ii, ee, pbs = [], [], []
    for sd in range(NS):
        ia, pb, _ = loo_ext(ZA, ZB, DA, DB, bA, lam, w, sd)
        ii.append(ia); ee.append(roc_auc_score(bB, pb)); pbs.append(pb)
    ii, ee = np.array(ii), np.array(ee)
    res[(K, w)] = dict(int_mean=ii.mean(), int_se=ii.std()/np.sqrt(NS), ext_mean=ee.mean(),
                       ext_sd=ee.std(), ens=np.mean(pbs, 0), internal=ii, external=ee)
    print(f"{K:3d} {w:6s} {ii.mean():.3f}+-{ii.std()/np.sqrt(NS):.3f} {ee.mean():.3f}+-{ee.std():.3f}", flush=True)

rng = np.random.RandomState(0); n = len(bB); n1 = int(bB.sum()); n0 = n - n1
ks = list(res); R = np.array([rankdata(res[k]['ens']) for k in ks])
auc = lambda y: (R @ y - n1*(n1+1)/2)/(n1*n0)
obs = auc(bB); mx = np.array([auc(bB[rng.permutation(n)]).max() for _ in range(20000)])
i = int(obs.argmax())
print(f"\nTEST 1 FAMILY-WISE NULL ({len(ks)} cells)")
print(f"  best external {obs[i]:.3f} (K={ks[i][0]}, omega={ks[i][1]})   "
      f"chance {mx.mean():.3f} (95th {np.percentile(mx,95):.3f})   p = {(mx>=obs[i]).mean():.4f}")

best = max(res, key=lambda k: res[k]['int_mean']); r = res[best]
pe = r['ens']; ae = roc_auc_score(bB, pe)
nl = np.array([roc_auc_score(bB[rng.permutation(n)], pe) for _ in range(20000)])
bs = [roc_auc_score(bB[ix], pe[ix]) for ix in rng.randint(0, n, (20000, n)) if len(np.unique(bB[ix])) > 1]
print(f"\nTEST 2 PRE-SPECIFIED CELL (best LOO internal)")
print(f"  cell K={best[0]} omega={best[1]}   internal {r['int_mean']:.3f} +- {r['int_se']:.3f}")
print(f"  external per-seed {r['ext_mean']:.3f} +- {r['ext_sd']:.3f}")
print(f"  seed-ensembled external {ae:.3f}   p = {(nl>=ae).mean():.4f}")
print(f"  bootstrap 95% CI [{np.percentile(bs,2.5):.3f}, {np.percentile(bs,97.5):.3f}]")
srt = sorted(res, key=lambda k: -res[k]['int_mean'])
if len(srt) > 1:
    d = res[srt[0]]['int_mean'] - res[srt[1]]['int_mean']
    se = np.hypot(res[srt[0]]['int_se'], res[srt[1]]['int_se'])
    print(f"  margin over 2nd: {d:+.3f} +- {se:.3f} -> {'separated' if d > 2*se else 'NOT separated'}")

ee = r['external']
print(f"\nTEST 3 SEED STABILITY ({NS} inits)")
print(f"  external {ee.mean():.3f} +- {ee.std():.3f}  min {ee.min():.3f}  max {ee.max():.3f}  below 0.5 in {(ee<0.5).sum()}/{NS}")
print(f"  internal {r['internal'].mean():.3f} +- {r['internal'].std():.3f}")

K, w = best
XA, XB, lam = feats(K)
sX = StandardScaler().fit(XA); sC = StandardScaler().fit(Ca)
ZA, ZB, DA, DB = sX.transform(XA), sX.transform(XB), sC.transform(Ca), sC.transform(Cb)
log(f"TEST 4: {NPERM} permutations, full LOO refit each")
null = []
for j in range(NPERM):
    yp = bA[rng.permutation(len(bA))]
    null.append(loo_ext(ZA, ZB, DA, DB, yp, lam, w, 0)[0])
    if (j+1) % 25 == 0: log(f"   {j+1}/{NPERM}  null mean {np.mean(null):.3f}")
null = np.array(null)
print(f"\nTEST 4 INTERNAL PERMUTATION")
print(f"  observed {r['int_mean']:.3f}   null {null.mean():.3f} +- {null.std():.3f}   p = {(null>=r['int_mean']).mean():.4f}")

res5 = {k: [] for k in ['intact', 'img0', 'imgshuf', 'cov0']}
for sd in range(10):
    _, m = fit(ZA, DA, bA.astype(float), ZB, DB, lam, w, sd)
    rr = np.random.RandomState(sd)
    with torch.no_grad():
        ex = lambda Z, D: roc_auc_score(bB, sig(m(T(Z), T(D)).cpu().numpy()))
        res5['intact'].append(ex(ZB, DB)); res5['img0'].append(ex(np.zeros_like(ZB), DB))
        res5['imgshuf'].append(ex(ZB[rr.permutation(len(ZB))], DB)); res5['cov0'].append(ex(ZB, np.zeros_like(DB)))
print(f"\nTEST 5 IMAGING ABLATION (10 seeds)")
for k, lab in [('intact','intact'), ('img0','imaging zeroed'), ('imgshuf','imaging shuffled'), ('cov0','covariates zeroed')]:
    v = np.array(res5[k]); print(f"  {lab:20s} {v.mean():.3f} +- {v.std():.3f}")
dd = np.array(res5['intact']) - np.array(res5['imgshuf'])
print(f"  imaging contribution (intact - shuffled): {dd.mean():+.3f} +- {dd.std():.3f}")
pickle.dump(dict(res={str(k): {kk: vv for kk, vv in v.items()} for k, v in res.items()},
                 null=null, ablation=res5, best=str(best)), open(os.path.join(OUT, 'scorecard_logfft.pkl'), 'wb'))
log("DONE")
