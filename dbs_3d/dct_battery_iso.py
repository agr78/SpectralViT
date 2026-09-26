"""Five-test battery on the 3D DCT basis, with isotropic mode selection.

Compares the original row-major truncation (which keeps only fx==0 modes)
against isotropic low-frequency selection, then runs the full battery on
whichever cell wins on seed-averaged internal AUC.
"""
import sys, os
SPDIR = os.path.dirname(os.path.abspath(globals().get('__file__', '/workspace/dbs_3d/x')))
REPO  = os.path.dirname(SPDIR)
OUT   = os.path.join(REPO, 'docs', 'dbs_3d')
DATA  = os.environ.get('DBS_CLINICAL_DIR', os.path.join(REPO, 'docs'))
CACHE = os.environ.get('DBS_CACHE_DIR', '/checkpoints')
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, REPO); sys.path.insert(0, SPDIR); os.chdir(REPO)
import numpy as np, pandas as pd, torch, torch.nn as nn, time, pickle, warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
exec(open(os.path.join(SPDIR, 'concat_attn.py')).read().split("rows=[]")[0])
from basis_fix import basis3_iso, low_modes, eigvals

dev = torch.device('cuda'); T = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
NS = 20

def feats(K, sel):
    if sel == 'iso':
        return basis3_iso(VA, '3DDCT', K, BB), basis3_iso(VB, '3DDCT', K, BB)
    return basis3(VA, '3DDCT', K), basis3(VB, '3DDCT', K)

def gfit(Xtr, Ctr, ytr, Xte, Cte, seed, ep=1500):
    torch.manual_seed(seed)
    m = ConcatAttn(Xtr.shape[1], Ctr.shape[1], typ=False, resid=True).to(dev)
    opt = torch.optim.AdamW(m.parameters(), 3e-3, weight_decay=0.1)
    lf = nn.BCEWithLogitsLoss()
    xt, ct, yt = T(Xtr), T(Ctr), T(ytr)
    for _ in range(ep):
        opt.zero_grad(); lf(m(xt, ct), yt).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        return m(T(Xte), T(Cte)).cpu().numpy(), m

def oof_ext(X0, X1, y, seed):
    pr = np.full(len(y), np.nan)
    for tr, va in StratifiedKFold(4, shuffle=True, random_state=0).split(X0, y):
        s2 = StandardScaler().fit(X0[tr]); s3 = StandardScaler().fit(Ca[tr])
        pr[va], _ = gfit(s2.transform(X0[tr]), s3.transform(Ca[tr]), y[tr].astype(float),
                         s2.transform(X0[va]), s3.transform(Ca[va]), seed)
    s2 = StandardScaler().fit(X0); s3 = StandardScaler().fit(Ca)
    pb, m = gfit(s2.transform(X0), s3.transform(Ca), y.astype(float),
                 s2.transform(X1), s3.transform(Cb), seed)
    return roc_auc_score(y, pr), pb, m

# ---------- the search space: DCT, two mode-selection schemes, two K ----------
rows, PRED, SEEDPB = [], {}, {}
for sel in ['rowmajor', 'iso']:
    for K in [8, 16]:
        X0, X1 = feats(K, sel)
        ii, ee, pbs = [], [], []
        for sd in range(NS):
            ai, pb, _ = oof_ext(X0, X1, bA, sd)
            ii.append(ai); ee.append(roc_auc_score(bB, pb)); pbs.append(pb)
        ii, ee = np.array(ii), np.array(ee)
        rows.append(dict(sel=sel, K=K, int_mean=ii.mean(), int_se=ii.std()/np.sqrt(NS),
                         int_sd=ii.std(), ext_mean=ee.mean(), ext_sd=ee.std(),
                         ens=roc_auc_score(bB, np.mean(pbs, 0))))
        PRED[(sel, K)] = np.mean(pbs, 0); SEEDPB[(sel, K)] = pbs
        log(f"{sel:9s} K={K:2d}: int {ii.mean():.3f}+-{ii.std()/np.sqrt(NS):.3f}  ext {ee.mean():.3f}+-{ee.std():.3f}")

df = pd.DataFrame(rows).sort_values('int_mean', ascending=False)
df.to_csv(os.path.join(OUT, 'dct_battery_iso.csv'), index=False)
print("\n=== DCT: row-major (fx==0 only) vs isotropic mode selection, 20 seeds ===")
print(df.round(3).to_string(index=False))

rng = np.random.RandomState(0); n = len(bB); n1 = int(bB.sum()); n0 = n - n1

ks = list(PRED); R = np.array([rankdata(PRED[k]) for k in ks])
auc = lambda y: (R @ y - n1*(n1+1)/2) / (n1*n0)
obs = auc(bB); mx = np.array([auc(bB[rng.permutation(n)]).max() for _ in range(20000)])
i = int(obs.argmax())
print(f"\nTEST 1 FAMILY-WISE NULL ({len(ks)} cells)")
print(f"  best external {obs[i]:.3f} ({ks[i]})   chance {mx.mean():.3f} (95th {np.percentile(mx,95):.3f})   p = {(mx>=obs[i]).mean():.4f}")

b = df.iloc[0]; key = (b['sel'], int(b['K'])); pe = PRED[key]; ae = roc_auc_score(bB, pe)
nl = np.array([roc_auc_score(bB[rng.permutation(n)], pe) for _ in range(20000)])
bs = [roc_auc_score(bB[ix], pe[ix]) for ix in rng.randint(0, n, (20000, n)) if len(np.unique(bB[ix])) > 1]
print(f"\nTEST 2 PRE-SPECIFIED (best seed-averaged internal)")
print(f"  cell: {b['sel']} K={int(b['K'])}   internal {b['int_mean']:.3f} +- {b['int_se']:.3f}")
print(f"  external per-seed {b['ext_mean']:.3f} +- {b['ext_sd']:.3f}")
print(f"  seed-ensembled external {ae:.3f}   p = {(nl>=ae).mean():.4f}   bootstrap CI [{np.percentile(bs,2.5):.3f},{np.percentile(bs,97.5):.3f}]")
d = df.iloc[0]['int_mean'] - df.iloc[1]['int_mean']
se = np.hypot(df.iloc[0]['int_se'], df.iloc[1]['int_se'])
print(f"  margin over 2nd ({df.iloc[1]['sel']} K={int(df.iloc[1]['K'])}): {d:+.3f} +- {se:.3f} -> {'separated' if d > 2*se else 'NOT separated'}")

ee = np.array([roc_auc_score(bB, p) for p in SEEDPB[key]])
print(f"\nTEST 3 SEED STABILITY ({NS} inits)")
print(f"  external {ee.mean():.3f} +- {ee.std():.3f}   min {ee.min():.3f}  max {ee.max():.3f}   below 0.5 in {(ee<0.5).sum()}/{NS}")

X0, X1 = feats(int(b['K']), b['sel'])
obs_int = b['int_mean']
log("TEST 4: 200 label permutations")
null = []
for j in range(200):
    yp = bA[rng.permutation(len(bA))]
    null.append(oof_ext(X0, X1, yp, 0)[0])
    if (j+1) % 50 == 0: log(f"   {j+1}/200 null mean {np.mean(null):.3f}")
null = np.array(null)
print(f"\nTEST 4 INTERNAL PERMUTATION")
print(f"  seed-averaged internal {obs_int:.3f}   null {null.mean():.3f} +- {null.std():.3f}   p = {(null>=obs_int).mean():.4f}")

s2 = StandardScaler().fit(X0); s3 = StandardScaler().fit(Ca)
ZB, DB = s2.transform(X1), s3.transform(Cb)
res = {k: [] for k in ['intact', 'img0', 'imgshuf', 'cov0']}
for sd in range(5):
    _, _, m = gfit(s2.transform(X0), s3.transform(Ca), bA.astype(float), ZB, DB, sd)
    r = np.random.RandomState(sd)
    with torch.no_grad():
        ex = lambda Z, D: roc_auc_score(bB, m(T(Z), T(D)).cpu().numpy())
        res['intact'].append(ex(ZB, DB)); res['img0'].append(ex(np.zeros_like(ZB), DB))
        res['imgshuf'].append(ex(ZB[r.permutation(len(ZB))], DB)); res['cov0'].append(ex(ZB, np.zeros_like(DB)))
print(f"\nTEST 5 IMAGING ABLATION (5 seeds)")
for k, lab in [('intact','intact'), ('img0','imaging zeroed'), ('imgshuf','imaging shuffled'), ('cov0','covariates zeroed')]:
    v = np.array(res[k]); print(f"  {lab:20s} {v.mean():.3f} +- {v.std():.3f}")
dd = np.array(res['intact']) - np.array(res['imgshuf'])
print(f"  imaging contribution (intact - shuffled): {dd.mean():+.3f} +- {dd.std():.3f}")

pickle.dump(dict(PRED=PRED, bB=bB, df=df), open(os.path.join(OUT, 'dct_battery_iso.pkl'), 'wb'))
log("DONE")
