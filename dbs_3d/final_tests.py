"""Tests 4 and 5 on one selected cell: internal permutation and imaging ablation.

usage: final_tests.py <ordering> <K> <weight> [nperm]
"""
import sys, os
SPDIR = os.path.dirname(os.path.abspath(__file__))
REPO  = os.path.dirname(SPDIR)
OUT   = os.path.join(REPO, 'docs', 'dbs_3d')
DATA  = os.environ.get('DBS_CLINICAL_DIR', os.path.join(REPO, 'docs'))
CACHE = os.environ.get('DBS_CACHE_DIR', '/checkpoints')
sys.path.insert(0, REPO); sys.path.insert(0, SPDIR); os.chdir(REPO)
import numpy as np, torch, torch.nn as nn, pickle, time, warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

ordering, K, weight = sys.argv[1], int(sys.argv[2]), sys.argv[3]
NPERM = int(sys.argv[4]) if len(sys.argv) > 4 else 200
exec(open(os.path.join(SPDIR, 'concat_attn.py')).read().split("rows=[]")[0])
from basis_fix import basis3_iso, low_modes, eigvals
from rank_model import RankAttn

dev = torch.device('cuda'); T = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
SHAPE = (BB[0].stop-BB[0].start, BB[1].stop-BB[1].start, BB[2].stop-BB[2].start)


def make_feats(K, ordering):
    if ordering == 'rowmajor':
        k = max(1, int(round(K ** (1/3)))); n = k + 1
        idx = [(a, b, c) for a in range(n) for b in range(n) for c in range(n)][:K]
        from scipy.fft import dctn
        TA = dctn(VA[:, BB[0], BB[1], BB[2]], axes=(1, 2, 3), norm='ortho')
        TB = dctn(VB[:, BB[0], BB[1], BB[2]], axes=(1, 2, 3), norm='ortho')
        A = np.stack([TA[:, a, b, c] for a, b, c in idx], 1)
        B = np.stack([TB[:, a, b, c] for a, b, c in idx], 1)
    else:
        o = 'l1' if ordering == 'l1' else 'lap'
        idx = low_modes(K, SHAPE, o)
        A = basis3_iso(VA, '3DDCT', K, BB, o); B = basis3_iso(VB, '3DDCT', K, BB, o)
    return A, B, eigvals(idx, SHAPE)


def gfit(Xtr, Ctr, ytr, Xte, Cte, lam, seed, ep=1500):
    torch.manual_seed(seed)
    m = RankAttn(Xtr.shape[1], Ctr.shape[1], lam, weight=weight).to(dev)
    opt = torch.optim.AdamW(m.parameters(), 3e-3, weight_decay=0.1)
    lf = nn.BCEWithLogitsLoss()
    xt, ct, yt = T(Xtr), T(Ctr), T(ytr)
    for _ in range(ep):
        opt.zero_grad(); lf(m(xt, ct), yt).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        return m(T(Xte), T(Cte)).cpu().numpy(), m


def oof(X0, X1, y, lam, seed):
    pr = np.full(len(y), np.nan)
    for tr, va in StratifiedKFold(4, shuffle=True, random_state=0).split(X0, y):
        s2 = StandardScaler().fit(X0[tr]); s3 = StandardScaler().fit(Ca[tr])
        pr[va], _ = gfit(s2.transform(X0[tr]), s3.transform(Ca[tr]), y[tr].astype(float),
                         s2.transform(X0[va]), s3.transform(Ca[va]), lam, seed)
    return roc_auc_score(y, pr)


X0, X1, lam = make_feats(K, ordering)
cell = pickle.load(open((os.path.join(OUT, f'{ordering}_{K}_{weight}.pkl') if os.path.exists(os.path.join(OUT, f'{ordering}_{K}_{weight}.pkl')) else os.path.join(OUT, 'cells', f'{ordering}_{K}_{weight}.pkl')), 'rb'))
obs_int = cell['int_mean']
print(f"cell {ordering} K={K} w={weight}: seed-averaged internal {obs_int:.3f}, external {cell['ext_mean']:.3f}")

rng = np.random.RandomState(0)
log(f"TEST 4: {NPERM} label permutations, full OOF refit")
null = []
for j in range(NPERM):
    null.append(oof(X0, X1, bA[rng.permutation(len(bA))], lam, 0))
    if (j + 1) % 50 == 0: log(f"   {j+1}/{NPERM}  null mean {np.mean(null):.3f}")
null = np.array(null)
print(f"\nTEST 4 INTERNAL PERMUTATION")
print(f"  seed-averaged internal {obs_int:.3f}   null {null.mean():.3f} +- {null.std():.3f}   p = {(null >= obs_int).mean():.4f}")

s2 = StandardScaler().fit(X0); s3 = StandardScaler().fit(Ca)
ZB, DB = s2.transform(X1), s3.transform(Cb)
res = {k: [] for k in ['intact', 'img0', 'imgshuf', 'cov0', 'covshuf']}
for sd in range(10):
    _, m = gfit(s2.transform(X0), s3.transform(Ca), bA.astype(float), ZB, DB, lam, sd)
    r = np.random.RandomState(sd)
    with torch.no_grad():
        ex = lambda Z, D: roc_auc_score(bB, m(T(Z), T(D)).cpu().numpy())
        res['intact'].append(ex(ZB, DB)); res['img0'].append(ex(np.zeros_like(ZB), DB))
        res['imgshuf'].append(ex(ZB[r.permutation(len(ZB))], DB))
        res['cov0'].append(ex(ZB, np.zeros_like(DB)))
        res['covshuf'].append(ex(ZB, DB[r.permutation(len(DB))]))
print(f"\nTEST 5 IMAGING ABLATION (10 seeds)")
for k, lab in [('intact','intact'), ('img0','imaging zeroed'), ('imgshuf','imaging shuffled'),
               ('cov0','covariates zeroed'), ('covshuf','covariates shuffled')]:
    v = np.array(res[k]); print(f"  {lab:20s} {v.mean():.3f} +- {v.std():.3f}")
d = np.array(res['intact']) - np.array(res['imgshuf'])
print(f"  imaging contribution (intact - shuffled): {d.mean():+.3f} +- {d.std():.3f}")
pickle.dump(dict(null=null, obs=obs_int, ablation=res),
            open(os.path.join(OUT, f'final_tests_{ordering}_{K}_{weight}.pkl'), 'wb'))
log("DONE")
