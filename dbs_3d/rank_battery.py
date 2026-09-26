"""Five-test battery with and without the rank prior, on a coherently ordered basis.

Mode ordering and rank weight are varied together, because omega_i only means
anything if rank tracks frequency:

  ordering   rowmajor  the original truncation (corr(rank, lambda) = 0.70)
             l1        isotropic, sorted by fx+fy+fz          (0.87)
             lap       sorted by Neumann-Laplacian eigenvalue (1.00)

  weight     none      omega_i = 1
             inv       omega_i = 1/i                (the PCA form)
             lap       omega_i = exp(-tau*lambda_i) (the Laplacian form)
             learned   omega_i free, initialised at 1/i
"""
import sys, os
SPDIR = os.path.dirname(os.path.abspath(globals().get('__file__', '/workspace/dbs_3d/x')))
REPO  = os.path.dirname(SPDIR)
OUT   = os.path.join(REPO, 'docs', 'dbs_3d')
DATA  = os.environ.get('DBS_CLINICAL_DIR', os.path.join(REPO, 'docs'))
CACHE = os.environ.get('DBS_CACHE_DIR', '/checkpoints')
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, REPO); sys.path.insert(0, SPDIR); os.chdir(REPO)
import numpy as np, pandas as pd, torch, torch.nn as nn, itertools, time, pickle, warnings
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
SHAPE = (BB[0].stop-BB[0].start, BB[1].stop-BB[1].start, BB[2].stop-BB[2].start)


def make_feats(K, ordering):
    """returns (train, test, lambda) for the chosen mode ordering"""
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
        A = basis3_iso(VA, '3DDCT', K, BB, o)
        B = basis3_iso(VB, '3DDCT', K, BB, o)
    return A, B, eigvals(idx, SHAPE)


class RankAttn(nn.Module):
    def __init__(s, K, C, lam, weight='none', d=16, tau=1.0):
        super().__init__(); s.K, s.C, s.weight = K, C, weight
        l = torch.tensor(lam, dtype=torch.float32)
        s.register_buffer('lam', l / (l.max() + 1e-12))
        i = torch.arange(1, K + 1, dtype=torch.float32)
        if weight == 'none':      s.register_buffer('w', torch.ones(K));            s.p = None
        elif weight == 'inv':     s.register_buffer('w', 1.0 / i);                  s.p = None
        elif weight == 'lap':     s.register_buffer('w', torch.exp(-tau * s.lam));  s.p = None
        elif weight == 'lap_tau': s.w = None; s.p = nn.Parameter(torch.tensor(float(tau)))
        elif weight == 'learned': s.w = nn.Parameter(1.0 / i);                      s.p = None
        s.tok = nn.Linear(1, d); s.pos = nn.Parameter(torch.zeros(1, K + C, d))
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 2, d*2, 0.1, batch_first=True), 1)
        s.hd = nn.Linear(d, 1); s.lin = nn.Linear(K + C, 1)
    def forward(s, x, c):
        w = torch.exp(-nn.functional.softplus(s.p) * s.lam) if s.p is not None else s.w
        v = torch.cat([x * w, c], 1)
        z = s.tok(v.unsqueeze(-1)) + s.pos
        return s.hd(s.enc(z).mean(1)).squeeze(-1) + s.lin(v).squeeze(-1)


def gfit(Xtr, Ctr, ytr, Xte, Cte, lam, weight, seed, ep=1500):
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


def oof_ext(X0, X1, y, lam, weight, seed):
    pr = np.full(len(y), np.nan)
    for tr, va in StratifiedKFold(4, shuffle=True, random_state=0).split(X0, y):
        s2 = StandardScaler().fit(X0[tr]); s3 = StandardScaler().fit(Ca[tr])
        pr[va], _ = gfit(s2.transform(X0[tr]), s3.transform(Ca[tr]), y[tr].astype(float),
                         s2.transform(X0[va]), s3.transform(Ca[va]), lam, weight, seed)
    s2 = StandardScaler().fit(X0); s3 = StandardScaler().fit(Ca)
    pb, m = gfit(s2.transform(X0), s3.transform(Ca), y.astype(float),
                 s2.transform(X1), s3.transform(Cb), lam, weight, seed)
    return roc_auc_score(y, pr), pb, m


rows, PRED, SEEDPB = [], {}, {}
for ordering, K, weight in itertools.product(['rowmajor', 'l1', 'lap'], [8, 16],
                                             ['none', 'inv', 'lap', 'learned']):
    X0, X1, lam = make_feats(K, ordering)
    ii, ee, pbs = [], [], []
    for sd in range(NS):
        ai, pb, _ = oof_ext(X0, X1, bA, lam, weight, sd)
        ii.append(ai); ee.append(roc_auc_score(bB, pb)); pbs.append(pb)
    ii, ee = np.array(ii), np.array(ee)
    rows.append(dict(ordering=ordering, K=K, weight=weight,
                     int_mean=ii.mean(), int_se=ii.std()/np.sqrt(NS), int_sd=ii.std(),
                     ext_mean=ee.mean(), ext_sd=ee.std(), ens=roc_auc_score(bB, np.mean(pbs, 0))))
    PRED[(ordering, K, weight)] = np.mean(pbs, 0); SEEDPB[(ordering, K, weight)] = pbs
    log(f"{ordering:9s} K={K:2d} w={weight:8s}  int {ii.mean():.3f}+-{ii.std()/np.sqrt(NS):.3f}  ext {ee.mean():.3f}+-{ee.std():.3f}")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'rank_battery.csv'), index=False)

dfa = pd.DataFrame(rows)
df  = dfa[(dfa.ordering=='lap') & (dfa.weight.isin(['none','lap']))].sort_values('int_mean', ascending=False)
print("\n=== DESCRIPTIVE sweep: all 24 cells (not used for cell selection) ===")
print(dfa.sort_values('int_mean',ascending=False).round(3).to_string(index=False))
print("\n=== PRE-SPECIFIED space for the battery: lambda-sorted, omega in {none, exp(-tau*lambda)} ===")
print(df.round(3).to_string(index=False))
print("\n=== marginal effect of the RANK PRIOR ===")
print(dfa.groupby('weight').agg(int_best=('int_mean','max'), int_mean=('int_mean','mean'),
                               ext_best=('ext_mean','max'), ext_mean=('ext_mean','mean')).round(3).to_string())
print("\n=== marginal effect of MODE ORDERING ===")
print(dfa.groupby('ordering').agg(int_best=('int_mean','max'), int_mean=('int_mean','mean'),
                                 ext_best=('ext_mean','max'), ext_mean=('ext_mean','mean')).round(3).to_string())
print("\n=== rank prior WITHIN the coherent (lambda-sorted) ordering ===")
print(dfa[dfa.ordering=='lap'][['K','weight','int_mean','int_se','ext_mean','ext_sd']].round(3).to_string(index=False))

rng = np.random.RandomState(0); n = len(bB); n1 = int(bB.sum()); n0 = n - n1
ks = [k for k in PRED if k[0]=='lap' and k[2] in ('none','lap')]
R = np.array([rankdata(PRED[k]) for k in ks])
auc = lambda y: (R @ y - n1*(n1+1)/2) / (n1*n0)
obs = auc(bB); mx = np.array([auc(bB[rng.permutation(n)]).max() for _ in range(20000)])
i = int(obs.argmax())
print(f"\nTEST 1 FAMILY-WISE NULL ({len(ks)} cells)")
print(f"  best external {obs[i]:.3f} ({ks[i]})   chance {mx.mean():.3f} (95th {np.percentile(mx,95):.3f})   p = {(mx>=obs[i]).mean():.4f}")

b = df.iloc[0]; key = (b['ordering'], int(b['K']), b['weight']); pe = PRED[key]; ae = roc_auc_score(bB, pe)
nl = np.array([roc_auc_score(bB[rng.permutation(n)], pe) for _ in range(20000)])
bs = [roc_auc_score(bB[ix], pe[ix]) for ix in rng.randint(0, n, (20000, n)) if len(np.unique(bB[ix])) > 1]
print(f"\nTEST 2 PRE-SPECIFIED (best seed-averaged internal)")
print(f"  cell: {b['ordering']} K={int(b['K'])} w={b['weight']}   internal {b['int_mean']:.3f} +- {b['int_se']:.3f}")
print(f"  external per-seed {b['ext_mean']:.3f} +- {b['ext_sd']:.3f}")
print(f"  seed-ensembled external {ae:.3f}   p = {(nl>=ae).mean():.4f}   bootstrap CI [{np.percentile(bs,2.5):.3f},{np.percentile(bs,97.5):.3f}]")
d = df.iloc[0]['int_mean'] - df.iloc[1]['int_mean']; se = np.hypot(df.iloc[0]['int_se'], df.iloc[1]['int_se'])
print(f"  margin over 2nd: {d:+.3f} +- {se:.3f} -> {'separated' if d > 2*se else 'NOT separated'}")

ee = np.array([roc_auc_score(bB, p) for p in SEEDPB[key]])
print(f"\nTEST 3 SEED STABILITY ({NS} inits)")
print(f"  external {ee.mean():.3f} +- {ee.std():.3f}   min {ee.min():.3f}  max {ee.max():.3f}   below 0.5 in {(ee<0.5).sum()}/{NS}")

X0, X1, lam = make_feats(int(b['K']), b['ordering'])
log("TEST 4: 200 label permutations")
null = []
for j in range(200):
    yp = bA[rng.permutation(len(bA))]
    null.append(oof_ext(X0, X1, yp, lam, b['weight'], 0)[0])
    if (j+1) % 50 == 0: log(f"   {j+1}/200 null mean {np.mean(null):.3f}")
null = np.array(null)
print(f"\nTEST 4 INTERNAL PERMUTATION")
print(f"  seed-averaged internal {b['int_mean']:.3f}   null {null.mean():.3f} +- {null.std():.3f}   p = {(null>=b['int_mean']).mean():.4f}")

s2 = StandardScaler().fit(X0); s3 = StandardScaler().fit(Ca)
ZB, DB = s2.transform(X1), s3.transform(Cb)
res = {k: [] for k in ['intact', 'img0', 'imgshuf', 'cov0']}
for sd in range(5):
    _, _, m = gfit(s2.transform(X0), s3.transform(Ca), bA.astype(float), ZB, DB, lam, b['weight'], sd)
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

pickle.dump(dict(PRED=PRED, bB=bB, df=df), open(os.path.join(OUT, 'rank_battery.pkl'), 'wb'))
log("DONE")
