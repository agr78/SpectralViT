"""Regression version of the winning architecture, on continuous UPDRS improvement.

The binary endpoint is limited by CHH having 3 non-responders: external AUC must
exceed 0.79 for p<0.05 regardless of the model. The continuous endpoint uses all
37 CHH patients, so the same effect size buys far more power -- at n=37, r=0.40
gives p~0.014 and r=0.50 gives p~0.002.

usage: reg_worker.py <ordering> <K> <weight> [nseeds]
"""
import sys, os
SPDIR = os.path.dirname(os.path.abspath(__file__))
REPO  = os.path.dirname(SPDIR)
OUT   = os.path.join(REPO, 'docs', 'dbs_3d')
DATA  = os.environ.get('DBS_CLINICAL_DIR', os.path.join(REPO, 'docs'))
CACHE = os.environ.get('DBS_CACHE_DIR', '/checkpoints')
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, REPO); sys.path.insert(0, SPDIR); os.chdir(REPO)
import numpy as np, torch, torch.nn as nn, pickle, time, warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.stats import pearsonr, spearmanr

ordering, K, weight = sys.argv[1], int(sys.argv[2]), sys.argv[3]
NS = int(sys.argv[4]) if len(sys.argv) > 4 else 20
tag = f"reg_{ordering}_{K}_{weight}"

exec(open(os.path.join(SPDIR, 'concat_attn.py')).read().split("rows=[]")[0])
from basis_fix import basis3_iso, low_modes, eigvals
from rank_model import RankAttn

dev = torch.device('cuda'); T = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
torch.set_num_threads(2)
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
    lf = nn.MSELoss()
    xt, ct, yt = T(Xtr), T(Ctr), T(ytr)
    for _ in range(ep):
        opt.zero_grad(); lf(m(xt, ct), yt).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        return m(T(Xte), T(Cte)).cpu().numpy()


def oof_ext(X0, X1, lam, seed):
    pr = np.full(len(yA), np.nan)
    for tr, va in KFold(5, shuffle=True, random_state=0).split(X0):
        s2 = StandardScaler().fit(X0[tr]); s3 = StandardScaler().fit(Ca[tr])
        pr[va] = gfit(s2.transform(X0[tr]), s3.transform(Ca[tr]), yA[tr],
                      s2.transform(X0[va]), s3.transform(Ca[va]), lam, seed)
    s2 = StandardScaler().fit(X0); s3 = StandardScaler().fit(Ca)
    pb = gfit(s2.transform(X0), s3.transform(Ca), yA, s2.transform(X1), s3.transform(Cb), lam, seed)
    return pearsonr(yA, pr)[0], pb


t0 = time.time()
X0, X1, lam = make_feats(K, ordering)
ii, ee, pbs = [], [], []
for sd in range(NS):
    ri, pb = oof_ext(X0, X1, lam, sd)
    ii.append(ri); ee.append(pearsonr(yB, pb)[0]); pbs.append(pb)
ii, ee = np.array(ii), np.array(ee)
pe = np.mean(pbs, 0)
r_ens, p_ens = pearsonr(yB, pe)
rho = spearmanr(yB, pe).statistic
r2 = 1 - np.sum((yB - pe) ** 2) / np.sum((yB - yB.mean()) ** 2)
res = dict(ordering=ordering, K=K, weight=weight, nseeds=NS, internal=ii, external=ee,
           preds=np.array(pbs), ens=pe, int_mean=ii.mean(), int_sd=ii.std(),
           ext_mean=ee.mean(), ext_sd=ee.std(), ens_r=r_ens, ens_p=p_ens,
           ens_rho=rho, ens_r2=r2, yA=yA, yB=yB, secs=time.time()-t0)
pickle.dump(res, open(os.path.join(OUT, tag + '.pkl'), 'wb'))
print(f"{tag}: int r {ii.mean():+.3f}+-{ii.std():.3f}  ext r {ee.mean():+.3f}+-{ee.std():.3f}  "
      f"ens r {r_ens:+.3f} (p={p_ens:.4f}, rho={rho:+.3f}, R2={r2:+.3f})  n={len(yB)}  ({res['secs']:.0f}s)",
      flush=True)
