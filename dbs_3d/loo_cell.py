"""One cell under leave-one-out CV, for the tests that failed or sat on the boundary.

Tests 1 (family-wise null), 2 (pre-specified cell) and 4 (internal permutation)
all depend on the internal estimate. Under 4-fold each model trains on 44 of 59
patients with 3 non-responders, while the external model trains on all 59 with 4
-- which may be why internal (0.694) came in below external (0.799). LOO trains
on 58 and removes that gap.

Tests 3 and 5 are unaffected: both use the model trained on all 59, so the CV
scheme does not enter.

usage: loo_cell.py <ordering> <K> <weight> [nseeds]
"""
import sys, os
SPDIR = os.path.dirname(os.path.abspath(__file__))
REPO  = os.path.dirname(SPDIR)
OUT   = os.path.join(REPO, 'docs', 'dbs_3d', 'loo')
DATA  = os.environ.get('DBS_CLINICAL_DIR', os.path.join(REPO, 'docs'))
CACHE = os.environ.get('DBS_CACHE_DIR', '/checkpoints')
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, REPO); sys.path.insert(0, SPDIR); os.chdir(REPO)
import numpy as np, torch, torch.nn as nn, pickle, time, warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

ordering, K, weight = sys.argv[1], int(sys.argv[2]), sys.argv[3]
NS = int(sys.argv[4]) if len(sys.argv) > 4 else 5
tag = f"loo_{ordering}_{K}_{weight}"

exec(open(os.path.join(SPDIR, 'concat_attn.py')).read().split("rows=[]")[0])
from basis_fix import basis3_iso, low_modes, eigvals
from rank_model import RankAttn

dev = torch.device('cuda'); T = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
torch.set_num_threads(2)
t0 = time.time()
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
        return m(T(Xte), T(Cte)).cpu().numpy()


def loo_oof(X0, y, lam, seed):
    """leave-one-out: each model trains on 58 patients"""
    pr = np.empty(len(y))
    for i in range(len(y)):
        tr = np.delete(np.arange(len(y)), i)
        s2 = StandardScaler().fit(X0[tr]); s3 = StandardScaler().fit(Ca[tr])
        pr[i] = gfit(s2.transform(X0[tr]), s3.transform(Ca[tr]), y[tr].astype(float),
                     s2.transform(X0[i:i+1]), s3.transform(Ca[i:i+1]), lam, seed)[0]
    return pr


X0, X1, lam = make_feats(K, ordering)
ii, ee, pbs, oofs = [], [], [], []
for sd in range(NS):
    pr = loo_oof(X0, bA, lam, sd)
    s2 = StandardScaler().fit(X0); s3 = StandardScaler().fit(Ca)
    pb = gfit(s2.transform(X0), s3.transform(Ca), bA.astype(float),
              s2.transform(X1), s3.transform(Cb), lam, sd)
    ii.append(roc_auc_score(bA, pr)); ee.append(roc_auc_score(bB, pb))
    oofs.append(pr); pbs.append(pb)
ii, ee = np.array(ii), np.array(ee)
res = dict(ordering=ordering, K=K, weight=weight, nseeds=NS, cv='loo',
           internal=ii, external=ee, oof=np.array(oofs), preds=np.array(pbs),
           ens=np.mean(pbs, 0), int_mean=ii.mean(), int_se=ii.std()/np.sqrt(NS),
           int_sd=ii.std(), ext_mean=ee.mean(), ext_sd=ee.std(),
           ens_auc=roc_auc_score(bB, np.mean(pbs, 0)), bA=bA, bB=bB, secs=time.time()-t0)
pickle.dump(res, open(os.path.join(OUT, tag + '.pkl'), 'wb'))
print(f"{tag}: LOO internal {ii.mean():.3f}+-{ii.std()/np.sqrt(NS):.3f}   "
      f"external {ee.mean():.3f}+-{ee.std():.3f}   ens {res['ens_auc']:.3f}  ({res['secs']:.0f}s)", flush=True)
