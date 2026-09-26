"""Run one cell (ordering, K, weight) over NS seeds and save its predictions.

Cells are independent, so several workers run concurrently on the same GPU --
each model is tiny (~2.7k parameters) and launch-latency bound, so throughput
comes from concurrency rather than from larger kernels.

usage: cell_worker.py <ordering> <K> <weight> [nseeds]
"""
import sys, os
SPDIR = os.path.dirname(os.path.abspath(__file__))
REPO  = os.path.dirname(SPDIR)
OUT   = os.path.join(REPO, 'docs', 'dbs_3d', 'cells')
DATA  = os.environ.get('DBS_CLINICAL_DIR', os.path.join(REPO, 'docs'))
CACHE = os.environ.get('DBS_CACHE_DIR', '/checkpoints')
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, REPO); sys.path.insert(0, SPDIR); os.chdir(REPO)
import numpy as np, torch, torch.nn as nn, pickle, time, warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

ordering, K, weight = sys.argv[1], int(sys.argv[2]), sys.argv[3]
DESHARP = os.environ.get('DESHARPEN','0') == '1'
NS = int(sys.argv[4]) if len(sys.argv) > 4 else 20
tag = f"{ordering}_{K}_{weight}" + ("_desharp" if DESHARP else "")

exec(open(os.path.join(SPDIR, 'concat_attn.py')).read().split("rows=[]")[0])
from basis_fix import basis3_iso, low_modes, eigvals
from rank_model import RankAttn
from desharpen import desharpen, mode_freq, unsharp_gain

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
        if DESHARP: B = desharpen(B, idx, SHAPE)
    else:
        o = 'l1' if ordering == 'l1' else 'lap'
        idx = low_modes(K, SHAPE, o)
        A = basis3_iso(VA, '3DDCT', K, BB, o)
        B = basis3_iso(VB, '3DDCT', K, BB, o)
        if DESHARP: B = desharpen(B, idx, SHAPE)
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


def oof_ext(X0, X1, y, lam, seed):
    """returns (internal AUC, external preds, MSW out-of-fold preds)"""
    pr = np.full(len(y), np.nan)
    for tr, va in StratifiedKFold(4, shuffle=True, random_state=0).split(X0, y):
        s2 = StandardScaler().fit(X0[tr]); s3 = StandardScaler().fit(Ca[tr])
        pr[va], _ = gfit(s2.transform(X0[tr]), s3.transform(Ca[tr]), y[tr].astype(float),
                         s2.transform(X0[va]), s3.transform(Ca[va]), lam, seed)
    s2 = StandardScaler().fit(X0); s3 = StandardScaler().fit(Ca)
    pb, m = gfit(s2.transform(X0), s3.transform(Ca), y.astype(float),
                 s2.transform(X1), s3.transform(Cb), lam, seed)
    return roc_auc_score(y, pr), pb, pr


t0 = time.time()
X0, X1, lam = make_feats(K, ordering)
ii, ee, pbs = [], [], []
oofs = []
for sd in range(NS):
    ai, pb, pr = oof_ext(X0, X1, bA, lam, sd)
    ii.append(ai); ee.append(roc_auc_score(bB, pb)); pbs.append(pb); oofs.append(pr)
ii, ee = np.array(ii), np.array(ee)

res = dict(ordering=ordering, K=K, weight=weight, nseeds=NS,
           internal=ii, external=ee, preds=np.array(pbs), ens=np.mean(pbs, 0),
           int_mean=ii.mean(), int_se=ii.std()/np.sqrt(NS), int_sd=ii.std(),
           ext_mean=ee.mean(), ext_sd=ee.std(), ens_auc=roc_auc_score(bB, np.mean(pbs, 0)),
           oof=np.array(oofs), bB=bB, bA=bA, secs=time.time()-t0)
pickle.dump(res, open(os.path.join(OUT, tag + '.pkl'), 'wb'))
print(f"{tag}: int {ii.mean():.3f}+-{ii.std()/np.sqrt(NS):.3f}  ext {ee.mean():.3f}+-{ee.std():.3f}  ens {res['ens_auc']:.3f}  ({res['secs']:.0f}s)", flush=True)
