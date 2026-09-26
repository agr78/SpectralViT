"""Leave-one-out internal estimate, to test whether 4-fold CV is the reason
test 4 fails.

4-fold CV on 59 patients trains each model on 44 with 3 non-responders, while
the external model trains on all 59 with 4. At this scale that gap is large, so
the out-of-fold estimate may be pessimistic rather than the external estimate
optimistic -- which would explain external 0.799 > internal 0.694.

Leave-one-out trains on 58 patients per fold, matching the external model much
more closely. If the internal estimate rises toward the external one, 4-fold was
the problem; if it stays at ~0.69, the internal effect really is weak.

usage: loo_internal.py <ordering> <K> <weight> [nseeds]
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
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.metrics import roc_auc_score

ordering, K, weight = sys.argv[1], int(sys.argv[2]), sys.argv[3]
NS = int(sys.argv[4]) if len(sys.argv) > 4 else 5
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
        return m(T(Xte), T(Cte)).cpu().numpy()


def oof(X0, y, lam, seed, scheme):
    pr = np.full(len(y), np.nan)
    splits = LeaveOneOut().split(X0) if scheme == 'loo' else \
             StratifiedKFold(int(scheme), shuffle=True, random_state=0).split(X0, y)
    for tr, va in splits:
        s2 = StandardScaler().fit(X0[tr]); s3 = StandardScaler().fit(Ca[tr])
        pr[va] = gfit(s2.transform(X0[tr]), s3.transform(Ca[tr]), y[tr].astype(float),
                      s2.transform(X0[va]), s3.transform(Ca[va]), lam, seed)
    return pr


X0, X1, lam = make_feats(K, ordering)
print(f"cell {ordering} K={K} w={weight}   n={len(bA)}  ({bA.sum()} resp / {(bA==0).sum()} non-resp)\n")
res = {}
for scheme, label in [('4', '4-fold  (trains on 44)'), ('10', '10-fold (trains on 53)'), ('loo', 'LOO     (trains on 58)')]:
    aucs = []
    for sd in range(NS):
        pr = oof(X0, bA, lam, sd, scheme)
        aucs.append(roc_auc_score(bA, pr))
    aucs = np.array(aucs); res[scheme] = aucs
    log(f"{label}: internal AUC {aucs.mean():.3f} +- {aucs.std():.3f}")
print()
print("external for reference: 0.799 +- 0.032 (model trains on all 59)")
pickle.dump(res, open(os.path.join(OUT, f'loo_{ordering}_{K}_{weight}.pkl'), 'wb'))
log("DONE")
