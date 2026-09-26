"""One chunk of LOO label permutations, for test 4 against the LOO observed value.

usage: loo_perm.py <ordering> <K> <weight> <chunk_id> <n_perms>
"""
import sys, os
SPDIR = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(SPDIR)
OUT = os.path.join(REPO, 'docs', 'dbs_3d', 'loo'); os.makedirs(OUT, exist_ok=True)
DATA = os.environ.get('DBS_CLINICAL_DIR', os.path.join(REPO, 'docs'))
CACHE = os.environ.get('DBS_CACHE_DIR', '/checkpoints')
sys.path.insert(0, REPO); sys.path.insert(0, SPDIR); os.chdir(REPO)
import numpy as np, torch, torch.nn as nn, pickle, time, warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
ordering, K, weight, cid, NP = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
exec(open(os.path.join(SPDIR, 'concat_attn.py')).read().split("rows=[]")[0])
from basis_fix import basis3_iso, low_modes, eigvals
from rank_model import RankAttn
dev = torch.device('cuda'); T = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
torch.set_num_threads(2)
SHAPE = (BB[0].stop-BB[0].start, BB[1].stop-BB[1].start, BB[2].stop-BB[2].start)
idx = low_modes(K, SHAPE, 'lap'); lam = eigvals(idx, SHAPE)
X0 = basis3_iso(VA, '3DDCT', K, BB, 'lap')
def gfit(Xtr, Ctr, ytr, Xte, Cte, seed, ep=1500):
    torch.manual_seed(seed)
    m = RankAttn(Xtr.shape[1], Ctr.shape[1], lam, weight=weight).to(dev)
    opt = torch.optim.AdamW(m.parameters(), 3e-3, weight_decay=0.1); lf = nn.BCEWithLogitsLoss()
    xt, ct, yt = T(Xtr), T(Ctr), T(ytr)
    for _ in range(ep):
        opt.zero_grad(); lf(m(xt, ct), yt).backward(); opt.step()
    m.eval()
    with torch.no_grad(): return m(T(Xte), T(Cte)).cpu().numpy()
def loo_auc(y, seed):
    pr = np.empty(len(y))
    for i in range(len(y)):
        tr = np.delete(np.arange(len(y)), i)
        s2 = StandardScaler().fit(X0[tr]); s3 = StandardScaler().fit(Ca[tr])
        pr[i] = gfit(s2.transform(X0[tr]), s3.transform(Ca[tr]), y[tr].astype(float),
                     s2.transform(X0[i:i+1]), s3.transform(Ca[i:i+1]), seed)[0]
    return roc_auc_score(y, pr)
rng = np.random.RandomState(1000 + cid); t0 = time.time()
vals = [loo_auc(bA[rng.permutation(len(bA))], 0) for _ in range(NP)]
pickle.dump(np.array(vals), open(os.path.join(OUT, f'perm_{ordering}_{K}_{weight}_{cid}.pkl'), 'wb'))
print(f"chunk {cid}: {NP} perms, mean {np.mean(vals):.3f} ({time.time()-t0:.0f}s)", flush=True)
