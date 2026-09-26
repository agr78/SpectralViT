"""Select d and optimisation steps by LOO on the final configuration.

K was selected this way; d and steps were swept only on other configurations
before that sweep was interrupted. This runs the same protocol on
'both' ROI + log|FFT| + K=16 so the capacity choices are cross-validated on the
model actually used.
"""
import sys, os
SPDIR = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(SPDIR)
OUT = os.path.join(REPO, 'docs', 'dbs_3d')
DATA = os.environ.get('DBS_CLINICAL_DIR', os.path.join(REPO, 'docs'))
CACHE = os.environ.get('DBS_CACHE_DIR', '/checkpoints')
sys.path.insert(0, REPO); sys.path.insert(0, SPDIR); os.chdir(REPO)
import numpy as np, pandas as pd, torch, torch.nn as nn, itertools, warnings
warnings.filterwarnings('ignore')
from scipy.fft import fftn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from basis_fix import low_modes
from rank_model import RankAttn

NS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
src = open(os.path.join(SPDIR, 'roi_x_models.py')).read().split('IDX = low_modes')[0]
g = {'__file__': os.path.join(SPDIR, 'roi_x_models.py')}
_a = sys.argv; sys.argv = [_a[0]]; exec(compile(src, 'x', 'exec'), g); sys.argv = _a
HA, LA, HB, LB, bA, bB, Ca, Cb = g['HA'], g['LA'], g['HB'], g['LB'], g['bA'], g['bB'], g['Ca'], g['Cb']
H, DZ, K = 10, 8, 16
dev = torch.device('cuda'); T = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
sig = lambda z: 1/(1+np.exp(-z))
IDX = low_modes(K, (2*H, 2*H, 2*DZ))if False else low_modes(K, (2*H, 2*H, 2*DZ), 'lap')
lg = lambda V: np.stack([np.log1p(np.abs(fftn(V, axes=(1,2,3))))[:, a, b, c] for a, b, c in IDX], 1)
XA = np.hstack([lg(HA), lg(LA)]); XB = np.hstack([lg(HB), lg(LB)])
sX = StandardScaler().fit(XA); ZA, ZB = sX.transform(XA), sX.transform(XB)
sC = StandardScaler().fit(Ca); DA, DB = sC.transform(Ca), sC.transform(Cb)
LAM = np.ones(ZA.shape[1])


def fit(Xtr, Ctr, ytr, Xte, Cte, seed, ep, d):
    torch.manual_seed(seed)
    m = RankAttn(Xtr.shape[1], Ctr.shape[1], LAM, weight='none', d=d).to(dev)
    opt = torch.optim.AdamW(m.parameters(), 3e-3, weight_decay=0.1); lf = nn.BCEWithLogitsLoss()
    xt, ct = T(Xtr), T(Ctr); yt = torch.tensor(ytr, dtype=torch.float32, device=dev)
    for _ in range(ep):
        opt.zero_grad(); lf(m(xt, ct), yt).backward(); opt.step()
    m.eval()
    with torch.no_grad(): return sig(m(T(Xte), T(Cte)).cpu().numpy())


print(f"final configuration: both ROI, log|FFT|, K={K}, lambda-sorted; LOO, {NS} seeds")
print(f"{'d':>4} {'steps':>6} {'params':>8} {'LOO int':>16} {'ext':>16}")
rows = []
for d, ep in itertools.product([8, 16, 32], [100, 200, 600]):
    ii, ee = [], []
    for sd in range(NS):
        pr = np.empty(len(bA))
        for i in range(len(bA)):
            tr = np.delete(np.arange(len(bA)), i)
            pr[i] = fit(ZA[tr], DA[tr], bA[tr].astype(float), ZA[i:i+1], DA[i:i+1], sd, ep, d)[0]
        pb = fit(ZA, DA, bA.astype(float), ZB, DB, sd, ep, d)
        ii.append(roc_auc_score(bA, pr)); ee.append(roc_auc_score(bB, pb))
    ii, ee = np.array(ii), np.array(ee)
    npar = sum(p.numel() for p in RankAttn(ZA.shape[1], DA.shape[1], LAM, weight='none', d=d).parameters())
    print(f"{d:4d} {ep:6d} {npar:8d} {ii.mean():7.3f}+-{ii.std()/np.sqrt(NS):.3f} "
          f"{ee.mean():9.3f}+-{ee.std():.3f}", flush=True)
    rows.append(dict(d=d, steps=ep, params=npar, loo_int=ii.mean(),
                     int_se=ii.std()/np.sqrt(NS), ext=ee.mean(), ext_sd=ee.std()))
df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, 'tune_capacity.csv'), index=False)
b = df.loc[df.loo_int.idxmax()]
print(f"\nselected by LOO internal: d={int(b['d'])}, steps={int(b['steps'])}  "
      f"(int {b['loo_int']:.3f}, ext {b['ext']:.3f})")
