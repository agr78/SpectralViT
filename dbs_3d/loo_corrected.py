"""LOO internal on the corrected 66-patient cohort, for the main architecture.

The 59-patient cohort gave LOO 0.821. That cohort silently excluded 7 labelled
patients (including one non-responder) whose nucleus sat near the volume edge.
This repeats the estimate with them restored.
"""
import sys, os
SPDIR = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(SPDIR)
OUT = os.path.join(REPO, 'docs', 'dbs_3d')
DATA = os.environ.get('DBS_CLINICAL_DIR', os.path.join(REPO, 'docs'))
CACHE = os.environ.get('DBS_CACHE_DIR', '/checkpoints')
sys.path.insert(0, REPO); sys.path.insert(0, SPDIR); os.chdir(REPO)
import numpy as np, torch, torch.nn as nn, warnings, pickle
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
src = open(os.path.join(SPDIR, 'roi_x_models.py')).read().split('class TokenViT')[0]
g = {'__file__': os.path.join(SPDIR, 'roi_x_models.py')}
exec(compile(src, 'x', 'exec'), g)
from rank_model import RankAttn
bA, bB, Ca, Cb, ROIS, LAM = g['bA'], g['bB'], g['Ca'], g['Cb'], g['ROIS'], g['LAM']
dev = torch.device('cuda'); T = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
sig = lambda z: 1/(1+np.exp(-z))
NS = int(sys.argv[1]) if len(sys.argv) > 1 else 5

def fit(Xtr, Ctr, ytr, Xte, Cte, seed, ep=200):
    torch.manual_seed(seed)
    m = RankAttn(Xtr.shape[1], Ctr.shape[1], np.resize(LAM, Xtr.shape[1]), weight='lap').to(dev)
    opt = torch.optim.AdamW(m.parameters(), 3e-3, weight_decay=0.1); lf = nn.BCEWithLogitsLoss()
    xt, ct = T(Xtr), T(Ctr); yt = torch.tensor(ytr, dtype=torch.float32, device=dev)
    for _ in range(ep):
        opt.zero_grad(); lf(m(xt, ct), yt).backward(); opt.step()
    m.eval()
    with torch.no_grad(): return sig(m(T(Xte), T(Cte)).cpu().numpy())

print(f"corrected cohort: MSW {len(bA)} ({bA.sum()}/{(bA==0).sum()})  CHH {len(bB)} ({bB.sum()}/{(bB==0).sum()})\n")
for roi in ['higher', 'both', 'diff']:
    XA, XB = ROIS[roi]
    sX = StandardScaler().fit(XA); sC = StandardScaler().fit(Ca)
    ZA, ZB, DA, DB = sX.transform(XA), sX.transform(XB), sC.transform(Ca), sC.transform(Cb)
    ii, ee = [], []
    for sd in range(NS):
        pr = np.empty(len(bA))
        for i in range(len(bA)):
            tr = np.delete(np.arange(len(bA)), i)
            pr[i] = fit(ZA[tr], DA[tr], bA[tr].astype(float), ZA[i:i+1], DA[i:i+1], sd)[0]
        pb = fit(ZA, DA, bA.astype(float), ZB, DB, sd)
        ii.append(roc_auc_score(bA, pr)); ee.append(roc_auc_score(bB, pb))
    print(f"  {roi:7s} LOO internal {np.mean(ii):.3f}+-{np.std(ii):.3f}   external {np.mean(ee):.3f}+-{np.std(ee):.3f}", flush=True)
print("\nfor comparison, 59-patient cohort gave LOO internal 0.821, external 0.816")
