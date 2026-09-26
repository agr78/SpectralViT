"""Paired comparison of SpectralViT against the clinical-only baseline.

The seed SD in the baseline table measures initialisation variance. The question
"does imaging add anything over covariates on these patients" needs patient
variance instead, so this resamples the 37 CHH patients and recomputes the AUC
difference on each resample, with both models scored on the same patients.

usage: paired_vs_clinical.py [nseeds]
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

NS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
exec(open(os.path.join(SPDIR, 'concat_attn.py')).read().split("rows=[]")[0])
from basis_fix import basis3_iso, low_modes, eigvals
from rank_model import RankAttn

dev = torch.device('cuda'); T = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
SHAPE = (BB[0].stop-BB[0].start, BB[1].stop-BB[1].start, BB[2].stop-BB[2].start)
sig = lambda z: 1.0/(1.0+np.exp(-z))
K = 16
IDX = low_modes(K, SHAPE, 'lap'); LAM = eigvals(IDX, SHAPE)
SA = basis3_iso(VA, '3DDCT', K, BB, 'lap'); SB = basis3_iso(VB, '3DDCT', K, BB, 'lap')

# clinical only, deterministic
sC = StandardScaler().fit(Ca)
clin = LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced').fit(sC.transform(Ca), bA)
p_clin = sig(clin.decision_function(sC.transform(Cb)))

# ours, seed-ensembled
sS = StandardScaler().fit(SA)
preds = []
for sd in range(NS):
    torch.manual_seed(sd)
    m = RankAttn(K, Ca.shape[1], LAM, weight='lap').to(dev)
    opt = torch.optim.AdamW(m.parameters(), 3e-3, weight_decay=0.1)
    lf = nn.BCEWithLogitsLoss()
    xt, ct = T(sS.transform(SA)), T(sC.transform(Ca))
    yt = torch.tensor(bA, dtype=torch.float32, device=dev)
    for _ in range(200):                        # early stop: keeps probabilities unsaturated
        opt.zero_grad(); lf(m(xt, ct), yt).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        preds.append(sig(m(T(sS.transform(SB)), T(sC.transform(Cb))).cpu().numpy()))
p_ours = np.mean(preds, 0)

a_ours, a_clin = roc_auc_score(bB, p_ours), roc_auc_score(bB, p_clin)
print(f"n = {len(bB)}  ({int(bB.sum())} responders / {int((bB==0).sum())} non-responders)")
print(f"  SpectralViT (ours)  external AUC {a_ours:.3f}")
print(f"  Clinical only       external AUC {a_clin:.3f}")
print(f"  difference          {a_ours - a_clin:+.3f}\n")

rng = np.random.RandomState(0); n = len(bB)
d = []
for _ in range(20000):
    ix = rng.randint(0, n, n)
    if len(np.unique(bB[ix])) < 2: continue
    d.append(roc_auc_score(bB[ix], p_ours[ix]) - roc_auc_score(bB[ix], p_clin[ix]))
d = np.array(d)
print(f"paired bootstrap over PATIENTS ({len(d)} resamples):")
print(f"  mean difference {d.mean():+.3f}")
print(f"  95% CI [{np.percentile(d,2.5):+.3f}, {np.percentile(d,97.5):+.3f}]")
print(f"  P(ours > clinical) = {(d > 0).mean():.3f}")
print()
# how many patients drive it
dis = np.where((p_ours > p_clin) != (bB == 1))[0]
print(f"patients where the two models disagree in rank order: {len(dis)} of {n}")
pickle.dump(dict(p_ours=p_ours, p_clin=p_clin, bB=bB, diff=d),
            open(os.path.join(OUT, 'paired_vs_clinical.pkl'), 'wb'))
