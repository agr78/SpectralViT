"""Fix the probability saturation and calibrate the threshold on MSW only.

Training 1500 full-batch steps on 59 separable samples drives every predicted
probability to ~1.0, so specificity is 0 at any fixed threshold even though the
ranking is informative (AUC 0.83). Two remedies are compared:

  steps     fewer optimisation steps, so the logits never saturate
  smooth    label smoothing, which caps the achievable logit magnitude
  posw      pos_weight in the loss, countering the 55:4 imbalance

The threshold is always calibrated on the MSW out-of-fold predictions by
maximising balanced accuracy, then applied unchanged to CHH. The test cohort is
never used to pick it.

usage: calib_worker.py <ordering> <K> <weight> <variant> [nseeds]
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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score, confusion_matrix

ordering, K, weight, variant = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
NS = int(sys.argv[5]) if len(sys.argv) > 5 else 20
tag = f"calib_{ordering}_{K}_{weight}_{variant}"

exec(open(os.path.join(SPDIR, 'concat_attn.py')).read().split("rows=[]")[0])
from basis_fix import basis3_iso, low_modes, eigvals
from rank_model import RankAttn

dev = torch.device('cuda'); T = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
torch.set_num_threads(2)
SHAPE = (BB[0].stop-BB[0].start, BB[1].stop-BB[1].start, BB[2].stop-BB[2].start)
sig = lambda z: 1.0 / (1.0 + np.exp(-z))

CFG = {'base':   dict(ep=1500, smooth=0.0,  posw=False),
       'steps':  dict(ep=200,  smooth=0.0,  posw=False),
       'smooth': dict(ep=1500, smooth=0.10, posw=False),
       'posw':   dict(ep=1500, smooth=0.0,  posw=True),
       'both':   dict(ep=300,  smooth=0.10, posw=True)}[variant]


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


def gfit(Xtr, Ctr, ytr, Xte, Cte, lam, seed):
    torch.manual_seed(seed)
    m = RankAttn(Xtr.shape[1], Ctr.shape[1], lam, weight=weight).to(dev)
    opt = torch.optim.AdamW(m.parameters(), 3e-3, weight_decay=0.1)
    pw = torch.tensor((ytr == 0).sum() / max(1, (ytr == 1).sum()), dtype=torch.float32, device=dev)
    lf = nn.BCEWithLogitsLoss(pos_weight=pw) if CFG['posw'] else nn.BCEWithLogitsLoss()
    s = CFG['smooth']
    yt = torch.tensor(ytr * (1 - s) + 0.5 * s, dtype=torch.float32, device=dev)
    xt, ct = T(Xtr), T(Ctr)
    for _ in range(CFG['ep']):
        opt.zero_grad(); lf(m(xt, ct), yt).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        return m(T(Xte), T(Cte)).cpu().numpy()


def calibrate(y, p):
    """threshold maximising balanced accuracy on the TRAINING cohort"""
    ts = np.unique(np.concatenate([p, np.linspace(p.min(), p.max(), 200)]))
    best, bt = -1, np.median(p)
    for t in ts:
        b = balanced_accuracy_score(y, (p >= t).astype(int))
        if b > best: best, bt = b, t
    return bt


def scores(y, p, thr):
    yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    return dict(AUC=roc_auc_score(y, p), BalAcc=balanced_accuracy_score(y, yh),
                Sens=tp/(tp+fn) if (tp+fn) else np.nan,
                Spec=tn/(tn+fp) if (tn+fp) else np.nan,
                F1=f1_score(y, yh, zero_division=0), TN=tn, FP=fp, FN=fn, TP=tp)


t0 = time.time()
X0, X1, lam = make_feats(K, ordering)
rows, sat = [], []
for sd in range(NS):
    pr = np.full(len(bA), np.nan)
    for tr, va in StratifiedKFold(4, shuffle=True, random_state=0).split(X0, bA):
        s2 = StandardScaler().fit(X0[tr]); s3 = StandardScaler().fit(Ca[tr])
        pr[va] = gfit(s2.transform(X0[tr]), s3.transform(Ca[tr]), bA[tr].astype(float),
                      s2.transform(X0[va]), s3.transform(Ca[va]), lam, sd)
    s2 = StandardScaler().fit(X0); s3 = StandardScaler().fit(Ca)
    pb = gfit(s2.transform(X0), s3.transform(Ca), bA.astype(float),
              s2.transform(X1), s3.transform(Cb), lam, sd)
    thr = calibrate(bA, sig(pr))                       # MSW only
    rows.append((scores(bA, sig(pr), thr), scores(bB, sig(pb), thr), thr))
    sat.append(sig(pb))

sat = np.array(sat)
agg = lambda which, k: np.array([r[which][k] for r in rows])
print(f"\n{tag}   ({CFG})")
print(f"  predicted prob range on CHH: [{sat.min():.4f}, {sat.max():.4f}]   "
      f"spread {sat.max()-sat.min():.4f}   (saturated if ~0)")
print(f"  threshold (MSW-calibrated): {np.mean([r[2] for r in rows]):.4f} +- {np.std([r[2] for r in rows]):.4f}")
for lab, w in [('INTERNAL (MSW)', 0), ('EXTERNAL (CHH)', 1)]:
    print(f"  {lab}")
    for k in ('AUC', 'BalAcc', 'Sens', 'Spec', 'F1'):
        v = agg(w, k); print(f"     {k:7s} {v.mean():.3f} +- {v.std():.3f}")
    print(f"     confusion  TN {agg(w,'TN').mean():.2f}  FP {agg(w,'FP').mean():.2f}"
          f"  FN {agg(w,'FN').mean():.2f}  TP {agg(w,'TP').mean():.2f}")
pickle.dump(dict(rows=rows, sat=sat, cfg=CFG, tag=tag),
            open(os.path.join(OUT, tag + '.pkl'), 'wb'))
print(f"  ({time.time()-t0:.0f}s)", flush=True)
