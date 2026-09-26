"""Operating-point analysis: what specificity is attainable, and at what cost.

Specificity is set by the decision threshold, not by the loss. AUC is
threshold-free, so at a fixed AUC the only way to raise specificity is to move
the cut and give up sensitivity. This maps the trade-off and compares threshold
rules, all calibrated on MSW out-of-fold and applied unchanged to CHH:

  balacc      maximise balanced accuracy            (what we have been using)
  spec_at_X   maximise specificity s.t. sensitivity >= X
  youden      maximise sensitivity + specificity - 1
  prevalence  cut at the training responder rate

usage: operating_point.py [nseeds]
"""
import sys, os
SPDIR = os.path.dirname(os.path.abspath(__file__))
REPO  = os.path.dirname(SPDIR)
OUT   = os.path.join(REPO, 'docs', 'dbs_3d')
DATA  = os.environ.get('DBS_CLINICAL_DIR', os.path.join(REPO, 'docs'))
CACHE = os.environ.get('DBS_CACHE_DIR', '/checkpoints')
sys.path.insert(0, REPO); sys.path.insert(0, SPDIR); os.chdir(REPO)
import numpy as np, pandas as pd, torch, torch.nn as nn, pickle, warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, balanced_accuracy_score, f1_score, confusion_matrix

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
from sklearn.model_selection import StratifiedKFold


def gfit(Xtr, Ctr, ytr, Xte, Cte, seed, ep=200):
    torch.manual_seed(seed)
    m = RankAttn(Xtr.shape[1], Ctr.shape[1], LAM, weight='lap').to(dev)
    opt = torch.optim.AdamW(m.parameters(), 3e-3, weight_decay=0.1)
    lf = nn.BCEWithLogitsLoss()
    xt, ct, yt = T(Xtr), T(Ctr), T(ytr)
    for _ in range(ep):
        opt.zero_grad(); lf(m(xt, ct), yt).backward(); opt.step()
    m.eval()
    with torch.no_grad(): return sig(m(T(Xte), T(Cte)).cpu().numpy())


def rules(y, p):
    """threshold rules, all fitted on the TRAINING cohort only"""
    fpr, tpr, thr = roc_curve(y, p)
    spec, sens = 1 - fpr, tpr
    out = {}
    best = -1
    for t in np.unique(p):
        b = balanced_accuracy_score(y, (p >= t).astype(int))
        if b > best: best, out['balacc'] = b, float(t)
    j = np.argmax(sens + spec - 1); out['youden'] = float(thr[j])
    for floor in (0.60, 0.70, 0.80):
        ok = sens >= floor
        out[f'spec_at_sens{int(floor*100)}'] = float(thr[ok][np.argmax(spec[ok])]) if ok.any() else 1.0
    out['prevalence'] = float(np.quantile(p, 1 - y.mean()))
    return out


def sc(y, p, t):
    yh = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    return dict(BalAcc=balanced_accuracy_score(y, yh),
                Sens=tp/(tp+fn) if (tp+fn) else np.nan,
                Spec=tn/(tn+fp) if (tn+fp) else np.nan,
                F1=f1_score(y, yh, zero_division=0), TN=tn, FP=fp)


acc = {}
ext_auc = []
for sd in range(NS):
    pr = np.full(len(bA), np.nan)
    for tr, va in StratifiedKFold(4, shuffle=True, random_state=0).split(SA, bA):
        s2 = StandardScaler().fit(SA[tr]); s3 = StandardScaler().fit(Ca[tr])
        pr[va] = gfit(s2.transform(SA[tr]), s3.transform(Ca[tr]), bA[tr].astype(float),
                      s2.transform(SA[va]), s3.transform(Ca[va]), sd)
    s2 = StandardScaler().fit(SA); s3 = StandardScaler().fit(Ca)
    pb = gfit(s2.transform(SA), s3.transform(Ca), bA.astype(float),
              s2.transform(SB), s3.transform(Cb), sd)
    ext_auc.append(roc_auc_score(bB, pb))
    for name, t in rules(bA, pr).items():        # calibrated on MSW
        acc.setdefault(name, []).append(sc(bB, pb, t))

print(f"external AUC {np.mean(ext_auc):.3f} +- {np.std(ext_auc):.3f}   "
      f"(CHH: {int(bB.sum())} responders / {int((bB==0).sum())} non-responders)\n")
print("threshold rule (fitted on MSW)      ext Sens   ext Spec   ext BalAcc   ext F1    TN/3")
for name, ss in acc.items():
    m = {k: np.mean([d[k] for d in ss]) for k in ('Sens', 'Spec', 'BalAcc', 'F1', 'TN')}
    s = {k: np.std([d[k] for d in ss]) for k in ('Sens', 'Spec')}
    print(f"  {name:32s} {m['Sens']:.3f}+-{s['Sens']:.2f}  {m['Spec']:.3f}+-{s['Spec']:.2f}   "
          f"{m['BalAcc']:.3f}      {m['F1']:.3f}   {m['TN']:.1f}")
print("\nspecificity is quantised to thirds: 3 external non-responders")
pickle.dump(acc, open(os.path.join(OUT, 'operating_point.pkl'), 'wb'))
