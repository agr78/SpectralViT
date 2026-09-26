"""Full metric table for one model, corrected 66-patient cohort.

Reports AUC, accuracy, balanced accuracy, sensitivity, specificity, PPV, NPV and
F1, internal and external, with the decision threshold calibrated on the MSW
leave-one-out predictions only -- never on CHH.

Specificity is quantised: 5 non-responders internally (steps of 0.2), 3
externally (steps of 1/3). Confusion counts are reported so that granularity is
visible rather than hidden behind a mean.

usage: metrics_worker.py <model> <K> <nseeds>
"""
import sys, os
SPDIR = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(SPDIR)
OUT = os.path.join(REPO, 'docs', 'dbs_3d', 'metrics'); os.makedirs(OUT, exist_ok=True)
DATA = os.environ.get('DBS_CLINICAL_DIR', os.path.join(REPO, 'docs'))
CACHE = os.environ.get('DBS_CACHE_DIR', '/checkpoints')
sys.path.insert(0, REPO); sys.path.insert(0, SPDIR); os.chdir(REPO)
import numpy as np, torch, torch.nn as nn, pickle, time, warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import (roc_auc_score, balanced_accuracy_score, f1_score,
                             accuracy_score, confusion_matrix)

MODEL, K, NS = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
_argv = sys.argv; sys.argv = [_argv[0], 'cell', MODEL, str(K), '1']   # the prefix parses these
exec(open(os.path.join(SPDIR, 'sc_worker.py')).read().split('t0 = time.time()')[0])
sys.argv = _argv


def calibrate(y, p):
    """threshold maximising balanced accuracy on the TRAINING cohort only"""
    ts = np.unique(np.concatenate([p, np.linspace(p.min(), p.max(), 300)]))
    best, bt = -1, float(np.median(p))
    for t in ts:
        b = balanced_accuracy_score(y, (p >= t).astype(int))
        if b > best: best, bt = b, float(t)
    return bt


def full(y, p, thr):
    yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    return dict(AUC=roc_auc_score(y, p), Acc=accuracy_score(y, yh),
                BalAcc=balanced_accuracy_score(y, yh),
                Sens=tp/(tp+fn) if (tp+fn) else np.nan,
                Spec=tn/(tn+fp) if (tn+fp) else np.nan,
                PPV=tp/(tp+fp) if (tp+fp) else np.nan,
                NPV=tn/(tn+fn) if (tn+fn) else np.nan,
                F1=f1_score(y, yh, zero_division=0),
                TN=tn, FP=fp, FN=fn, TP=tp)


t0 = time.time()
I, E, thrs = [], [], []
for sd in range(NS):
    pr = loo(bA, sd)                                   # MSW leave-one-out
    pb = predict(np.arange(len(bA)), 'EXT', bA, sd)    # trained on all MSW
    thr = calibrate(bA, pr)                            # MSW only
    I.append(full(bA, pr, thr)); E.append(full(bB, pb, thr)); thrs.append(thr)

agg = lambda S, k: (float(np.mean([d[k] for d in S])), float(np.std([d[k] for d in S])))
res = dict(model=MODEL, K=K, nseeds=NS, thr=float(np.mean(thrs)),
           internal={k: agg(I, k) for k in I[0]}, external={k: agg(E, k) for k in E[0]},
           secs=time.time()-t0)
pickle.dump(res, open(os.path.join(OUT, f'{MODEL.replace(" ","_")}_{K}.pkl'), 'wb'))
ks = ['AUC', 'Acc', 'BalAcc', 'Sens', 'Spec', 'PPV', 'NPV', 'F1']
for lab, S in [('int', res['internal']), ('ext', res['external'])]:
    vals = "  ".join(f"{k} {S[k][0]:.3f}" for k in ks)
    cm = f"TN {S['TN'][0]:.1f} FP {S['FP'][0]:.1f} FN {S['FN'][0]:.1f} TP {S['TP'][0]:.1f}"
    print(f"{MODEL:20s} K={K:2d} {lab}  {vals}  | {cm}", flush=True)
