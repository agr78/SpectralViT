"""Full metric table with quantile-transferred operating point.

usage: metrics2.py <model> <K> <nseeds>"""
import sys, os
SPDIR = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(SPDIR)
MOUT = os.path.join(REPO, 'docs', 'dbs_3d', 'metrics2'); os.makedirs(MOUT, exist_ok=True)
DATA = os.environ.get('DBS_CLINICAL_DIR', os.path.join(REPO, 'docs'))
CACHE = os.environ.get('DBS_CACHE_DIR', '/checkpoints')
sys.path.insert(0, REPO); sys.path.insert(0, SPDIR); os.chdir(REPO)
import numpy as np, pickle, time, warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import (roc_auc_score, balanced_accuracy_score, f1_score,
                             accuracy_score, confusion_matrix)
MODEL, K, NS = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
_argv = sys.argv; sys.argv = [_argv[0], 'cell', MODEL, str(K), '1']
exec(open(os.path.join(SPDIR, 'sc_worker.py')).read().split('t0 = time.time()')[0])
sys.argv = _argv


def cal(y, p):
    best, bt = -1, float(np.median(p))
    for t in np.unique(np.concatenate([p, np.linspace(p.min(), p.max(), 300)])):
        b = balanced_accuracy_score(y, (p >= t).astype(int))
        if b > best: best, bt = b, float(t)
    return bt


def full(y, p, thr):
    yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    return dict(AUC=roc_auc_score(y, p), Acc=accuracy_score(y, yh),
                BalAcc=balanced_accuracy_score(y, yh),
                Spec=tn/(tn+fp) if (tn+fp) else np.nan,
                PPV=tp/(tp+fp) if (tp+fp) else np.nan,
                NPV=tn/(tn+fn) if (tn+fn) else np.nan,
                F1=f1_score(y, yh, zero_division=0), TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp))


t0 = time.time()
PR, PB = [], []
for sd in range(NS):
    PR.append(loo(bA, sd)); PB.append(predict(np.arange(len(bA)), 'EXT', bA, sd))
pr, pb = np.mean(PR, 0), np.mean(PB, 0)
thr = cal(bA, pr)
q = float((pr < thr).mean())
tq = float(np.quantile(pb, q))
res = dict(model=MODEL, K=K, nseeds=NS, pred_msw=pr, pred_chh=pb, y_msw=bA, y_chh=bB, thr=thr, quantile=q, thr_chh=tq,
           internal=full(bA, pr, thr),
           external_raw=full(bB, pb, thr),
           external_quantile=full(bB, pb, tq),
           shift_sd=float((pb.mean()-pr.mean())/(pr.std()+1e-12)),
           secs=time.time()-t0)
pickle.dump(res, open(os.path.join(MOUT, f'{MODEL.replace(" ","_")}.pkl'), 'wb'))
f = lambda d: (f"AUC {d['AUC']:.3f} Acc {d['Acc']:.3f} BalAcc {d['BalAcc']:.3f} "
               f"Spec {d['Spec']:.3f} PPV {d['PPV'] if not np.isnan(d['PPV']) else float('nan'):.3f} "
               f"NPV {d['NPV']:.3f} F1 {d['F1']:.3f} | TN {d['TN']} FP {d['FP']}")
print(f"{MODEL:20s} int      {f(res['internal'])}", flush=True)
print(f"{MODEL:20s} ext raw  {f(res['external_raw'])}", flush=True)
print(f"{MODEL:20s} ext qtl  {f(res['external_quantile'])}   (shift {res['shift_sd']:+.2f} SD)", flush=True)
