"""Every model uses an MSW-calibrated threshold, so the procedure is not the
explanation for differing external specificity. This measures how far each
model's score distribution moves between sites, and whether that tracks the
specificity drop.

usage: shift_by_model.py <model> <K> <nseeds>
"""
import sys, os
SPDIR = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(SPDIR)
OUT = os.path.join(REPO, 'docs', 'dbs_3d', 'shift'); os.makedirs(OUT, exist_ok=True)
DATA = os.environ.get('DBS_CLINICAL_DIR', os.path.join(REPO, 'docs'))
CACHE = os.environ.get('DBS_CACHE_DIR', '/checkpoints')
sys.path.insert(0, REPO); sys.path.insert(0, SPDIR); os.chdir(REPO)
import numpy as np, pickle, warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix
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


PR, PB, TH = [], [], []
for sd in range(NS):
    PR.append(loo(bA, sd)); PB.append(predict(np.arange(len(bA)), 'EXT', bA, sd)); TH.append(cal(bA, PR[-1]))
pr, pb, thr = np.mean(PR, 0), np.mean(PB, 0), float(np.mean(TH))
sp = lambda y, p, t: (lambda c: c[0]/(c[0]+c[1]))(confusion_matrix(y, (p >= t).astype(int), labels=[0, 1]).ravel()[:2])
q = float((pr < thr).mean())                       # the MSW quantile the threshold sits at
tq = float(np.quantile(pb, q))                     # same quantile in CHH
res = dict(model=MODEL, K=K, thr=thr,
           msw_mean=float(pr.mean()), msw_sd=float(pr.std()),
           chh_mean=float(pb.mean()), chh_sd=float(pb.std()),
           shift_sd=float((pb.mean()-pr.mean())/(pr.std()+1e-12)),
           frac_above_msw=float((pr >= thr).mean()), frac_above_chh=float((pb >= thr).mean()),
           spec_int=sp(bA, pr, thr), spec_ext=sp(bB, pb, thr),
           spec_ext_quantile=sp(bB, pb, tq), spec_ext_best=sp(bB, pb, cal(bB, pb)),
           auc_ext=roc_auc_score(bB, pb))
pickle.dump(res, open(os.path.join(OUT, f'{MODEL.replace(" ","_")}.pkl'), 'wb'))
print(f"{MODEL:20s} shift {res['shift_sd']:+.2f} MSW-SD | above-thr MSW {res['frac_above_msw']:.2f} "
      f"CHH {res['frac_above_chh']:.2f} | spec int {res['spec_int']:.2f} ext {res['spec_ext']:.2f} "
      f"quantile-matched {res['spec_ext_quantile']:.2f} best-possible {res['spec_ext_best']:.2f}", flush=True)
