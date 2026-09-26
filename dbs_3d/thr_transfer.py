"""Why is SpectralViT's external specificity low when its external AUC is high?

Specificity is set by where the threshold lands, and the threshold is calibrated
on MSW. If CHH scores sit systematically higher or lower, the cut lands in the
wrong place even though the ranking is fine.
"""
import sys, os
SPDIR = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(SPDIR)
DATA = os.environ.get('DBS_CLINICAL_DIR', os.path.join(REPO, 'docs'))
CACHE = os.environ.get('DBS_CACHE_DIR', '/checkpoints')
sys.path.insert(0, REPO); sys.path.insert(0, SPDIR); os.chdir(REPO)
import numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix
MODEL, K, NS = 'SpectralViT', 16, 10
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
print(f"MSW-calibrated threshold: {thr:.4f}\n")
print(f"score distribution")
print(f"  MSW  all      mean {pr.mean():.4f}  sd {pr.std():.4f}   [{pr.min():.3f}, {pr.max():.3f}]")
print(f"  CHH  all      mean {pb.mean():.4f}  sd {pb.std():.4f}   [{pb.min():.3f}, {pb.max():.3f}]")
print(f"  MSW  non-resp mean {pr[bA==0].mean():.4f}   CHH non-resp mean {pb[bB==0].mean():.4f}")
print(f"  MSW  resp     mean {pr[bA==1].mean():.4f}   CHH resp     mean {pb[bB==1].mean():.4f}")
print(f"\n  shift in mean score, CHH - MSW: {pb.mean()-pr.mean():+.4f}"
      f"   ({(pb.mean()-pr.mean())/pr.std():+.2f} MSW SDs)")
print(f"  fraction of CHH above the MSW threshold: {(pb>=thr).mean():.3f}"
      f"   (MSW: {(pr>=thr).mean():.3f})")

for lab, y, p, t in [('MSW  @MSW thr', bA, pr, thr), ('CHH  @MSW thr', bB, pb, thr),
                     ('CHH  @its own optimum', bB, pb, cal(bB, pb))]:
    tn, fp, fn, tp = confusion_matrix(y, (p >= t).astype(int), labels=[0, 1]).ravel()
    n0 = tn + fp
    print(f"\n{lab}: thr {t:.4f}  spec {tn/n0:.3f} ({tn}/{n0})  "
          f"balacc {balanced_accuracy_score(y,(p>=t).astype(int)):.3f}  AUC {roc_auc_score(y,p):.3f}")
# where would the threshold have to be for CHH?
q = np.quantile(pr, 1 - (pr >= thr).mean())
print(f"\nMSW threshold is at MSW quantile {(pr < thr).mean():.3f}")
print(f"the same QUANTILE applied to CHH would be thr {np.quantile(pb, (pr < thr).mean()):.4f}")
tq = np.quantile(pb, (pr < thr).mean())
tn, fp, fn, tp = confusion_matrix(bB, (pb >= tq).astype(int), labels=[0, 1]).ravel()
print(f"  -> CHH spec {tn/(tn+fp):.3f} ({tn}/{tn+fp})  balacc {balanced_accuracy_score(bB,(pb>=tq).astype(int)):.3f}")
