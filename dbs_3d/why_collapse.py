"""Why did internal/external correspondence collapse when 7 patients were restored?

59-patient cohort: LOO internal 0.821, external 0.816  (corresponded)
66-patient cohort: internal ~0.5,     external 0.69-0.86 (inverted)

The external set is unchanged, so the cause is the 7 restored patients. Two
possibilities, distinguished here:

  (a) they are simply hard    -> AUC on the original 59 stays high, AUC on the 7
                                 is poor, and the model itself is unharmed
  (b) they corrupt the fit    -> AUC on the original 59 also falls

Also reports where each restored patient lands in the ranking, and what happens
if the restored non-responder (subject 56) is excluded.
"""
import sys, os
SPDIR = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(SPDIR)
OUT = os.path.join(REPO, 'docs', 'dbs_3d')
DATA = os.environ.get('DBS_CLINICAL_DIR', os.path.join(REPO, 'docs'))
CACHE = os.environ.get('DBS_CACHE_DIR', '/checkpoints')
sys.path.insert(0, REPO); sys.path.insert(0, SPDIR); os.chdir(REPO)
import numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

src = open(os.path.join(SPDIR, 'roi_x_models.py')).read().split('class TokenViT')[0]
g = {'__file__': os.path.join(SPDIR, 'roi_x_models.py')}
exec(compile(src, 'x', 'exec'), g)
bA, bB, Ca, Cb, ROIS, SA_ = g['bA'], g['bB'], g['Ca'], g['Cb'], g['ROIS'], g['SA_']
RESTORED = {1, 56, 62, 93, 100, 111, 115}
isnew = np.array([s in RESTORED for s in SA_])
print(f"cohort {len(bA)}  ({bA.sum()} resp / {(bA==0).sum()} non-resp);  restored {isnew.sum()}")
print(f"  restored subjects: {[int(s) for s in SA_[isnew]]}")
print(f"  of which non-responders: {[int(s) for s in SA_[isnew & (bA==0)]]}\n")


def loo(Z, y):
    p = np.empty(len(y))
    for i in range(len(y)):
        tr = np.delete(np.arange(len(y)), i)
        m = LogisticRegression(C=1.0, max_iter=3000, class_weight='balanced').fit(Z[tr], y[tr])
        p[i] = m.decision_function(Z[i:i+1])[0]
    return p


for roi in ['higher', 'both']:
    XA, XB = ROIS[roi]
    Zc = np.hstack([XA, Ca])
    s = StandardScaler().fit(Zc); Z = s.transform(Zc)
    p = loo(Z, bA)
    old = ~isnew
    a_all = roc_auc_score(bA, p)
    a_old = roc_auc_score(bA[old], p[old]) if len(np.unique(bA[old])) > 1 else np.nan
    print(f"[{roi}]  LOO on all {len(bA)} -> AUC {a_all:.3f}")
    print(f"        same predictions, scored on the original 59 only -> {a_old:.3f}")
    # train WITHOUT the restored patients, score the original 59 (reproduces the old setting)
    Zo, yo = Z[old], bA[old]
    po = loo(Zo, yo)
    print(f"        trained and scored on the original 59 only       -> {roc_auc_score(yo, po):.3f}")
    # where do the restored patients rank?
    r = np.argsort(np.argsort(p))
    print(f"        restored patients' ranks (of {len(bA)}, higher = predicted responder):")
    for i in np.where(isnew)[0]:
        print(f"           subject {int(SA_[i]):3d}  rank {r[i]:2d}   {'responder' if bA[i] else 'NON-RESPONDER'}")
    # drop only the restored non-responder
    keep = ~(isnew & (bA == 0))
    if keep.sum() < len(bA):
        pk = loo(Z[keep], bA[keep])
        print(f"        excluding only the restored non-responder (n={keep.sum()}) -> {roc_auc_score(bA[keep], pk):.3f}")
    print()
