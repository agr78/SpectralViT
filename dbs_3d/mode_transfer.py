"""Which individual spectral modes carry outcome signal that is not site signal?

The 16-mode feature vector identifies the site at AUC 0.940 while predicting
outcome externally at 0.729. If the two live in different modes, selecting on
site-invariance should retain the outcome signal and drop the site signal.

Per mode, two scores:
  outcome AUC   |AUC - 0.5| against responder status, MSW only
  site AUC      |AUC - 0.5| against MSW-vs-CHH membership

Neither uses a CHH outcome label, so selecting on both is leak-free with respect
to the external endpoint. Site membership is known for every scan by
construction.

A larger mode budget is scanned than the K=16 used so far, since restricting to
16 may simply have kept the site-dominated ones.
"""
import sys, os
SPDIR = os.path.dirname(os.path.abspath(__file__))
REPO  = os.path.dirname(SPDIR)
OUT   = os.path.join(REPO, 'docs', 'dbs_3d')
DATA  = os.environ.get('DBS_CLINICAL_DIR', os.path.join(REPO, 'docs'))
CACHE = os.environ.get('DBS_CACHE_DIR', '/checkpoints')
sys.path.insert(0, REPO); sys.path.insert(0, SPDIR); os.chdir(REPO)
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

exec(open(os.path.join(SPDIR, 'concat_attn.py')).read().split("rows=[]")[0])
from basis_fix import basis3_iso, low_modes, eigvals

SHAPE = (BB[0].stop-BB[0].start, BB[1].stop-BB[1].start, BB[2].stop-BB[2].start)
KBIG = 64
IDX = low_modes(KBIG, SHAPE, 'lap')
A = basis3_iso(VA, '3DDCT', KBIG, BB, 'lap')
B = basis3_iso(VB, '3DDCT', KBIG, BB, 'lap')
lam = eigvals(IDX, SHAPE)
site = np.r_[np.zeros(len(A)), np.ones(len(B))]
AB = np.vstack([A, B])

rows = []
for j in range(KBIG):
    oa = roc_auc_score(bA, A[:, j])                 # MSW outcome only
    sa = roc_auc_score(site, AB[:, j])              # site membership only
    rows.append(dict(mode=j, fx=IDX[j][0], fy=IDX[j][1], fz=IDX[j][2], lam=lam[j],
                     outcome=abs(oa - 0.5), site=abs(sa - 0.5),
                     outcome_raw=oa, site_raw=sa))
d = pd.DataFrame(rows)
d['ratio'] = d.outcome / (d.site + 1e-6)

print(f"{KBIG} modes, MSW {len(A)} / CHH {len(B)}\n")
print("most SITE-dominated modes (drop these):")
print(d.nlargest(6, 'site')[['mode', 'fx', 'fy', 'fz', 'outcome', 'site']].round(3).to_string(index=False))
print("\nmost OUTCOME-informative modes:")
print(d.nlargest(6, 'outcome')[['mode', 'fx', 'fy', 'fz', 'outcome', 'site']].round(3).to_string(index=False))
print("\nbest outcome-to-site RATIO (candidate transferable modes):")
print(d.nlargest(10, 'ratio')[['mode', 'fx', 'fy', 'fz', 'lam', 'outcome', 'site', 'ratio']].round(3).to_string(index=False))

print(f"\ncorrelation between per-mode outcome and site scores: "
      f"{np.corrcoef(d.outcome, d.site)[0,1]:+.3f}")
print("  (strongly positive would mean outcome and site are carried by the same modes)")


def evaluate(cols, label):
    """logistic probe on a mode subset: internal OOF and external"""
    Xa, Xb = A[:, cols], B[:, cols]
    s = StandardScaler().fit(Xa); Za, Zb = s.transform(Xa), s.transform(Xb)
    pr = np.full(len(bA), np.nan)
    for tr, va in StratifiedKFold(4, shuffle=True, random_state=0).split(Za, bA):
        m = LogisticRegression(C=1.0, max_iter=3000, class_weight='balanced').fit(Za[tr], bA[tr])
        pr[va] = m.decision_function(Za[va])
    m = LogisticRegression(C=1.0, max_iter=3000, class_weight='balanced').fit(Za, bA)
    pb = m.decision_function(Zb)
    # site separability of this subset
    sa = []
    Z = np.vstack([Za, Zb])
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=0).split(Z, site):
        ms = LogisticRegression(C=1.0, max_iter=3000).fit(Z[tr], site[tr])
        sa.append(roc_auc_score(site[va], ms.decision_function(Z[va])))
    print(f"  {label:34s} n={len(cols):2d}  int {roc_auc_score(bA, pr):.3f}  "
          f"ext {roc_auc_score(bB, pb):.3f}  site {np.mean(sa):.3f}")


print("\n=== imaging-only logistic probes on mode subsets ===")
print("  (clinical-only external reference 0.725)")
evaluate(list(range(16)), 'first 16 (what we used)')
evaluate(list(range(KBIG)), f'all {KBIG}')
for thr in (0.10, 0.15, 0.20):
    keep = d.index[d.site < thr].tolist()
    if len(keep) >= 4:
        evaluate(keep, f'site-invariant (|site AUC-0.5| < {thr})')
evaluate(d.nlargest(16, 'ratio').index.tolist(), 'top 16 by outcome/site ratio')
evaluate(d.nlargest(8, 'ratio').index.tolist(), 'top 8 by outcome/site ratio')
d.to_csv(os.path.join(OUT, 'mode_transfer.csv'), index=False)
