"""Is there anything left for imaging to explain after the clinical covariates?

Externally the imaging adds +0.004 AUC over covariates alone, while internally it
adds a lot (LOO 0.821 vs 0.691). Two explanations:

  (a) the imaging carries no information orthogonal to the covariates, in which
      case no architecture recovers anything, or
  (b) it carries orthogonal information that is site-specific, in which case
      domain adaptation is the right target.

This distinguishes them. Fit the clinical model, take its residual, and ask
whether the imaging predicts that residual -- internally and externally. Also
measures how much of the imaging signal is linearly explainable by the
covariates in the first place.
"""
import sys, os
SPDIR = os.path.dirname(os.path.abspath(__file__))
REPO  = os.path.dirname(SPDIR)
OUT   = os.path.join(REPO, 'docs', 'dbs_3d')
DATA  = os.environ.get('DBS_CLINICAL_DIR', os.path.join(REPO, 'docs'))
CACHE = os.environ.get('DBS_CACHE_DIR', '/checkpoints')
sys.path.insert(0, REPO); sys.path.insert(0, SPDIR); os.chdir(REPO)
import numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, RidgeCV
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

exec(open(os.path.join(SPDIR, 'concat_attn.py')).read().split("rows=[]")[0])
from basis_fix import basis3_iso, low_modes

K = 16
SA = basis3_iso(VA, '3DDCT', K, BB, 'lap')
SB = basis3_iso(VB, '3DDCT', K, BB, 'lap')
sS = StandardScaler().fit(SA); ZA, ZB = sS.transform(SA), sS.transform(SB)
sC = StandardScaler().fit(Ca); DA, DB = sC.transform(Ca), sC.transform(Cb)

print(f"MSW {len(bA)} ({bA.sum()}/{(bA==0).sum()})   CHH {len(bB)} ({bB.sum()}/{(bB==0).sum()})\n")

# 1. how much of each imaging feature is already explained by the covariates?
r2 = []
for j in range(ZA.shape[1]):
    m = LinearRegression().fit(DA, ZA[:, j])
    r2.append(m.score(DA, ZA[:, j]))
r2 = np.array(r2)
print("1. imaging features regressed on clinical covariates (MSW)")
print(f"   mean R2 {r2.mean():.3f}   max {r2.max():.3f}   "
      f"features with R2>0.3: {(r2 > 0.3).sum()}/{len(r2)}")
print("   -> how much imaging information is redundant with the covariates\n")

# 2. clinical out-of-fold residual, then can imaging predict it?
def clin_oof(y):
    p = np.full(len(y), np.nan)
    for tr, va in StratifiedKFold(4, shuffle=True, random_state=0).split(DA, y):
        m = LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced').fit(DA[tr], y[tr])
        p[va] = m.decision_function(DA[va])
    return p

pc = clin_oof(bA)
resid = bA - (1 / (1 + np.exp(-pc)))               # signed clinical error
print("2. does imaging predict the CLINICAL RESIDUAL?  (MSW, out-of-fold)")
rr = np.full(len(bA), np.nan)
for tr, va in KFold(5, shuffle=True, random_state=0).split(ZA):
    m = RidgeCV(alphas=np.logspace(-2, 4, 20)).fit(ZA[tr], resid[tr])
    rr[va] = m.predict(ZA[va])
r, p = pearsonr(resid, rr)
print(f"   internal residual r = {r:+.3f}  (p={p:.4f})")

m = RidgeCV(alphas=np.logspace(-2, 4, 20)).fit(ZA, resid)
mc = LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced').fit(DA, bA)
resid_B = bB - (1 / (1 + np.exp(-mc.decision_function(DB))))
rb = m.predict(ZB)
r2e, p2e = pearsonr(resid_B, rb)
print(f"   EXTERNAL residual r = {r2e:+.3f}  (p={p2e:.4f})")
print("   -> internal>0 external~0 means orthogonal but site-specific\n")

# 3. how site-specific are the imaging features? can we tell MSW from CHH?
X = np.vstack([ZA, ZB]); site = np.r_[np.zeros(len(ZA)), np.ones(len(ZB))]
sd = []
for tr, va in StratifiedKFold(5, shuffle=True, random_state=0).split(X, site):
    m = LogisticRegression(C=1.0, max_iter=2000).fit(X[tr], site[tr])
    sd.append(roc_auc_score(site[va], m.decision_function(X[va])))
print("3. can a classifier tell which SITE an imaging feature vector came from?")
print(f"   site-discrimination AUC = {np.mean(sd):.3f}   (0.5 = indistinguishable)")
same = []
for tr, va in StratifiedKFold(5, shuffle=True, random_state=0).split(np.vstack([DA, DB]), site):
    Xc = np.vstack([DA, DB])
    m = LogisticRegression(C=1.0, max_iter=2000).fit(Xc[tr], site[tr])
    same.append(roc_auc_score(site[va], m.decision_function(Xc[va])))
print(f"   same for the CLINICAL covariates = {np.mean(same):.3f}")
print("   -> if imaging is far more site-separable than clinical, that is the")
print("      quantity a domain-adversarial term would target")
