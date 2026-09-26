"""Patient-level rerun of the rebuttal Table 3 baselines, matched protocol.

Table 3 as submitted was slice-wise, which is why specificities like 0.948 and
0.967 appear despite 3 external non-responders. Everything here is patient-level
(59 MSW / 37 CHH), thresholds are calibrated on MSW out-of-fold only, and all
models see the same folds and the same 20 seeds.

Models, following the rebuttal:
  Clinical       covariates only
  Spectral LR    logistic regression on spectral coefficients
  Spectral MLP   MLP on spectral coefficients
  Spatial ViT    transformer over spatial patches of the same crop
  SpectralViT    spectral tokens + covariates, concat-token attention (ours)

Attention U-Net is omitted: it is a segmentation architecture whose DBS variant
is not reconstructable from this repo's committed code.
"""
import sys, os
SPDIR = os.path.dirname(os.path.abspath(__file__))
REPO  = os.path.dirname(SPDIR)
OUT   = os.path.join(REPO, 'docs', 'dbs_3d')
DATA  = os.environ.get('DBS_CLINICAL_DIR', os.path.join(REPO, 'docs'))
CACHE = os.environ.get('DBS_CACHE_DIR', '/checkpoints')
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, REPO); sys.path.insert(0, SPDIR); os.chdir(REPO)
import numpy as np, pandas as pd, torch, torch.nn as nn, pickle, time, warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score, confusion_matrix

NS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
exec(open(os.path.join(SPDIR, 'concat_attn.py')).read().split("rows=[]")[0])
from basis_fix import basis3_iso, low_modes, eigvals
from rank_model import RankAttn

dev = torch.device('cuda'); T = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
SHAPE = (BB[0].stop-BB[0].start, BB[1].stop-BB[1].start, BB[2].stop-BB[2].start)
sig = lambda z: 1.0 / (1.0 + np.exp(-z))
K = 16
IDX = low_modes(K, SHAPE, 'lap')
LAM = eigvals(IDX, SHAPE)
SPEC_A = basis3_iso(VA, '3DDCT', K, BB, 'lap')
SPEC_B = basis3_iso(VB, '3DDCT', K, BB, 'lap')
# spatial patches of the same crop, matched token count
PS = 5
def patches(V):
    c = V[:, BB[0], BB[1], BB[2]]
    n = c.shape[0]
    p = [c[:, i:i+PS, j:j+PS, k:k+PS].reshape(n, -1).mean(1)
         for i in range(0, SHAPE[0]-PS+1, PS)
         for j in range(0, SHAPE[1]-PS+1, PS)
         for k in range(0, SHAPE[2]-PS+1, PS)]
    return np.stack(p, 1)
PAT_A, PAT_B = patches(VA), patches(VB)


class TokenViT(nn.Module):
    """same body as ours, tokens supplied by the caller (spatial or spectral)"""
    def __init__(s, n, d=16):
        super().__init__()
        s.tok = nn.Linear(1, d); s.pos = nn.Parameter(torch.zeros(1, n, d))
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 2, d*2, 0.1, batch_first=True), 1)
        s.hd = nn.Linear(d, 1); s.lin = nn.Linear(n, 1)
    def forward(s, x):
        z = s.tok(x.unsqueeze(-1)) + s.pos
        return s.hd(s.enc(z).mean(1)).squeeze(-1) + s.lin(x).squeeze(-1)


def fit_torch(model_fn, Xtr, ytr, Xte, seed, ep=300, smooth=0.10):
    torch.manual_seed(seed); m = model_fn().to(dev)
    opt = torch.optim.AdamW(m.parameters(), 3e-3, weight_decay=0.1)
    pw = torch.tensor((ytr == 0).sum()/max(1, (ytr == 1).sum()), dtype=torch.float32, device=dev)
    lf = nn.BCEWithLogitsLoss(pos_weight=pw)
    yt = torch.tensor(ytr*(1-smooth) + 0.5*smooth, dtype=torch.float32, device=dev)
    xt = T(Xtr)
    for _ in range(ep):
        opt.zero_grad(); lf(m(xt), yt).backward(); opt.step()
    m.eval()
    with torch.no_grad(): return sig(m(T(Xte)).cpu().numpy())


def fit_ours(Xtr, Ctr, ytr, Xte, Cte, seed, ep=300, smooth=0.10):
    torch.manual_seed(seed)
    m = RankAttn(Xtr.shape[1], Ctr.shape[1], LAM, weight='lap').to(dev)
    opt = torch.optim.AdamW(m.parameters(), 3e-3, weight_decay=0.1)
    pw = torch.tensor((ytr == 0).sum()/max(1, (ytr == 1).sum()), dtype=torch.float32, device=dev)
    lf = nn.BCEWithLogitsLoss(pos_weight=pw)
    yt = torch.tensor(ytr*(1-smooth) + 0.5*smooth, dtype=torch.float32, device=dev)
    xt, ct = T(Xtr), T(Ctr)
    for _ in range(ep):
        opt.zero_grad(); lf(m(xt, ct), yt).backward(); opt.step()
    m.eval()
    with torch.no_grad(): return sig(m(T(Xte), T(Cte)).cpu().numpy())


def calibrate(y, p):
    ts = np.unique(np.concatenate([p, np.linspace(p.min(), p.max(), 200)]))
    best, bt = -1, float(np.median(p))
    for t in ts:
        b = balanced_accuracy_score(y, (p >= t).astype(int))
        if b > best: best, bt = b, float(t)
    return bt


def scores(y, p, thr):
    yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    return dict(AUC=roc_auc_score(y, p), BalAcc=balanced_accuracy_score(y, yh),
                Sens=tp/(tp+fn) if (tp+fn) else np.nan,
                Spec=tn/(tn+fp) if (tn+fp) else np.nan,
                F1=f1_score(y, yh, zero_division=0))


MODELS = {
    'Clinical':     ('sk',   None),
    'Spectral LR':  ('sk',   None),
    'Spectral MLP': ('sk',   None),
    'Spatial ViT':  ('torch', None),
    'SpectralViT (ours)': ('ours', None),
}
rows = []
for name in MODELS:
    ii, ee = [], []
    for sd in range(NS):
        pr = np.full(len(bA), np.nan)
        for tr, va in StratifiedKFold(4, shuffle=True, random_state=0).split(SPEC_A, bA):
            sC = StandardScaler().fit(Ca[tr]); sS = StandardScaler().fit(SPEC_A[tr])
            sP = StandardScaler().fit(PAT_A[tr])
            if name == 'Clinical':
                m = LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced').fit(sC.transform(Ca[tr]), bA[tr])
                pr[va] = sig(m.decision_function(sC.transform(Ca[va])))
            elif name == 'Spectral LR':
                m = LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced').fit(sS.transform(SPEC_A[tr]), bA[tr])
                pr[va] = sig(m.decision_function(sS.transform(SPEC_A[va])))
            elif name == 'Spectral MLP':
                m = MLPClassifier(hidden_layer_sizes=(32,), max_iter=2000, random_state=sd).fit(sS.transform(SPEC_A[tr]), bA[tr])
                pr[va] = m.predict_proba(sS.transform(SPEC_A[va]))[:, 1]
            elif name == 'Spatial ViT':
                pr[va] = fit_torch(lambda: TokenViT(PAT_A.shape[1]), sP.transform(PAT_A[tr]),
                                   bA[tr].astype(float), sP.transform(PAT_A[va]), sd)
            else:
                pr[va] = fit_ours(sS.transform(SPEC_A[tr]), sC.transform(Ca[tr]),
                                  bA[tr].astype(float), sS.transform(SPEC_A[va]), sC.transform(Ca[va]), sd)
        sC = StandardScaler().fit(Ca); sS = StandardScaler().fit(SPEC_A); sP = StandardScaler().fit(PAT_A)
        if name == 'Clinical':
            m = LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced').fit(sC.transform(Ca), bA)
            pb = sig(m.decision_function(sC.transform(Cb)))
        elif name == 'Spectral LR':
            m = LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced').fit(sS.transform(SPEC_A), bA)
            pb = sig(m.decision_function(sS.transform(SPEC_B)))
        elif name == 'Spectral MLP':
            m = MLPClassifier(hidden_layer_sizes=(32,), max_iter=2000, random_state=sd).fit(sS.transform(SPEC_A), bA)
            pb = m.predict_proba(sS.transform(SPEC_B))[:, 1]
        elif name == 'Spatial ViT':
            pb = fit_torch(lambda: TokenViT(PAT_A.shape[1]), sP.transform(PAT_A), bA.astype(float), sP.transform(PAT_B), sd)
        else:
            pb = fit_ours(sS.transform(SPEC_A), sC.transform(Ca), bA.astype(float),
                          sS.transform(SPEC_B), sC.transform(Cb), sd)
        thr = calibrate(bA, pr)                       # MSW only
        ii.append(scores(bA, pr, thr)); ee.append(scores(bB, pb, thr))
    for lab, ss in [('internal', ii), ('external', ee)]:
        rows.append(dict(model=name, split=lab,
                         **{k: np.mean([d[k] for d in ss]) for k in ('AUC', 'BalAcc', 'Sens', 'Spec', 'F1')},
                         **{k + '_sd': np.std([d[k] for d in ss]) for k in ('AUC', 'BalAcc', 'Spec')}))
    log(f"{name} done")

df = pd.DataFrame(rows)
for split in ('internal', 'external'):
    print(f"\n=== {split.upper()}  (patient level, MSW-calibrated threshold, {NS} seeds) ===")
    d = df[df.split == split][['model', 'AUC', 'AUC_sd', 'BalAcc', 'Sens', 'Spec', 'Spec_sd', 'F1']]
    print(d.round(3).to_string(index=False))
df.to_csv(os.path.join(OUT, 'baselines.csv'), index=False)
log("DONE")
