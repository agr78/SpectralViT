"""Every model against every ROI construction.

Results so far used one box on the higher-susceptibility nucleus. The both-sides
probe suggested the left-right difference is far less site-separable (0.749 vs
0.922), which would make it transfer better -- but that was an imaging-only
logistic probe, so it does not tell us which input is better for the full
models. This crosses them.

ROI variants
  higher     the higher-susceptibility nucleus        (used in all prior results)
  both       both nuclei, features concatenated
  mean       the two boxes averaged                   (symmetric)
  diff       their difference                         (asymmetric; cancels site)

Models: Clinical, Spectral LR, Spectral MLP, Spatial ViT, SpectralViT (ours).
Clinical does not depend on the ROI so it is reported once.

Crop windows are clamped to fit inside the volume rather than clipped and
dropped, which recovers subjects whose nucleus sits near the edge -- including
one MSW non-responder.

usage: roi_x_models.py [nseeds]
"""
import sys, os
SPDIR = os.path.dirname(os.path.abspath(__file__))
REPO  = os.path.dirname(SPDIR)
OUT   = os.path.join(REPO, 'docs', 'dbs_3d')
DATA  = os.environ.get('DBS_CLINICAL_DIR', os.path.join(REPO, 'docs'))
CACHE = os.environ.get('DBS_CACHE_DIR', '/checkpoints')
sys.path.insert(0, REPO); sys.path.insert(0, SPDIR); os.chdir(REPO)
import numpy as np, pandas as pd, torch, torch.nn as nn, pickle, time, warnings
warnings.filterwarnings('ignore')
from scipy import ndimage
from scipy.fft import dctn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score, confusion_matrix
from reader import prepare_qsm_dataset
from util import mask_crop
from basis_fix import low_modes, eigvals
from rank_model import RankAttn
from _clin import clin_tables, FULL

NS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
H, DZ, K = 10, 8, 16
dev = torch.device('cuda'); T = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
sig = lambda z: 1.0/(1.0+np.exp(-z))
t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
mcf = lambda d, m, p: mask_crop(d, m, p)


def outcomes():
    d = pd.read_csv(os.path.join(DATA, 'dbs_03292024.csv'), header=1)
    d.columns = [str(c).strip().replace('\n', ' ') for c in d.columns]
    o = pd.to_numeric(d['OFF (pre-dbs updrs)'], errors='coerce')
    q = pd.to_numeric(d['OFF meds ON stim 6mo'], errors='coerce')
    s = pd.to_numeric(d['CORNELL ID'], errors='coerce')
    A = {int(a): (b-c)/b for a, b, c in zip(s, o, q)
         if pd.notna(a) and pd.notna(b) and pd.notna(c) and b > 0}
    c = pd.read_csv(os.path.join(DATA, 'chh_subjects_table1_20240729.csv'), header=None).iloc[2:]
    B = {}
    for _, r in c.iterrows():
        try:
            sid = int(float(r[0])); x = pd.to_numeric(r[9], errors='coerce'); y = pd.to_numeric(r[11], errors='coerce')
            if pd.notna(x) and pd.notna(y) and x > 0: B[sid] = (x-y)/x
        except Exception: pass
    return A, B


def crops(ds, imp):
    HI, LO, S = [], [], []
    for s in sorted(ds.volumes):
        s = int(s)
        if s not in imp: continue
        v = np.asarray(ds.volumes[s], np.float32); m = np.asarray(ds.seg_masks[s]) > 0
        lb, n = ndimage.label(m)
        if n < 2 or m.sum() > 0.5*m.size: continue
        sz = ndimage.sum(m, lb, range(1, n+1)); big = np.argsort(sz)[::-1][:2] + 1
        info = sorted([(float(v[lb == b].mean()), b) for b in big], key=lambda t: t[0])
        cs = []
        for _, b in info:
            cx, cy, cz = [int(round(x)) for x in ndimage.center_of_mass(m, lb, [b])[0]]
            x0 = min(max(0, cx-H), v.shape[0]-2*H)       # clamp, do not clip-and-drop
            y0 = min(max(0, cy-H), v.shape[1]-2*H)
            z0 = min(max(0, cz-DZ), v.shape[2]-2*DZ)
            cs.append(v[x0:x0+2*H, y0:y0+2*H, z0:z0+2*DZ])
        if any(c.shape != (2*H, 2*H, 2*DZ) for c in cs): continue
        LO.append(cs[0]); HI.append(cs[1]); S.append(s)
    return np.array(HI, np.float32), np.array(LO, np.float32), np.array(S)


impA, impB = outcomes()
dsA = prepare_qsm_dataset('MSW', '/x', '/y', os.path.join(DATA, 'dbs_03292024.csv'),
                          os.path.join(CACHE, 'msw_cache_6d_cv.pt'), load_cache=True, mask_crop_fn=mcf, cv_pad=False)
dsB = prepare_qsm_dataset('CHH', '/x', '/y', os.path.join(DATA, 'chh_subjects_table1_20240729.csv'),
                          os.path.join(CACHE, 'chh_cache_6d_cv.pt'), load_cache=True, mask_crop_fn=mcf, cv_pad=False)
HA, LA, SA_ = crops(dsA, impA); HB, LB, SB_ = crops(dsB, impB)
yA = np.array([impA[s] for s in SA_]); yB = np.array([impB[s] for s in SB_])
bA = (yA >= 0.30).astype(int); bB = (yB >= 0.30).astype(int)
CA, CB = clin_tables()


def cmat(tab, subs):
    M = np.array([[tab.loc[s, c] if (s in tab.index and pd.notna(tab.loc[s, c])) else np.nan
                   for c in FULL] for s in subs], float)
    med = np.nanmedian(M, 0); ix = np.where(np.isnan(M)); M[ix] = np.take(med, ix[1]); return M


Ca, Cb = cmat(CA, list(SA_)), cmat(CB, list(SB_))
print(f"MSW {HA.shape}  {bA.sum()} resp / {(bA==0).sum()} NON-resp")
print(f"CHH {HB.shape}  {bB.sum()} resp / {(bB==0).sum()} NON-resp\n")

IDX = low_modes(K, (2*H, 2*H, 2*DZ), 'lap'); LAM = eigvals(IDX, (2*H, 2*H, 2*DZ))
f = lambda V: np.stack([dctn(V, axes=(1, 2, 3), norm='ortho')[:, a, b, c] for a, b, c in IDX], 1)
ROIS = {
    'higher':  (f(HA), f(HB)),
    'both':    (np.hstack([f(HA), f(LA)]), np.hstack([f(HB), f(LB)])),
    'mean':    (f((HA+LA)/2), f((HB+LB)/2)),
    'diff':    (f(HA-LA), f(HB-LB)),
}
PS = 5
patch = lambda V: np.stack([V[:, i:i+PS, j:j+PS, k:k+PS].reshape(len(V), -1).mean(1)
                            for i in range(0, 2*H-PS+1, PS) for j in range(0, 2*H-PS+1, PS)
                            for k in range(0, 2*DZ-PS+1, PS)], 1)
PATCH = {'higher': (patch(HA), patch(HB)), 'both': (np.hstack([patch(HA), patch(LA)]), np.hstack([patch(HB), patch(LB)])),
         'mean': (patch((HA+LA)/2), patch((HB+LB)/2)), 'diff': (patch(HA-LA), patch(HB-LB))}


class TokenViT(nn.Module):
    def __init__(s, n, d=16):
        super().__init__()
        s.tok = nn.Linear(1, d); s.pos = nn.Parameter(torch.zeros(1, n, d))
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 2, d*2, 0.1, batch_first=True), 1)
        s.hd = nn.Linear(d, 1); s.lin = nn.Linear(n, 1)
    def forward(s, x):
        z = s.tok(x.unsqueeze(-1)) + s.pos
        return s.hd(s.enc(z).mean(1)).squeeze(-1) + s.lin(x).squeeze(-1)


def tfit(mk, Xtr, ytr, Xte, seed, ep=200):
    torch.manual_seed(seed); m = mk().to(dev)
    opt = torch.optim.AdamW(m.parameters(), 3e-3, weight_decay=0.1); lf = nn.BCEWithLogitsLoss()
    xt, yt = T(Xtr), torch.tensor(ytr, dtype=torch.float32, device=dev)
    for _ in range(ep):
        opt.zero_grad(); lf(m(xt), yt).backward(); opt.step()
    m.eval()
    with torch.no_grad(): return sig(m(T(Xte)).cpu().numpy())


def ofit(Xtr, Ctr, ytr, Xte, Cte, seed, ep=200):
    torch.manual_seed(seed)
    m = RankAttn(Xtr.shape[1], Ctr.shape[1], np.resize(LAM, Xtr.shape[1]), weight='lap').to(dev)
    opt = torch.optim.AdamW(m.parameters(), 3e-3, weight_decay=0.1); lf = nn.BCEWithLogitsLoss()
    xt, ct, yt = T(Xtr), T(Ctr), torch.tensor(ytr, dtype=torch.float32, device=dev)
    for _ in range(ep):
        opt.zero_grad(); lf(m(xt, ct), yt).backward(); opt.step()
    m.eval()
    with torch.no_grad(): return sig(m(T(Xte), T(Cte)).cpu().numpy())


def site_auc(A, B):
    X = np.vstack([A, B]); y = np.r_[np.zeros(len(A)), np.ones(len(B))]
    s = StandardScaler().fit_transform(X); o = []
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=0).split(s, y):
        m = LogisticRegression(C=1.0, max_iter=3000).fit(s[tr], y[tr])
        o.append(roc_auc_score(y[va], m.decision_function(s[va])))
    return float(np.mean(o))


def run(model, XA, XB, PA_, PB_):
    ii, ee = [], []
    for sd in range(NS):
        pr = np.full(len(bA), np.nan)
        for tr, va in StratifiedKFold(4, shuffle=True, random_state=0).split(XA, bA):
            sX = StandardScaler().fit(XA[tr]); sP = StandardScaler().fit(PA_[tr]); sC = StandardScaler().fit(Ca[tr])
            if model == 'Clinical':
                m = LogisticRegression(C=1.0, max_iter=3000, class_weight='balanced').fit(sC.transform(Ca[tr]), bA[tr])
                pr[va] = sig(m.decision_function(sC.transform(Ca[va])))
            elif model == 'Spectral LR':
                m = LogisticRegression(C=1.0, max_iter=3000, class_weight='balanced').fit(sX.transform(XA[tr]), bA[tr])
                pr[va] = sig(m.decision_function(sX.transform(XA[va])))
            elif model == 'Spectral MLP':
                m = MLPClassifier(hidden_layer_sizes=(32,), max_iter=2000, random_state=sd).fit(sX.transform(XA[tr]), bA[tr])
                pr[va] = m.predict_proba(sX.transform(XA[va]))[:, 1]
            elif model == 'Spatial ViT':
                pr[va] = tfit(lambda: TokenViT(PA_.shape[1]), sP.transform(PA_[tr]), bA[tr].astype(float), sP.transform(PA_[va]), sd)
            else:
                pr[va] = ofit(sX.transform(XA[tr]), sC.transform(Ca[tr]), bA[tr].astype(float),
                              sX.transform(XA[va]), sC.transform(Ca[va]), sd)
        sX = StandardScaler().fit(XA); sP = StandardScaler().fit(PA_); sC = StandardScaler().fit(Ca)
        if model == 'Clinical':
            m = LogisticRegression(C=1.0, max_iter=3000, class_weight='balanced').fit(sC.transform(Ca), bA)
            pb = sig(m.decision_function(sC.transform(Cb)))
        elif model == 'Spectral LR':
            m = LogisticRegression(C=1.0, max_iter=3000, class_weight='balanced').fit(sX.transform(XA), bA)
            pb = sig(m.decision_function(sX.transform(XB)))
        elif model == 'Spectral MLP':
            m = MLPClassifier(hidden_layer_sizes=(32,), max_iter=2000, random_state=sd).fit(sX.transform(XA), bA)
            pb = m.predict_proba(sX.transform(XB))[:, 1]
        elif model == 'Spatial ViT':
            pb = tfit(lambda: TokenViT(PA_.shape[1]), sP.transform(PA_), bA.astype(float), sP.transform(PB_), sd)
        else:
            pb = ofit(sX.transform(XA), sC.transform(Ca), bA.astype(float), sX.transform(XB), sC.transform(Cb), sd)
        ii.append(roc_auc_score(bA, pr)); ee.append(roc_auc_score(bB, pb))
    return np.mean(ii), np.std(ii), np.mean(ee), np.std(ee)


rows = []
print(f"{'model':20s} {'ROI':8s} {'int AUC':>14} {'ext AUC':>14} {'site AUC':>9}")
for roi, (XA, XB) in ROIS.items():
    sa = site_auc(StandardScaler().fit_transform(XA), StandardScaler().fit(XA).transform(XB))
    PA_, PB_ = PATCH[roi]
    for model in ['Clinical', 'Spectral LR', 'Spectral MLP', 'Spatial ViT', 'SpectralViT (ours)']:
        if model == 'Clinical' and roi != 'higher': continue
        im, isd, em, esd = run(model, XA, XB, PA_, PB_)
        print(f"{model:20s} {roi:8s} {im:7.3f}+-{isd:.3f} {em:7.3f}+-{esd:.3f} {sa:9.3f}", flush=True)
        rows.append(dict(model=model, roi=roi, int_auc=im, int_sd=isd, ext_auc=em, ext_sd=esd, site_auc=sa))
pd.DataFrame(rows).to_csv(os.path.join(OUT, 'roi_x_models.csv'), index=False)
print("DONE")
