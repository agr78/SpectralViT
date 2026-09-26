"""Re-tune on the corrected 66-patient cohort.

Every hyperparameter in use was chosen when the cohort was 59 patients with 4
non-responders: K=16, d=16, 200 epochs, plain BCE. This sweeps them on the
corrected data, and adds ROI configurations we have not separated:

  higher / lower   ordered by mean susceptibility  (what we have used)
  left / right     ordered by position             (anatomical, a different split)
  both / mean / diff

The two mask components are left and right hemispheres -- x-centroids cluster
at 13 and 52 of 64 -- so susceptibility-ordering and anatomical ordering are
different groupings of the same two boxes.

LOO throughout, one seed for the sweep; the winner is then re-run multi-seed.
"""
import sys, os
SPDIR = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(SPDIR)
OUT = os.path.join(REPO, 'docs', 'dbs_3d')
DATA = os.environ.get('DBS_CLINICAL_DIR', os.path.join(REPO, 'docs'))
CACHE = os.environ.get('DBS_CACHE_DIR', '/checkpoints')
sys.path.insert(0, REPO); sys.path.insert(0, SPDIR); os.chdir(REPO)
import numpy as np, pandas as pd, torch, torch.nn as nn, itertools, time, warnings
warnings.filterwarnings('ignore')
from scipy import ndimage
from scipy.fft import dctn, fftn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from reader import prepare_qsm_dataset
from util import mask_crop
from basis_fix import low_modes, eigvals
from rank_model import RankAttn
from _clin import clin_tables, FULL

H, DZ = 10, 8
dev = torch.device('cuda'); T = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
sig = lambda z: 1/(1+np.exp(-z))
t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
mcf = lambda d, m, p: mask_crop(d, m, p)


def outcomes():
    d = pd.read_csv(os.path.join(DATA, 'dbs_03292024.csv'), header=1)
    d.columns = [str(c).strip().replace('\n', ' ') for c in d.columns]
    o = pd.to_numeric(d['OFF (pre-dbs updrs)'], errors='coerce')
    q = pd.to_numeric(d['OFF meds ON stim 6mo'], errors='coerce')
    s = pd.to_numeric(d['CORNELL ID'], errors='coerce')
    A = {int(a): (b-c)/b for a, b, c in zip(s, o, q) if pd.notna(a) and pd.notna(b) and pd.notna(c) and b > 0}
    c = pd.read_csv(os.path.join(DATA, 'chh_subjects_table1_20240729.csv'), header=None).iloc[2:]
    B = {}
    for _, r in c.iterrows():
        try:
            sid = int(float(r[0])); x = pd.to_numeric(r[9], errors='coerce'); y = pd.to_numeric(r[11], errors='coerce')
            if pd.notna(x) and pd.notna(y) and x > 0: B[sid] = (x-y)/x
        except Exception: pass
    return A, B


def crops(ds, imp):
    """returns higher, lower, left, right boxes per subject"""
    HI, LO, LF, RT, S = [], [], [], [], []
    for s in sorted(ds.volumes):
        s = int(s)
        if s not in imp: continue
        v = np.asarray(ds.volumes[s], np.float32); m = np.asarray(ds.seg_masks[s]) > 0
        lb, n = ndimage.label(m)
        if n < 2 or m.sum() > 0.5*m.size: continue
        sz = ndimage.sum(m, lb, range(1, n+1)); big = np.argsort(sz)[::-1][:2] + 1
        boxes = []
        for b in big:
            cx, cy, cz = [int(round(x)) for x in ndimage.center_of_mass(m, lb, [b])[0]]
            x0 = min(max(0, cx-H), v.shape[0]-2*H); y0 = min(max(0, cy-H), v.shape[1]-2*H)
            z0 = min(max(0, cz-DZ), v.shape[2]-2*DZ)
            boxes.append((v[x0:x0+2*H, y0:y0+2*H, z0:z0+2*DZ], cx, float(v[lb == b].mean())))
        if any(b[0].shape != (2*H, 2*H, 2*DZ) for b in boxes): continue
        by_sus = sorted(boxes, key=lambda t: t[2])       # low, high
        by_pos = sorted(boxes, key=lambda t: t[1])       # left, right
        LO.append(by_sus[0][0]); HI.append(by_sus[1][0])
        LF.append(by_pos[0][0]); RT.append(by_pos[1][0]); S.append(s)
    return [np.array(x, np.float32) for x in (HI, LO, LF, RT)] + [np.array(S)]


impA, impB = outcomes()
dsA = prepare_qsm_dataset('MSW', '/x', '/y', os.path.join(DATA, 'dbs_03292024.csv'),
                          os.path.join(CACHE, 'msw_cache_6d_cv.pt'), load_cache=True, mask_crop_fn=mcf, cv_pad=False)
dsB = prepare_qsm_dataset('CHH', '/x', '/y', os.path.join(DATA, 'chh_subjects_table1_20240729.csv'),
                          os.path.join(CACHE, 'chh_cache_6d_cv.pt'), load_cache=True, mask_crop_fn=mcf, cv_pad=False)
HA, LA, FA_, RA, SA_ = crops(dsA, impA)
HB, LB, FB_, RB, SB_ = crops(dsB, impB)
yA = np.array([impA[s] for s in SA_]); yB = np.array([impB[s] for s in SB_])
bA = (yA >= 0.30).astype(int); bB = (yB >= 0.30).astype(int)
CA, CB = clin_tables()
def cm(tab, subs):
    M = np.array([[tab.loc[s, c] if (s in tab.index and pd.notna(tab.loc[s, c])) else np.nan
                   for c in FULL] for s in subs], float)
    med = np.nanmedian(M, 0)
    ix = np.where(np.isnan(M))
    M[ix] = np.take(med, ix[1])
    return M


Ca, Cb = cm(CA, list(SA_)), cm(CB, list(SB_))
print(f"MSW {len(bA)} ({bA.sum()}/{(bA==0).sum()})   CHH {len(bB)} ({bB.sum()}/{(bB==0).sum()})\n")

ROI = {'higher': (HA, HB), 'lower': (LA, LB), 'left': (FA_, FB_), 'right': (RA, RB),
       'both': None, 'mean': ((HA+LA)/2, (HB+LB)/2), 'diff': (HA-LA, HB-LB)}


def feats(V, K, basis='logFFT'):
    idx = low_modes(K, (2*H, 2*H, 2*DZ), 'lap')
    Tm = (dctn(V, axes=(1, 2, 3), norm='ortho') if basis == 'DCT'
          else np.log1p(np.abs(fftn(V, axes=(1, 2, 3)))))
    return np.stack([Tm[:, a, b, c] for a, b, c in idx], 1), eigvals(idx, (2*H, 2*H, 2*DZ))


def fit(Xtr, Ctr, ytr, Xte, Cte, lam, seed, ep, d):
    torch.manual_seed(seed)
    m = RankAttn(Xtr.shape[1], Ctr.shape[1], np.resize(lam, Xtr.shape[1]), weight='lap', d=d).to(dev)
    opt = torch.optim.AdamW(m.parameters(), 3e-3, weight_decay=0.1); lf = nn.BCEWithLogitsLoss()
    xt, ct = T(Xtr), T(Ctr); yt = torch.tensor(ytr, dtype=torch.float32, device=dev)
    for _ in range(ep):
        opt.zero_grad(); lf(m(xt, ct), yt).backward(); opt.step()
    m.eval()
    with torch.no_grad(): return sig(m(T(Xte), T(Cte)).cpu().numpy())


rows = []
print(f"{'ROI':8s} {'basis':7s} {'K':>3} {'ep':>4} {'d':>3} {'LOO int':>8} {'ext':>7}")
for roi, basis, K, ep, d in itertools.product(['higher', 'both', 'left', 'right', 'diff'], ['logFFT', 'DCT'], [8, 16, 32], [200, 600], [16, 32]):
    if roi == 'both':
        fa, lam = feats(HA, K, basis); fb, _ = feats(HB, K, basis)
        fa2, _ = feats(LA, K, basis); fb2, _ = feats(LB, K, basis)
        XA, XB = np.hstack([fa, fa2]), np.hstack([fb, fb2])
    else:
        VA_, VB_ = ROI[roi]
        XA, lam = feats(VA_, K, basis); XB, _ = feats(VB_, K, basis)
    sX = StandardScaler().fit(XA); sC = StandardScaler().fit(Ca)
    ZA, ZB, DA, DB = sX.transform(XA), sX.transform(XB), sC.transform(Ca), sC.transform(Cb)
    pr = np.empty(len(bA))
    for i in range(len(bA)):
        tr = np.delete(np.arange(len(bA)), i)
        pr[i] = fit(ZA[tr], DA[tr], bA[tr].astype(float), ZA[i:i+1], DA[i:i+1], lam, 0, ep, d)[0]
    pb = fit(ZA, DA, bA.astype(float), ZB, DB, lam, 0, ep, d)
    ia, ea = roc_auc_score(bA, pr), roc_auc_score(bB, pb)
    print(f"{roi:8s} {basis:7s} {K:3d} {ep:4d} {d:3d} {ia:8.3f} {ea:7.3f}", flush=True)
    rows.append(dict(roi=roi, basis=basis, K=K, ep=ep, d=d, loo_int=ia, ext=ea))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'tune_corrected.csv'), index=False)
df = pd.DataFrame(rows)
print("\ntop 10 by LOO internal:")
print(df.nlargest(10, 'loo_int').round(3).to_string(index=False))
log("DONE")
