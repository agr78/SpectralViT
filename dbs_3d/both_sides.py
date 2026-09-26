"""Does using both nuclei help, or only the higher-susceptibility side?

Every result so far used a single 20x20x16 box centred on the higher-
susceptibility nucleus. This compares that against the alternatives:

  higher     the side with higher mean susceptibility   (what we have used)
  lower      the other side
  both       both boxes, features concatenated
  mean       the two boxes averaged voxelwise
  diff       their voxelwise difference (asymmetry)

Asymmetry is worth testing on its own: susceptibility is higher on one side in
~79% of patients, and the left-right difference is a different quantity from
either side alone.
"""
import sys, os
SPDIR = os.path.dirname(os.path.abspath(__file__))
REPO  = os.path.dirname(SPDIR)
OUT   = os.path.join(REPO, 'docs', 'dbs_3d')
DATA  = os.environ.get('DBS_CLINICAL_DIR', os.path.join(REPO, 'docs'))
CACHE = os.environ.get('DBS_CACHE_DIR', '/checkpoints')
sys.path.insert(0, REPO); sys.path.insert(0, SPDIR); os.chdir(REPO)
import numpy as np, pandas as pd, torch, torch.nn as nn, pickle, warnings
warnings.filterwarnings('ignore')
from scipy import ndimage
from scipy.fft import dctn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from reader import prepare_qsm_dataset
from util import mask_crop

NS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
H, DZ, K = 10, 8, 16
mcf = lambda d, m, p: mask_crop(d, m, p)


def load(site, csv, cache):
    return prepare_qsm_dataset(site, '/x', '/y', os.path.join(DATA, csv),
                               os.path.join(CACHE, cache), load_cache=True,
                               mask_crop_fn=mcf, cv_pad=False)


def both_crops(ds, imp):
    """returns (higher, lower) crops per subject"""
    HI, LO, S = [], [], []
    for s in sorted(ds.volumes):
        s = int(s)
        if s not in imp: continue
        v = np.asarray(ds.volumes[s], np.float32); m = np.asarray(ds.seg_masks[s]) > 0
        lb, n = ndimage.label(m)
        if n < 2 or m.sum() > 0.5 * m.size: continue
        sz = ndimage.sum(m, lb, range(1, n + 1))
        big = np.argsort(sz)[::-1][:2] + 1
        info = sorted([(float(v[lb == b].mean()), b) for b in big], key=lambda t: t[0])
        cs = []
        for _, b in info:                                   # low first, high second
            cx, cy, cz = [int(round(x)) for x in ndimage.center_of_mass(m, lb, [b])[0]]
            sl = (slice(max(0, cx-H), max(0, cx-H)+2*H), slice(max(0, cy-H), max(0, cy-H)+2*H),
                  slice(max(0, cz-DZ), max(0, cz-DZ)+2*DZ))
            cs.append(v[sl])
        if any(c.shape != (2*H, 2*H, 2*DZ) for c in cs): continue
        LO.append(cs[0]); HI.append(cs[1]); S.append(s)
    return np.array(HI, np.float32), np.array(LO, np.float32), np.array(S)


def outcomes_msw():
    d = pd.read_csv(os.path.join(DATA, 'dbs_03292024.csv'), header=1)
    d.columns = [str(c).strip().replace('\n', ' ') for c in d.columns]
    o = pd.to_numeric(d['OFF (pre-dbs updrs)'], errors='coerce')
    q = pd.to_numeric(d['OFF meds ON stim 6mo'], errors='coerce')
    s = pd.to_numeric(d['CORNELL ID'], errors='coerce')
    return {int(a): (b - c) / b for a, b, c in zip(s, o, q)
            if pd.notna(a) and pd.notna(b) and pd.notna(c) and b > 0}


def outcomes_chh():
    c = pd.read_csv(os.path.join(DATA, 'chh_subjects_table1_20240729.csv'), header=None).iloc[2:]
    out = {}
    for _, r in c.iterrows():
        try:
            sid = int(float(r[0])); a = pd.to_numeric(r[9], errors='coerce'); b = pd.to_numeric(r[11], errors='coerce')
            if pd.notna(a) and pd.notna(b) and a > 0: out[sid] = (a - b) / a
        except Exception: pass
    return out


impA, impB = outcomes_msw(), outcomes_chh()
dsA = load('MSW', 'dbs_03292024.csv', 'msw_cache_6d_cv.pt')
dsB = load('CHH', 'chh_subjects_table1_20240729.csv', 'chh_cache_6d_cv.pt')
HA, LA, SA_ = both_crops(dsA, impA)
HB, LB, SB_ = both_crops(dsB, impB)
yA = np.array([impA[s] for s in SA_]); yB = np.array([impB[s] for s in SB_])
bA = (yA >= 0.30).astype(int); bB = (yB >= 0.30).astype(int)
print(f"MSW {HA.shape} ({bA.sum()}/{(bA==0).sum()})   CHH {HB.shape} ({bB.sum()}/{(bB==0).sum()})\n")

from basis_fix import low_modes
BBb = (slice(0, 2*H), slice(0, 2*H), slice(0, 2*DZ))
IDX = low_modes(K, (2*H, 2*H, 2*DZ), 'lap')


def feats(V):
    T = dctn(V, axes=(1, 2, 3), norm='ortho')
    return np.stack([T[:, a, b, c] for a, b, c in IDX], 1)


VARIANTS = {
    'higher (used so far)': (feats(HA), feats(HB)),
    'lower': (feats(LA), feats(LB)),
    'both concatenated': (np.hstack([feats(HA), feats(LA)]), np.hstack([feats(HB), feats(LB)])),
    'mean of sides': (feats((HA + LA) / 2), feats((HB + LB) / 2)),
    'difference (asymmetry)': (feats(HA - LA), feats(HB - LB)),
}
site = np.r_[np.zeros(len(HA)), np.ones(len(HB))]
print("imaging-only logistic probe (clinical-only external reference 0.725)\n")
print(f"{'variant':24s} {'dim':>4} {'int AUC':>8} {'ext AUC':>8} {'site AUC':>9}")
rows = []
for name, (XA, XB) in VARIANTS.items():
    s = StandardScaler().fit(XA); ZA, ZB = s.transform(XA), s.transform(XB)
    pr = np.full(len(bA), np.nan)
    for tr, va in StratifiedKFold(4, shuffle=True, random_state=0).split(ZA, bA):
        m = LogisticRegression(C=1.0, max_iter=3000, class_weight='balanced').fit(ZA[tr], bA[tr])
        pr[va] = m.decision_function(ZA[va])
    m = LogisticRegression(C=1.0, max_iter=3000, class_weight='balanced').fit(ZA, bA)
    ia, ea = roc_auc_score(bA, pr), roc_auc_score(bB, m.decision_function(ZB))
    Z = np.vstack([ZA, ZB]); sa = []
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=0).split(Z, site):
        ms = LogisticRegression(C=1.0, max_iter=3000).fit(Z[tr], site[tr])
        sa.append(roc_auc_score(site[va], ms.decision_function(Z[va])))
    print(f"{name:24s} {XA.shape[1]:4d} {ia:8.3f} {ea:8.3f} {np.mean(sa):9.3f}")
    rows.append(dict(variant=name, dim=XA.shape[1], int_auc=ia, ext_auc=ea, site_auc=np.mean(sa)))
pd.DataFrame(rows).to_csv(os.path.join(OUT, 'both_sides.csv'), index=False)
