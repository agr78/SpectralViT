"""Recover the three CHH cases with missing masks, for harmonization and regression.

Subjects 12, 15 and 17 have all-ones masks (294912 voxels = the whole 64x64x72
volume), so the SN/STN centroid cannot be taken from their own segmentation.
All CHH volumes sit in the same resampled grid, so the population-mean centroid
from the other 37 gives a usable crop box.

They are all responders, so they do not help the 3-non-responder limit on
classification. They do help regression (40 patients contributing to a
correlation instead of 37) and they add samples to the covariance estimate that
CORAL-style harmonisation depends on.
"""
import sys, os
SPDIR = os.path.dirname(os.path.abspath(__file__))
REPO  = os.path.dirname(SPDIR)
OUT   = os.path.join(REPO, 'docs', 'dbs_3d')
DATA  = os.environ.get('DBS_CLINICAL_DIR', os.path.join(REPO, 'docs'))
CACHE = os.environ.get('DBS_CACHE_DIR', '/checkpoints')
sys.path.insert(0, REPO); sys.path.insert(0, SPDIR); os.chdir(REPO)
import numpy as np, pandas as pd, pickle, warnings
warnings.filterwarnings('ignore')
from scipy import ndimage
from reader import prepare_qsm_dataset
from util import mask_crop

H, DZ = 10, 8
mcf = lambda d, m, p: mask_crop(d, m, p)
ds = prepare_qsm_dataset('CHH', '/x', '/y', os.path.join(DATA, 'chh_subjects_table1_20240729.csv'),
                         os.path.join(CACHE, 'chh_cache_6d_cv.pt'), load_cache=True,
                         mask_crop_fn=mcf, cv_pad=False)

c = pd.read_csv(os.path.join(DATA, 'chh_subjects_table1_20240729.csv'), header=None).iloc[2:]
imp = {}
for _, r in c.iterrows():
    try:
        sid = int(float(r[0])); off = pd.to_numeric(r[9], errors='coerce'); post = pd.to_numeric(r[11], errors='coerce')
        if pd.notna(off) and pd.notna(post) and off > 0:
            imp[sid] = (off - post) / off
    except Exception:
        pass

good, missing, cents = [], [], []
for s in sorted(ds.volumes):
    s = int(s)
    m = np.asarray(ds.seg_masks[s]) > 0
    lb, n = ndimage.label(m)
    if n >= 2 and m.sum() < 0.5 * m.size:
        v = np.asarray(ds.volumes[s], np.float32)
        sz = ndimage.sum(m, lb, range(1, n + 1))
        big = np.argsort(sz)[::-1][:2] + 1
        info = sorted([(float(v[lb == b].mean()), b) for b in big], key=lambda t: t[0])
        cx, cy, cz = [int(round(x)) for x in ndimage.center_of_mass(m, lb, [info[1][1]])[0]]
        good.append((s, cx, cy, cz)); cents.append((cx, cy, cz))
    else:
        missing.append(s)

cents = np.array(cents)
print(f"CHH: {len(good)} with usable masks, {len(missing)} without ({missing})")
# the x-centroid is bimodal: we pick the higher-susceptibility side per patient,
# so it sits left for some subjects and right for others. Averaging lands between
# the nuclei. Split into the two clusters instead.
xm = cents[:, 0]
split = np.median(xm)
left = np.round(cents[xm <= split].mean(0)).astype(int)
right = np.round(cents[xm > split].mean(0)).astype(int)
print(f"  x-centroid is bimodal (SD {xm.std():.1f}): left cluster n={(xm<=split).sum()}, right n={(xm>split).sum()}")
print(f"  left  centre {tuple(left)}   (within-cluster SD {cents[xm<=split].std(0).round(1)})")
print(f"  right centre {tuple(right)}  (within-cluster SD {cents[xm>split].std(0).round(1)})")


def crop_at(v, cx, cy, cz):
    sl = (slice(max(0, cx-H), max(0, cx-H)+2*H),
          slice(max(0, cy-H), max(0, cy-H)+2*H),
          slice(max(0, cz-DZ), max(0, cz-DZ)+2*DZ))
    p = v[sl]
    return p if p.shape == (2*H, 2*H, 2*DZ) else None


V, S, Y, SRC = [], [], [], []
for s, cx, cy, cz in good:
    if s not in imp: continue
    p = crop_at(np.asarray(ds.volumes[s], np.float32), cx, cy, cz)
    if p is not None:
        V.append(p); S.append(s); Y.append(imp[s]); SRC.append('mask')
for s in missing:
    if s not in imp: continue
    v = np.asarray(ds.volumes[s], np.float32)
    # mirror the real rule: take both candidate sides, keep the higher-susceptibility one
    cands = [(crop_at(v, *left), 'L'), (crop_at(v, *right), 'R')]
    cands = [(c, side) for c, side in cands if c is not None]
    if not cands: continue
    p, side = max(cands, key=lambda t: float(t[0].mean()))
    V.append(p); S.append(s); Y.append(imp[s]); SRC.append(f'population-{side}')
V = np.array(V, np.float32); Y = np.array(Y); S = np.array(S); SRC = np.array(SRC)
print(f"\nCHH-40 built: {V.shape}   recovered by population centroid: {np.char.startswith(SRC.astype(str),'population').sum()}")
print(f"  outcome range [{Y.min():.3f}, {Y.max():.3f}]   mean {Y.mean():.3f}")
print(f"  responders (>=0.30): {(Y>=0.30).sum()}   non-responders: {(Y<0.30).sum()}")

# sanity: do the recovered crops look like the others, or are they outliers?
from scipy.fft import dctn
f = lambda X: dctn(X, axes=(1, 2, 3), norm='ortho').reshape(len(X), -1)[:, :64]
F = f(V)
zm = (F - F[SRC == 'mask'].mean(0)) / (F[SRC == 'mask'].std(0) + 1e-8)
print(f"\nrecovered crops vs the mask-derived ones (z across the first 64 DCT modes):")
for i in np.where(np.char.startswith(SRC.astype(str), 'population'))[0]:
    print(f"  subject {S[i]}: mean |z| {np.abs(zm[i]).mean():.2f}   max |z| {np.abs(zm[i]).max():.2f}")
print(f"  mask-derived subjects for comparison: mean |z| "
      f"{np.abs(zm[SRC=='mask']).mean():.2f}")
pickle.dump(dict(V=V, S=S, Y=Y, SRC=SRC), open(os.path.join(OUT, 'chh40.pkl'), 'wb'))
print(f"\nwrote {os.path.join(OUT, 'chh40.pkl')}")
