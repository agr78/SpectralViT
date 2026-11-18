"""Rebuild the inputs for the DBS diagram figure from the source data.

usage: python figures/make_fig_dbs_inputs.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repository root: shared modules and paths.py
import os, sys
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.fft import fftn, ifftn

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, 'dbs_3d'))
from reader import prepare_qsm_dataset
from util import mask_crop
from dbs_3d.method import H, DZ, DEFAULT_K
from dbs_3d.basis_fix import low_modes

DATA = os.environ.get('DBS_CLINICAL_DIR', os.path.join(REPO, 'docs'))
CACHE = os.environ.get('DBS_CACHE_DIR', '/checkpoints')
FIGDATA = os.environ.get('DBS_FIGDATA', os.path.join(REPO, 'paper', 'cvpr', 'fig', 'data'))
THR = float(os.environ.get('DBS_MODE_THR', 0.4))
RESP = 0.30


def improvements():
    d1 = pd.read_csv(os.path.join(DATA, 'dbs_03292024.csv'), header=1)
    d1.columns = [str(c).strip().replace('\n', ' ') for c in d1.columns]
    o = pd.to_numeric(d1['OFF (pre-dbs updrs)'], errors='coerce')
    q = pd.to_numeric(d1['OFF meds ON stim 6mo'], errors='coerce')
    s = pd.to_numeric(d1['CORNELL ID'], errors='coerce')
    A = {int(a): (b - c) / b for a, b, c in zip(s, o, q)
         if pd.notna(a) and pd.notna(b) and pd.notna(c) and b > 0}
    d2 = pd.read_csv(os.path.join(DATA, 'chh_subjects_table1_20240729.csv'),
                     header=None).iloc[2:]
    B = {}
    for _, r in d2.iterrows():
        try:
            sid = int(float(r[0]))
            a = pd.to_numeric(r[9], errors='coerce'); b = pd.to_numeric(r[11], errors='coerce')
            if pd.notna(a) and pd.notna(b) and a > 0:
                B[sid] = (a - b) / a
        except Exception:
            pass
    return A, B


def crops_padded(volume, mask):
    lb, n = ndimage.label(mask)
    if n < 2 or mask.sum() > 0.5 * mask.size:
        return None
    sz = ndimage.sum(mask, lb, range(1, n + 1))
    big = np.argsort(sz)[::-1][:2] + 1
    order = sorted([(float(volume[lb == b].mean()), b) for b in big], key=lambda t: t[0])
    half = (H, H, DZ)
    out, padded = [], 0
    for _, b in order:
        c = [int(round(v)) for v in ndimage.center_of_mass(mask, lb, [b])[0]]
        lo_pad = [max(0, half[d] - c[d]) for d in range(3)]
        hi_pad = [max(0, c[d] + half[d] - volume.shape[d]) for d in range(3)]
        if any(lo_pad) or any(hi_pad):
            padded += 1
            V = np.pad(volume, tuple((lo_pad[d], hi_pad[d]) for d in range(3)), mode='edge')
        else:
            V = volume
        cc = [c[d] + lo_pad[d] for d in range(3)]
        out.append(V[cc[0] - H:cc[0] + H, cc[1] - H:cc[1] + H, cc[2] - DZ:cc[2] + DZ])
    if any(o.shape != (2 * H, 2 * H, 2 * DZ) for o in out):
        return None
    return out[1], out[0], padded


def cohort(ds, imp):
    hi, lo, y, npad = [], [], [], 0
    for s in sorted(ds.volumes):
        s = int(s)
        if s not in imp:
            continue
        v = np.asarray(ds.volumes[s], np.float32)
        m = np.asarray(ds.seg_masks[s]) > 0
        c = crops_padded(v, m)
        if c is None:
            continue
        hi.append(c[0]); lo.append(c[1]); y.append(imp[s] >= RESP); npad += c[2]
    print(f'    {len(y)} patients, {npad} windows needed edge padding')
    return np.array(hi, np.float32), np.array(lo, np.float32), np.array(y, bool)


def tokens(V, idx):
    F = np.log1p(np.abs(fftn(V, axes=(1, 2, 3))))
    return np.stack([F[:, a, b, c] for a, b, c in idx], 1)


def effect(T, y):
    R, N = T[y], T[~y]
    nr, nn = len(R), len(N)
    sd = np.sqrt((R.var(0) * (nr - 1) + N.var(0) * (nn - 1)) / (nr + nn - 2))
    return (R.mean(0) - N.mean(0)) / (sd + 1e-12)


def reconstruct(V, idx, keep):
    Fv = fftn(V, axes=(1, 2, 3))
    M = np.zeros(V.shape[1:], bool)
    for j, (a, b, c) in enumerate(idx):
        if not keep[j]:
            continue
        M[a, b, c] = True
        M[(-a) % V.shape[1], (-b) % V.shape[2], (-c) % V.shape[3]] = True
    return np.real(ifftn(Fv * M[None], axes=(1, 2, 3)))


mcf = lambda d, m, p: mask_crop(d, m, p)
impA, impB = improvements()
ds_tr = prepare_qsm_dataset('MSW', '/x', '/y', os.path.join(DATA, 'dbs_03292024.csv'),
                            os.path.join(CACHE, 'msw_cache_6d_cv.pt'), load_cache=True,
                            mask_crop_fn=mcf, cv_pad=False)
ds_te = prepare_qsm_dataset('CHH', '/x', '/y',
                            os.path.join(DATA, 'chh_subjects_table1_20240729.csv'),
                            os.path.join(CACHE, 'chh_cache_6d_cv.pt'), load_cache=True,
                            mask_crop_fn=mcf, cv_pad=False)

idx = low_modes(DEFAULT_K, (2 * H, 2 * H, 2 * DZ), 'lap')
out, kept_counts = {}, {}
S = {}
for site, ds, imp in (('int', ds_tr, impA), ('ext', ds_te, impB)):
    S[site] = cohort(ds, imp)
    print(f'{site}: {len(S[site][2])} patients, {int(S[site][2].sum())} responders')

from sklearn.preprocessing import StandardScaler
from dbs_3d.method import crops as crops_clamped


def cohort_clamped(ds, imp):
    hi, lo = [], []
    for s in sorted(ds.volumes):
        s = int(s)
        if s not in imp:
            continue
        c = crops_clamped(np.asarray(ds.volumes[s], np.float32),
                          np.asarray(ds.seg_masks[s]) > 0)
        if c is None:
            continue
        hi.append(c[0]); lo.append(c[1])
    return np.array(hi, np.float32), np.array(lo, np.float32)


TOK = {}
for site, ds, imp in (('int', ds_tr, impA), ('ext', ds_te, impB)):
    hi, lo = cohort_clamped(ds, imp)
    assert len(hi) == len(S[site][2]), (site, len(hi), len(S[site][2]))
    TOK[site] = np.hstack([tokens(hi, idx), tokens(lo, idx)])
scaler = StandardScaler().fit(TOK['int'])
d_int = effect(scaler.transform(TOK['int']), S['int'][2])
d_ext = effect(scaler.transform(TOK['ext']), S['ext'][2])
rule = (np.sign(d_int) == np.sign(d_ext)) & (np.abs(d_int) > THR) & (np.abs(d_ext) > THR)
cons = np.where(rule)[0]
cons = cons[np.argsort(-(np.abs(d_int) + np.abs(d_ext))[cons])]
print(f'consistent modes (|d| > {THR} both sites, same sign):')
for t in cons:
    a_, b_, c_ = idx[t % len(idx)]
    print(f'  mode {t:2d}  {"higher" if t < len(idx) else "lower"} nucleus  k=({a_},{b_},{c_})'
          f'  d_int {d_int[t]:+.2f}  d_ext {d_ext[t]:+.2f}'
          f'  -> responders {"higher" if d_int[t] > 0 else "lower"} magnitude')
KEEP = {'hi': rule[:len(idx)], 'lo': rule[len(idx):]}
assert sorted(cons) == [5, 7, 11, 12, 20, 21, 22, 26], sorted(cons)

for nucleus, slot in (('hi', 0), ('lo', 1)):
    keep = KEEP[nucleus]
    kept_counts[nucleus] = int(keep.sum())
    for site in ('int', 'ext'):
        V, y = S[site][slot], S[site][2]
        R = reconstruct(V, idx, keep)
        out[f'{site}_{nucleus}'] = (R[y].mean(0) - R[~y].mean(0)).astype(np.float32)
    print(f'  {nucleus}: {keep.sum()} of {len(idx)} modes consistent at |d| > {THR}')

from scipy.stats import pearsonr
for nucleus in ('hi', 'lo'):
    r, _ = pearsonr(out[f'int_{nucleus}'].ravel(), out[f'ext_{nucleus}'].ravel())
    print(f'  {"higher" if nucleus == "hi" else "lower"} nucleus: cross-site r={r:+.3f}'
          f'  amplitude int {np.abs(out[f"int_{nucleus}"]).max():.4f}'
          f' ext {np.abs(out[f"ext_{nucleus}"]).max():.4f}')

import collections as _c
vols = [np.asarray(ds_tr.volumes[s], np.float32) for s in sorted(ds_tr.volumes)]
shp = _c.Counter(v.shape for v in vols).most_common(1)[0][0]
use = [s for s in sorted(ds_tr.volumes) if np.asarray(ds_tr.volumes[s]).shape == shp]
print(f'  cohort average over {len(use)} of {len(vols)} volumes at {shp}')
mean_vol = np.mean([np.asarray(ds_tr.volumes[s], np.float32) for s in use], 0).astype(np.float32)
pts = {'lo': [], 'hi': []}
for sid in sorted(ds_tr.volumes):
    v = np.asarray(ds_tr.volumes[sid], np.float32)
    m = np.asarray(ds_tr.seg_masks[sid]) > 0
    lb, n = ndimage.label(m)
    if n < 2:
        continue
    sz = ndimage.sum(m, lb, range(1, n + 1))
    big = np.argsort(sz)[::-1][:2] + 1
    o = sorted([(float(v[lb == b].mean()), ndimage.center_of_mass(m, lb, [b])[0])
                for b in big], key=lambda t: t[0])
    pts['lo'].append(o[0][1]); pts['hi'].append(o[1][1])
LOC = np.mean(pts['lo'], 0)
HIC = np.mean(pts['hi'], 0)
print(f'  centroids from {len(pts["lo"])} subjects: lo {LOC.round(1)}  hi {HIC.round(1)}')

os.makedirs(FIGDATA, exist_ok=True)
np.savez_compressed(os.path.join(FIGDATA, 'padded_maps.npz'), **out)
np.savez_compressed(os.path.join(FIGDATA, 'atlas.npz'), mean=mean_vol)
np.save(os.path.join(FIGDATA, 'atlas_centroids.npy'), np.stack([LOC, HIC]))
print(f'wrote padded_maps.npz ({kept_counts}), atlas.npz {mean_vol.shape}, '
      f'atlas_centroids.npy -> {FIGDATA}')
