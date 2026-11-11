"""Cache HCA T1w volumes into a memmapped array the sweep can resample without disk reads.

One scan per subject (earliest V visit, per participants_hca.tsv). The T1w is found by
recursive glob inside each visit directory, so the internal layout does not matter.
Volumes are center-cropped and z-scored per subject.

Written row by row into a float16 memmap: np.stack would need the list and the output in
memory at once, which is twice the array and will not fit. float16 is ample for z-scored
intensities and halves both memory and disk.

usage: python hca_cache.py [participants_hca.tsv] [out_prefix]
  writes <prefix>.npy (volumes) and <prefix>_meta.npz (labels, ages, ids)
"""
import os, sys, glob, re
import numpy as np, pandas as pd
import nibabel as nib

TSV = sys.argv[1] if len(sys.argv) > 1 else 'participants_hca.tsv'
PRE = sys.argv[2] if len(sys.argv) > 2 else 'hca_vols_96'
IMG = os.environ.get('HCA_IMG_DIR',
                     '/autofs/vast/lemon/data_instantsurfer_uncurated/ahcp_data')
VOL = int(os.environ.get('HCA_VOL_SIZE', '96'))

d = pd.read_csv(TSV, sep='\t')
print(f'{len(d)} subjects in {TSV}; root {IMG}; crop {VOL}^3; dtype float16')

dirs = {}
for e in os.listdir(IMG):
    m = re.match(r'(HCA\d+)_(V\d)_', e)
    if m:
        dirs[f'{m.group(1)}_{m.group(2)}'] = os.path.join(IMG, e)
print(f'{len(dirs)} visit directories indexed')

npy = PRE + '.npy'
X = np.lib.format.open_memmap(npy, mode='w+', dtype=np.float16,
                              shape=(len(d), VOL, VOL, VOL))
print(f'allocated {npy}  ({X.nbytes/1e9:.2f} GB on disk)')

k = 0
ages, groups, ids, miss = [], [], [], []
for i, r in enumerate(d.itertuples()):
    dp = dirs.get(r.id_event)
    if dp is None:
        miss.append((r.id_event, 'no directory')); continue
    cand = glob.glob(os.path.join(dp, '**', '*T1w*.nii*'), recursive=True)
    if not cand:
        miss.append((r.id_event, 'no T1w')); continue
    try:
        img = nib.load(sorted(cand)[0]).get_fdata(dtype=np.float32)
    except Exception as ex:
        miss.append((r.id_event, type(ex).__name__)); continue
    pad = [max(0, VOL - s) for s in img.shape[:3]]
    if any(pad):
        img = np.pad(img, [(0, p) for p in pad], mode='constant')
    c = np.array(img.shape[:3]) // 2; h = VOL // 2
    crop = img[c[0]-h:c[0]+h, c[1]-h:c[1]+h, c[2]-h:c[2]+h]
    if crop.shape != (VOL, VOL, VOL):
        miss.append((r.id_event, f'shape {crop.shape}')); continue
    s = crop.std()
    X[k] = ((crop - crop.mean()) / (s if s > 0 else 1)).astype(np.float16)
    ages.append(r.age); groups.append(r.AgeGroup); ids.append(r.id)
    k += 1
    del img, crop
    if (i + 1) % 100 == 0:
        print(f'  {i+1}/{len(d)} read, {k} kept', flush=True)

X.flush(); del X
groups = np.array(groups)
y = (groups == sorted(set(groups.tolist()))[0]).astype(np.int64)
np.savez(PRE + '_meta.npz', n=k, y=y, age=np.array(ages),
         subject=np.array(ids), group=groups, vol=VOL,
         classes=np.array(sorted(set(groups.tolist()))))
print(f'\nkept {k} of {len(d)}; failed {len(miss)}')
from collections import Counter
print('  failure reasons:', Counter(r for _, r in miss).most_common())
print(f'  label counts {np.bincount(y)} for classes {sorted(set(groups.tolist()))}')
print(f'wrote {npy} (first {k} rows valid) and {PRE}_meta.npz')
