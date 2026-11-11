"""Is synapse's class signal spatially concentrated, and do the bases dilute it?

Per-voxel F-statistic against the label, then two concentration measures:
  top1%   share of total F carried by the most discriminative 1% of voxels
  PR      participation ratio (sum F)^2 / sum(F^2), the effective number of voxels
          carrying signal -- 21952 means perfectly spread, small means concentrated

A basis that averages over a region dilutes a concentrated signal in proportion to the
region size, so if synapse is the concentrated one, the 14^3 block DCT (2744 voxels per
coefficient) and whole-volume PCA are exactly the wrong representations for it.
"""
import numpy as np, glob, os
from sklearn.feature_selection import f_classif

S = {'organmnist3d': '/agent-home/medmnist/organmnist3d.npz'}
for f in sorted(glob.glob('/workspace/data/medmnist/*.npz')):
    S[os.path.basename(f)[:-4]] = f

print('%-17s %8s %9s %10s %12s' % ('dataset', 'cls', 'top1% F', 'PR', 'PR/21952'))
for name, path in S.items():
    z = np.load(path)
    X = z['train_images'].astype(np.float32).reshape(len(z['train_images']), -1)
    y = z['train_labels'].ravel()
    F, _ = f_classif(X, y)
    F = np.nan_to_num(F)
    F = np.maximum(F, 0)
    srt = np.sort(F)[::-1]
    k = max(1, len(F) // 100)
    top1 = srt[:k].sum() / (F.sum() + 1e-12)
    pr = (F.sum() ** 2) / ((F ** 2).sum() + 1e-12)
    print('%-17s %8d %8.1f%% %10.0f %11.3f' % (
        name, len(np.unique(y)), 100 * top1, pr, pr / len(F)), flush=True)
