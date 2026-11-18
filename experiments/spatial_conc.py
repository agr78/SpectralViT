"""Test whether the class signal is spatially concentrated."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repository root: shared modules and paths.py
from paths import DATA
import numpy as np, glob, os
from sklearn.feature_selection import f_classif

S = {'organmnist3d': f'{DATA}/medmnist/organmnist3d.npz'}
for f in sorted(glob.glob(f'{DATA}/medmnist/*.npz')):
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
