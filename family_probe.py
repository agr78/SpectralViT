"""Pick the FAMILY by a cheap diagnostic, then the config by CV inside it.

Flat argmax over 115 candidates fails: the inner CV ranks the true best 5th on synapse and
44th on vessel, and its split-half reliability is +0.184. But the diagnostic that found
fracture's fix -- comparing a probe on signed coefficients against one on magnitudes --
worked from training data alone, and it chooses between a handful of FAMILIES rather than
115 configs, so it carries far less selection variance.

This asks whether that diagnostic works from 25 volumes: for each (dataset, seed) it
builds four feature families from the 25 training volumes, scores each by repeated inner
CV with a convex probe, and reports which family it picks against which family actually
contains the best config on test. Everything is fitted inside the 25; the test split is
only used to say whether the pick was right.
"""
import os, glob
import numpy as np, pandas as pd
from scipy import fft, ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

DS = ['organmnist3d', 'nodulemnist3d', 'adrenalmnist3d',
      'fracturemnist3d', 'vesselmnist3d', 'synapsemnist3d']
SETS = {'organmnist3d': '/agent-home/medmnist/organmnist3d.npz'}
for d in ('nodule', 'adrenal', 'fracture', 'vessel', 'synapse'):
    SETS[d + 'mnist3d'] = f'/workspace/data/medmnist/{d}mnist3d.npz'


def allocate(avail, n):
    take = np.zeros(len(avail), int)
    for c in np.argsort(avail):
        take[c] = min(avail[c], (n - take.sum()) // max(1, (take == 0).sum()))
    i = 0
    while take.sum() < n:
        c = np.argsort(-avail)[i % len(avail)]
        if take[c] < avail[c]:
            take[c] += 1
        i += 1
    return take


def feats(V):
    C = np.stack([fft.dctn(v, type=2, norm='ortho').ravel() for v in V])
    side = V.shape[1]
    k = np.arange(side)
    kz, ky, kx = np.meshgrid(k, k, k, indexing='ij')
    kr = np.sqrt(kz ** 2 + ky ** 2 + kx ** 2).ravel()
    e = np.linspace(0, kr.max() + 1e-6, 13)
    b = np.digitize(kr, e) - 1
    E = np.stack([np.bincount(b, weights=c ** 2, minlength=12) /
                  np.bincount(b, minlength=12) for c in C])
    T = np.zeros((len(V), 12, 3), np.float32)
    for i, v in enumerate(V):
        lo, hi = np.percentile(v, [5, 95])
        for j, t in enumerate(np.linspace(lo, hi, 12)):
            B = v > t
            T[i, j] = (ndimage.label(B)[1], ndimage.label(~B)[1], B.mean())
    return {'signed': np.sign(C) * np.log1p(np.abs(C)),
            'magnitude': np.log1p(np.abs(C)),
            'energy': np.log1p(E),
            'topology': np.log1p(T.reshape(len(V), -1))}


def score(F, y, k=12):
    mn = np.bincount(y).min()
    if mn < 2:
        return np.nan
    cv = RepeatedStratifiedKFold(n_splits=min(4, mn), n_repeats=3, random_state=0)
    a = []
    for tr, te in cv.split(F, y):
        sc = StandardScaler().fit(F[tr])
        nc = max(1, min(k, len(tr) - 1))
        pc = PCA(n_components=nc, random_state=0).fit(sc.transform(F[tr]))
        m = LogisticRegression(max_iter=2000, C=0.1).fit(pc.transform(sc.transform(F[tr])), y[tr])
        p = m.predict_proba(pc.transform(sc.transform(F[te])))
        try:
            a.append(roc_auc_score(y[te], p[:, 1]) if p.shape[1] == 2 else
                     roc_auc_score(y[te], p, multi_class='ovr', labels=np.arange(p.shape[1])))
        except Exception:
            pass
    return float(np.mean(a)) if a else np.nan


FAM = {'signed': ('dct', 'pca_c', 'concat', 'blockpca_c', 'blockpca_b', 'triple', 'cross', 'rbf',
                  'pca24', 'pca16', 'oriented', 'scatter', 'square', 'dct_oriented', 'dct_scatter'),
       'magnitude': ('absfft', 'magblockpca'),
       'energy': ('energy', 'dct_energy', 'pca_energy', 'energy_cat', 'blockpca_energy',
                  'energy_gram', 'pca_sq_energy'),
       'topology': ('topo',)}


def fam_of(c):
    c = c.replace('OLD_', '')
    best, bl = 'signed', -1
    for f, pre in FAM.items():
        for p in pre:
            if c.startswith(p) and len(p) > bl:
                best, bl = f, len(p)
    return best


rows = []
for name in DS:
    z = np.load(SETS[name])
    X, Y = z['train_images'].astype(np.float32), z['train_labels'].ravel().astype(int)
    ncls = len(np.unique(Y))
    pool = {c: np.where(Y == c)[0] for c in range(ncls)}
    avail = np.array([len(pool[c]) for c in range(ncls)])
    for seed in range(int(os.environ.get('PROBE_SEED0',0)), int(os.environ.get('PROBE_SEED0',0))+int(os.environ.get('PROBE_SEEDS',5))):
        np.random.seed(seed)
        take = allocate(avail, 25)
        idx = np.concatenate([np.random.choice(pool[c], take[c], replace=False)
                              for c in range(ncls) if take[c]])
        F = feats(X[idx])
        y = Y[idx]
        s = {k: score(v, y) for k, v in F.items()}
        rows.append(dict(dataset=name, seed=seed, pick=max(s, key=lambda k: s[k]), **s))
        print('  %-16s s%d  ' % (name, seed)
              + '  '.join('%s %.3f' % (k, s[k]) for k in F) + '  -> %s' % rows[-1]['pick'],
              flush=True)
pd.DataFrame(rows).to_csv('results/family_probe%s.csv' % os.environ.get('PROBE_TAG',''), index=False)
print('\nwrote results/family_probe%s.csv' % os.environ.get('PROBE_TAG', ''))
