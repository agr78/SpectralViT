"""Which structure the spectral bases do not represent on SynapseMNIST3D."""
from paths import DATA
import numpy as np, sys
from scipy import fft
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score

SETS = {'synapsemnist3d': f'{DATA}/medmnist/synapsemnist3d.npz',
        'nodulemnist3d':  f'{DATA}/medmnist/nodulemnist3d.npz'}
SIDE, GRID = 14, 2


def blocks(V, freq):
    offs = list(range(0, V.shape[1] - SIDE + 1, SIDE))
    out = np.zeros((len(V), len(offs) ** 3, freq ** 3), dtype=np.float32)
    for i in range(len(V)):
        bi = 0
        for a in offs:
            for b in offs:
                for c in offs:
                    t = fft.dctn(V[i, a:a+SIDE, b:b+SIDE, c:c+SIDE], type=2, norm='ortho')
                    out[i, bi] = t[:freq, :freq, :freq].ravel(); bi += 1
    return np.sign(out) * np.log1p(np.abs(out))


def patch_means(V, side):
    o = range(0, V.shape[1], side)
    return np.stack([V[:, a:a+side, b:b+side, c:c+side].mean((1, 2, 3))
                     for a in o for b in o for c in o], 1)


for name, path in SETS.items():
    z = np.load(path)
    X = z['train_images'].astype(np.float32); y = z['train_labels'].ravel()
    X = (X - X.mean()) / X.std()
    ncls = len(np.unique(y))
    flat = X.reshape(len(X), -1)
    reps = {
        'raw voxels (21952)': flat,
        'DCT complete (21952)': blocks(X, SIDE).reshape(len(X), -1),
        'DCT 3^3 (216)': blocks(X, 3).reshape(len(X), -1),
        'patch means 4^3 (343)': patch_means(X, 4),
        'patch means 2^3 (2744)': patch_means(X, 2),
    }
    print(f'=== {name}  ({len(X)} volumes, {ncls} classes)', flush=True)
    print('%-24s %8s' % ('representation', 'probe'), flush=True)
    for lab, R in reps.items():
        sc = []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(R, y):
            s = StandardScaler().fit(R[tr])
            lr = LogisticRegression(max_iter=3000, C=1.0).fit(s.transform(R[tr]), y[tr])
            p = lr.predict_proba(s.transform(R[te]))
            sc.append(roc_auc_score(y[te], p[:, 1]) if ncls == 2
                      else accuracy_score(y[te], p.argmax(1)))
        print('%-24s %8.3f' % (lab, np.mean(sc)), flush=True)
    for k in (16, 128, 512):
        sc = []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(flat, y):
            s = StandardScaler().fit(flat[tr])
            pc = PCA(n_components=min(k, len(tr) - 1), random_state=0).fit(s.transform(flat[tr]))
            lr = LogisticRegression(max_iter=3000, C=1.0).fit(pc.transform(s.transform(flat[tr])), y[tr])
            p = lr.predict_proba(pc.transform(s.transform(flat[te])))
            sc.append(roc_auc_score(y[te], p[:, 1]) if ncls == 2
                      else accuracy_score(y[te], p.argmax(1)))
        print('%-24s %8.3f' % (f'PCA {k}', np.mean(sc)), flush=True)
    print(flush=True)
