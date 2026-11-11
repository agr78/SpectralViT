"""Is synapse's class signal in band ENERGY rather than in coefficient values?

The images are dense texture, not an object on a background. For texture the phase of a
high-frequency coefficient is near-random across samples, so a signed coefficient carries
little, while the POWER in a frequency band is a stable descriptor. Every basis in the
sweep is linear in the coefficients and so cannot express |c|^2 at all -- the encoder
would have to learn it from 25 volumes.

Band energy per block: sum of c^2 over coefficients in each radial |k| shell.
"""
import numpy as np
from scipy import fft
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

SIDE = 14
for name in ('synapsemnist3d', 'nodulemnist3d', 'organmnist3d'):
    path = (f'/workspace/data/medmnist/{name}.npz' if name != 'organmnist3d'
            else '/agent-home/medmnist/organmnist3d.npz')
    z = np.load(path); X = z['train_images'].astype(np.float32); y = z['train_labels'].ravel()
    X = (X - X.mean()) / X.std()
    ncls = len(np.unique(y))
    offs = list(range(0, X.shape[1] - SIDE + 1, SIDE))
    nb = len(offs) ** 3
    C = np.zeros((len(X), nb, SIDE ** 3), dtype=np.float32)
    for i in range(len(X)):
        bi = 0
        for a in offs:
            for b in offs:
                for c in offs:
                    C[i, bi] = fft.dctn(X[i, a:a+SIDE, b:b+SIDE, c:c+SIDE],
                                        type=2, norm='ortho').ravel(); bi += 1
    kz, ky, kx = np.meshgrid(*[np.arange(SIDE)] * 3, indexing='ij')
    kr = np.sqrt(kz ** 2 + ky ** 2 + kx ** 2).ravel()
    NB = 12
    edges = np.linspace(0, kr.max() + 1e-6, NB + 1)
    band = np.digitize(kr, edges) - 1

    E = np.zeros((len(X), nb, NB), dtype=np.float32)
    for b in range(NB):
        m = band == b
        if m.sum(): E[:, :, b] = (C[:, :, m] ** 2).mean(2)
    E = E.reshape(len(X), -1)

    def probe(R):
        sc = []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(R, y):
            lr = LogisticRegression(max_iter=4000, C=1.0).fit(R[tr], y[tr])
            p = lr.predict_proba(R[te])
            sc.append(roc_auc_score(y[te], p[:, 1]) if ncls == 2
                      else roc_auc_score(y[te], p, multi_class='ovr'))
        return float(np.mean(sc))

    reps = {
        f'band energy  ({E.shape[1]} feats)': E,
        f'log band energy': np.log1p(E),
        f'low 216 coefficients (signed)': C.reshape(len(X), -1)[:, np.argsort(np.tile(kr, nb))[:216]],
        f'all 21952 coefficients (signed)': C.reshape(len(X), -1),
    }
    print(f'=== {name}  ({ncls} classes)', flush=True)
    for lab, R in reps.items():
        print('  %-36s %6.3f' % (lab, probe(R)), flush=True)
    print(flush=True)
