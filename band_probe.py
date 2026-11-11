"""Which frequencies carry the class signal? Keep 216 coefficients chosen different ways.

The tokenizer keeps the 3^3 lowest frequencies of each block, but that is a choice. If
synapse's signal is high-frequency, a high band should probe better at the same budget.
The fold-wise F selection is the ceiling: the best 216 coefficients any fixed rule could
pick, chosen on the training fold alone so it stays honest.

Run unscaled and linear: per-feature standardisation costs 0.09 AUC on synapse by
amplifying near-zero coefficients, and the log compression makes almost no difference.
"""
import numpy as np
from scipy import fft
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import f_classif
from sklearn.metrics import roc_auc_score

SIDE, K = 14, 216
for name in ('synapsemnist3d', 'nodulemnist3d'):
    z = np.load(f'/workspace/data/medmnist/{name}.npz')
    X = z['train_images'].astype(np.float32); y = z['train_labels'].ravel()
    X = (X - X.mean()) / X.std()
    offs = list(range(0, X.shape[1] - SIDE + 1, SIDE))
    C = np.zeros((len(X), len(offs) ** 3, SIDE ** 3), dtype=np.float32)
    for i in range(len(X)):
        bi = 0
        for a in offs:
            for b in offs:
                for c in offs:
                    C[i, bi] = fft.dctn(X[i, a:a+SIDE, b:b+SIDE, c:c+SIDE],
                                        type=2, norm='ortho').ravel(); bi += 1
    nb = C.shape[1]; C = C.reshape(len(X), -1)
    kz, ky, kx = np.meshgrid(*[np.arange(SIDE)] * 3, indexing='ij')
    k = np.sqrt(kz ** 2 + ky ** 2 + kx ** 2).ravel()
    kk = np.tile(k, nb)
    order = np.argsort(kk)

    def probe(idx, foldwise=None):
        sc = []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(C, y):
            if foldwise is not None:
                F, _ = f_classif(C[tr], y[tr]); F = np.nan_to_num(F)
                idx = np.argsort(F)[::-1][:K]
            lr = LogisticRegression(max_iter=3000, C=1.0).fit(C[tr][:, idx], y[tr])
            sc.append(roc_auc_score(y[te], lr.predict_proba(C[te][:, idx])[:, 1]))
        return float(np.mean(sc))

    rs = np.random.RandomState(0)
    mid = len(order) // 2
    bands = {
        'lowest |k|  (what we keep)': order[:K],
        'mid |k|':                    order[mid - K // 2: mid + K // 2],
        'highest |k|':                order[-K:],
        'random':                     rs.choice(len(kk), K, replace=False),
    }
    print(f'=== {name}   (216 of {len(kk)} coefficients, linear, unscaled)', flush=True)
    for lab, idx in bands.items():
        print('  %-28s %6.3f   mean |k| %5.1f' % (lab, probe(idx), kk[idx].mean()), flush=True)
    print('  %-28s %6.3f   (ceiling: best 216 by training-fold F)'
          % ('fold-wise F selection', probe(None, foldwise=True)), flush=True)
    print('  %-28s %6.3f' % ('all 21952', probe(np.arange(len(kk)))), flush=True)
    print(flush=True)
