"""Three follow-ups on synapse.

1. Does the log compression explain why a complete DCT loses to raw voxels? A complete
   block DCT is an orthogonal rotation and L2 logistic regression is rotation-invariant,
   so 'DCT complete, linear' MUST match raw voxels. If it does and the log version does
   not, the compression is the cause, not the basis.
2. Why do patch means beat raw voxels? Averaging is a low-pass filter and a 8x dimension
   cut, so it may simply be denoising. Sweeping the patch side separates "locality helps"
   from "fewer, smoother features help".
3. The overlapping DCT was never probed; it is included here at matched coefficient count.
"""
import numpy as np
from scipy import fft
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

Z = np.load('/workspace/data/medmnist/synapsemnist3d.npz')
X = Z['train_images'].astype(np.float32); y = Z['train_labels'].ravel()
X = (X - X.mean()) / X.std()
SIDE = 14


def blocks(V, freq, stride=SIDE, log=True):
    offs = list(range(0, V.shape[1] - SIDE + 1, stride))
    out = np.zeros((len(V), len(offs) ** 3, freq ** 3), dtype=np.float32)
    for i in range(len(V)):
        bi = 0
        for a in offs:
            for b in offs:
                for c in offs:
                    t = fft.dctn(V[i, a:a+SIDE, b:b+SIDE, c:c+SIDE], type=2, norm='ortho')
                    out[i, bi] = t[:freq, :freq, :freq].ravel(); bi += 1
    out = out.reshape(len(V), -1)
    return np.sign(out) * np.log1p(np.abs(out)) if log else out


def pmean(V, s):
    o = range(0, V.shape[1], s)
    return np.stack([V[:, a:a+s, b:b+s, c:c+s].mean((1, 2, 3))
                     for a in o for b in o for c in o], 1)


def probe(R, scale=True):
    sc = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(R, y):
        A, B = R[tr], R[te]
        if scale:
            s = StandardScaler().fit(A); A, B = s.transform(A), s.transform(B)
        lr = LogisticRegression(max_iter=3000, C=1.0).fit(A, y[tr])
        sc.append(roc_auc_score(y[te], lr.predict_proba(B)[:, 1]))
    return float(np.mean(sc))


flat = X.reshape(len(X), -1)
print('%-34s %7s %8s' % ('representation', 'dims', 'probe'), flush=True)
print('%-34s %7d %8.3f' % ('raw voxels', flat.shape[1], probe(flat)), flush=True)
print('%-34s %7d %8.3f' % ('raw voxels, no scaling', flat.shape[1], probe(flat, False)), flush=True)
for log in (False, True):
    R = blocks(X, SIDE, log=log)
    print('%-34s %7d %8.3f' % (f'DCT complete, log={log}', R.shape[1], probe(R)), flush=True)
R = blocks(X, SIDE, log=False)
print('%-34s %7d %8.3f' % ('DCT complete, linear, no scaling', R.shape[1], probe(R, False)), flush=True)
for f in (3, 6):
    for log in (False, True):
        R = blocks(X, f, log=log)
        print('%-34s %7d %8.3f' % (f'DCT {f}^3 disjoint, log={log}', R.shape[1], probe(R)), flush=True)
for f in (2, 3):
    R = blocks(X, f, stride=7)
    print('%-34s %7d %8.3f' % (f'DCT {f}^3 OVERLAPPING stride 7', R.shape[1], probe(R)), flush=True)
for s in (1, 2, 4, 7):
    R = pmean(X, s)
    print('%-34s %7d %8.3f' % (f'patch means {s}^3', R.shape[1], probe(R)), flush=True)
