"""Hand-engineered feature baseline on MNIST, on the subsets mnist_sweep.py draws.

usage: python experiments/eval_handfeatures_mnist.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repository root: shared modules and paths.py
import os
import sys

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from scipy import ndimage, stats
from skimage.feature import graycomatrix, graycoprops
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util import get_stratified_subset, compute_multiclass_metrics

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.environ.get('MNIST_ROOT', '/checkpoints/data')
OUT = os.environ.get('HANDFEAT_OUT', os.path.join(HERE, 'results', 'mnist_handfeat.txt'))
N_SEEDS = int(os.environ.get('MNIST_SEEDS', 5))
PER_CLASS = [int(v) for v in os.environ.get('MNIST_PER_CLASS', '5,20,100,500').split(',')]
LEVELS = 32
PROPS = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']


def first_order(v):
    f = v.ravel()
    q = np.percentile(f, [10, 25, 50, 75, 90])
    h = np.histogram(f, bins=LEVELS)[0].astype(float)
    h /= max(h.sum(), 1)
    return [f.mean(), f.std(), stats.skew(f), stats.kurtosis(f), np.square(f).mean(),
            stats.entropy(h + 1e-12), f.min(), f.max(), f.max() - f.min(),
            *q, q[3] - q[1], np.abs(f - f.mean()).mean(), np.square(h).sum()]


def features(img):
    v = img[0].numpy() if hasattr(img, 'numpy') else img[0]
    lo, hi = v.min(), v.max()
    q = np.clip(((v - lo) / max(hi - lo, 1e-9) * (LEVELS - 1)).round(), 0, LEVELS - 1)
    g = graycomatrix(q.astype(np.uint8), [1, 2], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                     levels=LEVELS, symmetric=True, normed=True)
    f = first_order(v) + [float(graycoprops(g, p).mean()) for p in PROPS]
    for s in (1.0, 2.0):
        f += first_order(ndimage.gaussian_laplace(v, sigma=s))
    return f


def table(imgs):
    return np.asarray([features(x) for x in imgs], dtype=np.float32)


tf = transforms.Compose([transforms.ToTensor(),
                         transforms.Normalize((0.1307,), (0.3081,))])
train_full = datasets.MNIST(root=ROOT, train=True, download=False, transform=tf)
test_full = datasets.MNIST(root=ROOT, train=False, download=False, transform=tf)
te_imgs = torch.stack([test_full[i][0] for i in range(len(test_full))])
te_y = test_full.targets.numpy()
print(f'MNIST {len(train_full)} train / {len(te_imgs)} test', flush=True)
Fte = table(te_imgs)
print(f'test features {Fte.shape}', flush=True)

ARMS = [('hand features + linear', lambda: LogisticRegression(max_iter=5000, n_jobs=-1)),
        ('hand features + forest',
         lambda: RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1))]
L = ['Hand-engineered feature baseline on MNIST.',
     f'{Fte.shape[1]} features, same subsets and seeds as mnist_sweep.py, fixed settings.',
     '', '%-24s' % 'arm' + ''.join('%16s' % f'n={p * 10}' for p in PER_CLASS)]
rows = {}
for name, build in ARMS:
    cells = []
    for per_class in PER_CLASS:
        t1 = []
        for seed in range(N_SEEDS):
            torch.manual_seed(seed); np.random.seed(seed)
            tr_imgs, tr_y = get_stratified_subset(train_full, per_class)
            Ftr = table(tr_imgs)
            keep = Ftr.std(0) > 1e-6
            sc = StandardScaler().fit(Ftr[:, keep])
            m = build().fit(sc.transform(Ftr[:, keep]), tr_y.numpy())
            p = m.predict_proba(sc.transform(Fte[:, keep]))
            t1.append(compute_multiclass_metrics(p, te_y)['T1'])
        rows[(name, per_class)] = np.array(t1)
        cells.append('%.3f $\\pm$ %.3f' % (np.mean(t1), np.std(t1)))
        print(f'  {name} n={per_class * 10}: {np.mean(t1):.3f}', flush=True)
    L.append('%-24s' % name + ''.join('%16s' % c for c in cells))
PUB = {50: 0.409, 200: 0.555, 1000: 0.654, 5000: 0.697}
bad = [f'n={p * 10}: {rows[("hand features + linear", p)].mean():.3f} '
       f'against the text\'s {PUB[p * 10]:.3f}'
       for p in PER_CLASS
       if p * 10 in PUB
       and f'{rows[("hand features + linear", p)].mean():.3f}' != f'{PUB[p * 10]:.3f}']

report = '\n'.join(L)
print('\n' + report)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, 'w').write(report + '\n')
print(f'\nwrote {OUT}')
if bad:
    raise SystemExit('DOES NOT MATCH sec:dc:\n  ' + '\n  '.join(bad))
print(f'matches sec:dc: all {len(PUB)} quoted values')
