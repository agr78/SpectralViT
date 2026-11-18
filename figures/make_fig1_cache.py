"""Build the Figure 1 eigenimage cache from a volume cohort.

usage: FIG1_SET=mnist python figures/make_fig1_cache.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repository root: shared modules and paths.py
from paths import DATA
import os
import numpy as np
from sklearn.decomposition import PCA

SRC = os.path.expanduser(os.environ.get('HCA_VOLS', f'{DATA}/hca_smoke/hca_vols_96.npy'))
SET = os.environ.get('FIG1_SET', 'hca')
OUT = os.environ.get('FIG1_CACHE', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                'paper', 'cvpr', 'fig', 'data',
                                                f'fig1_cache_{SET}.npz'))
NMODE = int(os.environ.get('FIG1_MODES', 16 if SET == 'mnist' else 8))

if SET == 'mnist':
    import torch
    from torchvision import datasets, transforms
    from sklearn.preprocessing import StandardScaler
    from util import get_stratified_subset
    ROOT = os.environ.get('MNIST_ROOT', '/checkpoints/data')
    tr = datasets.MNIST(root=ROOT, train=True, download=False)
    sag = (tr.data.numpy().astype(np.float32) / 255.0)
    SPLIT = os.environ.get('FIG1_SPLIT', 'test')
    DIGIT = int(os.environ.get('FIG1_DIGIT', 0))
    ex_set = datasets.MNIST(root=ROOT, train=(SPLIT == 'train'), download=False)
    exemplar = ex_set.data.numpy().astype(np.float32)[DIGIT] / 255.0
    N_DRAW = int(os.environ.get('FIG1_MNIST_N', 1000))
    SEED = int(os.environ.get('FIG1_SEED', 0))
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))])
    train_full = datasets.MNIST(root=ROOT, train=True, download=False, transform=tf)
    torch.manual_seed(SEED); np.random.seed(SEED)
    tr_imgs, tr_y = get_stratified_subset(train_full, N_DRAW // 10)
    sc = StandardScaler().fit(tr_imgs.view(len(tr_imgs), -1).numpy())
    flat = sc.transform(tr_imgs.view(len(tr_imgs), -1).numpy())
    extra = {'mu': (sc.mean_ * 0.3081 + 0.1307).astype(np.float32),
             'sd': (sc.scale_ * 0.3081).astype(np.float32),
             'train_examples': (tr_imgs[::len(tr_imgs) // 10, 0] * 0.3081 + 0.1307)
                               .clamp(0, 1).numpy().astype(np.float32)}
    SRC = f'MNIST train, n={len(flat)} stratified draw seed {SEED}; subject {SPLIT}[{DIGIT}]'
    print(f'MNIST train: {sag.shape}; basis fitted on the n={len(flat)} draw, seed {SEED}')
    disp = lambda a: a
else:
    V = np.load(SRC, mmap_mode='r')
    print(f'{os.path.basename(SRC)}: {V.shape}')
    mid = V.shape[1] // 2
    sag = np.asarray(V[:, mid, :, :], dtype=np.float32)
    flat = sag.reshape(len(sag), -1)
    exemplar = sag[0]
    extra = {}
    disp = lambda a: a.T[::-1]
if SET != 'mnist':
    flat = flat - flat.mean(1, keepdims=True)
    flat = flat - flat.mean(0)

k = min(NMODE, len(flat) - 1)
p = PCA(n_components=k, svd_solver='full').fit(flat)
eig = p.components_.reshape(k, sag.shape[1], sag.shape[2])
eig = np.stack([disp(e) for e in eig])

os.makedirs(os.path.dirname(OUT), exist_ok=True)
if SET == 'mnist':
    extra['pca_mean'] = p.mean_.astype(np.float32)
np.savez_compressed(OUT, **extra, eig_sag=eig.astype(np.float32),
                    brain_sag=disp(exemplar).astype(np.float32),
                    var=p.explained_variance_ratio_[:k].astype(np.float32),
                    source=os.path.basename(SRC) if os.path.exists(SRC) else SRC,
                    n_subjects=len(flat))
print(f'wrote {OUT}: {k} modes from {len(flat)} subjects, '
      f'leading variance {p.explained_variance_ratio_[:4].round(3).tolist()}')
