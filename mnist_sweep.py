"""MNIST multi-class sweep by training-set size, with per-seed results persisted.

mnist.ipynb ran this and printed only a groupby mean, so the published table carries no
spread and no test. Here every seed's metrics are written out, so the result can be
reported as mean +/- sd with a paired test over seeds, matching the other experiments.

Both arms are left at fixed default capacity, as in the notebook: 16 PCA components for
the spectral ViT, patch 4 for the spatial ViT. Neither is tuned, so the comparison is
symmetric -- this is not the one-sided selection that had to be fixed on IXI -- but it is
also not nested selection, and the manuscript should say so plainly.

The two models are parameter balanced to within 0.6% (1,188,490 against 1,195,786): both
carry the same embed_dim=128, 2-layer, 4-head encoder, and differ only in the input
projection and positional embedding.

A single AdamW instance covers both models, as in the notebook. The loss is a sum of two
terms over disjoint parameter sets, so this is equivalent to two optimizers with identical
hyperparameters, not a coupling between the models.

env: MNIST_ROOT MNIST_SEEDS MNIST_OUT
usage: python mnist_sweep.py [--smoke]
"""
import os, sys, time
import numpy as np, pandas as pd, torch, torch.optim as optim, torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from scipy import fft

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from networks import (MultiClassSpectralViT, MultiClassSpatialViT,
                      MultiClassLinearHead)
import torch.nn as nn
from util import get_stratified_subset, compute_multiclass_metrics


def spectral_features(imgs, grid, comp, kind='dct'):
    """The comp lowest-frequency block coefficients, rank major and block minor.

    Duplicated from cifar_sweep.py in the style of the volumetric scripts, each of which
    carries its own feature machinery. 'dct' keeps the signed coefficient; 'fft' keeps the
    magnitude, which is translation invariant within a block at the cost of the phase. A
    graph Laplacian arm is not included: on a regular lattice with Neumann boundaries its
    eigenvectors are the cosine basis, so it would reproduce the 'dct' arm exactly.
    """
    x = imgs.numpy() if hasattr(imgs, 'numpy') else imgs
    B, C, H, W = x.shape
    bh, bw = H // grid, W // grid
    x = x.reshape(B, C, grid, bh, grid, bw).transpose(0, 1, 2, 4, 3, 5)
    if kind == 'dct':
        co = fft.dctn(x, type=2, norm='ortho', axes=(-2, -1))
        co = np.sign(co) * np.log1p(np.abs(co))
    else:
        co = np.log1p(np.abs(fft.fftn(x, axes=(-2, -1))))
    nf, nb = bh * bw, C * grid * grid
    ky, kx = np.unravel_index(np.arange(nf), (bh, bw))
    order = np.argsort(ky ** 2 + kx ** 2, kind='stable')
    perm = np.array([b * nf + order[r] for r in range(nf) for b in range(nb)])
    return co.reshape(B, nb * nf)[:, perm[:comp]].astype(np.float32)


def overlap_tokens(x, block, stride):
    """Overlapping block-DCT tokens, indexed by position so weight sharing is valid.

    Duplicated from cifar_sweep.py, matching how the volumetric scripts each carry their own
    feature machinery. Taken at a stride below its support the transform is overcomplete and
    therefore not a per-patch invertible map, so a learned projection cannot absorb it.
    """
    x = x.numpy() if hasattr(x, 'numpy') else x
    B, C, H, W = x.shape
    ys = range(0, H - block + 1, stride)
    xs = range(0, W - block + 1, stride)
    pat = np.stack([x[:, :, i:i + block, j:j + block] for i in ys for j in xs], axis=2)
    co = fft.dctn(pat, type=2, norm='ortho', axes=(-2, -1))
    co = np.sign(co) * np.log1p(np.abs(co))
    co = co.reshape(B, C, len(ys) * len(xs), block * block).transpose(0, 2, 1, 3)
    return co.reshape(B, -1).astype(np.float32)

SMOKE = '--smoke' in sys.argv
ROOT = os.environ.get('MNIST_ROOT', '/checkpoints/data')
N_SEEDS = 2 if SMOKE else int(os.environ.get('MNIST_SEEDS', 5))
PER_CLASS = [5, 20] if SMOKE else [5, 20, 100, 500]
# Component count is swept for the sensitivity analysis; 16 is the reported setting.
N_COMP = int(os.environ.get('MNIST_COMP', 16))
EPOCHS, BATCH, LR, WD = (2 if SMOKE else 100), 128, 1e-3, 0.01
# Ablation over the two SpectralViT mechanisms this variant leaves off by default: the
# 1/i variance hierarchy and the component-wise embedding of sec:ca. Default empty, so the
# published run is reproduced exactly.
BASIS = os.environ.get('MNIST_BASIS', 'pca')   # pca | dct | fft
GRID = int(os.environ.get('MNIST_GRID', 4))    # 28 -> 7x7 blocks
OVERLAP = os.environ.get('MNIST_OVERLAP') == '1'   # 4x4 DCT at stride 2, position tokens
# The coefficient-input controls. Attention over spectral tokens is the contribution, so
# the margin that supports it is the one over a head that receives exactly the same
# coefficients -- not the one over the patch tokenizer, which also changes the basis.
# Mirrors faces_sweep.py and cifar_sweep.py so the three benchmarks report the same arms.
class FeatureMLP(nn.Module):
    """Two hidden layers on the same components the spectral arm receives."""
    def __init__(self, input_dim, hidden, num_classes):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, num_classes))

    def forward(self, x):
        return self.net(x)


def match_hidden(input_dim, num_classes, target):
    """Hidden width whose parameter count matches the spectral arm's."""
    b = input_dim + 2 + num_classes
    return max(2, int((-b + np.sqrt(b * b - 4 * (num_classes - target))) / 2))


SPEC_KW = {}
if os.environ.get('MNIST_RANK') == '1':
    SPEC_KW['use_rank_weights'] = True
# Fixed rather than learnable 1/i, matching the three-way ablation on the volumetric
# cohort: omitted, enforced, or initialized and trained.
if os.environ.get('MNIST_FIXRANK') == '1':
    SPEC_KW['use_rank_weights'] = True
    SPEC_KW['learnable_rank_weights'] = False
if os.environ.get('MNIST_MODE') == '1':
    SPEC_KW['use_mode_weights'] = True
# The overlap tokenizer depends on a projection SHARED across position tokens; mode
# weights would instead give one token per coefficient, which is the opposite design
# and would make thousands of tokens. They are mutually exclusive.
if OVERLAP:
    SPEC_KW.pop('use_mode_weights', None)
OUT = os.environ.get('MNIST_OUT', os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                               'results', 'mnist'))
os.makedirs(OUT, exist_ok=True)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tf = transforms.Compose([transforms.ToTensor(),
                         transforms.Normalize((0.1307,), (0.3081,))])
train_full = datasets.MNIST(root=ROOT, train=True, download=False, transform=tf)
test_full = datasets.MNIST(root=ROOT, train=False, download=False, transform=tf)
# The full 10k test set is fixed across every size and seed, so it is built once.
te_imgs = torch.stack([test_full[i][0] for i in range(len(test_full))])
te_y = test_full.targets.numpy()
te_flat = te_imgs.view(len(te_imgs), -1).numpy()
print(f'MNIST {len(train_full)} train / {len(te_imgs)} test on {dev}', flush=True)

rows, t0 = [], time.time()
for per_class in PER_CLASS:
    for seed in range(N_SEEDS):
        torch.manual_seed(seed); np.random.seed(seed)
        tr_imgs, tr_y = get_stratified_subset(train_full, per_class)
        tr_flat = tr_imgs.view(len(tr_imgs), -1).numpy()

        if BASIS in ('dct', 'fft'):
            Ftr = spectral_features(tr_imgs, GRID, N_COMP, BASIS)
            Fte = spectral_features(te_imgs, GRID, N_COMP, BASIS)
            m, sv = Ftr.mean(0, keepdims=True), Ftr.std(0, keepdims=True) + 1e-6
            Ztr = torch.tensor((Ftr - m) / sv, dtype=torch.float32)
            Zte = torch.tensor((Fte - m) / sv, dtype=torch.float32)
        elif OVERLAP:
            Ftr = overlap_tokens(tr_imgs, 4, 2)
            Fte = overlap_tokens(te_imgs, 4, 2)
            m, sv = Ftr.mean(0, keepdims=True), Ftr.std(0, keepdims=True) + 1e-6
            Ztr = torch.tensor((Ftr - m) / sv, dtype=torch.float32)
            Zte = torch.tensor((Fte - m) / sv, dtype=torch.float32)
        else:
            sc = StandardScaler().fit(tr_flat)
            pca = PCA(n_components=min(N_COMP, len(tr_flat))).fit(sc.transform(tr_flat))
            Ztr = torch.tensor(pca.transform(sc.transform(tr_flat)), dtype=torch.float32)
            Zte = torch.tensor(pca.transform(sc.transform(te_flat)), dtype=torch.float32)

        spec = MultiClassSpectralViT(n_inputs=Ztr.shape[1],
                                     chunk_size=(16 if OVERLAP else 4), **SPEC_KW).to(dev)
        spat = MultiClassSpatialViT().to(dev)
        np_spec = sum(p.numel() for p in spec.parameters())
        mlp = FeatureMLP(Ztr.shape[1], match_hidden(Ztr.shape[1], 10, np_spec), 10).to(dev)
        lin = MultiClassLinearHead(Ztr.shape[1], 10).to(dev)
        # One optimizer over every arm, losses summed into one backward: identical lr,
        # weight decay, batches and step count, so a margin cannot be a tuning artifact.
        opt = optim.AdamW(list(spec.parameters()) + list(spat.parameters()) +
                          list(mlp.parameters()) + list(lin.parameters()), lr=LR,
                          weight_decay=WD)
        loader = DataLoader(TensorDataset(tr_imgs, Ztr, tr_y), batch_size=BATCH, shuffle=True)
        for _ in range(EPOCHS):
            spec.train(); spat.train(); mlp.train(); lin.train()
            for bi, bz, by in loader:
                bi, bz, by = bi.to(dev), bz.to(dev), by.to(dev)
                opt.zero_grad()
                (F.cross_entropy(spec(bz), by) + F.cross_entropy(spat(bi), by) +
                 F.cross_entropy(mlp(bz), by) + F.cross_entropy(lin(bz), by)).backward()
                opt.step()

        spec.eval(); spat.eval(); mlp.eval(); lin.eval()
        ps, pp, pm, pl = [], [], [], []
        with torch.no_grad():
            for bi, bz, _ in DataLoader(TensorDataset(te_imgs, Zte,
                                                      torch.from_numpy(te_y)),
                                        batch_size=512):
                ps.append(F.softmax(spec(bz.to(dev)), 1).cpu())
                pp.append(F.softmax(spat(bi.to(dev)), 1).cpu())
                pm.append(F.softmax(mlp(bz.to(dev)), 1).cpu())
                pl.append(F.softmax(lin(bz.to(dev)), 1).cpu())
        ms = compute_multiclass_metrics(torch.cat(ps).numpy(), te_y)
        mp = compute_multiclass_metrics(torch.cat(pp).numpy(), te_y)
        mm = compute_multiclass_metrics(torch.cat(pm).numpy(), te_y)
        ml = compute_multiclass_metrics(torch.cat(pl).numpy(), te_y)
        rows.append(dict(per_class=per_class, n=per_class * 10, seed=seed,
                         np_spec=np_spec,
                         np_mlp=sum(p.numel() for p in mlp.parameters()),
                         np_lin=sum(p.numel() for p in lin.parameters()),
                         **{f'spec_{k}': v for k, v in ms.items()},
                         **{f'spat_{k}': v for k, v in mp.items()},
                         **{f'mlp_{k}': v for k, v in mm.items()},
                         **{f'lin_{k}': v for k, v in ml.items()}))
        print(f'  {per_class:3d}/class seed {seed}: spec T1 {ms["T1"]:.3f} '
              f'spat T1 {mp["T1"]:.3f} mlp T1 {mm["T1"]:.3f} lin T1 {ml["T1"]:.3f} '
              f'| spec AUC {ms["AUC"]:.3f} spat AUC {mp["AUC"]:.3f} '
              f'({time.time()-t0:.0f}s)', flush=True)
        del spec, spat, mlp, lin
        if dev.type == 'cuda':
            torch.cuda.empty_cache()

        d = pd.DataFrame(rows)
        tmp = os.path.join(OUT, 'mnist_perseed.csv.tmp')
        d.to_csv(tmp, index=False); os.replace(tmp, os.path.join(OUT, 'mnist_perseed.csv'))

print(f'\nwrote {OUT}/mnist_perseed.csv ({len(rows)} cells)', flush=True)
