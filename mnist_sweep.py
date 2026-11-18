"""MNIST multi-class sweep by training-set size, with per-seed results persisted.

usage: python mnist_sweep.py [--smoke]"""
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


AUG = os.environ.get('MNIST_AUG') == '1'
AUG_COPIES = int(os.environ.get('MNIST_AUG_COPIES', 3))
AUG_DEG = float(os.environ.get('MNIST_AUG_DEG', 10.0))
AUG_SHIFT = float(os.environ.get('MNIST_AUG_SHIFT', 2.0))
AUG_SCALE = float(os.environ.get('MNIST_AUG_SCALE', 0.1))


def augment(imgs, y, copies, g):
    out_i, out_y, n = [imgs], [y], len(imgs)
    for _ in range(copies):
        th = (torch.rand(n, generator=g) * 2 - 1) * (AUG_DEG * 3.141592653589793 / 180)
        sc = 1 + (torch.rand(n, generator=g) * 2 - 1) * AUG_SCALE
        tx = (torch.rand(n, generator=g) * 2 - 1) * (2 * AUG_SHIFT / imgs.shape[-1])
        ty = (torch.rand(n, generator=g) * 2 - 1) * (2 * AUG_SHIFT / imgs.shape[-2])
        cos, sin = torch.cos(th) / sc, torch.sin(th) / sc
        theta = torch.stack([torch.stack([cos, -sin, tx], 1),
                             torch.stack([sin, cos, ty], 1)], 1)
        grid = F.affine_grid(theta, imgs.shape, align_corners=False)
        out_i.append(F.grid_sample(imgs, grid, align_corners=False, padding_mode='zeros'))
        out_y.append(y)
    return torch.cat(out_i), torch.cat(out_y)


def spectral_features(imgs, grid, comp, kind='dct'):
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
PER_CLASS = [5, 20] if SMOKE else [int(v) for v in
              os.environ.get('MNIST_PER_CLASS', '5,20,100,500').split(',')]
N_COMP = int(os.environ.get('MNIST_COMP', 16))
EPOCHS, BATCH, LR, WD = (2 if SMOKE else 100), 128, 1e-3, 0.01
BASIS = os.environ.get('MNIST_BASIS', 'pca')
GRID = int(os.environ.get('MNIST_GRID', 4))
OVERLAP = os.environ.get('MNIST_OVERLAP') == '1'
class FeatureMLP(nn.Module):
    def __init__(self, input_dim, hidden, num_classes):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, num_classes))

    def forward(self, x):
        return self.net(x)


def match_hidden(input_dim, num_classes, target):
    b = input_dim + 2 + num_classes
    return max(2, int((-b + np.sqrt(b * b - 4 * (num_classes - target))) / 2))


SPEC_KW = {}
if os.environ.get('MNIST_RANK') == '1':
    SPEC_KW['use_rank_weights'] = True
if os.environ.get('MNIST_FIXRANK') == '1':
    SPEC_KW['use_rank_weights'] = True
    SPEC_KW['learnable_rank_weights'] = False
if os.environ.get('MNIST_MODE') == '1':
    SPEC_KW['use_mode_weights'] = True
if OVERLAP:
    SPEC_KW.pop('use_mode_weights', None)
OUT = os.environ.get('MNIST_OUT', os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                               'results', 'mnist'))
os.makedirs(OUT, exist_ok=True)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tf = transforms.Compose([transforms.ToTensor(),
                         transforms.Normalize((0.1307,), (0.3081,))])
DL = os.environ.get('MNIST_DOWNLOAD') == '1'
train_full = datasets.MNIST(root=ROOT, train=True, download=DL, transform=tf)
test_full = datasets.MNIST(root=ROOT, train=False, download=DL, transform=tf)
te_imgs = torch.stack([test_full[i][0] for i in range(len(test_full))])
te_y = test_full.targets.numpy()
te_flat = te_imgs.view(len(te_imgs), -1).numpy()
print(f'MNIST {len(train_full)} train / {len(te_imgs)} test on {dev}', flush=True)

rows, t0 = [], time.time()
for per_class in PER_CLASS:
    for seed in range(N_SEEDS):
        torch.manual_seed(seed); np.random.seed(seed)
        tr_imgs, tr_y = get_stratified_subset(train_full, per_class)
        basis_flat = tr_imgs.view(len(tr_imgs), -1).numpy()
        if AUG:
            tr_imgs, tr_y = augment(tr_imgs, tr_y, AUG_COPIES,
                                    torch.Generator().manual_seed(seed * 7919 + per_class))
        tr_flat = tr_imgs.view(len(tr_imgs), -1).numpy()
        if os.environ.get('MNIST_AUG_BASIS') == 'aug':
            basis_flat = tr_flat

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
            sc = StandardScaler().fit(basis_flat)
            pca = PCA(n_components=min(N_COMP, len(basis_flat))).fit(sc.transform(basis_flat))
            Ztr = torch.tensor(pca.transform(sc.transform(tr_flat)), dtype=torch.float32)
            Zte = torch.tensor(pca.transform(sc.transform(te_flat)), dtype=torch.float32)

        spec = MultiClassSpectralViT(n_inputs=Ztr.shape[1],
                                     chunk_size=(16 if OVERLAP else 4), **SPEC_KW).to(dev)
        spat = MultiClassSpatialViT().to(dev)
        np_spec = sum(p.numel() for p in spec.parameters())
        mlp = FeatureMLP(Ztr.shape[1], match_hidden(Ztr.shape[1], 10, np_spec), 10).to(dev)
        lin = MultiClassLinearHead(Ztr.shape[1], 10).to(dev)
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
                         aug=AUG_COPIES if AUG else 0,
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
