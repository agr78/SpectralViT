"""CIFAR-10/100 multi-class sweep by training-set size, mirroring mnist_sweep.py.

usage: python cifar_sweep.py [--smoke]"""
import os, sys, time
import numpy as np, pandas as pd, torch, torch.optim as optim, torch.nn.functional as F
from scipy import fft
import warnings
from sklearn.decomposition import PCA, FastICA
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from networks import MultiClassSpectralViT, MultiClassSpatialViT
from sklearn.metrics import accuracy_score, top_k_accuracy_score, roc_auc_score

SMOKE = '--smoke' in sys.argv
ROOT = os.environ.get('CIFAR_ROOT', '/checkpoints/data')
N_SEEDS = 2 if SMOKE else int(os.environ.get('CIFAR_SEEDS', 5))
N_COMP = int(os.environ.get('CIFAR_COMP', 16))
CHUNK = int(os.environ.get('CIFAR_CHUNK', 4))
DL = os.environ.get('CIFAR_DOWNLOAD', '0') == '1'
EPOCHS, BATCH, LR, WD = (2 if SMOKE else 100), 128, 1e-3, 0.01
SIZE, PATCH, CHANS = 32, 4, 3
SPEC_KW = {}
if os.environ.get('CIFAR_MODE') == '1': SPEC_KW['use_mode_weights'] = True
if os.environ.get('CIFAR_RANK') == '1': SPEC_KW['use_rank_weights'] = True
BASIS = os.environ.get('CIFAR_BASIS', 'pca')
GRID = int(os.environ.get('CIFAR_GRID', 1))
LEVELS = int(os.environ.get('CIFAR_LEVELS', 3))
STRIDE = int(os.environ.get('CIFAR_STRIDE', 2))
SPAT_STRIDE = os.environ.get('CIFAR_SPAT_STRIDE')
SPAT_STRIDE = int(SPAT_STRIDE) if SPAT_STRIDE else None
OVERLAP_NOLOG = os.environ.get('CIFAR_OVERLAP_NOLOG') == '1'
NCLS = int(os.environ.get('CIFAR_CLASSES', 10))
DSET = {10: datasets.CIFAR10, 100: datasets.CIFAR100}[NCLS]
STATS = {10: ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
         100: ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))}[NCLS]
TOTALS = [int(v) for v in os.environ['CIFAR_N'].split(',')] if os.environ.get('CIFAR_N') else None
_default_pc = [10, 50, 100, 500] if NCLS == 100 else [5, 20, 100, 500]
PER_CLASS = [5, 20] if SMOKE else [int(v) for v in
                                   os.environ.get('CIFAR_PER_CLASS',
                                                  ','.join(map(str, _default_pc))).split(',')]
if N_COMP % CHUNK:
    sys.exit(f'CIFAR_COMP must be a multiple of CIFAR_CHUNK={CHUNK}; got {N_COMP}')
OUT = os.environ.get('CIFAR_OUT', os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                               'results', f'cifar{NCLS}'))
os.makedirs(OUT, exist_ok=True)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def stratified_subset(dataset, per_class):
    labels = np.asarray(dataset.targets)
    idx = []
    for c in range(NCLS):
        pool = np.where(labels == c)[0]
        idx.extend(np.random.choice(pool, per_class, replace=False))
    imgs = torch.stack([dataset[i][0] for i in idx])
    return imgs, torch.tensor([dataset[i][1] for i in idx])


def haar_features(imgs, levels, comp):
    x = (imgs.numpy() if hasattr(imgs, 'numpy') else imgs).astype(np.float64).copy()
    r2 = np.sqrt(2.0)
    h, w = x.shape[-2:]
    for _ in range(levels):
        a = x[..., :h, :w]
        a = np.concatenate([(a[..., 0::2, :] + a[..., 1::2, :]) / r2,
                            (a[..., 0::2, :] - a[..., 1::2, :]) / r2], axis=-2)
        a = np.concatenate([(a[..., :, 0::2] + a[..., :, 1::2]) / r2,
                            (a[..., :, 0::2] - a[..., :, 1::2]) / r2], axis=-1)
        x[..., :h, :w] = a
        h, w = h // 2, w // 2
    x = np.sign(x) * np.log1p(np.abs(x))
    B, C, H, W = x.shape
    ii, jj = np.unravel_index(np.arange(H * W), (H, W))
    order = np.argsort(np.maximum(ii, jj), kind='stable')
    nb, nf = C, H * W
    co = x.reshape(B, nb, nf)
    perm = np.array([b * nf + order[r] for r in range(nf) for b in range(nb)])
    return co.reshape(B, nb * nf)[:, perm[:comp]].astype(np.float32)


def overlap_tokens(x, block, stride):
    x = x.numpy() if hasattr(x, 'numpy') else x
    B, C, H, W = x.shape
    ys = range(0, H - block + 1, stride)
    xs = range(0, W - block + 1, stride)
    pat = np.stack([x[:, :, i:i + block, j:j + block] for i in ys for j in xs], axis=2)
    co = fft.dctn(pat, type=2, norm='ortho', axes=(-2, -1))
    if not OVERLAP_NOLOG:
        co = np.sign(co) * np.log1p(np.abs(co))
    co = co.reshape(B, C, len(ys) * len(xs), block * block).transpose(0, 2, 1, 3)
    return co.reshape(B, -1).astype(np.float32)


def bispectrum(x, comp, nfreq=8):
    x = x.numpy() if hasattr(x, 'numpy') else x
    B, C, H, W = x.shape
    z = fft.fftn(x, axes=(-2, -1))
    ky, kx = np.unravel_index(np.arange(H * W), (H, W))
    fy = np.minimum(ky, H - ky); fx = np.minimum(kx, W - kx)
    low = np.argsort(fy ** 2 + fx ** 2, kind='stable')[:nfreq]
    ly, lx = ky[low], kx[low]
    feats = []
    for a in range(nfreq):
        for b in range(a, nfreq):
            sy, sx = (ly[a] + ly[b]) % H, (lx[a] + lx[b]) % W
            v = (z[:, :, ly[a], lx[a]] * z[:, :, ly[b], lx[b]]
                 * np.conj(z[:, :, sy, sx]))
            feats += [v.real, v.imag]
    f = np.stack(feats, axis=-1)
    f = np.sign(f) * np.log1p(np.abs(f))
    return f.reshape(B, -1)[:, :comp].astype(np.float32)


def scattering(x, comp, levels=3):
    x = x.numpy() if hasattr(x, 'numpy') else x
    B, C = x.shape[:2]
    out, ll = [], x
    firsts = []
    for _ in range(levels):
        ll, lh, hl, hh = haar2d(ll)
        for d in (lh, hl, hh):
            m = np.abs(d)
            out.append(m.reshape(B, C, -1).mean(-1))
            firsts.append(m)
    out.insert(0, ll.reshape(B, C, -1).mean(-1))
    for m in firsts[:3]:
        if m.shape[-1] < 2:
            continue
        _, l2, h2, hh2 = haar2d(m)
        for d2 in (l2, h2, hh2):
            out.append(np.abs(d2).reshape(B, C, -1).mean(-1))
    f = np.stack(out, axis=-1)
    f = np.sign(f) * np.log1p(np.abs(f))
    return f.reshape(B, -1)[:, :comp].astype(np.float32)


def block_tokens(x, grid):
    x = x.numpy() if hasattr(x, 'numpy') else x
    B, C, H, W = x.shape
    bh, bw = H // grid, W // grid
    b = x.reshape(B, C, grid, bh, grid, bw).transpose(0, 1, 2, 4, 3, 5)
    co = fft.dctn(b, type=2, norm='ortho', axes=(-2, -1))
    co = np.sign(co) * np.log1p(np.abs(co))
    co = co.reshape(B, C, grid, grid, bh * bw).transpose(0, 2, 3, 1, 4)
    return co.reshape(B, -1).astype(np.float32)


def haar2d(a):
    r2 = np.sqrt(2.0)
    lo = (a[..., 0::2, :] + a[..., 1::2, :]) / r2
    hi = (a[..., 0::2, :] - a[..., 1::2, :]) / r2
    return ((lo[..., :, 0::2] + lo[..., :, 1::2]) / r2,
            (lo[..., :, 0::2] - lo[..., :, 1::2]) / r2,
            (hi[..., :, 0::2] + hi[..., :, 1::2]) / r2,
            (hi[..., :, 0::2] - hi[..., :, 1::2]) / r2)


def haar_tokens(x, levels=2):
    x = x.numpy() if hasattr(x, 'numpy') else x
    B, C = x.shape[:2]
    ll, details = x, []
    for _ in range(levels):
        ll, lh, hl, hh = haar2d(ll)
        details.append((lh, hl, hh))
    g = ll.shape[-1]
    toks = [ll.reshape(B, C, g, g, 1)]
    for k in range(levels - 1, -1, -1):
        for d in details[k]:
            r = d.shape[-1]
            f = r // g
            toks.append(d.reshape(B, C, g, f, g, f)
                         .transpose(0, 1, 2, 4, 3, 5).reshape(B, C, g, g, f * f))
    t = np.concatenate(toks, axis=-1)
    t = t.transpose(0, 2, 3, 1, 4).reshape(B, g * g, -1)
    t = np.sign(t) * np.log1p(np.abs(t))
    return t.reshape(B, -1).astype(np.float32)


def rand_features(flat, comp, seed=0):
    rs = np.random.RandomState(seed)
    q, _ = np.linalg.qr(rs.randn(flat.shape[1], comp))
    return (flat @ q).astype(np.float32)


def spectral_features(imgs, grid, comp, kind='dct'):
    x = imgs.numpy() if hasattr(imgs, 'numpy') else imgs
    B, C, H, W = x.shape
    bh, bw = H // grid, W // grid
    x = x.reshape(B, C, grid, bh, grid, bw).transpose(0, 1, 2, 4, 3, 5)
    if kind == 'dct':
        co = fft.dctn(x, type=2, norm='ortho', axes=(-2, -1))
        co = np.sign(co) * np.log1p(np.abs(co))
    elif kind == 'fftc':
        z = fft.rfftn(x, axes=(-2, -1))
        co = np.stack([z.real, z.imag], axis=-1)
        co = np.sign(co) * np.log1p(np.abs(co))
    else:
        co = np.log1p(np.abs(fft.fftn(x, axes=(-2, -1))))
    nf, nb = bh * bw, C * grid * grid
    ky, kx = np.unravel_index(np.arange(nf), (bh, bw))
    order = np.argsort(ky ** 2 + kx ** 2, kind='stable')
    if kind == 'fftc':
        hw = bw // 2 + 1
        rnf = bh * hw
        co = co.reshape(B, nb, rnf, 2)
        ry, rx = np.unravel_index(np.arange(rnf), (bh, hw))
        edge = (rx == 0) | (rx == bw // 2)
        uniq = ~edge | (ry <= bh // 2)
        ridx = np.where(uniq)[0]
        rorder = ridx[np.argsort(np.minimum(ry[ridx], bh - ry[ridx]) ** 2
                                 + rx[ridx] ** 2, kind='stable')]
        rnf_ord = len(rorder)
        take = max(1, comp // 2)
        perm = np.array([b * rnf + rorder[r] for r in range(rnf_ord)
                         for b in range(nb)])[:take]
        g = co.reshape(B, nb * rnf, 2)[:, perm, :]
        return g.reshape(B, -1)[:, :comp].astype(np.float32)
    co = co.reshape(B, nb, nf)
    perm = np.array([b * nf + order[r] for r in range(nf) for b in range(nb)])
    return co.reshape(B, nb * nf)[:, perm[:comp]].astype(np.float32)


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
    h = int((-b + np.sqrt(b * b - 4 * (num_classes - target))) / 2)
    return max(2, h)


def subset_by_total(dataset, n):
    labels = np.asarray(dataset.targets)
    base, rem = divmod(n, NCLS)
    counts = np.full(NCLS, base)
    counts[np.random.choice(NCLS, rem, replace=False)] += 1
    idx = []
    for c in range(NCLS):
        if counts[c]:
            pool = np.where(labels == c)[0]
            idx.extend(np.random.choice(pool, counts[c], replace=False))
    imgs = torch.stack([dataset[i][0] for i in idx])
    return imgs, torch.tensor([dataset[i][1] for i in idx])


def metrics(probs, y_true):
    lab = np.arange(NCLS)
    out = {'T1': accuracy_score(y_true, np.argmax(probs, axis=1)),
           'T5': top_k_accuracy_score(y_true, probs, k=5, labels=lab),
           'AUC': roc_auc_score(y_true, probs, multi_class='ovr', labels=lab)}
    if NCLS > 10:
        out['T10'] = top_k_accuracy_score(y_true, probs, k=10, labels=lab)
    return out


tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*STATS)])
train_full = DSET(root=ROOT, train=True, download=DL, transform=tf)
test_full = DSET(root=ROOT, train=False, download=DL, transform=tf)
te_imgs = torch.stack([test_full[i][0] for i in range(len(test_full))])
te_y = np.asarray(test_full.targets)
te_flat = te_imgs.view(len(te_imgs), -1).numpy()
print(f'CIFAR-{NCLS} {len(train_full)} train / {len(te_imgs)} test on {dev}; '
      f'basis {BASIS}' + (f' grid {GRID}' if BASIS in ('dct', 'fft', 'fftc') else '') +
      f', {N_COMP} coef', flush=True)
te_spec = (spectral_features(te_imgs, GRID, N_COMP, BASIS)
           if BASIS in ('dct', 'fft', 'fftc') else
           haar_features(te_imgs, LEVELS, N_COMP) if BASIS == 'haar' else
           haar_tokens(te_imgs, LEVELS) if BASIS == 'haartok' else
           block_tokens(te_imgs, GRID) if BASIS == 'blocktok' else
           bispectrum(te_imgs, N_COMP) if BASIS == 'bispec' else
           scattering(te_imgs, N_COMP) if BASIS == 'scatter' else
           overlap_tokens(te_imgs, 4, STRIDE) if BASIS == 'overlap' else None)

rows, t0 = [], time.time()
for per_class in (TOTALS if TOTALS else PER_CLASS):
    for seed in range(N_SEEDS):
        torch.manual_seed(seed); np.random.seed(seed)
        tr_imgs, tr_y = (subset_by_total(train_full, per_class) if TOTALS else
                         stratified_subset(train_full, per_class))
        tr_flat = tr_imgs.view(len(tr_imgs), -1).numpy()

        if BASIS == 'overlap':
            Ftr, Fte = overlap_tokens(tr_imgs, 4, STRIDE), te_spec
            mu, sd = Ftr.mean(0, keepdims=True), Ftr.std(0, keepdims=True) + 1e-6
            Ztr = torch.tensor((Ftr - mu) / sd, dtype=torch.float32)
            Zte = torch.tensor((Fte - mu) / sd, dtype=torch.float32)
        elif BASIS in ('bispec', 'scatter'):
            fn = bispectrum if BASIS == 'bispec' else scattering
            Ftr, Fte = fn(tr_imgs, N_COMP), te_spec
            mu, sd = Ftr.mean(0, keepdims=True), Ftr.std(0, keepdims=True) + 1e-6
            Ztr = torch.tensor((Ftr - mu) / sd, dtype=torch.float32)
            Zte = torch.tensor((Fte - mu) / sd, dtype=torch.float32)
        elif BASIS == 'pcasup':
            sc = StandardScaler().fit(tr_flat)
            wide = min(256, len(tr_flat) - 1)
            pw = PCA(n_components=wide).fit(sc.transform(tr_flat))
            Atr, Ate = pw.transform(sc.transform(tr_flat)), pw.transform(sc.transform(te_flat))
            Fv, _ = f_classif(Atr, tr_y.numpy())
            keep = np.argsort(-np.nan_to_num(Fv))[:N_COMP]
            Ztr = torch.tensor(Atr[:, keep], dtype=torch.float32)
            Zte = torch.tensor(Ate[:, keep], dtype=torch.float32)
        elif BASIS == 'blocktok':
            Ftr, Fte = block_tokens(tr_imgs, GRID), te_spec
            mu, sd = Ftr.mean(0, keepdims=True), Ftr.std(0, keepdims=True) + 1e-6
            Ztr = torch.tensor((Ftr - mu) / sd, dtype=torch.float32)
            Zte = torch.tensor((Fte - mu) / sd, dtype=torch.float32)
        elif BASIS == 'haartok':
            Ftr, Fte = haar_tokens(tr_imgs, LEVELS), te_spec
            mu, sd = Ftr.mean(0, keepdims=True), Ftr.std(0, keepdims=True) + 1e-6
            Ztr = torch.tensor((Ftr - mu) / sd, dtype=torch.float32)
            Zte = torch.tensor((Fte - mu) / sd, dtype=torch.float32)
        elif BASIS == 'haar':
            Ftr = haar_features(tr_imgs, LEVELS, N_COMP)
            Fte = te_spec
            mu, sd = Ftr.mean(0, keepdims=True), Ftr.std(0, keepdims=True) + 1e-6
            Ztr = torch.tensor((Ftr - mu) / sd, dtype=torch.float32)
            Zte = torch.tensor((Fte - mu) / sd, dtype=torch.float32)
        elif BASIS == 'ica':
            sc = StandardScaler().fit(tr_flat)
            ica = FastICA(n_components=min(N_COMP, len(tr_flat)), whiten='unit-variance',
                          max_iter=400, random_state=0, tol=1e-3)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                ica.fit(sc.transform(tr_flat))
            Ztr = torch.tensor(ica.transform(sc.transform(tr_flat)), dtype=torch.float32)
            Zte = torch.tensor(ica.transform(sc.transform(te_flat)), dtype=torch.float32)
        elif BASIS == 'rand':
            Ftr, Fte = rand_features(tr_flat, N_COMP), rand_features(te_flat, N_COMP)
            mu, sd = Ftr.mean(0, keepdims=True), Ftr.std(0, keepdims=True) + 1e-6
            Ztr = torch.tensor((Ftr - mu) / sd, dtype=torch.float32)
            Zte = torch.tensor((Fte - mu) / sd, dtype=torch.float32)
        elif BASIS in ('dct', 'fft', 'fftc'):
            Ftr, Fte = spectral_features(tr_imgs, GRID, N_COMP, BASIS), te_spec
            mu, sd = Ftr.mean(0, keepdims=True), Ftr.std(0, keepdims=True) + 1e-6
            Ztr = torch.tensor((Ftr - mu) / sd, dtype=torch.float32)
            Zte = torch.tensor((Fte - mu) / sd, dtype=torch.float32)
        else:
            sc = StandardScaler().fit(tr_flat)
            pca = PCA(n_components=min(N_COMP, len(tr_flat))).fit(sc.transform(tr_flat))
            Ztr = torch.tensor(pca.transform(sc.transform(tr_flat)), dtype=torch.float32)
            Zte = torch.tensor(pca.transform(sc.transform(te_flat)), dtype=torch.float32)

        spec = MultiClassSpectralViT(n_inputs=Ztr.shape[1], chunk_size=CHUNK,
                                     num_classes=NCLS, **SPEC_KW).to(dev)
        spat = MultiClassSpatialViT(num_classes=NCLS, size=SIZE, patch_size=PATCH,
                                    in_channels=CHANS, stride=SPAT_STRIDE).to(dev)
        mlp = FeatureMLP(Ztr.shape[1], match_hidden(
            Ztr.shape[1], NCLS, sum(p.numel() for p in spec.parameters())), NCLS).to(dev)
        np_mlp = sum(p.numel() for p in mlp.parameters())
        np_spec = sum(p.numel() for p in spec.parameters())
        np_spat = sum(p.numel() for p in spat.parameters())
        opt = optim.AdamW(list(spec.parameters()) + list(spat.parameters()) +
                          list(mlp.parameters()), lr=LR, weight_decay=WD)
        loader = DataLoader(TensorDataset(tr_imgs, Ztr, tr_y), batch_size=BATCH,
                            shuffle=True)
        for _ in range(EPOCHS):
            spec.train(); spat.train(); mlp.train()
            for bi, bz, by in loader:
                bi, bz, by = bi.to(dev), bz.to(dev), by.to(dev)
                opt.zero_grad()
                (F.cross_entropy(spec(bz), by) + F.cross_entropy(spat(bi), by) +
                 F.cross_entropy(mlp(bz), by)).backward()
                opt.step()

        spec.eval(); spat.eval(); mlp.eval()
        ps, pp, pm = [], [], []
        with torch.no_grad():
            for bi, bz, _ in DataLoader(TensorDataset(te_imgs, Zte,
                                                      torch.from_numpy(te_y)),
                                        batch_size=512):
                ps.append(F.softmax(spec(bz.to(dev)), 1).cpu())
                pp.append(F.softmax(spat(bi.to(dev)), 1).cpu())
                pm.append(F.softmax(mlp(bz.to(dev)), 1).cpu())
        ms = metrics(torch.cat(ps).numpy(), te_y)
        mp = metrics(torch.cat(pp).numpy(), te_y)
        mm = metrics(torch.cat(pm).numpy(), te_y)
        rows.append(dict(per_class=(None if TOTALS else per_class),
                         n=per_class if TOTALS else per_class * NCLS, ncls=NCLS, seed=seed,
                         comp=int(Ztr.shape[1]), basis=BASIS, grid=GRID,
                         mode=int('use_mode_weights' in SPEC_KW), chunk=CHUNK,
                         np_spec=np_spec, np_spat=np_spat, np_mlp=np_mlp,
                         **{f'mlp_{k}': v for k, v in mm.items()},
                         **{f'spec_{k}': v for k, v in ms.items()},
                         **{f'spat_{k}': v for k, v in mp.items()}))
        tag = f'n={per_class:5d}' if TOTALS else f'{per_class:3d}/class'
        print(f'  {tag} seed {seed}: spec T1 {ms["T1"]:.3f} '
              f'spat T1 {mp["T1"]:.3f} | spec AUC {ms["AUC"]:.3f} '
              f'spat AUC {mp["AUC"]:.3f} mlp AUC {mm["AUC"]:.3f} '
              f'({time.time()-t0:.0f}s)', flush=True)
        del spec, spat, mlp
        if dev.type == 'cuda':
            torch.cuda.empty_cache()

        d = pd.DataFrame(rows)
        tmp = os.path.join(OUT, f'cifar{NCLS}_perseed.csv.tmp')
        d.to_csv(tmp, index=False); os.replace(tmp, os.path.join(OUT, f'cifar{NCLS}_perseed.csv'))

print(f'\nparameters: spectral {np_spec:,}  spatial {np_spat:,}', flush=True)
print(f'wrote {OUT}/cifar{NCLS}_perseed.csv ({len(rows)} cells)', flush=True)
