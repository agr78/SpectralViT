"""MedMNIST3D classification by training-set size.

usage: python medmnist_sweep.py [--smoke]"""
from paths import DATA
import os, sys, time
import numpy as np, pandas as pd, torch, torch.optim as optim
import torch.nn.functional as F
from scipy import fft
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, top_k_accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from networks import MultiClassSpectralViT, MultiClassSpatialViT

_POS = os.environ.get('MEDMNIST_POS', '1') != '0'


SMOKE = '--smoke' in sys.argv
NPZ = os.environ.get('MEDMNIST_NPZ', f'{DATA}/medmnist/organmnist3d.npz')
N_SEEDS = 2 if SMOKE else int(os.environ.get('MEDMNIST_SEEDS', 5))
SEED0 = int(os.environ.get('MEDMNIST_SEED0', 0))
N_COMP = int(os.environ.get('MEDMNIST_COMP', 16))
FULL_RANK = N_COMP == 0
BASIS = os.environ.get('MEDMNIST_BASIS', 'pca')
GRID = int(os.environ.get('MEDMNIST_GRID', 2))
FREQ = int(os.environ.get('MEDMNIST_FREQ', 3))
BLOCK = int(os.environ.get('MEDMNIST_BLOCK', 0)) or None
STRIDE = int(os.environ.get('MEDMNIST_STRIDE', 0)) or None
SIZES = [25, 50] if SMOKE else [int(v) for v in
                                os.environ.get('MEDMNIST_N', '25,50,100,200,500').split(',')]
EPOCHS, BATCH, LR, WD = (2 if SMOKE else 100), 128, 1e-3, 0.01
PATCH = int(os.environ.get('MEDMNIST_PATCH', 4))
if 28 % PATCH:
    sys.exit(f'MEDMNIST_PATCH must divide 28; got {PATCH}')
OUT = os.environ.get('MEDMNIST_OUT',
                     os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'results', 'medmnist'))
if N_COMP and N_COMP % 4:
    sys.exit(f'MEDMNIST_COMP must be a multiple of 4 (chunk_size); got {N_COMP}')
os.makedirs(OUT, exist_ok=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
SPAT_CACHE = os.environ.get('MEDMNIST_SPAT_CACHE',
                           f'{DATA}/medmnist_spatial_cache')
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def block_offsets(side):
    b = BLOCK or side // GRID
    st = STRIDE or b
    return [o for o in range(0, side - b + 1, st)], b


def n_blocks(side=28):
    return len(block_offsets(side)[0]) ** 3


def block_spectra(vols):
    B = len(vols)
    offs, b = block_offsets(vols.shape[1])
    nb = len(offs) ** 3
    out = np.zeros((B, nb, FREQ ** 3), dtype=np.float32)
    for i in range(B):
        bi = 0
        for z in offs:
            for y in offs:
                for x in offs:
                    blk = vols[i, z:z+b, y:y+b, x:x+b]
                    c = fft.dctn(blk, type=2, norm='ortho')[:FREQ, :FREQ, :FREQ].ravel()
                    out[i, bi] = np.sign(c) * np.log1p(np.abs(c))
                    bi += 1
    return out.reshape(B, nb * FREQ ** 3)


def band_energy(vols, nbands=None):
    nbands = nbands or int(os.environ.get('MEDMNIST_BANDS', 12))
    offs, side = block_offsets(vols.shape[1])
    kz, ky, kx = np.meshgrid(*[np.arange(side)] * 3, indexing='ij')
    kr = np.sqrt(kz ** 2 + ky ** 2 + kx ** 2).ravel()
    edges = np.linspace(0, kr.max() + 1e-6, nbands + 1)
    masks = [np.digitize(kr, edges) - 1 == b for b in range(nbands)]
    out = np.zeros((len(vols), len(offs) ** 3, nbands), dtype=np.float32)
    for i in range(len(vols)):
        bi = 0
        for z in offs:
            for y in offs:
                for x in offs:
                    c2 = fft.dctn(vols[i, z:z+side, y:y+side, x:x+side],
                                  type=2, norm='ortho').ravel() ** 2
                    for b, m in enumerate(masks):
                        out[i, bi, b] = c2[m].mean()
                    bi += 1
    return np.log1p(out.reshape(len(vols), -1))


def scatter_energy(vols, J=None):
    J = J or int(os.environ.get('MEDMNIST_SCAT_J', 4))
    offs, side = block_offsets(vols.shape[1])
    f1 = np.fft.fftfreq(side)
    kz, ky, kx = np.meshgrid(f1, f1, f1, indexing='ij')
    kr = np.sqrt(kz ** 2 + ky ** 2 + kx ** 2)
    cents = 0.5 / 2.0 ** np.arange(J)[::-1]
    filts = [np.exp(-0.5 * ((kr - c) / (0.5 * c)) ** 2) for c in cents]
    pairs = [(a, b) for a in range(J) for b in range(a + 1, J)]
    out = np.zeros((len(vols), len(offs) ** 3, J + len(pairs)), dtype=np.float32)
    for i in range(len(vols)):
        bi = 0
        for z in offs:
            for y in offs:
                for x in offs:
                    Fb = np.fft.fftn(vols[i, z:z+side, y:y+side, x:x+side])
                    u = [np.abs(np.fft.ifftn(Fb * f)) for f in filts]
                    out[i, bi, :J] = [v.mean() for v in u]
                    for j, (a, b) in enumerate(pairs):
                        Fa = np.fft.fftn(u[a])
                        out[i, bi, J + j] = np.abs(np.fft.ifftn(Fa * filts[b])).mean()
                    bi += 1
    return np.log1p(out.reshape(len(vols), -1))


def oriented_energy(vols, nrad=None, nori=None):
    nrad = nrad or int(os.environ.get('MEDMNIST_ORAD', 6))
    nori = nori or int(os.environ.get('MEDMNIST_ORI', 6))
    offs, side = block_offsets(vols.shape[1])
    kz, ky, kx = [a.ravel() for a in
                  np.meshgrid(*[np.arange(side)] * 3, indexing='ij')]
    kr = np.sqrt(kz ** 2 + ky ** 2 + kx ** 2)
    th = np.arccos(np.clip(kz / (kr + 1e-9), -1, 1))
    ph = np.arctan2(ky, kx)
    rb = np.clip((kr / (kr.max() + 1e-9) * nrad).astype(int), 0, nrad - 1)
    nth = 2
    nph = max(1, nori // nth)
    tb = np.clip((th / (np.pi / 2 + 1e-9) * nth).astype(int), 0, nth - 1)
    pb = np.clip((ph / (np.pi / 2 + 1e-9) * nph).astype(int), 0, nph - 1)
    cell = rb * (nth * nph) + tb * nph + pb
    nc = nrad * nth * nph
    masks = [cell == c for c in range(nc)]
    keep = [m for m in masks if m.any()]
    out = np.zeros((len(vols), len(offs) ** 3, len(keep)), dtype=np.float32)
    for i in range(len(vols)):
        bi = 0
        for z in offs:
            for y in offs:
                for x in offs:
                    c2 = fft.dctn(vols[i, z:z+side, y:y+side, x:x+side],
                                  type=2, norm='ortho').ravel() ** 2
                    for b, m in enumerate(keep):
                        out[i, bi, b] = c2[m].mean()
                    bi += 1
    return np.log1p(out.reshape(len(vols), -1))


def topo_features(vols, nth=None):
    from scipy import ndimage
    nth = nth or int(os.environ.get('MEDMNIST_THRESH', 12))
    out = np.zeros((len(vols), nth, 3), dtype=np.float32)
    for i, v in enumerate(vols):
        lo, hi = np.percentile(v, [5, 95])
        for j, t in enumerate(np.linspace(lo, hi, nth)):
            B = v > t
            out[i, j, 0] = ndimage.label(B)[1]
            out[i, j, 1] = ndimage.label(~B)[1]
            out[i, j, 2] = B.mean()
    return np.log1p(out.reshape(len(vols), -1))


def block_mag(vols):
    B = len(vols)
    offs, b = block_offsets(vols.shape[1])
    nb = len(offs) ** 3
    out = np.zeros((B, nb, FREQ ** 3), dtype=np.float32)
    for i in range(B):
        bi = 0
        for z in offs:
            for y in offs:
                for x in offs:
                    m = np.abs(fft.fftn(vols[i, z:z+b, y:y+b, x:x+b]))
                    out[i, bi] = np.log1p(m[:FREQ, :FREQ, :FREQ].ravel())
                    bi += 1
    return out.reshape(B, nb * FREQ ** 3)


def moment_features(vols):
    out = np.zeros((len(vols), 3 * vols.shape[1] + 9), dtype=np.float32)
    for i, v in enumerate(vols):
        w = np.maximum(v, 0.0)
        parts = []
        mom = []
        for ax in (0, 1, 2):
            p = w.sum(axis=tuple(a for a in (0, 1, 2) if a != ax)).astype(np.float64)
            parts.append(np.log1p(p))
            q = p / (p.sum() + 1e-9)
            c = np.arange(len(q), dtype=np.float64)
            mu = (c * q).sum()
            sd = np.sqrt(((c - mu) ** 2 * q).sum()) + 1e-9
            mom += [mu, sd, ((c - mu) ** 3 * q).sum() / sd ** 3]
        out[i] = np.concatenate(parts + [np.asarray(mom)])
    return out


def haar_features(vols, signed=False):
    r2 = np.sqrt(2.0)

    def split(a, ax):
        s = np.take(a, np.arange(0, a.shape[ax] - 1, 2), axis=ax)
        d = np.take(a, np.arange(1, a.shape[ax], 2), axis=ax)
        return (s + d) / r2, (s - d) / r2

    def level(a):
        out = []
        for x0 in split(a, 0):
            for x1 in split(x0, 1):
                out += list(split(x1, 2))
        return out

    def octants(b):
        h = [max(1, s // 2) for s in b.shape]
        return np.array([b[i*h[0]:(i+1)*h[0] or None,
                           j*h[1]:(j+1)*h[1] or None,
                           k*h[2]:(k+1)*h[2] or None].mean()
                         for i in (0, 1) for j in (0, 1) for k in (0, 1)])

    out = np.zeros((len(vols), 15 * 8), dtype=np.float32)
    for i, v in enumerate(vols):
        l1 = level(v.astype(np.float64))
        bands = l1[1:] + level(l1[0])
        rows = []
        for b in bands:
            c = b if signed else np.abs(b)
            o = octants(c)
            rows.append(o if signed else np.log1p(o))
        out[i] = np.concatenate(rows)
    return out


def _freq_grid(side):
    k = np.fft.fftfreq(side)
    kz, ky, kx = np.meshgrid(k, k, k, indexing='ij')
    kr = np.sqrt(kz ** 2 + ky ** 2 + kx ** 2)
    return kz, ky, kx, kr


def steerable_features(vols, nsc=None):
    nsc = nsc or int(os.environ.get('MEDMNIST_SCALES', 4))
    side = vols.shape[1]
    kz, ky, kx, kr = _freq_grid(side)
    cents = 0.5 / 2.0 ** np.arange(nsc)[::-1]
    rad = [np.exp(-0.5 * ((kr - c) / (0.5 * c)) ** 2) for c in cents]
    k2 = kr ** 2 + 1e-12
    ang = [kz * kz / k2, ky * ky / k2, kx * kx / k2,
           kz * ky / k2, kz * kx / k2, ky * kx / k2]
    out = np.zeros((len(vols), nsc * 6 * 8), dtype=np.float32)
    h = side // 2
    for i, v in enumerate(vols):
        Fv = np.fft.fftn(v)
        row = []
        for r in rad:
            for a in ang:
                resp = np.abs(np.fft.ifftn(Fv * r * a))
                row += [resp[p::h if h else 1].mean() if False else
                        resp[zi*h:(zi+1)*h, yi*h:(yi+1)*h, xi*h:(xi+1)*h].mean()
                        for zi in (0, 1) for yi in (0, 1) for xi in (0, 1)
                        for p in (0,)]
        out[i] = np.log1p(np.asarray(row))
    return out


def bispectrum_features(vols, nsh=None):
    nsh = nsh or int(os.environ.get('MEDMNIST_SHELLS', 6))
    side = vols.shape[1]
    _, _, _, kr = _freq_grid(side)
    e = np.linspace(0, kr.max() + 1e-9, nsh + 1)
    masks = [(kr >= e[b]) & (kr < e[b + 1]) for b in range(nsh)]
    pairs = [(a, b) for a in range(nsh) for b in range(a, nsh)]
    out = np.zeros((len(vols), 2 * len(pairs)), dtype=np.float32)
    for i, v in enumerate(vols):
        Fv = np.fft.fftn(v)
        comp = [np.fft.ifftn(Fv * m) for m in masks]
        row = []
        for a, b in pairs:
            z = comp[a] * comp[b] * np.conj(comp[min(a + b, nsh - 1)])
            row += [np.sign(z.real.mean()) * np.log1p(abs(z.real.mean())),
                    np.log1p(abs(z.imag.mean()))]
        out[i] = np.asarray(row)
    return out


def shell_harmonic_features(vols, lmax=None, nsh=None):
    from scipy.special import sph_harm_y
    lmax = lmax or int(os.environ.get('MEDMNIST_LMAX', 3))
    nsh = nsh or int(os.environ.get('MEDMNIST_SHELLS', 8))
    side = vols.shape[1]
    kz, ky, kx, kr = _freq_grid(side)
    th = np.arccos(np.clip(kz / (kr + 1e-12), -1, 1)).ravel()
    ph = np.arctan2(ky, kx).ravel()
    e = np.linspace(0, kr.max() + 1e-9, nsh + 1)
    sh_idx = np.digitize(kr.ravel(), e) - 1
    Y = {}
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            Y[(l, m)] = sph_harm_y(l, m, th, ph)
    out = np.zeros((len(vols), nsh * (lmax + 1)), dtype=np.float32)
    for i, v in enumerate(vols):
        P = np.abs(np.fft.fftn(v)).ravel()
        row = np.zeros((nsh, lmax + 1))
        for b in range(nsh):
            sel = sh_idx == b
            if not sel.any():
                continue
            p = P[sel]
            for l in range(lmax + 1):
                acc = 0.0
                for m in range(-l, l + 1):
                    acc += abs((p * np.conj(Y[(l, m)][sel])).sum()) ** 2
                row[b, l] = acc / (sel.sum() ** 2 + 1e-12)
        out[i] = np.log1p(row.ravel())
    return out


def glcm_features(vols, nlev=None):
    nlev = nlev or int(os.environ.get('MEDMNIST_GLEV', 8))
    offs = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)]
    out = np.zeros((len(vols), len(offs) * 5), dtype=np.float32)
    lv = np.arange(nlev)
    ii, jj = np.meshgrid(lv, lv, indexing='ij')
    for i, v in enumerate(vols):
        q = np.clip(np.digitize(v, np.percentile(v, np.linspace(0, 100, nlev + 1)[1:-1])),
                    0, nlev - 1)
        row = []
        for dz, dy, dx in offs:
            a = q[dz:q.shape[0] or None, dy:q.shape[1] or None, dx:q.shape[2] or None]
            b = q[:q.shape[0] - dz or None, :q.shape[1] - dy or None, :q.shape[2] - dx or None]
            G = np.bincount((a.ravel() * nlev + b.ravel()), minlength=nlev ** 2)
            G = G.reshape(nlev, nlev).astype(np.float64)
            G = (G + G.T) / (G.sum() * 2 + 1e-12)
            mu = (G * ii).sum()
            sd = np.sqrt((G * (ii - mu) ** 2).sum()) + 1e-12
            row += [(G * (ii - jj) ** 2).sum(),
                    (G / (1 + (ii - jj) ** 2)).sum(),
                    (G ** 2).sum(),
                    (G * (ii - mu) * (jj - mu)).sum() / sd ** 2,
                    -(G * np.log(G + 1e-12)).sum()]
        out[i] = np.asarray(row)
    return out


def patch_dct(vols, k=None, patch=None):
    patch = patch or PATCH
    k = k or int(os.environ.get('MEDMNIST_PK', patch ** 3))
    side = vols.shape[1]
    npx = side // patch
    f = np.arange(patch)
    fz, fy, fx = np.meshgrid(f, f, f, indexing='ij')
    order = np.argsort((fz ** 2 + fy ** 2 + fx ** 2).ravel(), kind='stable')[:k]
    out = np.zeros((len(vols), npx ** 3, k), dtype=np.float32)
    for i, v in enumerate(vols):
        t = 0
        for z in range(npx):
            for y in range(npx):
                for x in range(npx):
                    blk = v[z*patch:(z+1)*patch, y*patch:(y+1)*patch, x*patch:(x+1)*patch]
                    out[i, t] = fft.dctn(blk, type=2, norm='ortho').ravel()[order]
                    t += 1
    return out.reshape(len(vols), npx ** 3 * k)


def rank_perm():
    nb, nf = n_blocks(), FREQ ** 3
    kz, ky, kx = np.unravel_index(np.arange(nf), (FREQ, FREQ, FREQ))
    order = np.argsort(kz ** 2 + ky ** 2 + kx ** 2, kind='stable')
    return np.array([b * nf + order[r] for r in range(nf) for b in range(nb)])


def metrics(probs, y_true, ncls):
    lab = np.arange(ncls)
    return {'T1': accuracy_score(y_true, np.argmax(probs, axis=1)),
            'T5': (top_k_accuracy_score(y_true, probs, k=5, labels=lab)
                   if ncls > 5 else float('nan')),
            'AUC': (roc_auc_score(y_true, probs[:, 1]) if ncls == 2 else
                    roc_auc_score(y_true, probs, multi_class='ovr', labels=lab))}


def allocate(avail, n):
    k = len(avail)
    take = np.minimum(avail, n // k)
    while take.sum() < n:
        room = np.where(avail - take > 0)[0]
        if not len(room):
            break
        pick = np.random.choice(room, min(n - take.sum(), len(room)), replace=False)
        take[pick] += 1
    return take


z = np.load(NPZ)
Xtr_all = z['train_images'].astype(np.float32)
ytr_all = z['train_labels'].ravel().astype(np.int64)
Xte = z['test_images'].astype(np.float32)
yte = z['test_labels'].ravel().astype(np.int64)
NCLS, VOL = int(ytr_all.max()) + 1, Xtr_all.shape[1]
mu, sd = Xtr_all.mean(), Xtr_all.std()
Xtr_all, Xte = (Xtr_all - mu) / sd, (Xte - mu) / sd

OCCLUDE = os.environ.get('MEDMNIST_OCCLUDE')
if OCCLUDE:
    _occ = np.load(OCCLUDE)
    Xte = (_occ['volumes'].astype(np.float32) - mu) / sd
    yte = _occ['labels'].ravel().astype(np.int64)
te_imgs = torch.from_numpy(Xte).unsqueeze(1)
te_flat = te_imgs.view(len(te_imgs), -1).numpy()
pool = {c: np.where(ytr_all == c)[0] for c in range(NCLS)}
avail = np.array([len(pool[c]) for c in range(NCLS)])
print(f'{os.path.basename(NPZ)}: {len(Xtr_all)} train / {len(Xte)} test, {VOL}^3, '
      f'{NCLS} classes on {dev}; per-class train {avail.tolist()}', flush=True)

rows, t0 = [], time.time()
for N in SIZES:
    if N > avail.sum():
        print(f'  skipping n={N}: only {avail.sum()} training volumes'); continue
    GROUPS = None
    for seed in range(SEED0, SEED0 + N_SEEDS):
        torch.manual_seed(seed); np.random.seed(seed)
        take = allocate(avail, N)
        tr_idx = np.concatenate([np.random.choice(pool[c], take[c], replace=False)
                                 for c in range(NCLS) if take[c]])
        tr_imgs = torch.from_numpy(Xtr_all[tr_idx]).unsqueeze(1)
        tr_y = torch.from_numpy(ytr_all[tr_idx])
        tr_flat = tr_imgs.view(len(tr_imgs), -1).numpy()

        if BASIS in ('energy', 'energy_cat', 'dct_energy', 'pca_energy', 'pca_sq_energy'):
            Etr, Ete = band_energy(Xtr_all[tr_idx]), band_energy(Xte)
            se = StandardScaler().fit(Etr)
            if BASIS == 'energy':
                Ztr = torch.tensor(se.transform(Etr), dtype=torch.float32)
                Zte = torch.tensor(se.transform(Ete), dtype=torch.float32)
            elif BASIS == 'dct_energy':
                perm = rank_perm()
                Ftr = block_spectra(Xtr_all[tr_idx])[:, perm]
                Fte = block_spectra(Xte)[:, perm]
                sd = StandardScaler().fit(Ftr)
                Ztr = torch.tensor(np.hstack([sd.transform(Ftr), se.transform(Etr)]),
                                   dtype=torch.float32)
                Zte = torch.tensor(np.hstack([sd.transform(Fte), se.transform(Ete)]),
                                   dtype=torch.float32)
            elif BASIS == 'pca_sq_energy':
                sp = StandardScaler().fit(tr_flat)
                nc = len(tr_flat) - 1 if FULL_RANK else min(N_COMP, len(tr_flat))
                pca = PCA(n_components=max(1, nc)).fit(sp.transform(tr_flat))
                P0 = pca.transform(sp.transform(tr_flat))
                P1 = pca.transform(sp.transform(te_flat))
                Ztr = torch.tensor(np.hstack([P0, P0 ** 2, se.transform(Etr)]),
                                   dtype=torch.float32)
                Zte = torch.tensor(np.hstack([P1, P1 ** 2, se.transform(Ete)]),
                                   dtype=torch.float32)
            elif BASIS == 'pca_energy':
                sp = StandardScaler().fit(tr_flat)
                nc = len(tr_flat) - 1 if FULL_RANK else min(N_COMP, len(tr_flat))
                pca = PCA(n_components=max(1, nc)).fit(sp.transform(tr_flat))
                Ztr = torch.tensor(np.hstack([pca.transform(sp.transform(tr_flat)),
                                              se.transform(Etr)]), dtype=torch.float32)
                Zte = torch.tensor(np.hstack([pca.transform(sp.transform(te_flat)),
                                              se.transform(Ete)]), dtype=torch.float32)
            else:
                perm = rank_perm()
                Ftr = block_spectra(Xtr_all[tr_idx])[:, perm]
                Fte = block_spectra(Xte)[:, perm]
                sd = StandardScaler().fit(Ftr); sp = StandardScaler().fit(tr_flat)
                nc = len(tr_flat) - 1 if FULL_RANK else min(N_COMP, len(tr_flat))
                pca = PCA(n_components=max(1, nc)).fit(sp.transform(tr_flat))
                Ztr = torch.tensor(np.hstack([pca.transform(sp.transform(tr_flat)),
                                              sd.transform(Ftr), se.transform(Etr)]),
                                   dtype=torch.float32)
                Zte = torch.tensor(np.hstack([pca.transform(sp.transform(te_flat)),
                                              sd.transform(Fte), se.transform(Ete)]),
                                   dtype=torch.float32)
            GROUPS = None
            if BASIS == 'dct_energy':
                GROUPS = [Ftr.shape[1], Etr.shape[1]]
            elif BASIS == 'pca_energy':
                GROUPS = [Ztr.shape[1] - Etr.shape[1], Etr.shape[1]]
            elif BASIS == 'energy_cat':
                GROUPS = [Ztr.shape[1] - Ftr.shape[1] - Etr.shape[1], Ftr.shape[1], Etr.shape[1]]
            elif BASIS == 'pca_sq_energy':
                _np = (Ztr.shape[1] - Etr.shape[1]) // 2
                GROUPS = [_np, _np, Etr.shape[1]]
        elif BASIS in ('square', 'rbf', 'triple'):
            perm = rank_perm()
            Ftr = block_spectra(Xtr_all[tr_idx])[:, perm]
            Fte = block_spectra(Xte)[:, perm]
            sd = StandardScaler().fit(Ftr)
            sp = StandardScaler().fit(tr_flat)
            nc = len(tr_flat) - 1 if FULL_RANK else min(N_COMP, len(tr_flat))
            pca = PCA(n_components=max(1, nc)).fit(sp.transform(tr_flat))
            if BASIS == 'triple':
                g0, s0 = BLOCK, STRIDE
                globals()['BLOCK'], globals()['STRIDE'] = 14, 7
                g2 = rank_perm()
                Otr, Ote = block_spectra(Xtr_all[tr_idx])[:, g2], block_spectra(Xte)[:, g2]
                globals()['BLOCK'], globals()['STRIDE'] = g0, s0
                so = StandardScaler().fit(Otr)
            W = b = None
            if BASIS == 'rbf':
                D = int(os.environ.get('MEDMNIST_RBF', 128))
                rs = np.random.RandomState(0)

            def build(flat, F, O=None):
                P = pca.transform(sp.transform(flat)); Dd = sd.transform(F)
                Z = np.hstack([P, Dd])
                if BASIS == 'square':
                    return np.hstack([Z, Z ** 2])
                if BASIS == 'triple':
                    return np.hstack([Z, so.transform(O)])
                return np.hstack([Z, np.cos(Z @ W + b)])

            if BASIS == 'rbf':
                Z0 = np.hstack([pca.transform(sp.transform(tr_flat)), sd.transform(Ftr)])
                sig = np.median(np.abs(Z0)) + 1e-9
                W = rs.normal(0, 1.0 / (sig * np.sqrt(Z0.shape[1])), (Z0.shape[1], D))
                b = rs.uniform(0, 2 * np.pi, D)
            Ztr = torch.tensor(build(tr_flat, Ftr, Otr if BASIS == 'triple' else None),
                               dtype=torch.float32)
            Zte = torch.tensor(build(te_flat, Fte, Ote if BASIS == 'triple' else None),
                               dtype=torch.float32)
            GROUPS = None
        elif BASIS == 'cross':
            K = int(os.environ.get('MEDMNIST_CROSS', 8))
            perm = rank_perm()
            Ftr = block_spectra(Xtr_all[tr_idx])[:, perm]
            Fte = block_spectra(Xte)[:, perm]
            sd = StandardScaler().fit(Ftr)
            sp = StandardScaler().fit(tr_flat)
            nc = len(tr_flat) - 1 if FULL_RANK else min(N_COMP, len(tr_flat))
            pca = PCA(n_components=max(1, nc)).fit(sp.transform(tr_flat))

            def build(flat, F):
                P = pca.transform(sp.transform(flat))
                D = sd.transform(F)
                X = np.einsum('ni,nj->nij', P[:, :K], D[:, :K]).reshape(len(P), -1)
                return np.hstack([P, D, X])

            Ztr = torch.tensor(build(tr_flat, Ftr), dtype=torch.float32)
            Zte = torch.tensor(build(te_flat, Fte), dtype=torch.float32)
            GROUPS = None
        elif BASIS == 'concat':
            perm = rank_perm()
            Ftr = block_spectra(Xtr_all[tr_idx])[:, perm]
            Fte = block_spectra(Xte)[:, perm]
            sd = StandardScaler().fit(Ftr)
            sp = StandardScaler().fit(tr_flat)
            nc = len(tr_flat) - 1 if FULL_RANK else min(N_COMP, len(tr_flat))
            pca = PCA(n_components=max(1, nc)).fit(sp.transform(tr_flat))
            Ptr = pca.transform(sp.transform(tr_flat))
            Ztr = torch.tensor(np.hstack([Ptr, sd.transform(Ftr)]), dtype=torch.float32)
            Zte = torch.tensor(np.hstack([pca.transform(sp.transform(te_flat)),
                                          sd.transform(Fte)]), dtype=torch.float32)
            GROUPS = [Ptr.shape[1], Ftr.shape[1]]
        elif BASIS == 'blockpca':
            offs, side = block_offsets(Xtr_all.shape[1])
            def tiles(V):
                return np.stack([V[:, z:z+side, y:y+side, x:x+side].reshape(len(V), -1)
                                 for z in offs for y in offs for x in offs], 1)
            Ttr, Tte = tiles(Xtr_all[tr_idx]), tiles(Xte)
            nb, dim = Ttr.shape[1], Ttr.shape[2]
            tiled = Ttr.reshape(-1, dim)
            k = min(*tiled.shape) if FULL_RANK else min(N_COMP, *tiled.shape)
            sc = StandardScaler().fit(tiled)
            pca = PCA(n_components=k, random_state=0).fit(sc.transform(tiled))
            def enc(T):
                return pca.transform(sc.transform(T.reshape(-1, dim))).reshape(len(T), nb * k)
            Ztr = torch.tensor(enc(Ttr), dtype=torch.float32)
            Zte = torch.tensor(enc(Tte), dtype=torch.float32)
        elif BASIS in ('scatter', 'dct_scatter', 'oriented', 'dct_oriented',
                       'dct_energy_scatter', 'energy_gram'):
            Str = Gtr = None
            if 'scatter' in BASIS:
                Str, Ste = scatter_energy(Xtr_all[tr_idx]), scatter_energy(Xte)
            if 'oriented' in BASIS:
                Str, Ste = oriented_energy(Xtr_all[tr_idx]), oriented_energy(Xte)
            if 'energy' in BASIS or BASIS == 'energy_gram':
                Gtr, Gte = band_energy(Xtr_all[tr_idx]), band_energy(Xte)
            parts_tr, parts_te, names = [], [], []
            if BASIS.startswith('dct'):
                perm = rank_perm()
                Ftr = block_spectra(Xtr_all[tr_idx])[:, perm]
                Fte = block_spectra(Xte)[:, perm]
                sdd = StandardScaler().fit(Ftr)
                parts_tr.append(sdd.transform(Ftr)); parts_te.append(sdd.transform(Fte))
                names.append('dct')
            if Gtr is not None:
                sg = StandardScaler().fit(Gtr)
                parts_tr.append(sg.transform(Gtr)); parts_te.append(sg.transform(Gte))
                names.append('energy')
            if BASIS == 'energy_gram':
                nb = n_blocks()
                def gram(E):
                    R = E.reshape(len(E), nb, -1)
                    G = np.einsum('nbi,nbj->nij', R, R) / nb
                    iu = np.triu_indices(G.shape[1])
                    return G[:, iu[0], iu[1]]
                Qtr, Qte = gram(Gtr), gram(Gte)
                sq = StandardScaler().fit(Qtr)
                parts_tr.append(sq.transform(Qtr)); parts_te.append(sq.transform(Qte))
                names.append('gram')
            if Str is not None:
                ss2 = StandardScaler().fit(Str)
                parts_tr.append(ss2.transform(Str)); parts_te.append(ss2.transform(Ste))
                names.append('scatter' if 'scatter' in BASIS else 'oriented')
            Ztr = torch.tensor(np.hstack(parts_tr), dtype=torch.float32)
            Zte = torch.tensor(np.hstack(parts_te), dtype=torch.float32)
            GROUPS = [p.shape[1] for p in parts_tr]
            print('      %s: ' % BASIS + ' '.join(
                '%s %d' % (n, p.shape[1]) for n, p in zip(names, parts_tr)), flush=True)
        elif BASIS.startswith('pool'):
            SELECT = os.environ.get('MEDMNIST_SELECT', 'none')
            SELK = int(os.environ.get('MEDMNIST_SELK', 312))

            sp = StandardScaler().fit(tr_flat)
            nc = len(tr_flat) - 1 if FULL_RANK else min(N_COMP, len(tr_flat))
            pca = PCA(n_components=max(1, nc), random_state=0).fit(sp.transform(tr_flat))
            Ptr = pca.transform(sp.transform(tr_flat))
            Pte = pca.transform(sp.transform(te_flat))
            perm = rank_perm()
            Ftr = block_spectra(Xtr_all[tr_idx])[:, perm]
            Fte = block_spectra(Xte)[:, perm]
            sd = StandardScaler().fit(Ftr)
            Dtr, Dte = sd.transform(Ftr), sd.transform(Fte)
            Etr, Ete = band_energy(Xtr_all[tr_idx]), band_energy(Xte)
            se = StandardScaler().fit(Etr)
            Gtr, Gte = se.transform(Etr), se.transform(Ete)

            g0, s0 = BLOCK, STRIDE
            globals()['BLOCK'], globals()['STRIDE'] = 14, 7
            pov = rank_perm()
            Otr = block_spectra(Xtr_all[tr_idx])[:, pov]
            Ote = block_spectra(Xte)[:, pov]
            offs, side = block_offsets(Xtr_all.shape[1])
            globals()['BLOCK'], globals()['STRIDE'] = g0, s0
            so = StandardScaler().fit(Otr)
            Vtr, Vte = so.transform(Otr), so.transform(Ote)

            def tiles(V):
                return np.stack([V[:, z:z+side, y:y+side, x:x+side].reshape(len(V), -1)
                                 for z in offs for y in offs for x in offs], 1)
            Ttr, Tte = tiles(Xtr_all[tr_idx]), tiles(Xte)
            nb, dim = Ttr.shape[1], Ttr.shape[2]
            tiled = Ttr.reshape(-1, dim)
            kb = min(N_COMP or min(*tiled.shape), *tiled.shape)
            sb = StandardScaler().fit(tiled)
            bp = PCA(n_components=kb, random_state=0).fit(sb.transform(tiled))
            def enc(T):
                return bp.transform(sb.transform(T.reshape(-1, dim))).reshape(len(T), nb * kb)
            Btr, Bte = enc(Ttr), enc(Tte)

            K = int(os.environ.get('MEDMNIST_CROSS', 8))
            def xprod(P, D):
                return np.einsum('ni,nj->nij', P[:, :K], D[:, :K]).reshape(len(P), -1)
            Xtr_c, Xte_c = xprod(Ptr, Dtr), xprod(Pte, Dte)
            Qtr = np.hstack([Ptr, Dtr]) ** 2
            Qte = np.hstack([Pte, Dte]) ** 2
            RD = int(os.environ.get('MEDMNIST_RBF', 128))
            rs = np.random.RandomState(0)
            Z0 = np.hstack([Ptr, Dtr])
            sig = np.median(np.abs(Z0)) + 1e-9
            Wr = rs.normal(0, 1.0 / (sig * np.sqrt(Z0.shape[1])), (Z0.shape[1], RD))
            br = rs.uniform(0, 2 * np.pi, RD)
            Rtr = np.cos(Z0 @ Wr + br)
            Rte = np.cos(np.hstack([Pte, Dte]) @ Wr + br)

            BLK = [('pca', Ptr, Pte), ('dct', Dtr, Dte), ('dct_ov', Vtr, Vte),
                   ('blockpca', Btr, Bte), ('energy', Gtr, Gte), ('cross', Xtr_c, Xte_c),
                   ('square', Qtr, Qte), ('rbf', Rtr, Rte)]
            Atr = np.hstack([b[1] for b in BLK]).astype(np.float64)
            Ate = np.hstack([b[2] for b in BLK]).astype(np.float64)
            owner = np.concatenate([[i] * b[1].shape[1] for i, b in enumerate(BLK)])
            ytr = tr_y.numpy()

            idx, proj = np.arange(Atr.shape[1]), None
            k = min(SELK, Atr.shape[1])
            if SELECT == 'fscore':
                from sklearn.feature_selection import f_classif
                sc = np.nan_to_num(f_classif(Atr, ytr)[0])
                idx = np.sort(np.argsort(-sc)[:k])
            elif SELECT == 'mi':
                from sklearn.feature_selection import mutual_info_classif
                sc = mutual_info_classif(Atr, ytr, random_state=0, n_neighbors=3)
                idx = np.sort(np.argsort(-sc)[:k])
            elif SELECT == 'mrmr':
                from sklearn.feature_selection import mutual_info_classif
                rel = mutual_info_classif(Atr, ytr, random_state=0, n_neighbors=3)
                C = np.abs(np.nan_to_num(np.corrcoef(Atr, rowvar=False)))
                pick = [int(np.argmax(rel))]
                while len(pick) < k:
                    red = C[:, pick].mean(1)
                    cand = rel - red
                    cand[pick] = -np.inf
                    pick.append(int(np.argmax(cand)))
                idx = np.sort(np.array(pick))
            elif SELECT == 'stab':
                from sklearn.linear_model import LogisticRegression
                B = int(os.environ.get('MEDMNIST_BOOT', 100))
                cnt = np.zeros(Atr.shape[1])
                rb = np.random.RandomState(seed)
                for _ in range(B):
                    bi = rb.choice(len(Atr), len(Atr), replace=True)
                    if len(np.unique(ytr[bi])) < 2:
                        continue
                    lr = LogisticRegression(penalty='l1', C=0.1, solver='liblinear',
                                            max_iter=2000)
                    try:
                        lr.fit(Atr[bi], ytr[bi])
                    except Exception:
                        continue
                    cnt += (np.abs(lr.coef_).max(0) > 1e-8)
                idx = np.sort(np.argsort(-cnt)[:k]) if cnt.any() else idx
            elif SELECT == 'group':
                from sklearn.linear_model import LogisticRegression
                from sklearn.model_selection import RepeatedStratifiedKFold
                from sklearn.metrics import roc_auc_score as _auc
                mn = np.bincount(ytr).min()
                cv = RepeatedStratifiedKFold(n_splits=min(5, mn), n_repeats=4,
                                             random_state=0) if mn >= 2 else None
                sc = np.zeros(len(BLK))
                for i, (nm, tb, _) in enumerate(BLK):
                    if cv is None:
                        sc[i] = 0.5; continue
                    aucs = []
                    for a, b in cv.split(tb, ytr):
                        if len(np.unique(ytr[a])) < 2:
                            continue
                        lr = LogisticRegression(max_iter=2000, C=1.0).fit(tb[a], ytr[a])
                        pr = lr.predict_proba(tb[b])
                        try:
                            aucs.append(_auc(ytr[b], pr[:, 1] if pr.shape[1] == 2 else pr,
                                             multi_class='ovr',
                                             labels=np.arange(NCLS)) if pr.shape[1] > 2
                                        else _auc(ytr[b], pr[:, 1]))
                        except Exception:
                            continue
                    sc[i] = np.mean(aucs) if aucs else 0.5
                keep = np.where(sc >= np.median(sc))[0]
                idx = np.sort(np.where(np.isin(owner, keep))[0])
                print('      group scores ' + ' '.join(
                    '%s=%.3f%s' % (BLK[i][0], sc[i], '*' if i in keep else '')
                    for i in range(len(BLK))), flush=True)
            elif SELECT == 'cons':
                keep = [i for i, b in enumerate(BLK)
                        if b[0] in ('dct', 'dct_ov', 'blockpca', 'energy')]
                idx = np.sort(np.where(np.isin(owner, keep))[0])
            elif SELECT == 'cons_lite':
                keep = [i for i, b in enumerate(BLK) if b[0] in ('dct', 'dct_ov', 'energy')]
                idx = np.sort(np.where(np.isin(owner, keep))[0])
            elif SELECT == 'pcared':
                ss = StandardScaler().fit(Atr)
                proj = PCA(n_components=min(k, len(Atr) - 1), random_state=0).fit(
                    ss.transform(Atr))
                Ztr = torch.tensor(proj.transform(ss.transform(Atr)), dtype=torch.float32)
                Zte = torch.tensor(proj.transform(ss.transform(Ate)), dtype=torch.float32)
            if proj is None:
                Ztr = torch.tensor(Atr[:, idx], dtype=torch.float32)
                Zte = torch.tensor(Ate[:, idx], dtype=torch.float32)
                kept = [int((owner[idx] == i).sum()) for i in range(len(BLK))]
                GROUPS = [c for c in kept if c]
                print('      pool %d -> %d via %s | ' % (Atr.shape[1], len(idx), SELECT)
                      + ' '.join('%s %d/%d' % (BLK[i][0], kept[i], BLK[i][1].shape[1])
                                 for i in range(len(BLK))), flush=True)
            else:
                GROUPS = None
        elif BASIS == 'blockpca_energy':
            Etr, Ete = band_energy(Xtr_all[tr_idx]), band_energy(Xte)
            se = StandardScaler().fit(Etr)
            offs, side = block_offsets(Xtr_all.shape[1])

            def tiles(V):
                return np.stack([V[:, z:z+side, y:y+side, x:x+side].reshape(len(V), -1)
                                 for z in offs for y in offs for x in offs], 1)
            Ttr, Tte = tiles(Xtr_all[tr_idx]), tiles(Xte)
            nb, dim = Ttr.shape[1], Ttr.shape[2]
            tiled = Ttr.reshape(-1, dim)
            kb = min(*tiled.shape) if FULL_RANK else min(N_COMP, *tiled.shape)
            sb = StandardScaler().fit(tiled)
            bp = PCA(n_components=kb, random_state=0).fit(sb.transform(tiled))

            def enc(T):
                return bp.transform(sb.transform(T.reshape(-1, dim))).reshape(len(T), nb * kb)
            Ztr = torch.tensor(np.hstack([enc(Ttr), se.transform(Etr)]), dtype=torch.float32)
            Zte = torch.tensor(np.hstack([enc(Tte), se.transform(Ete)]), dtype=torch.float32)
            GROUPS = [nb * kb, Etr.shape[1]]
        elif BASIS in ('pls', 'pls_energy'):
            from sklearn.cross_decomposition import PLSRegression
            nc = max(1, min(N_COMP or 16, len(tr_flat) - 2))
            sp = StandardScaler().fit(tr_flat)
            Y = np.eye(NCLS)[tr_y.numpy()] if NCLS > 2 else tr_y.numpy().astype(float)
            pls = PLSRegression(n_components=nc, scale=False).fit(sp.transform(tr_flat), Y)
            Ptr = pls.transform(sp.transform(tr_flat))
            Pte = pls.transform(sp.transform(te_flat))
            ss = StandardScaler().fit(Ptr)
            if BASIS == 'pls_energy':
                Etr, Ete = band_energy(Xtr_all[tr_idx]), band_energy(Xte)
                se = StandardScaler().fit(Etr)
                Ztr = torch.tensor(np.hstack([ss.transform(Ptr), se.transform(Etr)]),
                                   dtype=torch.float32)
                Zte = torch.tensor(np.hstack([ss.transform(Pte), se.transform(Ete)]),
                                   dtype=torch.float32)
                GROUPS = [Ptr.shape[1], Etr.shape[1]]
            else:
                Ztr = torch.tensor(ss.transform(Ptr), dtype=torch.float32)
                Zte = torch.tensor(ss.transform(Pte), dtype=torch.float32)
        elif BASIS in ('topo', 'topo_energy', 'topo_dct'):
            Ttr, Tte = topo_features(Xtr_all[tr_idx]), topo_features(Xte)
            st = StandardScaler().fit(Ttr)
            parts_tr, parts_te = [st.transform(Ttr)], [st.transform(Tte)]
            if BASIS == 'topo_energy':
                Etr, Ete = band_energy(Xtr_all[tr_idx]), band_energy(Xte)
                se2 = StandardScaler().fit(Etr)
                parts_tr.append(se2.transform(Etr)); parts_te.append(se2.transform(Ete))
            elif BASIS == 'topo_dct':
                perm = rank_perm()
                Ftr = block_spectra(Xtr_all[tr_idx])[:, perm]
                Fte = block_spectra(Xte)[:, perm]
                sd2 = StandardScaler().fit(Ftr)
                parts_tr.append(sd2.transform(Ftr)); parts_te.append(sd2.transform(Fte))
            Ztr = torch.tensor(np.hstack(parts_tr), dtype=torch.float32)
            Zte = torch.tensor(np.hstack(parts_te), dtype=torch.float32)
            GROUPS = [p.shape[1] for p in parts_tr] if len(parts_tr) > 1 else None
        elif BASIS in ('absfft', 'absfft_energy', 'absfft_pca'):
            Mtr, Mte = block_mag(Xtr_all[tr_idx]), block_mag(Xte)
            sm = StandardScaler().fit(Mtr)
            parts_tr, parts_te = [sm.transform(Mtr)], [sm.transform(Mte)]
            if BASIS == 'absfft_pca':
                At = np.abs(np.stack([fft.dctn(v, type=2, norm='ortho').ravel()
                                      for v in Xtr_all[tr_idx]]))
                Ae = np.abs(np.stack([fft.dctn(v, type=2, norm='ortho').ravel()
                                      for v in Xte]))
                sa = StandardScaler().fit(At)
                nc = max(1, min(N_COMP or 16, len(At) - 1))
                pa = PCA(n_components=nc, random_state=0).fit(sa.transform(At))
                parts_tr = [pa.transform(sa.transform(At))]
                parts_te = [pa.transform(sa.transform(Ae))]
            elif BASIS == 'absfft_energy':
                Etr, Ete = band_energy(Xtr_all[tr_idx]), band_energy(Xte)
                se3 = StandardScaler().fit(Etr)
                parts_tr.append(se3.transform(Etr)); parts_te.append(se3.transform(Ete))
            Ztr = torch.tensor(np.hstack(parts_tr), dtype=torch.float32)
            Zte = torch.tensor(np.hstack(parts_te), dtype=torch.float32)
            GROUPS = [p.shape[1] for p in parts_tr] if len(parts_tr) > 1 else None
        elif BASIS in ('magblockpca', 'magblockpca_energy'):
            g0, s0 = BLOCK, STRIDE
            offs, side = block_offsets(Xtr_all.shape[1])
            nbk = len(offs) ** 3

            def magtiles(V):
                out = np.zeros((len(V), nbk, FREQ ** 3), dtype=np.float32)
                for i in range(len(V)):
                    bi = 0
                    for z in offs:
                        for y in offs:
                            for x in offs:
                                m = np.abs(fft.fftn(V[i, z:z+side, y:y+side, x:x+side]))
                                out[i, bi] = np.log1p(m[:FREQ, :FREQ, :FREQ].ravel())
                                bi += 1
                return out
            Ttr, Tte = magtiles(Xtr_all[tr_idx]), magtiles(Xte)
            dim = Ttr.shape[2]
            pooled = Ttr.reshape(-1, dim)
            kk = min(N_COMP or min(*pooled.shape), *pooled.shape)
            sbm = StandardScaler().fit(pooled)
            pbm = PCA(n_components=kk, random_state=0).fit(sbm.transform(pooled))

            def encm(T):
                return pbm.transform(sbm.transform(T.reshape(-1, dim))).reshape(len(T), nbk * kk)
            parts_tr, parts_te = [encm(Ttr)], [encm(Tte)]
            if BASIS == 'magblockpca_energy':
                Etr, Ete = band_energy(Xtr_all[tr_idx]), band_energy(Xte)
                se4 = StandardScaler().fit(Etr)
                parts_tr.append(se4.transform(Etr)); parts_te.append(se4.transform(Ete))
            Ztr = torch.tensor(np.hstack(parts_tr), dtype=torch.float32)
            Zte = torch.tensor(np.hstack(parts_te), dtype=torch.float32)
            GROUPS = [p.shape[1] for p in parts_tr] if len(parts_tr) > 1 else None
        elif BASIS in ('moment', 'moment_energy', 'moment_dct'):
            Mo_tr, Mo_te = moment_features(Xtr_all[tr_idx]), moment_features(Xte)
            smo = StandardScaler().fit(Mo_tr)
            parts_tr = [np.clip(smo.transform(Mo_tr), -5, 5)]
            parts_te = [np.clip(smo.transform(Mo_te), -5, 5)]
            if BASIS == 'moment_energy':
                Etr, Ete = band_energy(Xtr_all[tr_idx]), band_energy(Xte)
                se5 = StandardScaler().fit(Etr)
                parts_tr.append(se5.transform(Etr)); parts_te.append(se5.transform(Ete))
            elif BASIS == 'moment_dct':
                perm = rank_perm()
                Ftr = block_spectra(Xtr_all[tr_idx])[:, perm]
                Fte = block_spectra(Xte)[:, perm]
                sd5 = StandardScaler().fit(Ftr)
                parts_tr.append(sd5.transform(Ftr)); parts_te.append(sd5.transform(Fte))
            Ztr = torch.tensor(np.hstack(parts_tr), dtype=torch.float32)
            Zte = torch.tensor(np.hstack(parts_te), dtype=torch.float32)
            GROUPS = [p.shape[1] for p in parts_tr] if len(parts_tr) > 1 else None
        elif BASIS in ('haar', 'haar_signed', 'haar_energy', 'haar_dct'):
            Ha_tr = haar_features(Xtr_all[tr_idx], signed=(BASIS == 'haar_signed'))
            Ha_te = haar_features(Xte, signed=(BASIS == 'haar_signed'))
            sha = StandardScaler().fit(Ha_tr)
            parts_tr = [np.clip(sha.transform(Ha_tr), -5, 5)]
            parts_te = [np.clip(sha.transform(Ha_te), -5, 5)]
            if BASIS == 'haar_energy':
                Etr, Ete = band_energy(Xtr_all[tr_idx]), band_energy(Xte)
                se6 = StandardScaler().fit(Etr)
                parts_tr.append(se6.transform(Etr)); parts_te.append(se6.transform(Ete))
            elif BASIS == 'haar_dct':
                perm = rank_perm()
                Ftr = block_spectra(Xtr_all[tr_idx])[:, perm]
                Fte = block_spectra(Xte)[:, perm]
                sd6 = StandardScaler().fit(Ftr)
                parts_tr.append(sd6.transform(Ftr)); parts_te.append(sd6.transform(Fte))
            Ztr = torch.tensor(np.hstack(parts_tr), dtype=torch.float32)
            Zte = torch.tensor(np.hstack(parts_te), dtype=torch.float32)
            GROUPS = [p.shape[1] for p in parts_tr] if len(parts_tr) > 1 else None
        elif BASIS == 'patchdct':
            Ztr = torch.tensor(patch_dct(Xtr_all[tr_idx]), dtype=torch.float32)
            Zte = torch.tensor(patch_dct(Xte), dtype=torch.float32)
            GROUPS = None
        elif BASIS == 'dct':
            perm = rank_perm()
            Ftr = block_spectra(Xtr_all[tr_idx])[:, perm]
            Fte = block_spectra(Xte)[:, perm]
            sc = StandardScaler().fit(Ftr)
            Ztr = torch.tensor(sc.transform(Ftr), dtype=torch.float32)
            Zte = torch.tensor(sc.transform(Fte), dtype=torch.float32)
        else:
            sc = StandardScaler().fit(tr_flat)
            nc = len(tr_flat) - 1 if FULL_RANK else min(N_COMP, len(tr_flat))
            pca = PCA(n_components=max(1, nc)).fit(sc.transform(tr_flat))
            Ztr = torch.tensor(pca.transform(sc.transform(tr_flat)), dtype=torch.float32)
            Zte = torch.tensor(pca.transform(sc.transform(te_flat)), dtype=torch.float32)

        rk = os.environ.get('MEDMNIST_RANK') == '1'
        groups = GROUPS if (rk and (BASIS == 'concat' or BASIS.startswith('pool') or GROUPS)) else None
        if BASIS == 'patchdct':
            spec = MultiClassSpectralViT(
                n_inputs=Ztr.shape[1], chunk_size=int(os.environ.get('MEDMNIST_PK', PATCH ** 3)),
                num_classes=NCLS, use_mode_weights=False).to(dev)
        else:
            spec = MultiClassSpectralViT(n_inputs=Ztr.shape[1], num_classes=NCLS,
                                         use_mode_weights=True, use_rank_weights=rk,
                                         rank_groups=groups,
                                         use_pos_embed=_POS).to(dev)
        torch.manual_seed(seed)
        spat = MultiClassSpatialViT(num_classes=NCLS, size=VOL, patch_size=PATCH,
                                    in_channels=1, volumetric=True).to(dev)
        def train(model, tensors, sd):
            torch.manual_seed(sd)
            o = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
            ld = DataLoader(TensorDataset(*tensors, tr_y), batch_size=BATCH, shuffle=True,
                            generator=torch.Generator().manual_seed(seed))
            model.train()
            for _ in range(EPOCHS):
                for bx, by in ld:
                    o.zero_grad()
                    F.cross_entropy(model(bx.to(dev)), by.to(dev)).backward()
                    o.step()

        _ps = '' if PATCH == 4 else f'_p{PATCH}'
        ck = os.path.join(SPAT_CACHE,
                          f'{os.path.basename(NPZ)[:-4]}_n{int(take.sum())}_s{seed}{_ps}.npy')
        if os.path.exists(ck) and not OCCLUDE:
            pp_arr = np.load(ck)
        else:
            train(spat, (tr_imgs,), seed)
            spat.eval()
            with torch.no_grad():
                pp_arr = torch.cat([F.softmax(spat(c.to(dev)), 1).cpu()
                                    for c in torch.split(te_imgs, 256)]).numpy()
            os.makedirs(SPAT_CACHE, exist_ok=True)
            np.save(ck, pp_arr)
        train(spec, (Ztr,), seed + 1_000_000)

        spec.eval()
        ps = []
        with torch.no_grad():
            for bz in torch.split(Zte, 256):
                ps.append(F.softmax(spec(bz.to(dev)), 1).cpu())
        pp = [torch.from_numpy(pp_arr)]
        if OCCLUDE:
            np.savez_compressed(os.path.join(OUT, 'occlusion_probs.npz'),
                                spec=torch.cat(ps).numpy(), spat=torch.cat(pp).numpy(),
                                labels=yte, comp=int(Ztr.shape[1]), patch=PATCH)
            print(f'  wrote {OUT}/occlusion_probs.npz '
                  f'({len(yte)} volumes, {Ztr.shape[1]} features)', flush=True)
            sys.exit(0)
        ms = metrics(torch.cat(ps).numpy(), yte, NCLS)
        mp = metrics(torch.cat(pp).numpy(), yte, NCLS)
        rows.append(dict(n=int(take.sum()), requested=N, seed=seed, ncls=NCLS,
                         comp=int(Ztr.shape[1]),
                         np_spec=sum(q.numel() for q in spec.parameters()),
                         np_spat=sum(q.numel() for q in spat.parameters()),
                         **{f'spec_{k}': v for k, v in ms.items()},
                         **{f'spat_{k}': v for k, v in mp.items()}))
        print(f'  n={int(take.sum()):4d} seed {seed}: spec T1 {ms["T1"]:.3f} '
              f'spat T1 {mp["T1"]:.3f} | AUC {ms["AUC"]:.3f}/{mp["AUC"]:.3f} '
              f'({time.time()-t0:.0f}s)', flush=True)
        del spec, spat
        if dev.type == 'cuda':
            torch.cuda.empty_cache()
        tmp = os.path.join(OUT, 'medmnist_perseed.csv.tmp')
        pd.DataFrame(rows).to_csv(tmp, index=False)
        os.replace(tmp, os.path.join(OUT, 'medmnist_perseed.csv'))

print(f'\nwrote {OUT}/medmnist_perseed.csv ({len(rows)} cells)', flush=True)
