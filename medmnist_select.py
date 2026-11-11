"""MedMNIST3D with the basis chosen by inner cross-validation on the training split.

The sweep in medmnist_sweep.py fixes the tokenizer in advance, and which one is fixed
decides the answer: on OrganMNIST3D a fold-wise PCA leads at n=25 while the fixed block
cosine transform loses at every size, and on NoduleMNIST3D the ordering reverses. Picking
per dataset after seeing test scores would be selection on the test set, so the basis is
instead chosen the way capacity already is in the volumetric experiments -- by inner
cross-validation inside the training subsample, with the fixed test set untouched.

Both arms get a grid, or the comparison stops being matched: the spectral arm selects its
basis, the spatial arm its patch size, on the identical inner folds and the identical
metric. Selection uses the one-standard-error rule over arms ordered by token count, so
the cheapest setting statistically indistinguishable from the best is the one that runs,
which is the rule the volumetric capacity selection already uses.

env: MEDMNIST_NPZ MEDMNIST_SEEDS MEDMNIST_N MEDMNIST_OUT MEDMNIST_INNER
usage: python medmnist_select.py
"""
import os, time
import numpy as np, pandas as pd, torch, torch.optim as optim
import torch.nn.functional as F
from scipy import fft
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from networks import MultiClassSpectralViT, MultiClassSpatialViT

NPZ = os.environ.get('MEDMNIST_NPZ', '/agent-home/medmnist/organmnist3d.npz')
OUT = os.environ.get('MEDMNIST_OUT', 'results/medmnist_select')
N_SEEDS = int(os.environ.get('MEDMNIST_SEEDS', 5))
INNER_K = int(os.environ.get('MEDMNIST_INNER', 5))
EPOCHS, BATCH, LR, WD = 100, 128, 1e-3, 0.01
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(OUT, exist_ok=True)

# Ordered by token count so the one-standard-error rule breaks ties toward the cheaper
# arm. 28 = 2^2 * 7, so a block grid must divide into {1,2,4,7,14,28}.
SPEC_GRID = [('pca16', ('pca', 16)), ('pca64', ('pca', 64)),
             ('dct216', ('dct', 2, 3)), ('dct512', ('dct', 4, 2))]
# Four points each, so neither arm gets a larger tuning budget. Patch sizes must divide
# 28; patch 28 is dropped as degenerate -- a single token, and a 21,952 to 128 embedding
# that triples the weight count -- in favour of patch 2.
SPAT_GRID = [('patch14', 14), ('patch7', 7), ('patch4', 4), ('patch2', 2)]


def block_spectra(vols, grid, freq):
    """Per-block low-frequency cosine coefficients, sign kept and log compressed."""
    B = len(vols)
    bz, by, bx = [t // grid for t in vols.shape[1:]]
    out = np.zeros((B, grid ** 3, freq ** 3), dtype=np.float32)
    for i in range(B):
        bi = 0
        for z in range(grid):
            for y in range(grid):
                for x in range(grid):
                    blk = vols[i, z*bz:(z+1)*bz, y*by:(y+1)*by, x*bx:(x+1)*bx]
                    c = fft.dctn(blk, type=2, norm='ortho')[:freq, :freq, :freq].ravel()
                    out[i, bi] = np.sign(c) * np.log1p(np.abs(c))
                    bi += 1
    return out.reshape(B, grid ** 3 * freq ** 3)


def rank_perm(grid, freq):
    """Block-major to frequency-major so index order is variance order for 1/i weights."""
    nb, nf = grid ** 3, freq ** 3
    kz, ky, kx = np.unravel_index(np.arange(nf), (freq, freq, freq))
    order = np.argsort(kz ** 2 + ky ** 2 + kx ** 2, kind='stable')
    return np.array([b * nf + order[r] for r in range(nf) for b in range(nb)])


def features(spec, Xfit, Xapply):
    """Build spectral features, fitting anything data-dependent on Xfit alone."""
    if spec[0] == 'pca':
        k = spec[1]
        fit_flat = Xfit.reshape(len(Xfit), -1)
        sc = StandardScaler().fit(fit_flat)
        pca = PCA(n_components=min(k, len(fit_flat))).fit(sc.transform(fit_flat))
        return [torch.tensor(pca.transform(sc.transform(x.reshape(len(x), -1))),
                             dtype=torch.float32) for x in Xapply]
    _, grid, freq = spec
    perm = rank_perm(grid, freq)
    Ffit = block_spectra(Xfit, grid, freq)[:, perm]
    sc = StandardScaler().fit(Ffit)
    return [torch.tensor(sc.transform(block_spectra(x, grid, freq)[:, perm]),
                         dtype=torch.float32) for x in Xapply]


def score(prob, y, ncls):
    return (roc_auc_score(y, prob[:, 1]) if ncls == 2
            else accuracy_score(y, prob.argmax(1)))


def train_eval(make, Ztr, ytr, Zev, ncls):
    model = make().to(dev)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    ld = DataLoader(TensorDataset(Ztr, torch.from_numpy(ytr)), batch_size=BATCH,
                    shuffle=True)
    model.train()
    for _ in range(EPOCHS):
        for bz, by in ld:
            opt.zero_grad()
            F.cross_entropy(model(bz.to(dev)), by.to(dev)).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        p = torch.cat([F.softmax(model(c.to(dev)), 1).cpu()
                       for c in torch.split(Zev, 256)]).numpy()
    return p


def choose(names, mat):
    """Both selection rules from the same inner scores, to separate their effects.

    The one-standard-error rule takes the cheapest arm within one SE of the best. On
    HCP-Aging that rule chose the smallest coefficient count in 59% of folds at n=25
    against 16% for the reported setting: a tiny inner split makes the SE wide enough
    that nearly every arm qualifies, and the rule then falls through to the cheapest.
    Logging the plain argmax beside it separates how much of a choice is signal from how
    much is that pull toward parsimony.

    mat is (arms x folds), higher is better; grids are ordered cheapest first.
    """
    mu = mat.mean(1)
    best = int(np.argmax(mu))
    se = mat[best].std(ddof=1) / np.sqrt(mat.shape[1]) if mat.shape[1] > 1 else 0.0
    ok = np.where(mu >= mu[best] - se)[0]
    return int(ok[0]), names[int(ok[0])], best, names[best]


z = np.load(NPZ)
Xtr_all = z['train_images'].astype(np.float32)
ytr_all = z['train_labels'].ravel().astype(np.int64)
Xte = z['test_images'].astype(np.float32)
yte = z['test_labels'].ravel().astype(np.int64)
NCLS, VOL = int(ytr_all.max()) + 1, Xtr_all.shape[1]
mu_, sd_ = Xtr_all.mean(), Xtr_all.std()
Xtr_all, Xte = (Xtr_all - mu_) / sd_, (Xte - mu_) / sd_
pool = {c: np.where(ytr_all == c)[0] for c in range(NCLS)}
avail = np.array([len(pool[c]) for c in range(NCLS)])
SIZES = [int(v) for v in os.environ.get(
    'MEDMNIST_N', '25,50,100,200,500,%d' % avail.sum()).split(',')]
print(f'{os.path.basename(NPZ)}: {len(Xtr_all)} train / {len(Xte)} test, {VOL}^3, '
      f'{NCLS} classes on {dev}', flush=True)

rows, t0 = [], time.time()
for N in SIZES:
    if N > avail.sum():
        continue
    for seed in range(N_SEEDS):
        torch.manual_seed(seed); np.random.seed(seed)
        take = np.minimum(avail, N // NCLS)
        while take.sum() < N:
            room = np.where(avail - take > 0)[0]
            pick = np.random.choice(room, min(N - take.sum(), len(room)), replace=False)
            take[pick] += 1
        idx = np.concatenate([np.random.choice(pool[c], take[c], replace=False)
                              for c in range(NCLS) if take[c]])
        Xs, ys = Xtr_all[idx], ytr_all[idx]

        # Inner folds come from the training subsample only; the MedMNIST test split is
        # the reporting set and is never seen during selection.
        counts = np.bincount(ys, minlength=NCLS)
        k = min(INNER_K, int(counts[counts > 0].min()))
        splitter = (StratifiedKFold(k, shuffle=True, random_state=seed) if k >= 2
                    else KFold(min(INNER_K, len(ys)), shuffle=True, random_state=seed))
        folds = list(splitter.split(Xs, ys) if k >= 2 else splitter.split(Xs))

        sm = np.zeros((len(SPEC_GRID), len(folds)))
        pm = np.zeros((len(SPAT_GRID), len(folds)))
        for fi, (itr, iva) in enumerate(folds):
            for ai, (_, spec) in enumerate(SPEC_GRID):
                Ztr, Zva = features(spec, Xs[itr], [Xs[itr], Xs[iva]])
                p = train_eval(lambda d=Ztr.shape[1]: MultiClassSpectralViT(
                    n_inputs=d, num_classes=NCLS, use_mode_weights=True),
                    Ztr, ys[itr], Zva, NCLS)
                sm[ai, fi] = score(p, ys[iva], NCLS)
            for ai, (_, patch) in enumerate(SPAT_GRID):
                Vtr = torch.from_numpy(Xs[itr]).unsqueeze(1)
                Vva = torch.from_numpy(Xs[iva]).unsqueeze(1)
                p = train_eval(lambda q=patch: MultiClassSpatialViT(
                    num_classes=NCLS, size=VOL, patch_size=q, in_channels=1,
                    volumetric=True), Vtr, ys[itr], Vva, NCLS)
                pm[ai, fi] = score(p, ys[iva], NCLS)
        si, sname, sa, saname = choose([n for n, _ in SPEC_GRID], sm)
        pi, pname, pa, paname = choose([n for n, _ in SPAT_GRID], pm)

        # Refit the selected arms on the whole subsample, then score the fixed test set.
        Ztr, Zte = features(SPEC_GRID[si][1], Xs, [Xs, Xte])
        ps = train_eval(lambda d=Ztr.shape[1]: MultiClassSpectralViT(
            n_inputs=d, num_classes=NCLS, use_mode_weights=True), Ztr, ys, Zte, NCLS)
        pp = train_eval(lambda q=SPAT_GRID[pi][1]: MultiClassSpatialViT(
            num_classes=NCLS, size=VOL, patch_size=q, in_channels=1, volumetric=True),
            torch.from_numpy(Xs).unsqueeze(1), ys, torch.from_numpy(Xte).unsqueeze(1),
            NCLS)
        a, b = score(ps, yte, NCLS), score(pp, yte, NCLS)

        # Where argmax disagrees with the one-SE pick, score that arm too, so the cost of
        # the parsimony rule is measured rather than inferred. Agreement costs nothing.
        if sa != si:
            Za, Zea = features(SPEC_GRID[sa][1], Xs, [Xs, Xte])
            aa = score(train_eval(lambda d=Za.shape[1]: MultiClassSpectralViT(
                n_inputs=d, num_classes=NCLS, use_mode_weights=True),
                Za, ys, Zea, NCLS), yte, NCLS)
        else:
            aa = a
        if pa != pi:
            ba = score(train_eval(lambda q=SPAT_GRID[pa][1]: MultiClassSpatialViT(
                num_classes=NCLS, size=VOL, patch_size=q, in_channels=1,
                volumetric=True), torch.from_numpy(Xs).unsqueeze(1), ys,
                torch.from_numpy(Xte).unsqueeze(1), NCLS), yte, NCLS)
        else:
            ba = b
        rows.append(dict(n=int(take.sum()), seed=seed, ncls=NCLS,
                         spec_basis=sname, spat_patch=pname,
                         spec_basis_argmax=saname, spat_patch_argmax=paname,
                         spec_score=a, spat_score=b,
                         spec_score_argmax=aa, spat_score_argmax=ba,
                         spec_inner=sm[si].mean(), spat_inner=pm[pi].mean()))
        print(f'  n={N:5d} s{seed}: 1SE {sname:7s} {a:.3f} / amax {saname:7s} {aa:.3f} '
              f'|| {pname:8s} {b:.3f} / {paname:8s} {ba:.3f} '
              f'| d {a-b:+.3f} ({time.time()-t0:.0f}s)', flush=True)
        tmp = os.path.join(OUT, 'medmnist_select.csv.tmp')
        pd.DataFrame(rows).to_csv(tmp, index=False)
        os.replace(tmp, os.path.join(OUT, 'medmnist_select.csv'))
print(f'\nwrote {OUT}/medmnist_select.csv ({len(rows)} cells)', flush=True)
