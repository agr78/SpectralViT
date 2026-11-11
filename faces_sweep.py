"""LFW identity classification by training-set size: the aligned natural-image test.

CIFAR showed the spectral advantage does not survive on unregistered natural images, and
the diagnostic reason was spatial correspondence: PCA on flattened pixels is only
meaningful if coordinate k refers to the same thing in every image. Brain volumes satisfy
that through registration; photographs of objects at arbitrary pose do not.

Aligned faces are the case that separates "natural images" from "no spatial
correspondence", and they are the oldest precedent for it -- eigenfaces is the canonical
demonstration that a global linear basis works on aligned photographs. If the advantage is
about correspondence rather than about medical imaging, it should appear here.

LFW 'funneled' images are aligned. Ten images per identity are held out as a fixed test set
(every identity has at least twenty), so the test set is balanced and seed independent, and
training draws come from the remainder.

The spectral arm defaults to the component-wise embedding of sec:ca (use_mode_weights),
which on MNIST is worth roughly twice what the chunked projection gives; FACES_MODE=0 falls
back to chunks for comparison. A parameter-matched MLP on the identical components is
carried as a control, since on CIFAR it beat both transformers.

env: FACES_COMP FACES_SEEDS FACES_PER_CLASS FACES_MODE FACES_RANK FACES_OUT FACES_HOME
usage: python faces_sweep.py [--smoke]
"""
import os, sys, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from scipy import fft
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, top_k_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from networks import MultiClassSpectralViT, MultiClassSpatialViT

# Token-identity switches; defaults reproduce every result already on disk.
_POS = os.environ.get('FACES_POS', '1') != '0'  # FACES_RANK already exists below



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
HOME = os.environ.get('FACES_HOME', '/agent-home/sk_data')
N_SEEDS = 2 if SMOKE else int(os.environ.get('FACES_SEEDS', 5))
N_COMP = int(os.environ.get('FACES_COMP', 16))
PER_CLASS = [1, 2] if SMOKE else [int(v) for v in
                                  os.environ.get('FACES_PER_CLASS', '1,2,3,5,10').split(',')]
EPOCHS, BATCH, LR, WD = (2 if SMOKE else 100), 128, 1e-3, 0.01
SIZE, PATCH, TEST_PER_CLASS = 48, 4, 10
OVERLAP = os.environ.get('FACES_OVERLAP') == '1'   # 4x4 DCT at stride 2, position tokens
SPEC_KW = {}
if os.environ.get('FACES_MODE', '1') == '1': SPEC_KW['use_mode_weights'] = True
if os.environ.get('FACES_RANK', '0') == '1': SPEC_KW['use_rank_weights'] = True
# The overlap tokenizer depends on a projection SHARED across position tokens; mode
# weights would instead give one token per coefficient, which is the opposite design
# and would make thousands of tokens. They are mutually exclusive.
if OVERLAP:
    SPEC_KW.pop('use_mode_weights', None)
OUT = os.environ.get('FACES_OUT', os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                               'results', 'faces'))
if N_COMP % 4:
    sys.exit(f'FACES_COMP must be a multiple of 4 (chunk_size); got {N_COMP}')
os.makedirs(OUT, exist_ok=True)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    b = input_dim + 2 + num_classes
    return max(2, int((-b + np.sqrt(b * b - 4 * (num_classes - target))) / 2))


def eigenfaces(Ztr, ytr, Zte, ncls):
    """Eigenfaces: nearest neighbours in the PCA subspace the spectral arm already uses.

    This is Turk and Pentland's method, not a new baseline -- the spectral arm on LFW is
    PCA of the faces, so 1-NN on those same coefficients IS eigenfaces, and the margin over
    it is what separates the encoder from a representation that predates it by decades.

    k is chosen by inner stratified CV over the training subsample alone. At one image per
    identity no split leaves a neighbour of the right class, so k falls back to 1, which is
    the classical rule anyway. predict_proba on k neighbours is coarse (steps of 1/k), so
    top-1 is the metric to read here and AUC is reported only for completeness.
    """
    Xtr, Xte = Ztr.numpy(), Zte.numpy()
    counts = np.bincount(ytr, minlength=ncls)
    nmin = counts[counts > 0].min()
    cands = [k for k in (1, 3, 5) if k <= max(1, nmin - 1)] or [1]
    if len(cands) > 1 and nmin >= 2:
        folds = StratifiedKFold(n_splits=min(3, nmin), shuffle=True, random_state=0)
        best, bk = -1.0, cands[0]
        for k in cands:
            acc = []
            for itr, iva in folds.split(Xtr, ytr):
                m = KNeighborsClassifier(n_neighbors=k).fit(Xtr[itr], ytr[itr])
                acc.append(accuracy_score(ytr[iva], m.predict(Xtr[iva])))
            if np.mean(acc) > best:
                best, bk = float(np.mean(acc)), k
    else:
        bk = cands[0]
    m = KNeighborsClassifier(n_neighbors=bk).fit(Xtr, ytr)
    pr = np.zeros((len(Xte), ncls))
    pr[:, m.classes_] = m.predict_proba(Xte)
    return pr, bk


def metrics(probs, y_true, ncls):
    lab = np.arange(ncls)
    return {'T1': accuracy_score(y_true, np.argmax(probs, axis=1)),
            'T5': top_k_accuracy_score(y_true, probs, k=5, labels=lab),
            'AUC': roc_auc_score(y_true, probs, multi_class='ovr', labels=lab)}


# ---------------------------------------------------------------- data
d = fetch_lfw_people(min_faces_per_person=20, resize=0.5, funneled=True, data_home=HOME)
imgs, y = d.images.astype(np.float32), d.target
NCLS = len(d.target_names)
# Square center crop to the narrower axis, then one resize, so every image is SIZE^2 and
# the spatial arm's patch grid divides evenly.
h, w = imgs.shape[1:]
s = min(h, w)
imgs = imgs[:, (h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
X = F.interpolate(torch.from_numpy(imgs).unsqueeze(1), size=SIZE, mode='bilinear',
                  align_corners=False)
X = (X - X.mean()) / X.std()
print(f'LFW {len(X)} aligned faces, {NCLS} identities, {SIZE}^2 on {dev}; '
      f'{N_COMP} comp, spec {SPEC_KW or "chunked"}', flush=True)

# A fixed, balanced, seed-independent test set: the first TEST_PER_CLASS of each identity.
te_idx, pool = [], {}
for c in range(NCLS):
    ix = np.where(y == c)[0]
    te_idx.extend(ix[:TEST_PER_CLASS]); pool[c] = ix[TEST_PER_CLASS:]
te_idx = np.array(te_idx)
te_imgs, te_y = X[te_idx], y[te_idx]
te_flat = te_imgs.view(len(te_imgs), -1).numpy()
print(f'test set {len(te_idx)} images ({TEST_PER_CLASS}/identity); '
      f'train pool min {min(len(v) for v in pool.values())}/identity', flush=True)

rows, t0 = [], time.time()
for per_class in PER_CLASS:
    for seed in range(N_SEEDS):
        torch.manual_seed(seed); np.random.seed(seed)
        tr_idx = np.concatenate([np.random.choice(pool[c], min(per_class, len(pool[c])),
                                                  replace=False) for c in range(NCLS)])
        tr_imgs, tr_y = X[tr_idx], torch.from_numpy(y[tr_idx]).long()
        tr_flat = tr_imgs.view(len(tr_imgs), -1).numpy()

        if OVERLAP:
            Ftr, Fte = overlap_tokens(tr_imgs, 4, 2), overlap_tokens(te_imgs, 4, 2)
            m, sv = Ftr.mean(0, keepdims=True), Ftr.std(0, keepdims=True) + 1e-6
            Ztr = torch.tensor((Ftr - m) / sv, dtype=torch.float32)
            Zte = torch.tensor((Fte - m) / sv, dtype=torch.float32)
        else:
            sc = StandardScaler().fit(tr_flat)
            pca = PCA(n_components=min(N_COMP, len(tr_flat))).fit(sc.transform(tr_flat))
            Ztr = torch.tensor(pca.transform(sc.transform(tr_flat)), dtype=torch.float32)
            Zte = torch.tensor(pca.transform(sc.transform(te_flat)), dtype=torch.float32)

        spec = MultiClassSpectralViT(n_inputs=Ztr.shape[1], num_classes=NCLS,
                                     chunk_size=(16 if OVERLAP else 4),
                                     use_pos_embed=_POS,
                                     **SPEC_KW).to(dev)
        spat = MultiClassSpatialViT(num_classes=NCLS, size=SIZE, patch_size=PATCH,
                                    in_channels=1).to(dev)
        np_spec = sum(p.numel() for p in spec.parameters())
        mlp = FeatureMLP(Ztr.shape[1], match_hidden(Ztr.shape[1], NCLS, np_spec),
                         NCLS).to(dev)
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
        ms = metrics(torch.cat(ps).numpy(), te_y, NCLS)
        mp = metrics(torch.cat(pp).numpy(), te_y, NCLS)
        mm = metrics(torch.cat(pm).numpy(), te_y, NCLS)
        pk, knn_k = eigenfaces(Ztr, tr_y.numpy() if hasattr(tr_y, 'numpy') else tr_y,
                               Zte, NCLS)
        mk = metrics(pk, te_y, NCLS)
        rows.append(dict(per_class=per_class, n=len(tr_idx), seed=seed, ncls=NCLS,
                         comp=int(Ztr.shape[1]), mode=int('use_mode_weights' in SPEC_KW),
                         rank=int('use_rank_weights' in SPEC_KW), np_spec=np_spec,
                         np_spat=sum(p.numel() for p in spat.parameters()),
                         np_mlp=sum(p.numel() for p in mlp.parameters()),
                         **{f'spec_{k}': v for k, v in ms.items()},
                         **{f'spat_{k}': v for k, v in mp.items()},
                         **{f'mlp_{k}': v for k, v in mm.items()},
                         knn_k=knn_k,
                         **{f'knn_{k}': v for k, v in mk.items()}))
        print(f'  {per_class:3d}/id n={len(tr_idx):4d} seed {seed}: '
              f'spec T1 {ms["T1"]:.3f} spat T1 {mp["T1"]:.3f} mlp T1 {mm["T1"]:.3f} '
              f'knn T1 {mk["T1"]:.3f} (k={knn_k}) | '
              f'AUC {ms["AUC"]:.3f}/{mp["AUC"]:.3f}/{mm["AUC"]:.3f} '
              f'({time.time()-t0:.0f}s)', flush=True)
        del spec, spat, mlp
        if dev.type == 'cuda':
            torch.cuda.empty_cache()
        tmp = os.path.join(OUT, 'faces_perseed.csv.tmp')
        pd.DataFrame(rows).to_csv(tmp, index=False)
        os.replace(tmp, os.path.join(OUT, 'faces_perseed.csv'))

print(f'\nwrote {OUT}/faces_perseed.csv ({len(rows)} cells)', flush=True)
