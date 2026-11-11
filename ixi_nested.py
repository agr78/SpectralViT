"""IXI sex classification with properly nested capacity selection.

The previous pipeline chose the component count and patch size with a 5-fold CV over the
whole dataset, then reported on the same KFold(random_state=0) partition -- so the
hyperparameter was selected on the reporting folds (model-selection bias; Varma & Simon
2006, Cawley & Talbot 2010). Here selection runs inside each outer fold on its training
split only, so no test subject influences any choice.

PCA components are nested: the first k columns of a 128-component whitened transform are
exactly the k-component whitened transform, so one PCA fit per inner fold serves every
candidate. That turns 60 PCA fits into 15.

usage: python ixi_nested.py [--smoke]
"""
import os, sys, json, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import OneCycleLR

from util import seed_everything
from networks import (SpectralViT, SpatialViT, AttentionUNet, SwinTransformer,
                      LogisticRegression, MultiLayerPerceptron)
from loader import load_ixi_data

SMOKE   = '--smoke' in sys.argv
IXI_DIR = os.environ.get('IXI_DIR', '/checkpoints/data/IXI_extracted/')
OUT     = os.environ.get('IXI_OUT', '/workspace/results/ixi_nested')
VOL, LR, BATCH, NF, PHYS, EFF = 96, 1e-4, 8, 5, 2, 8
ACC     = EFF // PHYS
EPOCHS  = 3 if SMOKE else int(os.environ.get('IXI_EPOCHS', 500))
SEL_EP  = 3 if SMOKE else int(os.environ.get('IXI_SEL_EPOCHS', 200))   # convergent, cheaper
INNER_K = 2 if SMOKE else int(os.environ.get('IXI_INNER_K', 3))
NCOMPS  = [16, 32] if SMOKE else [16, 32, 64, 128]
PATCHES = [8, 12] if SMOKE else [8, 12, 16]
# The U-net and Swin had no grid on IXI: Swin ran at library defaults and the U-net at
# base_channels=7, picked to match a published parameter count. Selecting four arms and
# hand-picking the strongest baseline is the asymmetry that invalidated the DBS table.
CHANS   = [4, 8] if SMOKE else [4, 8, 16]
SWEMB   = [16, 32] if SMOKE else [16, 32, 64]
# Simple spectral-feature baselines on the same per-fold PCA components, which the
# reviews asked for: without them the transformer's contribution cannot be separated
# from the representation's. Grids mirror hca_regress.py so the two experiments compare
# the same models. WD descends so a tie resolves to the more regularized fit.
WD      = [0.5, 0.05] if SMOKE else [0.5, 0.05, 0.005]
HID     = [2, 6] if SMOKE else [2, 6, 16]
# 'match' (the compact spatial ViT) is no longer reported: Swin is the better
# parameter-matched spatial baseline at 11k parameters against the compact ViT's 28k.
# The arm is still implemented and runs with IXI_ARMS=match.
ARMS_ON = [a for a in os.environ.get(
    'IXI_ARMS', 'spec,spat,swin,unet,logreg,mlp').split(',') if a]
TAGSUF = os.environ.get('IXI_SUFFIX', '')
# OUT lives on spacelink, which is not readable from everywhere the results are needed.
# Mirror each cell into the repo's results/ as hca_sweep.py and hca_regress.py do, so the
# curve can be read without reaching for the scratch filesystem.
MIRROR = os.environ.get('IXI_MIRROR', os.path.expanduser('~/SpectralViT/results/ixi_curve'))
os.makedirs(OUT, exist_ok=True)
if MIRROR:
    try:
        os.makedirs(MIRROR, exist_ok=True)
    except OSError:
        MIRROR = ''


def save_npz(name, **arrays):
    """Write a cell to OUT, then fail-soft to the mirror.

    The mirror sits on a quota-limited filesystem; a failure there once truncated a run's
    outputs, so it must never raise and never affect what lands in OUT.
    """
    np.savez(os.path.join(OUT, name), **arrays)
    if MIRROR:
        try:
            np.savez(os.path.join(MIRROR, name), **arrays)
        except OSError:
            pass
seed_everything(0)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# IXI_CACHE points at the prefix ixi_cache.py wrote. /checkpoints is a local disk on the
# container host, so a host without it (durian, the cluster) reads the cached array instead.
# The cache is saved straight from load_ixi_data, so both paths see identical inputs.
CACHE = os.environ.get('IXI_CACHE', '')
if CACHE and os.path.exists(CACHE + '.npy'):
    X = np.load(CACHE + '.npy')
    Y = np.load(CACHE + '_meta.npz')['y']
    assert X.ndim == 4 and X.shape[1:] == (VOL, VOL, VOL), (X.shape, VOL)
    assert len(X) == len(Y), (len(X), len(Y))
    print(f'cache {CACHE}.npy: {X.shape} {X.dtype}')
else:
    X, Y = load_ixi_data(IXI_DIR, os.path.join(IXI_DIR, 'IXI.csv'), VOL)
if SMOKE: X, Y = X[:40], Y[:40]
SUB = int(os.environ.get('IXI_N', 0))          # 0 = use every subject
SEED = int(os.environ.get('IXI_SEED', 0))      # subsample draw, varies the reported spread
if SUB and SUB < len(Y):
    # Stratified on sex so the class balance of the full cohort is preserved at every size.
    rng = np.random.RandomState(SEED)
    keep = []
    for c in np.unique(Y):
        pool = np.where(Y == c)[0]
        keep += list(rng.choice(pool, size=int(round(SUB * len(pool) / len(Y))), replace=False))
    keep = np.array(sorted(keep))[:SUB]
    X, Y = X[keep], Y[keep]
    print(f'subsampled to n={len(Y)} (seed {SEED}), labels {np.bincount(Y.astype(int))}',
          flush=True)
Xf = X.reshape(len(X), -1)
print(f'IXI: {len(X)} volumes {X.shape[1:]}, labels {np.bincount(Y.astype(int))}', flush=True)
def folds(n_splits, labels, seed=0):
    """Stratified folds, falling back to plain KFold if a class is too small.

    The original pipeline used unstratified KFold, which is harmless at n=566 (folds of
    113) but not on the subsampled curve: at n=25 a test fold holds 5 subjects and an
    inner validation fold about 7, so a single-class fold is likely and roc_auc_score
    would raise.
    """
    # IXI_STRATIFY=0 reproduces the unstratified partition the full-cohort table was
    # run on, so arms added later can be scored on the same folds as the ones already
    # reported rather than on a partition that merely resembles them.
    if (os.environ.get('IXI_STRATIFY', '1') != '0'
            and np.bincount(labels.astype(int), minlength=2).min() >= n_splits):
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed)


def auc_safe(y, p):
    """AUC, or nan when a fold is single-class; one_se already ignores nan."""
    y = np.asarray(y)
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else float('nan')


NF_ = 2 if SMOKE else NF
kf = folds(NF_, Y, seed=SEED)


def fit_train(model, xtr, ytr, epochs, bs, acc, wd=None):
    ld = DataLoader(TensorDataset(xtr, torch.from_numpy(ytr).float()),
                    batch_size=bs, shuffle=True)
    # wd=None keeps AdamW's default, so the arms that predate this are unchanged.
    opt = (optim.AdamW(model.parameters(), lr=LR) if wd is None
           else optim.AdamW(model.parameters(), lr=LR, weight_decay=wd))
    sch = OneCycleLR(opt, max_lr=LR, steps_per_epoch=max(1, len(ld) // acc), epochs=epochs)
    crit = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        model.train()
        for i, (bx, by) in enumerate(ld):
            (crit(model(bx.to(dev)), by.to(dev)) / acc).backward()
            if (i + 1) % acc == 0:
                opt.step(); sch.step(); opt.zero_grad()
    model.eval(); return model


def predict(model, xte, bs):
    with torch.no_grad():
        if xte.ndim < 5: return torch.sigmoid(model(xte.to(dev))).cpu().numpy()
        return torch.cat([torch.sigmoid(model(c.to(dev)))
                          for c in torch.split(xte, bs)]).cpu().numpy()


def one_se(scores, cands):
    """Smallest candidate whose mean is within one SE of the best (higher is better)."""
    means = {c: np.nanmean(v) for c, v in scores.items()}
    if all(m != m for m in means.values()): return cands[0]
    best = max(means, key=lambda c: means[c])
    se = np.nanstd(scores[best], ddof=1) / np.sqrt(len(scores[best])) if len(scores[best]) > 1 else 0.0
    thresh = means[best] - (se if se == se else 0.0)
    for c in cands:
        if means[c] == means[c] and means[c] >= thresh: return c
    return best


def select_spectral(tr):
    """Inner CV over component counts, one PCA fit per inner fold (components nest)."""
    inner = folds(INNER_K, Y[tr])
    # PCA rank is bounded by the inner-training sample count, so restrict the grid once.
    kmin = int(len(tr) * (INNER_K - 1) / INNER_K) - 1
    cands = [c for c in NCOMPS if c <= kmin] or [max(2, kmin)]
    scores = {c: [] for c in cands}
    for i_i, v_i in inner.split(tr, Y[tr]):
        itr, iva = tr[i_i], tr[v_i]
        # PCA cannot exceed the inner-training sample count; drop candidates above it.
        pca = PCA(n_components=max(cands), whiten=True, random_state=0).fit(Xf[itr])
        Ztr, Zva = pca.transform(Xf[itr]), pca.transform(Xf[iva])
        for c in cands:
            m = SpectralViT(n_inputs=c, embed_dim=16, use_mode_weights=True).to(dev)
            m = fit_train(m, torch.from_numpy(Ztr[:, :c]).float(), Y[itr], SEL_EP, BATCH, 1)
            p = predict(m, torch.from_numpy(Zva[:, :c]).float(), BATCH)
            scores[c].append(auc_safe(Y[iva], p))
            del m; torch.cuda.empty_cache()
    return one_se(scores, cands)


def select_volume(tr, cands, build):
    """Inner CV over a volume arm's capacity grid; build(c) returns the model."""
    inner = folds(INNER_K, Y[tr])
    scores = {c: [] for c in cands}
    for i_i, v_i in inner.split(tr, Y[tr]):
        itr, iva = tr[i_i], tr[v_i]
        xtr = torch.from_numpy(X[itr]).unsqueeze(1).float()
        xva = torch.from_numpy(X[iva]).unsqueeze(1).float()
        for c in cands:
            m = build(c).to(dev)
            m = fit_train(m, xtr, Y[itr], SEL_EP, PHYS, ACC)
            scores[c].append(auc_safe(Y[iva], predict(m, xva, PHYS)))
            del m; torch.cuda.empty_cache()
    return one_se(scores, cands)


class _Flat(nn.Module):
    """Drop the trailing singleton so these arms match the others' [B] output.

    LogisticRegression and MultiLayerPerceptron return [B, 1]; wrapping here rather than
    changing networks.py leaves the HCA and DBS experiments that use them untouched.
    """

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        return self.inner(x).squeeze(-1)


def build_simple(tag, ncomp, hp):
    """Logistic regression or a 2-layer perceptron on ncomp PCA components."""
    return _Flat(LogisticRegression(in_features=ncomp) if tag == 'logreg'
                 else MultiLayerPerceptron(input_dim=ncomp, hidden_dim=hp))


def select_simple(tr, tag):
    """Inner CV jointly over component count and the arm's own hyperparameter.

    Candidates are ordered smallest-and-most-regularized first so the one-standard-error
    rule breaks ties toward the simpler fit, as it does for every other arm. Components
    nest, so one PCA fit per inner fold still serves the whole grid.
    """
    inner = folds(INNER_K, Y[tr])
    kmin = int(len(tr) * (INNER_K - 1) / INNER_K) - 1
    ncs = [c for c in NCOMPS if c <= kmin] or [max(2, kmin)]
    hps = WD if tag == 'logreg' else HID
    cands = [(c, h) for c in ncs for h in hps]
    scores = {k: [] for k in cands}
    for i_i, v_i in inner.split(tr, Y[tr]):
        itr, iva = tr[i_i], tr[v_i]
        pca = PCA(n_components=max(ncs), whiten=True, random_state=0).fit(Xf[itr])
        Ztr, Zva = pca.transform(Xf[itr]), pca.transform(Xf[iva])
        for c, h in cands:
            m = build_simple(tag, c, h).to(dev)
            m = fit_train(m, torch.from_numpy(Ztr[:, :c]).float(), Y[itr], SEL_EP, BATCH,
                          1, wd=h if tag == 'logreg' else None)
            p = predict(m, torch.from_numpy(Zva[:, :c]).float(), BATCH)
            scores[(c, h)].append(auc_safe(Y[iva], p))
            del m; torch.cuda.empty_cache()
    return one_se(scores, cands)


def run_arm(tag):
    probs, ys, caps, nparam, sizes = [], [], [], None, []
    for fold, (tr, te) in enumerate(kf.split(Xf, Y), 1):
        t0 = time.time(); torch.cuda.empty_cache()
        if tag == 'spec':
            c = select_spectral(tr)
            pca = PCA(n_components=c, whiten=True, random_state=0).fit(Xf[tr])
            xtr = torch.from_numpy(pca.transform(Xf[tr])).float()
            xte = torch.from_numpy(pca.transform(Xf[te])).float()
            m = SpectralViT(n_inputs=c, embed_dim=16, use_mode_weights=True).to(dev)
            bs, acc, cap, wd = BATCH, 1, f'{c} comp', None
        elif tag in ('logreg', 'mlp'):
            c, hp = select_simple(tr, tag)
            pca = PCA(n_components=c, whiten=True, random_state=0).fit(Xf[tr])
            xtr = torch.from_numpy(pca.transform(Xf[tr])).float()
            xte = torch.from_numpy(pca.transform(Xf[te])).float()
            m = build_simple(tag, c, hp).to(dev)
            wd = hp if tag == 'logreg' else None
            cap = f'{c}c wd{hp:g}' if tag == 'logreg' else f'{c}c h{hp}'
            bs, acc = BATCH, 1
        else:
            cfg = dict(spat=(128, 2, 4), match=(12, 1, 1)).get(tag)
            xtr = torch.from_numpy(X[tr]).unsqueeze(1).float()
            xte = torch.from_numpy(X[te]).unsqueeze(1).float()
            if cfg:
                e, l, h = cfg
                ps = select_volume(tr, PATCHES, lambda c: SpatialViT(
                    vol_size=VOL, patch_size=c, embed_dim=e, n_layers=l, n_heads=h))
                m = SpatialViT(vol_size=VOL, patch_size=ps, embed_dim=e,
                               n_layers=l, n_heads=h).to(dev)
                cap = f'patch {ps}'
            elif tag == 'swin':
                ed = select_volume(tr, SWEMB, lambda c: SwinTransformer(
                    img_size=VOL, embed_dim=c, window_size=6))
                m = SwinTransformer(img_size=VOL, embed_dim=ed, window_size=6).to(dev)
                cap = f'embed {ed}'
            else:
                bc = select_volume(tr, CHANS, lambda c: AttentionUNet(
                    in_channels=1, base_channels=c))
                m = AttentionUNet(in_channels=1, base_channels=bc).to(dev)
                cap = f'{bc} ch'
            bs, acc, wd = PHYS, ACC, None
        nparam = sum(p.numel() for p in m.parameters())
        m = fit_train(m, xtr, Y[tr], EPOCHS, bs, acc, wd=wd)
        p = predict(m, xte, bs)
        probs.append(p); ys.append(Y[te]); caps.append(cap); sizes.append(len(te))
        print(f'  {tag} fold {fold}: {cap:9s} AUC {auc_safe(Y[te], p):.3f} '
              f'({time.time()-t0:.0f}s)', flush=True)
        del m; torch.cuda.empty_cache()
    save_npz(f'oof_{tag}{TAGSUF}.npz', probs=np.concatenate(probs),
             y=np.concatenate(ys), nparam=nparam, fold_sizes=sizes, caps=np.array(caps))
    return caps


sel = {}
for tag in ['spec', 'spat', 'match', 'swin', 'unet', 'logreg', 'mlp']:
    if tag not in ARMS_ON: continue
    print(f'=== {tag}', flush=True)
    sel[tag] = run_arm(tag)
cfg = dict(selected=sel, epochs=EPOCHS, sel_epochs=SEL_EP, inner_k=INNER_K, n=len(X))
json.dump(cfg, open(os.path.join(OUT, f'nested_config{TAGSUF}.json'), 'w'), indent=1)
if MIRROR:
    try:
        json.dump(cfg, open(os.path.join(MIRROR, f'nested_config{TAGSUF}.json'), 'w'),
                  indent=1)
    except OSError:
        pass
print('\nper-fold selections:', json.dumps(sel), flush=True)
