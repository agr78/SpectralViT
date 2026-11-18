"""IXI sex classification with properly nested capacity selection.

usage: python experiments/ixi_nested.py [--smoke]"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repository root: shared modules and paths.py
from paths import RESULTS
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
OUT     = os.environ.get('IXI_OUT', f'{RESULTS}/ixi_nested')
VOL, LR, BATCH, NF, PHYS, EFF = 96, 1e-4, 8, 5, 2, 8
ACC     = EFF // PHYS
EPOCHS  = 3 if SMOKE else int(os.environ.get('IXI_EPOCHS', 500))
SEL_EP  = 3 if SMOKE else int(os.environ.get('IXI_SEL_EPOCHS', 200))
INNER_K = 2 if SMOKE else int(os.environ.get('IXI_INNER_K', 3))
NCOMPS  = [16, 32] if SMOKE else [16, 32, 64, 128]
PATCHES = [8, 12] if SMOKE else [8, 12, 16]
CHANS   = [4, 8] if SMOKE else [4, 8, 16]
SWEMB   = [16, 32] if SMOKE else [16, 32, 64]
WD      = [0.5, 0.05] if SMOKE else [0.5, 0.05, 0.005]
HID     = [2, 6] if SMOKE else [2, 6, 16]
ARMS_ON = [a for a in os.environ.get(
    'IXI_ARMS', 'spec,spat,swin,unet,logreg,mlp').split(',') if a]
TAGSUF = os.environ.get('IXI_SUFFIX', '')
MIRROR = os.environ.get('IXI_MIRROR', os.path.expanduser('~/SpectralViT/results/ixi_curve'))
os.makedirs(OUT, exist_ok=True)
if MIRROR:
    try:
        os.makedirs(MIRROR, exist_ok=True)
    except OSError:
        MIRROR = ''


def save_npz(name, **arrays):
    np.savez(os.path.join(OUT, name), **arrays)
    if MIRROR:
        try:
            np.savez(os.path.join(MIRROR, name), **arrays)
        except OSError:
            pass
seed_everything(0)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
SUB = int(os.environ.get('IXI_N', 0))
SEED = int(os.environ.get('IXI_SEED', 0))
if SUB and SUB < len(Y):
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
    if (os.environ.get('IXI_STRATIFY', '1') != '0'
            and np.bincount(labels.astype(int), minlength=2).min() >= n_splits):
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed)


def auc_safe(y, p):
    y = np.asarray(y)
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else float('nan')


NF_ = 2 if SMOKE else NF
kf = folds(NF_, Y, seed=SEED)


def fit_train(model, xtr, ytr, epochs, bs, acc, wd=None):
    ld = DataLoader(TensorDataset(xtr, torch.from_numpy(ytr).float()),
                    batch_size=bs, shuffle=True)
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
    means = {c: np.nanmean(v) for c, v in scores.items()}
    if all(m != m for m in means.values()): return cands[0]
    best = max(means, key=lambda c: means[c])
    se = np.nanstd(scores[best], ddof=1) / np.sqrt(len(scores[best])) if len(scores[best]) > 1 else 0.0
    thresh = means[best] - (se if se == se else 0.0)
    for c in cands:
        if means[c] == means[c] and means[c] >= thresh: return c
    return best


def select_spectral(tr):
    inner = folds(INNER_K, Y[tr])
    kmin = int(len(tr) * (INNER_K - 1) / INNER_K) - 1
    cands = [c for c in NCOMPS if c <= kmin] or [max(2, kmin)]
    scores = {c: [] for c in cands}
    for i_i, v_i in inner.split(tr, Y[tr]):
        itr, iva = tr[i_i], tr[v_i]
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
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        return self.inner(x).squeeze(-1)


def build_simple(tag, ncomp, hp):
    return _Flat(LogisticRegression(in_features=ncomp) if tag == 'logreg'
                 else MultiLayerPerceptron(input_dim=ncomp, hidden_dim=hp))


def select_simple(tr, tag):
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
