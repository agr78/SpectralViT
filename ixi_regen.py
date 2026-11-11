"""Regenerate the IXI sex-classification table and its significance tests in one run.

The published table mixes rows from two runs, so its p-values and point estimates do not
come from the same experiment. This reproduces the notebook protocol end to end and emits
the table, parameter counts and DeLong/McNemar tests together.

usage: python ixi_regen.py [--smoke]
"""
import os, sys, json, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import OneCycleLR

from util import seed_everything, compute_delong_paired, compute_mcnemar
from networks import SpectralViT, SpatialViT, AttentionUNet, SwinTransformer
from loader import load_ixi_data
from validate import SpectralViT_cv, SpatialViT_cv

SMOKE   = '--smoke' in sys.argv
IXI_DIR = os.environ.get('IXI_DIR', '/checkpoints/data/IXI_extracted/')
CSV     = os.path.join(IXI_DIR, 'IXI.csv')
OUT     = os.environ.get('IXI_OUT', '/workspace/results/ixi_regen')
VOL, LR, BATCH, NF = 96, 1e-4, 8, 5
EPOCHS  = 3 if SMOKE else int(os.environ.get('IXI_EPOCHS', 500))
PHYS, EFF = 2, 8
ACC = EFF // PHYS
os.makedirs(OUT, exist_ok=True)
seed_everything(0)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X, Y = load_ixi_data(IXI_DIR, CSV, VOL)
if SMOKE: X, Y = X[:40], Y[:40]
Xf = X.reshape(len(X), -1)
print(f'IXI: {len(X)} volumes {X.shape[1:]}, labels {np.bincount(Y.astype(int))}', flush=True)
kf = KFold(n_splits=2 if SMOKE else NF, shuffle=True, random_state=0)

# Capacity by cross-validation, as the notebook did.
if os.environ.get('IXI_NCOMP'):           # override the probe, e.g. the paper's 128
    NCOMP = int(os.environ['IXI_NCOMP'])
    PATCH = int(os.environ.get('IXI_PATCH', 12))
elif SMOKE:
    NCOMP, PATCH = 16, 12
else:
    NCOMP = SpectralViT_cv(X_flat=Xf, Y=Y, pca_candidates=[16, 32, 64, 128], device=dev,
                           epochs=EPOCHS, lr=LR, batch_size=BATCH)
    PATCH = SpatialViT_cv(X=X, Y=Y, patch_candidates=[8, 12, 16], device=dev, vol_size=VOL,
                          epochs=EPOCHS, lr=LR, physical_bs=PHYS, effective_bs=EFF)
    if os.environ.get('IXI_SELECT_ONLY'):
        json.dump(dict(N_COMP=int(NCOMP), patch=int(PATCH), epochs=EPOCHS, lr=LR),
                  open(os.path.join(OUT, 'selection.json'), 'w'), indent=1)
        print(f'selection written: N_COMP={NCOMP} patch={PATCH}'); sys.exit(0)
print(f'selected: N_COMP={NCOMP}  patch={PATCH}', flush=True)


def run(tag, build, spectral=False):
    """Train one arm across folds, returning pooled out-of-fold probabilities."""
    probs, ytrue, nparam = [], [], None
    for fold, (tr, te) in enumerate(kf.split(Xf), 1):
        t0 = time.time(); torch.cuda.empty_cache()
        if spectral:
            pca = PCA(n_components=NCOMP, whiten=True).fit(Xf[tr])
            xtr = torch.from_numpy(pca.transform(Xf[tr])).float()
            xte = torch.from_numpy(pca.transform(Xf[te])).float().to(dev)
            bs, acc = BATCH, 1
        else:
            xtr = torch.from_numpy(X[tr]).unsqueeze(1).float()
            xte = torch.from_numpy(X[te]).unsqueeze(1).float()
            bs, acc = PHYS, ACC
        ld = DataLoader(TensorDataset(xtr, torch.from_numpy(Y[tr]).float()),
                        batch_size=bs, shuffle=True)
        m = build().to(dev); nparam = sum(p.numel() for p in m.parameters())
        opt = optim.AdamW(m.parameters(), lr=LR)
        sch = OneCycleLR(opt, max_lr=LR, steps_per_epoch=max(1, len(ld) // acc), epochs=EPOCHS)
        crit = nn.BCEWithLogitsLoss()
        for ep in range(EPOCHS):
            m.train()
            for i, (bx, by) in enumerate(ld):
                loss = crit(m(bx.to(dev)), by.to(dev)) / acc
                loss.backward()
                if (i + 1) % acc == 0:
                    opt.step(); sch.step(); opt.zero_grad()
        m.eval()
        with torch.no_grad():
            if spectral:
                p = torch.sigmoid(m(xte)).cpu().numpy()
            else:
                p = torch.cat([torch.sigmoid(m(c.to(dev)))
                               for c in torch.split(xte, PHYS)]).cpu().numpy()
        probs.append(p); ytrue.append(Y[te])
        print(f'  {tag} fold {fold}: AUC {roc_auc_score(Y[te], p):.3f} '
              f'({time.time()-t0:.0f}s)', flush=True)
        del m; torch.cuda.empty_cache()
    np.savez(os.path.join(OUT, f'oof_{tag}.npz'), probs=np.concatenate(probs),
             y=np.concatenate(ytrue), nparam=nparam,
             fold_sizes=[len(p) for p in probs])
    return probs, ytrue, nparam


ARMS = [
    ('spec',  'Spectral ViT (proposed)',
     lambda: SpectralViT(n_inputs=NCOMP, embed_dim=16, use_mode_weights=True), True),
    ('spat',  'Standard spatial ViT',
     lambda: SpatialViT(vol_size=VOL, patch_size=PATCH, embed_dim=128, n_layers=2, n_heads=4), False),
    ('match', 'Compact spatial ViT',
     lambda: SpatialViT(vol_size=VOL, patch_size=PATCH, embed_dim=12, n_layers=1, n_heads=1), False),
    ('swin',  'Swin ViT', lambda: SwinTransformer(img_size=VOL), False),
    ('unet',  'U-net with attention',
     lambda: AttentionUNet(in_channels=1, base_channels=7), False),
]

res = {}
ONLY = [a for a in os.environ.get('IXI_ARMS', '').split(',') if a]
if ONLY: ARMS = [a for a in ARMS if a[0] in ONLY]
for tag, label, build, spec in ARMS:
    print(f'=== {label}', flush=True)
    res[tag] = (label,) + run(tag, build, spectral=spec)

rows, base = [], None
for tag, label, _, _ in ARMS:
    label, probs, ytrue, nparam = res[tag]
    yy, pp = np.concatenate(ytrue), np.concatenate(probs)
    accs = [accuracy_score(y, (p > 0.5).astype(int)) for y, p in zip(ytrue, probs)]
    aucs = [roc_auc_score(y, p) for y, p in zip(ytrue, probs)]
    r = dict(Model=label, Params=nparam, Accuracy=np.mean(accs), Acc_sd=np.std(accs),
             AUC=np.mean(aucs), AUC_sd=np.std(aucs),
             Efficiency=np.mean(aucs) / (nparam / 1e6),
             AUC_pooled=roc_auc_score(yy, pp))
    if tag == 'spec':
        base = (yy, pp)
        r['DeLong_z'] = r['DeLong_p'] = float('nan')
        r['McNemar_chi2'] = r['McNemar_p'] = float('nan')
    else:
        # both helpers return (statistic, p); the p-value is symmetric in argument order
        r['DeLong_z'], r['DeLong_p'] = compute_delong_paired(base[0], pp, base[1])
        r['McNemar_chi2'], r['McNemar_p'] = compute_mcnemar(
            base[0], (pp > 0.5).astype(int), (base[1] > 0.5).astype(int))
    rows.append(r)

d = pd.DataFrame(rows)
d.to_csv(os.path.join(OUT, 'ixi_table.csv'), index=False)
json.dump(dict(N_COMP=int(NCOMP), patch=int(PATCH), epochs=EPOCHS, folds=kf.n_splits,
               n=len(X)), open(os.path.join(OUT, 'config.json'), 'w'), indent=1)
pd.set_option('display.width', 200)
print('\n' + d.round(4).to_string(index=False))
print(f'\nwrote {OUT}/ixi_table.csv', flush=True)
