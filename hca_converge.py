"""Does the HCA small-n budget overtrain, and does the budget convention matter?

hca_regress.py fixes the gradient-step budget at ~10,000 for every sample size, which at
n=25 means 2,000 epochs over 20 subjects. The conventional alternative fixes epochs and
lets steps scale with n. These are not nested: OneCycleLR anneals over the whole budget,
so a 500-epoch run is a different schedule rather than a prefix of a 2,000-epoch one, and
the two have to be run separately.

The curve is traced on a validation split carved from the TRAINING fold, never on the
test fold. Watching test error over training would leak test information into the choice
of schedule even though no model is fit to it, and the minimum of such a curve is not
attainable without peeking, so it must not be computed or reported.

Every outer fold of every subsample draw trains the proposed arm and logistic regression
under both budgets, recording train and validation MAE throughout, so that two questions
can be answered with an error bar: whether held-out error degrades late (overtraining),
and whether the two budgets land in materially different places (whether the convention
matters). A single fold at n=25 holds five test subjects, which is far too noisy to read.

env: HCA_PREFIX HCA_N HCA_SEEDS HCA_EVERY HCA_OUT
usage: python hca_converge.py
"""
import os, sys, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
from scipy import fft
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import OneCycleLR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from networks import SpectralViT, LogisticRegression

PREFIX = os.environ.get('HCA_PREFIX',
                        os.path.expanduser('~/spacelink/aHCP/derivatives/hca_vols_96'))
N = int(os.environ.get('HCA_N', 25))
SEEDS = int(os.environ.get('HCA_SEEDS', 5))
EVERY = int(os.environ.get('HCA_EVERY', 25))      # evaluate every EVERY epochs
OUT = os.environ.get('HCA_OUT', os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             'results', 'converge'))
GRID, FREQ, BATCH, LR = 4, 4, 4, 1e-3
os.makedirs(OUT, exist_ok=True)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

meta = np.load(PREFIX + '_meta.npz', allow_pickle=True)
n_valid = int(meta['n'])
Xall = np.load(PREFIX + '.npy', mmap_mode='r')[:n_valid]
yall, aall = meta['y'][:n_valid], meta['age'][:n_valid].astype(np.float32)


def block_spectra(vols):
    B, nb, nf = len(vols), GRID ** 3, FREQ ** 3
    bz, by, bx = [s // GRID for s in vols.shape[1:]]
    out = np.zeros((B, nb, nf), dtype=np.float32)
    for i in range(B):
        bi = 0
        for z in range(GRID):
            for yy in range(GRID):
                for xx in range(GRID):
                    blk = vols[i][z*bz:(z+1)*bz, yy*by:(yy+1)*by, xx*bx:(xx+1)*bx]
                    c = fft.dctn(blk, type=2, norm='ortho')[:FREQ, :FREQ, :FREQ].ravel()
                    out[i, bi] = np.sign(c) * np.log1p(np.abs(c)); bi += 1
    return out.reshape(B, nb * nf)


def rank_perm():
    nb, nf = GRID ** 3, FREQ ** 3
    kz, ky, kx = np.unravel_index(np.arange(nf), (FREQ, FREQ, FREQ))
    order = np.argsort(kz ** 2 + ky ** 2 + kx ** 2, kind='stable')
    return np.array([b * nf + order[r] for r in range(nf) for b in range(nb)])



def build(kind):
    if kind == 'spectral':
        return SpectralViT(n_inputs=4096, patch_size=64, embed_dim=32, n_layers=2,
                           n_heads=2, use_rank_weights=True, use_pos_embed=False,
                           learnable_rank_weights=True)
    m = LogisticRegression(in_features=4096)
    m._wd = 0.05
    return m


rows = []
for seed in range(SEEDS):
    rng = np.random.RandomState(seed)
    idx = []
    for c in np.unique(yall):
        pool = np.where(yall == c)[0]
        idx += list(rng.choice(pool, size=int(round(N * len(pool) / len(yall))),
                               replace=False))
    idx = np.array(sorted(idx))[:N]
    X, y, a = np.asarray(Xall[idx], dtype=np.float32), yall[idx], aall[idx]
    Fs = torch.from_numpy(block_spectra(X)[:, rank_perm()])
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    for fold, (tr, _te) in enumerate(kf.split(Fs, y)):
        # The test fold is discarded here. Fit on 80% of the training fold, trace the
        # curve on the remaining 20%, so nothing about the schedule is chosen on test.
        cut = max(2, int(0.2 * len(tr)))
        va, fit = tr[:cut], tr[cut:]
        mu, sd = Fs[fit].mean(0, keepdim=True), Fs[fit].std(0, keepdim=True) + 1e-6
        F = (Fs - mu) / sd
        amu, asd = float(a[fit].mean()), float(a[fit].std() + 1e-8)
        for kind in ('spectral', 'logreg'):
            for epochs in (2000, 500):   # equal-step budget vs conventional fixed epochs
                torch.manual_seed(0)
                m = build(kind).to(dev)  # both arms already end in a single-unit head
                ld = DataLoader(TensorDataset(F[fit], torch.tensor((a[fit] - amu) / asd)),
                                batch_size=BATCH, shuffle=True)
                opt = optim.AdamW(m.parameters(), lr=LR,
                                  weight_decay=getattr(m, '_wd', 0.05))
                sch = OneCycleLR(opt, max_lr=LR, steps_per_epoch=len(ld), epochs=epochs)
                crit = nn.L1Loss()
                t0 = time.time()
                for ep in range(1, epochs + 1):
                    m.train()
                    for bx, by in ld:
                        bx, by = bx.to(dev), by.to(dev, dtype=torch.float32)
                        opt.zero_grad(); l = crit(m(bx).squeeze(-1), by)
                        if not torch.isfinite(l):
                            continue
                        l.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                        opt.step(); sch.step()
                    if ep % EVERY == 0 or ep == epochs:
                        m.eval()
                        with torch.no_grad():
                            def mae(ix):
                                p = m(F[ix].to(dev)).squeeze(-1).cpu().numpy() * asd + amu
                                return float(np.abs(p - a[ix]).mean())
                            rows.append(dict(seed=seed, fold=fold, model=kind,
                                             budget=epochs, epoch=ep, frac=ep / epochs,
                                             train_mae=mae(fit), val_mae=mae(va),
                                             lr=sch.get_last_lr()[0]))
                print(f'  seed {seed} fold {fold} {kind:9s} {epochs:5d}ep  '
                      f'train {rows[-1]["train_mae"]:5.2f}  val '
                      f'{rows[-1]["val_mae"]:5.2f}  ({time.time()-t0:.0f}s)', flush=True)
                pd.DataFrame(rows).to_csv(
                    os.path.join(OUT, f'converge_n{N}.csv'), index=False)
print(f'\nwrote {OUT}/converge_n{N}.csv  ({len(rows)} checkpoints)', flush=True)
