"""Object-classification distribution shift, replicated over seeds.

objects.ipynb ran this once at seed 0 and reported a single pair of accuracies. Here each
seed is an independent draw of data and initialisation, and every seed's scores are saved,
so the result carries a spread and a paired test rather than one number.

Two cubes sit at distance 4 (class 0) or 12 (class 1). In training, class 0 is on the left
and class 1 on the right, so position is spuriously correlated with the label; at test the
positions are swapped. A model keyed to position therefore scores BELOW chance under the
swap rather than merely at it, which is the signature of a learned confound. To show that
is what happens, each seed is scored twice: on a held-out set from the training
distribution (in-distribution) and on the swapped set (shifted).

The spectral arm sees |FFT| of the volume, whose magnitude is translation invariant, so
the confound is unavailable to it by construction.

env: OC_BLOCK OC_BLOCK_SIZE OC_OUT OC_EPOCHS
usage: python oc_sweep.py [--smoke]
"""
import os, sys, time
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from util import seed_everything, generate_distance_data
from networks import SpectralViT, SpatialViT

SMOKE = '--smoke' in sys.argv
VOL, N_TRAIN, N_TEST = 32, 10000, 200
EPOCHS = 2 if SMOKE else int(os.environ.get('OC_EPOCHS', 100))
BATCH, LR = 32, 1e-3
TOTAL_BINS, N_SAMP, N_TOK = 32768, 2048, 32

BLOCK = int(os.environ.get('OC_BLOCK', 0))
BLOCK_SIZE = 2 if SMOKE else int(os.environ.get('OC_BLOCK_SIZE', 10))
OUT = os.environ.get('OC_OUT', os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            'results', 'oc_sweep'))
SEEDS = range(BLOCK * BLOCK_SIZE, (BLOCK + 1) * BLOCK_SIZE)
os.makedirs(OUT, exist_ok=True)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IDX = torch.linspace(0, TOTAL_BINS - 1, N_SAMP).long()
print(f'seeds {SEEDS.start}-{SEEDS.stop - 1}, {EPOCHS} epochs on {dev}', flush=True)


def build():
    """The two models exactly as objects.ipynb configured them."""
    spec = SpectralViT(n_inputs=N_TOK, patch_size=N_SAMP // N_TOK, sampling_indices=IDX,
                       embed_dim=64, n_heads=4, n_layers=2, use_rank_weights=False,
                       use_pos_embed=False, use_layer_norm=False).to(dev)
    spat = SpatialViT(vol_size=VOL, patch_size=8, embed_dim=64, n_heads=4, n_layers=2,
                      use_cls_token=False, use_layer_norm=False, use_sigmoid=False).to(dev)
    return spec, spat


def score(spec, spat, X, Y):
    """Accuracy and AUC for both arms on one evaluation set."""
    spec.eval(); spat.eval()
    ps, pp = [], []
    with torch.no_grad():
        for bv, _ in DataLoader(TensorDataset(X, Y), batch_size=64):
            bv = bv.to(dev)
            bf = torch.abs(torch.fft.fftn(bv, dim=(-3, -2, -1)))
            ps.append(torch.sigmoid(spec(bf)).cpu().numpy())
            pp.append(torch.sigmoid(spat(bv)).cpu().numpy())
    y = Y.numpy()
    ps, pp = np.concatenate(ps), np.concatenate(pp)
    return ((ps > .5) == y).mean(), roc_auc_score(y, ps), \
           ((pp > .5) == y).mean(), roc_auc_score(y, pp)


# columns: spec_acc_id, spec_auc_id, spat_acc_id, spat_auc_id,
#          spec_acc_ood, spec_auc_ood, spat_acc_ood, spat_auc_ood
res = np.zeros((len(SEEDS), 8))
t0 = time.time()
for row, seed in enumerate(SEEDS):
    seed_everything(seed)
    Xtr, Ytr = generate_distance_data(N_TRAIN, VOL, swapped=False)
    Xid, Yid = generate_distance_data(N_TEST, VOL, swapped=False)   # same distribution
    Xood, Yood = generate_distance_data(N_TEST, VOL, swapped=True)  # positions flipped

    spec, spat = build()
    o_spec, o_spat = optim.Adam(spec.parameters(), lr=LR), optim.Adam(spat.parameters(), lr=LR)
    crit = nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=BATCH, shuffle=True)
    for _ in range(EPOCHS):
        spec.train(); spat.train()
        for bv, by in loader:
            bv, by = bv.to(dev), by.to(dev)
            bf = torch.abs(torch.fft.fftn(bv, dim=(-3, -2, -1)))
            o_spec.zero_grad(); crit(spec(bf), by).backward(); o_spec.step()
            o_spat.zero_grad(); crit(spat(bv), by).backward(); o_spat.step()

    res[row, 0:4] = score(spec, spat, Xid, Yid)
    res[row, 4:8] = score(spec, spat, Xood, Yood)
    print(f'  seed {seed}: in-dist spec {res[row,0]:.3f} spat {res[row,2]:.3f} | '
          f'shifted spec {res[row,4]:.3f} spat {res[row,6]:.3f} '
          f'({time.time()-t0:.0f}s)', flush=True)
    del spec, spat
    if dev.type == 'cuda':
        torch.cuda.empty_cache()

f = os.path.join(OUT, f'oc_b{BLOCK}.npz')
np.savez(f + '.tmp.npz', res=res, seeds=np.array(list(SEEDS)),
         cols=np.array(['spec_acc_id', 'spec_auc_id', 'spat_acc_id', 'spat_auc_id',
                        'spec_acc_ood', 'spec_auc_ood', 'spat_acc_ood', 'spat_auc_ood']))
os.replace(f + '.tmp.npz', f)
print(f'wrote {f}', flush=True)
