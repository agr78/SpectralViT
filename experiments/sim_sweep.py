"""Simulated pattern-classification sweep.

usage: python experiments/sim_sweep.py [--smoke]"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repository root: shared modules and paths.py
import os, sys, time
import numpy as np, torch, torch.optim as optim, torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util import seed_everything
from train import generate_checkerboard_data
from networks import SpectralViT, SpatialViT

SMOKE = '--smoke' in sys.argv
N_COMPONENTS, EPOCHS, LR, N_TEST = 16, 100, 1e-3, 500
SIZES = [10, 100] if SMOKE else [10, 100, 1000, 10000, 100000]
NOISE_LEVELS = [1.0, 2.0, 3.0]

NL_IDX = int(os.environ.get('SIM_NL_IDX', 0))
BLOCK = int(os.environ.get('SIM_BLOCK', 0))
BLOCK_SIZE = 2 if SMOKE else int(os.environ.get('SIM_BLOCK_SIZE', 20))
OUT = os.environ.get('SIM_OUT', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                             'results', 'sim_sweep'))
NL = NOISE_LEVELS[NL_IDX]
SEEDS = range(BLOCK * BLOCK_SIZE, (BLOCK + 1) * BLOCK_SIZE)
os.makedirs(OUT, exist_ok=True)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'noise {NL} seeds {SEEDS.start}-{SEEDS.stop - 1} sizes {SIZES} on {dev}', flush=True)


def build(ncomp):
    spec = SpectralViT(n_inputs=ncomp, embed_dim=16, n_heads=2, n_layers=2,
                       use_sigmoid=True, pooling='mean',
                       learnable_rank_weights=False, use_layer_norm=False).to(dev)
    spat = SpatialViT(size=28, patch_size=7, embed_dim=16, n_heads=2, n_layers=2,
                      is_2d=True, use_cls_token=False, use_sigmoid=True,
                      use_layer_norm=False).to(dev)
    return spec, spat


spec_res = np.zeros((len(SEEDS), len(SIZES)))
spat_res = np.zeros((len(SEEDS), len(SIZES)))
t0 = time.time()
for row, seed in enumerate(SEEDS):
    seed_everything(seed)
    for col, n in enumerate(SIZES):
        X_tr, y_tr = generate_checkerboard_data(n, noise_level=NL)
        X_te, y_te = generate_checkerboard_data(N_TEST, noise_level=NL)
        sc = StandardScaler().fit(X_tr)
        X_tr_s, X_te_s = sc.transform(X_tr), sc.transform(X_te)
        pca = PCA(n_components=min(n - 1, N_COMPONENTS), whiten=True).fit(X_tr_s)

        yt = torch.tensor(y_tr, dtype=torch.float32).to(dev)
        m_spec, m_spat = build(pca.n_components_)
        o_spec = optim.Adam(m_spec.parameters(), lr=LR)
        o_spat = optim.Adam(m_spat.parameters(), lr=LR)
        xt_spec = torch.tensor(pca.transform(X_tr_s), dtype=torch.float32).to(dev)
        xt_spat = torch.tensor(X_tr_s, dtype=torch.float32).to(dev)

        with sdpa_kernel(SDPBackend.MATH):
            for _ in range(EPOCHS):
                F.binary_cross_entropy(m_spec(xt_spec), yt).backward()
                o_spec.step(); o_spec.zero_grad()
                F.binary_cross_entropy(m_spat(xt_spat.view(-1, 1, 28, 28)), yt).backward()
                o_spat.step(); o_spat.zero_grad()

            with torch.no_grad():
                p_spec = m_spec(torch.tensor(pca.transform(X_te_s),
                                             dtype=torch.float32).to(dev)).cpu().numpy()
                p_spat = m_spat(torch.tensor(X_te_s, dtype=torch.float32)
                                .to(dev).view(-1, 1, 28, 28)).cpu().numpy()
        spec_res[row, col] = roc_auc_score(y_te, p_spec)
        spat_res[row, col] = roc_auc_score(y_te, p_spat)
        del m_spec, m_spat, xt_spec, xt_spat
        if dev.type == 'cuda':
            torch.cuda.empty_cache()
    print(f'  seed {seed}: spec {spec_res[row].round(3)} spat {spat_res[row].round(3)} '
          f'({time.time() - t0:.0f}s)', flush=True)

f = os.path.join(OUT, f'sim_nl{NL:g}_b{BLOCK}.npz')
np.savez(f + '.tmp.npz', spec=spec_res, spat=spat_res, sizes=np.array(SIZES),
         noise_level=NL, seeds=np.array(list(SEEDS)))
os.replace(f + '.tmp.npz', f)
print(f'wrote {f}', flush=True)
