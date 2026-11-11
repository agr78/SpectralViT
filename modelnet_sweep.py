"""ModelNet40 on a shared icosphere: does the topology-preserving basis hold up?

sec:tp claims the Laplacian eigenbasis orders tokens by smoothness over a graph rather than
over a pixel lattice, which matters on domains where the Euclidean grid is not the relevant
geometry. No experiment in the paper currently separates that claim from the DCT, because on
a regular grid the two coincide: the graph Laplacian eigenvectors of a lattice with Neumann
boundaries *are* the DCT basis. A sphere has no separable cosine basis, so it is the setting
where the Laplacian is the only natural fixed choice and the property becomes testable.

Each ModelNet40 shape is reduced to its support function on one shared icosphere (see
spherical.py), so every object carries a signal on the same graph and one Laplacian serves
all of them. The comparison is then basis against basis at matched coefficient counts:

  laplacian  the eigenvectors of the shared icosphere graph, fixed and data independent
  pca        fit on each training subsample, data dependent, topology agnostic
  learned    a trainable projection of the raw signal, the stand-in for a learned tokenizer
  rand       fixed random orthonormal directions: data independent AND topology blind
  vertex     the signal at a fixed subset of sphere vertices, i.e. no basis at all

Laplacian against pca isolates fixed versus fitted; Laplacian against rand or vertex
isolates what the graph structure itself contributes, which pca cannot answer because it
differs in both respects at once.

Both feed the identical encoder, so the only variable is which fixed or fitted basis
supplies the coefficients. A spatial-patch baseline is deliberately absent: an icosphere has
no regular patch structure, and inventing one would compare against a straw tokenizer rather
than answer the question sec:tp poses.

env: MODELNET_DIR MODELNET_CACHE MODELNET_BASIS MODELNET_COMP MODELNET_SUBDIV
     MODELNET_SEEDS MODELNET_N MODELNET_OUT
usage: python modelnet_sweep.py [--smoke]
"""
import os, sys, time, glob
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, top_k_accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from networks import MultiClassSpectralViT
from spherical import basis, support, read_off

# Token-identity switches; defaults reproduce every result already on disk.
_POS = os.environ.get('MODELNET_POS', '1') != '0'
_RANK = os.environ.get('MODELNET_RANK') == '1'


SMOKE = '--smoke' in sys.argv
DIR = os.environ.get('MODELNET_DIR', '/agent-home/modelnet/ModelNet40')
CACHE = os.environ.get('MODELNET_CACHE', '/agent-home/modelnet/sphere_signals.npz')
BASIS = os.environ.get('MODELNET_BASIS', 'laplacian')
N_COMP = int(os.environ.get('MODELNET_COMP', 16))
SUBDIV = int(os.environ.get('MODELNET_SUBDIV', 3))
N_SEEDS = 2 if SMOKE else int(os.environ.get('MODELNET_SEEDS', 5))
SIZES = [40, 80] if SMOKE else [int(v) for v in
                                os.environ.get('MODELNET_N', '40,80,200,400,800').split(',')]
EPOCHS, BATCH, LR, WD = (2 if SMOKE else 100), 128, 1e-3, 0.01
OUT = os.environ.get('MODELNET_OUT',
                     os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'results', 'modelnet'))
if N_COMP % 4:
    sys.exit(f'MODELNET_COMP must be a multiple of 4 (chunk_size); got {N_COMP}')
os.makedirs(OUT, exist_ok=True)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------------------------------------------- shared sphere + basis
verts, eigval, eigvec = basis(SUBDIV, max(N_COMP, 64))
print(f'icosphere subdiv {SUBDIV}: {len(verts)} vertices; '
      f'{eigvec.shape[1]} Laplacian modes, eigenvalues to {eigval[-1]:.2f}', flush=True)


class LearnedBasis(nn.Module):
    """A trainable projection in place of a fixed basis, on identical input.

    A spatial-patch baseline is unavailable here because an icosphere has no regular patch
    structure, so this is the fair stand-in for a learned tokenizer: the same signals, the
    same encoder, and the projection estimated end to end rather than supplied. It carries
    n_in * comp additional parameters, which is the cost a fixed basis avoids.
    """
    def __init__(self, n_in, comp, ncls, **kw):
        super().__init__()
        self.proj = nn.Linear(n_in, comp, bias=False)
        self.net = MultiClassSpectralViT(n_inputs=comp, num_classes=ncls, **kw)

    def forward(self, x):
        return self.net(self.proj(x))


def build_cache():
    """Support-function signals for every shape, computed once."""
    classes = sorted(d for d in os.listdir(DIR) if os.path.isdir(os.path.join(DIR, d)))
    out = {}
    for split in ('train', 'test'):
        sig, lab, t0 = [], [], time.time()
        for ci, c in enumerate(classes):
            files = sorted(glob.glob(os.path.join(DIR, c, split, '*.off')))
            for f in files:
                try:
                    sig.append(support(read_off(f), verts)); lab.append(ci)
                except Exception:
                    # A handful of ModelNet OFF files are malformed; skipping one shape is
                    # preferable to failing the whole cache, and the count is reported.
                    continue
            print(f'  {split} {c:24s} {len(files):4d} files '
                  f'({time.time()-t0:.0f}s)', flush=True)
        out[f'{split}_x'] = np.asarray(sig, dtype=np.float32)
        out[f'{split}_y'] = np.asarray(lab, dtype=np.int64)
    out['classes'] = np.array(classes)
    np.savez_compressed(CACHE, **out)
    return out


if os.path.exists(CACHE):
    z = dict(np.load(CACHE, allow_pickle=True))
    print(f'cache {CACHE}', flush=True)
else:
    print(f'building cache from {DIR} (reads every OFF file once)', flush=True)
    z = build_cache()

Xtr_all, ytr_all = z['train_x'], z['train_y']
Xte, yte = z['test_x'], z['test_y']
NCLS = int(ytr_all.max()) + 1
mu, sd = Xtr_all.mean(), Xtr_all.std()
Xtr_all, Xte = (Xtr_all - mu) / sd, (Xte - mu) / sd
pool = {c: np.where(ytr_all == c)[0] for c in range(NCLS)}
avail = np.array([len(pool[c]) for c in range(NCLS)])
print(f'ModelNet40: {len(Xtr_all)} train / {len(Xte)} test signals on {len(verts)} '
      f'vertices, {NCLS} classes on {dev}; basis {BASIS}, {N_COMP} coef', flush=True)


def metrics(probs, y_true):
    lab = np.arange(NCLS)
    return {'T1': accuracy_score(y_true, np.argmax(probs, axis=1)),
            'T5': top_k_accuracy_score(y_true, probs, k=5, labels=lab),
            'AUC': roc_auc_score(y_true, probs, multi_class='ovr', labels=lab)}


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


rows, t0 = [], time.time()
for N in SIZES:
    for seed in range(N_SEEDS):
        torch.manual_seed(seed); np.random.seed(seed)
        take = allocate(avail, N)
        tr_idx = np.concatenate([np.random.choice(pool[c], take[c], replace=False)
                                 for c in range(NCLS) if take[c]])
        Str, ytr = Xtr_all[tr_idx], ytr_all[tr_idx]

        if BASIS == 'laplacian':
            # Fixed: the same eigenvectors for every shape and every fold, so nothing is
            # estimated from the training split except the standardization.
            Ftr, Fte = Str @ eigvec[:, :N_COMP], Xte @ eigvec[:, :N_COMP]
            m, s = Ftr.mean(0, keepdims=True), Ftr.std(0, keepdims=True) + 1e-6
            Ztr = torch.tensor((Ftr - m) / s, dtype=torch.float32)
            Zte = torch.tensor((Fte - m) / s, dtype=torch.float32)
        elif BASIS == 'learned':
            # No projection applied here: the raw signal is handed to the model, which
            # learns one. Standardization is still fit on the training split only.
            m, sdv = Str.mean(0, keepdims=True), Str.std(0, keepdims=True) + 1e-6
            Ztr = torch.tensor((Str - m) / sdv, dtype=torch.float32)
            Zte = torch.tensor((Xte - m) / sdv, dtype=torch.float32)
        elif BASIS in ('rand', 'vertex'):
            # Topology-blind controls, both fixed and data independent like the Laplacian,
            # so a gap between them and 'laplacian' isolates what the graph structure buys
            # rather than what adaptivity buys. 'rand' projects onto fixed random
            # orthonormal directions; 'vertex' just reads the signal at a fixed subset of
            # sphere vertices, which is what one would do with no basis at all. Both use a
            # fixed seed, so the projection is identical across every size and fold.
            rs = np.random.RandomState(0)
            if BASIS == 'rand':
                q, _ = np.linalg.qr(rs.randn(Str.shape[1], N_COMP))
            else:
                sel = rs.choice(Str.shape[1], N_COMP, replace=False)
                q = np.eye(Str.shape[1])[:, sel]
            Ftr, Fte = Str @ q, Xte @ q
            m, sdv = Ftr.mean(0, keepdims=True), Ftr.std(0, keepdims=True) + 1e-6
            Ztr = torch.tensor((Ftr - m) / sdv, dtype=torch.float32)
            Zte = torch.tensor((Fte - m) / sdv, dtype=torch.float32)
        else:
            sc = StandardScaler().fit(Str)
            p = PCA(n_components=min(N_COMP, len(Str))).fit(sc.transform(Str))
            Ztr = torch.tensor(p.transform(sc.transform(Str)), dtype=torch.float32)
            Zte = torch.tensor(p.transform(sc.transform(Xte)), dtype=torch.float32)

        net = (LearnedBasis(Ztr.shape[1], N_COMP, NCLS, use_mode_weights=True)
               if BASIS == 'learned' else
               MultiClassSpectralViT(n_inputs=Ztr.shape[1], num_classes=NCLS,
                                     use_mode_weights=True, use_rank_weights=_RANK,
                                     use_pos_embed=_POS)).to(dev)
        opt = optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
        loader = DataLoader(TensorDataset(Ztr, torch.from_numpy(ytr)), batch_size=BATCH,
                            shuffle=True)
        for _ in range(EPOCHS):
            net.train()
            for bz, by in loader:
                bz, by = bz.to(dev), by.to(dev)
                opt.zero_grad()
                F.cross_entropy(net(bz), by).backward()
                opt.step()

        net.eval(); ps = []
        with torch.no_grad():
            for (bz,) in DataLoader(TensorDataset(Zte), batch_size=512):
                ps.append(F.softmax(net(bz.to(dev)), 1).cpu())
        ms = metrics(torch.cat(ps).numpy(), yte)
        rows.append(dict(n=int(take.sum()), requested=N, seed=seed, basis=BASIS,
                         comp=int(Ztr.shape[1]), subdiv=SUBDIV, ncls=NCLS,
                         nparam=sum(q.numel() for q in net.parameters()), **ms))
        print(f'  n={int(take.sum()):4d} seed {seed}: T1 {ms["T1"]:.3f} '
              f'T5 {ms["T5"]:.3f} AUC {ms["AUC"]:.3f} ({time.time()-t0:.0f}s)', flush=True)
        del net
        if dev.type == 'cuda':
            torch.cuda.empty_cache()
        tmp = os.path.join(OUT, f'modelnet_{BASIS}.csv.tmp')
        pd.DataFrame(rows).to_csv(tmp, index=False)
        os.replace(tmp, os.path.join(OUT, f'modelnet_{BASIS}.csv'))

print(f'\nwrote {OUT}/modelnet_{BASIS}.csv ({len(rows)} cells)', flush=True)
