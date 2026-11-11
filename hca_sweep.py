"""Data-efficiency sweep on HCP-Aging: spectral vs spatial tokenization by sample size.

One arm per (basis, model). Both spectral bases are run at every n so a difference in the
curve is attributable to the basis rather than to the sample size:

  Block-DCT    signed low-frequency DCT coefficients, per 4x4x4 block  (phase sensitive)
  Block-|FFT|  log magnitude of the same block spectra               (translation invariant)

Volumes are unregistered and center-cropped, so only the magnitude readout is shift
invariant; the paper's argument predicts its advantage should grow as n shrinks.

Training, the classification wrapper and the early-stopping rule are taken unchanged from
the existing HCP experiment so the numbers stay comparable.

usage: python hca_sweep.py <n> [prefix] [--smoke]
  --smoke runs 2 epochs and 2 folds to check shapes without spending GPU time
"""
import os, sys, time, copy, json, warnings, traceback
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import (StratifiedKFold, KFold, LeaveOneOut,
                                     train_test_split)
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score,
                             roc_auc_score)
import scipy.fft as fft

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from networks import (SpectralViT, SpatialViT, AttentionUNet, SwinTransformer,
                      LogisticRegression, MultiLayerPerceptron)

N          = int(sys.argv[1])
PREFIX     = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') \
             else os.path.expanduser('~/spacelink/aHCP/derivatives/hca_vols_96')
SMOKE      = '--smoke' in sys.argv
EPOCHS     = 2 if SMOKE else int(os.environ.get('HCA_EPOCHS', '500'))
LR         = 1e-3
BATCH      = 4
NUM_CLASSES = 2
# FREQ is the CACHE LAYOUT, not the per-run knob: trunc() reshapes the stored features to
# (GRID^3, FREQ, FREQ, FREQ) before slicing, so lowering HCA_FREQ misreads a cache written
# at 4 rather than truncating it. The feature count is GRID^3 * FREQ^3, i.e. 512 / 1728 /
# 4096 for 2 / 3 / 4, but to RUN at one of those counts set HCA_FREQS below and leave this
# at 4. Overridable only if the cache itself is rebuilt to match.
GRID = 4
FREQ = int(os.environ.get('HCA_FREQ', 4))
NFEAT = GRID ** 3 * FREQ ** 3
# The coefficient count is a capacity like any other, and IXI and DBS both select theirs
# by cross-validation. HCA fixed it at 4096, which handed a linear head 4097 parameters
# regardless of n. Listing more than one value here puts it in the inner-CV grid.
FREQS = sorted({int(v) for v in os.environ.get('HCA_FREQS', str(FREQ)).split(',')})


def nfeat(f):
    return GRID ** 3 * f ** 3


def trunc(x, f, key='dct'):
    """Low-frequency sub-cube of the FREQ-truncated features.

    The cube truncations nest, so no recomputation is needed, but the f-cube is not a
    prefix of the raveled vector: the block axis has to be restored before slicing.

    The 'dct_rank' features are additionally permuted to frequency-major order by PERM,
    and the f-cube is not a prefix of that order either -- (0,0,3) sorts before (1,1,1)
    by |k| yet lies outside the 3-cube. Those are therefore un-permuted, truncated in
    block-major order, and re-permuted at the smaller size.
    """
    if f == FREQ:
        return x
    ranked = key.startswith('dct_rank')
    xb = x[:, INV_PERM] if ranked else x
    xb = xb.reshape(len(x), GRID ** 3, FREQ, FREQ, FREQ)[:, :, :f, :f, :f] \
           .reshape(len(x), nfeat(f))
    return xb[:, rank_perm(f)] if ranked else xb
OUT = os.environ.get('HCA_OUT', os.path.expanduser('~/spacelink/aHCP/derivatives/sweep'))
N_SEEDS = int(os.environ.get('HCA_SEEDS', 5))   # matches the MNIST sweep
# Resume a partially finished sweep: run seeds [SEED_START, N_SEEDS). draw(seed) is
# independent per seed, so a later seed is reproducible without rerunning the earlier
# ones. Outputs are suffixed when SEED_START > 0 so completed seeds are never overwritten.
SEED_START = int(os.environ.get('HCA_SEED_START', 0))
# HCA_RANK=1 turns the paper's variance hierarchy ON for the two SpectralViT arms. Off by
# default, which is how every HCA result on disk was produced. When set, those arms are
# RENAMED ('+w') and read the frequency-major features, because omega_i = 1/i is only a
# variance hierarchy if index i is variance rank -- block-major order would downweight whole
# spatial blocks instead of high frequencies. The rename is the safety property: cells are
# keyed by (Seed, Fold, Model), so a '+w' arm can never be merged with its omega-off twin by
# a combiner, whatever suffix the run uses.
RANKON = os.environ.get('HCA_RANK') == '1'
INNER_K = int(os.environ.get('HCA_INNER_K', 3))   # inner folds for capacity selection
# Results also land in the repo's gitignored results/ so they are readable without
# reaching onto spacelink. Summaries only; volumes and caches stay on spacelink.
MIRROR = os.environ.get('HCA_MIRROR', os.path.expanduser('~/SpectralViT/results/sweep'))
SUF = '' if SEED_START == 0 else f'_s{SEED_START}'
if os.environ.get('HCA_SUFFIX'): SUF = os.environ['HCA_SUFFIX']
# Belt and braces: even with no HCA_SUFFIX given, an omega-on run gets its own files.
if RANKON and not os.environ.get('HCA_SUFFIX'): SUF = '_rankon' + SUF
os.makedirs(OUT, exist_ok=True)
if MIRROR:
    try:
        os.makedirs(MIRROR, exist_ok=True)
    except OSError:
        MIRROR = ''


def save(df, name, **kw):
    df.to_csv(os.path.join(OUT, name), **kw)
    if MIRROR:
        # The mirror is a convenience copy on a quota-limited filesystem. A failure here
        # once truncated every per-cell CSV and killed the run, so it must never raise.
        try:
            df.to_csv(os.path.join(MIRROR, name), **kw)
        except OSError:
            pass


dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------------------------------------------------- data
meta = np.load(PREFIX + '_meta.npz', allow_pickle=True)
n_valid, VOL = int(meta['n']), int(meta['vol'])
# Read the volumes into RAM rather than leaving them memmapped on spacelink. The array
# is n_valid x 96^3 float32 -- 3.5 GB at n=1000, against the 32 GB the job requests --
# while the memmap re-reads every volume from the network on every step: BATCH=4 over
# HCA_EPOCHS=500 is on the order of a terabyte of reads per fit, which held GPU
# utilisation at ~5% and put the n=1000 tasks on track for 118 h against a 120 h wall.
# HCA_MMAP=1 restores the old behaviour, and anything larger than HCA_RAM_GB stays
# memmapped so a bigger cache cannot push the job into swap.
_raw = np.load(PREFIX + '.npy', mmap_mode='r')
_need = _raw[:n_valid].nbytes
if os.environ.get('HCA_MMAP') == '1' or _need > float(os.environ.get('HCA_RAM_GB', 24)) * 2 ** 30:
    Xall = _raw[:n_valid]
    print(f'volumes left memmapped ({_need / 2 ** 30:.1f} GB)', flush=True)
else:
    Xall = np.ascontiguousarray(_raw[:n_valid])
    print(f'volumes loaded into RAM ({_need / 2 ** 30:.1f} GB)', flush=True)
yall = meta['y'][:n_valid]
print(f'cache {PREFIX}: {n_valid} subjects, {VOL}^3, labels {np.bincount(yall)} '
      f'for {list(meta["classes"])}')

if N > n_valid:
    sys.exit(f'requested n={N} but only {n_valid} subjects are cached')
def draw(seed):
    """Stratified subsample of N subjects. The draw varies with seed so the
    reported spread covers subject sampling, not just initialization."""
    rng = np.random.RandomState(seed)
    idx = []
    for c in np.unique(yall):
        pool = np.where(yall == c)[0]
        take = int(round(N * len(pool) / len(yall)))
        idx += list(rng.choice(pool, size=min(take, len(pool)), replace=False))
    idx = np.array(sorted(idx))[:N]
    return np.asarray(Xall[idx], dtype=np.float32), yall[idx]


# ---------------------------------------------------------------- bases
def block_spectra(vols, kind):
    """Per-block low-frequency features. 'dct' keeps sign; 'fft' keeps magnitude only."""
    B = len(vols)
    bz, by, bx = [s // GRID for s in vols.shape[1:]]
    nb, nf = GRID ** 3, FREQ ** 3
    out = np.zeros((B, nb, nf), dtype=np.float32)
    for i in range(B):
        v, bi = vols[i], 0
        for z in range(GRID):
            for yy in range(GRID):
                for xx in range(GRID):
                    blk = v[z*bz:(z+1)*bz, yy*by:(yy+1)*by, xx*bx:(xx+1)*bx]
                    if kind == 'dct':
                        c = fft.dctn(blk, type=2, norm='ortho')[:FREQ, :FREQ, :FREQ].ravel()
                        out[i, bi] = np.sign(c) * np.log1p(np.abs(c))
                    else:
                        m = np.abs(fft.fftn(blk))[:FREQ, :FREQ, :FREQ].ravel()
                        out[i, bi] = np.log1p(m)
                    bi += 1
    return out.reshape(B, nb * nf)


def rank_perm(f=None):
    """Reorder block-major features to frequency-major, frequencies sorted by |k|.

    Rank weights are 1/i over the flat feature vector, so index order must be variance
    order. Block-major order would downweight whole spatial blocks instead of high
    frequencies, which carries no variance interpretation.
    """
    f = FREQ if f is None else f
    nb, nf = GRID ** 3, f ** 3
    kz, ky, kx = np.unravel_index(np.arange(nf), (f, f, f))
    order = np.argsort(kz ** 2 + ky ** 2 + kx ** 2, kind='stable')
    return np.array([b * nf + order[r] for r in range(nf) for b in range(nb)])


PERM = rank_perm()
INV_PERM = np.argsort(PERM)


def build_inputs(V):
    t0 = time.time()
    F = {k: torch.tensor(block_spectra(V, k)) for k in ('dct', 'fft')}
    Vt = torch.tensor(V).unsqueeze(1)              # (B,1,D,H,W) for spatial arms
    print(f'features built in {time.time()-t0:.0f}s; {F["dct"].shape}')
    return dict(F, dct_rank=F['dct'][:, PERM], fft_rank=F['fft'][:, PERM], vol=Vt,
                vol32=torch.nn.functional.interpolate(Vt, size=32))


# ------------------------------------------------- wrapper + training (unchanged)
class ClassificationWrapper(nn.Module):
    """Wraps regression/feature nets for multi-class classification."""
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model, self.num_classes, self.classifier = base_model, num_classes, None

    def forward(self, x):
        out = self.base_model(x)
        if out.ndim > 2:  out = out.mean(dim=1)
        elif out.ndim == 1: out = out.unsqueeze(1)
        if out.shape[1] == self.num_classes: return out
        if self.classifier is None or self.classifier.in_features != out.shape[1]:
            self.classifier = nn.Linear(out.shape[1], self.num_classes).to(out.device)
        return self.classifier(out)


# Preemptible partitions requeue rather than kill, and a single n=1000 cell is ~63 h, so
# a restart that recomputed from fold 0 would lose the run. Completed cells live in the
# CSV already; the per-fold predictions behind the pooled AUC do not, so they are
# checkpointed beside it. Without them a resumed run would silently report a pooled AUC
# over only the folds it happened to recompute.
CKPT = os.path.join(OUT, f'hca_ckpt_n{N}{SUF}.npz')


def ckpt_load():
    """(rows already computed, per-cell held-out predictions) from a previous attempt."""
    rowf = os.path.join(OUT, f'hca_sweep_n{N}{SUF}.csv')
    if not (os.path.exists(CKPT) and os.path.exists(rowf)):
        return [], {}
    store = {k: np.load(CKPT)[k] for k in np.load(CKPT).files}
    done = pd.read_csv(rowf).to_dict('records')
    # Keep only rows whose predictions survived, so the two never disagree.
    keep = [r for r in done
            if f"{r['Seed']}|{r['Fold']}|{r['Model']}|y" in store
            or not np.isfinite(r.get('AUC', float('nan')))]
    print(f'resuming: {len(keep)} cells already done in {rowf}', flush=True)
    return keep, store


def ckpt_save(store):
    tmp = CKPT + '.tmp.npz'
    np.savez_compressed(tmp, **store)
    os.replace(tmp, CKPT)


def auc2(y_true, p_prob):
    if len(np.unique(y_true)) < 2: return np.nan
    return roc_auc_score(y_true, p_prob[:, 1])


def train_eval(model, loader, val, epochs, lr, class_weights):
    model.eval()                                   # build the lazy head before AdamW
    with torch.no_grad():
        px, _ = next(iter(loader)); model(px.to(dev))
    model.train()
    # A linear model's capacity knob is regularisation, not parameter count, so the
    # logistic-regression arm varies weight decay where the others vary width. Read off
    # the wrapped module so no plumbing changes for arms that keep the default.
    wd = getattr(getattr(model, 'base_model', model), '_wd', 0.05)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sch = OneCycleLR(opt, max_lr=lr, steps_per_epoch=len(loader), epochs=epochs)
    crit = nn.CrossEntropyLoss(weight=class_weights)
    # Fixed budget, no checkpoint selection. The only held-out data is the test fold, so
    # early stopping on it would pick the epoch that best fits what is being reported. A
    # validation slice is unavailable at n=10, where leave-one-out leaves 9 subjects.
    # Running the full budget also lets OneCycleLR complete, as it is designed to.
    for ep in range(1, epochs + 1):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(dev), by.to(dev, dtype=torch.long)
            opt.zero_grad(); o = model(bx); l = crit(o, by)
            if not torch.isfinite(l): continue
            l.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step()
    model.eval()
    with torch.no_grad():
        vx, vy = val
        p = (model(vx.to(dev)) if vx.ndim < 5
             else torch.cat([model(c.to(dev)) for c in torch.split(vx, 2)]))
        pp = torch.softmax(p, 1).cpu().numpy(); pc = np.argmax(pp, 1)
    return dict(Accuracy=accuracy_score(vy, pc),
                Macro_F1=f1_score(vy, pc, average='macro', zero_division=0),
                Balanced_Acc=balanced_accuracy_score(vy, pc),
                AUC=auc2(vy, pp)), pp


# ---------------------------------------------------------------- arms
def spectral_hcp():
    net = SpectralViT(n_inputs=NFEAT, patch_size=FREQ ** 3, embed_dim=32, n_layers=2,
                      n_heads=2, use_rank_weights=False, use_pos_embed=False)
    net.use_pos_embed = True
    net.pos_embed = nn.Parameter(torch.randn(64, 1, 32) * 0.02)
    net.mlp_head = nn.Identity()
    return net


def spectral_small():
    """Half the width and one layer instead of two -- the DBS capacity, not its head.

    use_linear_residual is deliberately omitted: it adds linear_residual(raw) to the
    logit, which presumes mlp_head has reduced to a scalar. Here mlp_head is Identity so
    the wrapper can build the 2-class head from pooled features, and the residual would
    be added to a (B, embed_dim) tensor instead. Porting it would mean redesigning the
    classification head, so only the capacity difference is tested.
    """
    net = SpectralViT(n_inputs=NFEAT, patch_size=FREQ ** 3, embed_dim=16, n_layers=1,
                      n_heads=2, use_rank_weights=False, use_pos_embed=False)
    net.use_pos_embed = True
    net.pos_embed = nn.Parameter(torch.randn(64, 1, 16) * 0.02)
    net.mlp_head = nn.Identity()
    return net


def spectral_rank():
    """SpectralViT with 1/i rank weights on frequency-ordered tokens."""
    net = SpectralViT(n_inputs=NFEAT, patch_size=FREQ ** 3, embed_dim=32, n_layers=2,
                      n_heads=2, use_rank_weights=True, use_pos_embed=False)
    net.mlp_head = nn.Identity()
    return net


def svit(embed, rank=False, layers=2, learnable=True):
    def f(nf=NFEAT):
        net = SpectralViT(n_inputs=nf, patch_size=nf // GRID ** 3, embed_dim=embed, n_layers=layers,
                          n_heads=2, use_rank_weights=rank, use_pos_embed=False,
                          learnable_rank_weights=learnable)
        # Rank weights and positional embeddings are NOT alternatives. One is a scalar gain
        # on the raw coefficients before the input projection; the other is an additive
        # d_e-dimensional code on the embedded tokens after it. Neither expresses the other,
        # and the methods section says omega "complements" the positional embedding. The
        # original `if not rank` made them exclusive, so every rank-vs-no-rank comparison
        # toggled two factors at once and could attribute nothing. Under HCA_RANK=1 both are
        # on, which is the cell the 2x2 was missing.
        if (not rank) or RANKON:
            net.use_pos_embed = True
            net.pos_embed = nn.Parameter(torch.randn(64, 1, embed) * 0.02)
        net.mlp_head = nn.Identity()
        return net
    return f


# Every arm carries a capacity grid chosen by internal cross-validation on the training
# split alone. Logistic regression has no capacity knob, so its grid is a singleton; the
# selection rule is still the same one applied to every arm.
def _wd(m, w):
    m._wd = w
    return m


WD = [0.005, 0.05, 0.5]   # logistic regression's grid: regularisation, not width
EMB = [16, 32, 64]
HID = [2, 6, 16]        # floor lowered so the MLP grid overlaps the others
ARMS = []
# Features are standardised per fold by default: the IXI pipeline whitens via
# PCA(whiten=True), while block_spectra only log-compresses, so without this the spectral
# arms are the one place features reach a model unstandardised -- and AttentionUNet gets
# the equivalent for free from BatchNorm3d.
for tag, key in [('Block-DCT', 'dct:std'), ('Block-|FFT|', 'fft:std')]:
    ARMS += [
        (f'SpectralViT hcp{" +w+p" if RANKON else ""} ({tag})',
         [(f'embed {e}', svit(e, layers=2, rank=RANKON)) for e in EMB],
         key.replace(':std', '_rank:std') if RANKON else key),
        (f'SpectralViT small{" +w+p" if RANKON else ""} ({tag})',
         [(f'embed {e}', svit(e, layers=1, rank=RANKON)) for e in EMB],
         key.replace(':std', '_rank:std') if RANKON else key),
        (f'LogisticRegression ({tag})',
         [(f'wd {w:g}', (lambda nf=NFEAT, w=w: _wd(LogisticRegression(in_features=nf), w)))
          for w in WD], key),
        (f'MLP ({tag})',
         [(f'hidden {h}', (lambda nf=NFEAT, h=h: MultiLayerPerceptron(input_dim=nf,
                                                            hidden_dim=h))) for h in HID], key),
    ]
ARMS += [
    ('SpatialViT (Micro)',
     [(f'patch {p}', (lambda p=p: SpatialViT(vol_size=32, patch_size=p, embed_dim=32,
                                             n_layers=1, n_heads=2))) for p in (4, 8, 16)],
     'vol32'),
    ('AttnUNet (Micro)',
     [(f'{c} ch', (lambda c=c: AttentionUNet(in_channels=1, base_channels=c)))
      for c in (4, 8, 16)], 'vol'),
    # Vary embed, not window_size: the window does not change the parameter count here, so
    # the old grid gave only two distinct capacities and capped Swin far below the others.
    # window_size must divide the token grid (img_size // patch_size = 12), so 6 is kept.
    ('Swin (Micro)',
     [(f'embed {e}', (lambda e=e: SwinTransformer(img_size=VOL, embed_dim=e,
                                                  window_size=6))) for e in EMB], 'vol'),
]
# Appended last so the arms above are unaffected. The only arms with the paper's
# variance-hierarchy weighting enabled; they read frequency-major features so that rank i
# is frequency rank.
ARMS += [('SpectralViT rank (Block-DCT)',
          [(f'embed {e}', svit(e, rank=True)) for e in EMB], 'dct_rank:std')]
# Fixed 1/i weights are a buffer rather than a parameter, so this arm adds none of the
# 4096 weights the learnable version does. It separates the variance hierarchy from the
# capacity increase that comes with it.
ARMS += [('SpectralViT fixrank (Block-DCT)',
          [(f'embed {e}', svit(e, rank=True, learnable=False)) for e in EMB], 'dct_rank:std')]
# Same model and basis as the rank arm, but WITHOUT per-feature standardisation: the key
# is 'dct_rank' rather than 'dct_rank:std', and only the ':std' suffix triggers it. An
# earlier version of this comment said the opposite. Appended so arms above are unaffected.
ARMS += [('SpectralViT rank no-std (Block-DCT)',
          [(f'embed {e}', svit(e, rank=True)) for e in EMB], 'dct_rank')]
# De-confounds the rank-weight ablation. The three arms above read 'dct_rank', which is
# frequency-major, while the reported arms read 'dct:std', which is block-major. Every
# coefficient gets its own phi_i and positional embedding, so the ORDER of the feature
# vector matters on its own, independently of omega. The learnable-rank arm therefore beats
# the reported arm by +0.07 to +0.10 AUC for two reasons at once and the ablation cannot
# separate them. This arm is the missing cell: frequency-major features, rank weights OFF.
#   near the rank arms  -> the permutation did the work; nothing needs rerunning with omega
#   near the hcp arm    -> the learnable hierarchy is real and worth the decision
# Cheap at n <= 100; there is no reason to run it at n=1000 to answer this.
ARMS += [('SpectralViT perm no-rank (Block-DCT)',
          [(f'embed {e}', svit(e, rank=False)) for e in EMB], 'dct_rank:std')]

# The configuration the methods text actually describes: BOTH mechanisms, each matched to
# the structure it can speak about. Tokens stay block-major, so one token is one spatial
# block and the positional embedding encodes where in the volume it sits -- the spatial
# ViT's use of it, and meaningful. The 1/i profile then RESTARTS inside each block, so it
# is a frequency hierarchy within a block rather than a claim that every coefficient of
# block 1 outranks every coefficient of block 2 -- which is not a statement either ordering
# makes. That is the same argument rank_groups makes for concatenated bases in
# MultiClassSpectralViT; SpectralViT has no rank_groups parameter, so the profile is built
# here and assigned, the way pos_embed already is above.
def svit_grouped(embed, layers=2):
    def f(nf=NFEAT):
        nb = GRID ** 3
        per = nf // nb
        net = SpectralViT(n_inputs=nf, patch_size=per, embed_dim=embed, n_layers=layers,
                          n_heads=2, use_rank_weights=True, use_pos_embed=False,
                          learnable_rank_weights=True)
        ranks = torch.cat([torch.arange(1, per + 1, dtype=torch.float32) for _ in range(nb)])
        assert len(ranks) == nf, (len(ranks), nf)
        net.rank_weights = nn.Parameter(1.0 / ranks)
        net.use_pos_embed = True
        net.pos_embed = nn.Parameter(torch.randn(nb, 1, embed) * 0.02)
        net.mlp_head = nn.Identity()
        return net
    return f


ARMS += [('SpectralViT grouped (Block-DCT)',
          [(f'embed {e}', svit_grouped(e)) for e in EMB], 'dct:std')]

# Equation 2 as written: one embedding vector per coefficient, W_phi in R^{n x d_e}, which
# is use_mode_weights. Every other experiment implements this; HCA does not, because with
# FREQ=4 it means 4,096 tokens instead of 64. sec:nsens reports the arm varies by only
# 0.030-0.041 pooled AUC across an eightfold range of coefficient counts, so HCA_FREQ=2
# (512 coefficients, 512 tokens) is a defensible setting at which Eq. 2 may be affordable.
# This arm exists to measure that: the CSV's `secs` column answers whether it is feasible,
# and the AUC answers whether it is worth it. Note it must NOT reuse the 64-row pos_embed
# override the other arms apply -- with one token per coefficient the sequence is n_inputs
# long, so SpectralViT builds its own positional embedding at the right length.
def svit_phi(embed, layers=2):
    def f(nf=NFEAT):
        net = SpectralViT(n_inputs=nf, patch_size=1, embed_dim=embed, n_layers=layers,
                          n_heads=2, use_rank_weights=False, use_mode_weights=True,
                          use_pos_embed=True)
        net.mlp_head = nn.Identity()
        return net
    return f


ARMS += [('SpectralViT phi (Block-DCT)',
          [(f'embed {e}', svit_phi(e)) for e in EMB], 'dct:std')]
# Run only arms whose name contains one of these substrings, so a changed or added arm
# can be backfilled across every seed without recomputing the ones already on disk.
# Outputs take a suffix when filtering, keeping the completed runs intact.
ONLY = [a for a in os.environ.get('HCA_ARMS', '').split(',') if a]
if ONLY:
    ARMS = [a for a in ARMS if any(o in a[0] for o in ONLY)]
    if not ARMS: sys.exit(f'HCA_ARMS matched no arms: {ONLY}')
if SMOKE: ARMS = ARMS[:2] + ARMS[-3:]

# Optional arm filter, so one array task can run a subset of the models instead of all 14.
# At n=1000 a fold costs 13.4 h and a seed 67 h, which puts a five-fold seed within 53 h of
# the 120 h wall when it runs alone and past the wall when two share a node. Splitting by
# arm caps the longest job at Swin's share -- about 12 h for five folds -- and removes the
# timeout risk entirely. Unset, the behaviour is exactly as before, so running jobs and
# every result already on disk are unaffected. Comma-separated substrings, matched
# CASE-SENSITIVELY: two filters are applied in sequence and the first one (above, at the
# ONLY/ARMS block) is case-sensitive, so a lowercase pattern exits with 'matched no arms'
# before anything is written, however the second filter is spelled. Use the arm names as
# they appear, e.g. 'SpectralViT rank (Block-DCT)', not 'spectralvit rank'.
#   HCA_ARMS='swin'                      one arm
#   HCA_ARMS='logistic,mlp'              the cheap arms together
#   HCA_ARMS='spectralvit hcp,spatialvit'  the pair table:hca-cls actually compares
# Each subset MUST get its own HCA_SUFFIX, or concurrent tasks race on the same CSV and
# checkpoint; the combiner then has to merge per-arm files and check completeness by
# (arm, fold) rather than by row count.
_only = os.environ.get('HCA_ARMS', '').strip()
if _only:
    _pats = [q.strip().lower() for q in _only.split(',') if q.strip()]
    _keep = [a for a in ARMS if any(q in a[0].lower() for q in _pats)]
    if not _keep:
        sys.exit(f'HCA_ARMS={_only!r} matched none of: ' + ', '.join(a[0] for a in ARMS))
    print(f'arm filter {_pats}: {len(_keep)} of {len(ARMS)} arms -> '
          + ', '.join(a[0] for a in _keep), flush=True)
    ARMS = _keep


scheme = 'leave-one-out' if N <= 10 else f'{2 if SMOKE else 5}-fold'
print(f'\nCV: {scheme}; {len(ARMS)} arms; epochs {EPOCHS}; '
      f'{N_SEEDS} seeds; device {dev}\n')

def standardize(x, tr):
    """Per-feature standardisation fit on the training rows only.

    SpectralViT applies no input normalisation, while AttentionUNet has BatchNorm3d and
    normalises internally. The block-DCT features span ~3 orders of magnitude -- log1p of
    a 24^3 block's DC term lands near 10, high frequencies near 0.01 -- so the spectral
    arms see badly conditioned inputs the convolutional baseline does not. AUC is
    invariant to monotone rescaling and barely notices; MAE needs calibrated magnitudes
    and does. This arm tests whether that asymmetry explains the regression gap.
    """
    mu = x[tr].mean(0, keepdim=True)
    sd = x[tr].std(0, keepdim=True) + 1e-6
    return (x - mu) / sd


def select_capacity(name, builds, x, y, tr, seed, featarm=False, base_key='dct'):
    """Pick a capacity by inner cross-validated AUC on the training split alone.

    A single inner holdout made selection arbitrary: the capacities perform within noise
    of one another, so the top choice won only 36-48% of folds even at n=1000. Averaging
    over INNER_K folds gives a spread, and the one-standard-error rule then takes the
    smallest capacity statistically indistinguishable from the best, so ties resolve to
    parsimony rather than noise. Builds are ordered smallest-first.
    """
    # Feature arms also select the coefficient count; volume arms never see these
    # features, so their grid is unchanged. Candidates run smallest-first in both axes,
    # so the one-standard-error rule resolves ties toward fewer coefficients.
    freqs = FREQS if featarm else [FREQ]
    cands = [((lbl if len(freqs) == 1 else f'{lbl}, {nfeat(f)} coef'), b, f)
             for f in freqs for lbl, b in builds]
    if len(cands) == 1:
        return cands[0]
    strat = np.bincount(y[tr], minlength=2).min() >= INNER_K
    inner = (StratifiedKFold(n_splits=INNER_K, shuffle=True, random_state=seed) if strat
             else KFold(n_splits=INNER_K, shuffle=True, random_state=seed))
    scores = {lbl: [] for lbl, _, _ in cands}
    for itr_i, iva_i in inner.split(np.zeros(len(tr)), y[tr]):
        itr, iva = tr[itr_i], tr[iva_i]
        icw = torch.tensor(len(y[itr]) / (2 * np.bincount(y[itr], minlength=2) + 1e-9),
                           dtype=torch.float32, device=dev)
        for lbl, b, f in cands:
            try:
                xf = trunc(x, f, base_key)
                m = ClassificationWrapper(b(nfeat(f)) if featarm else b(),
                                          NUM_CLASSES).to(dev)
                ld = DataLoader(TensorDataset(xf[itr], torch.tensor(y[itr])),
                                batch_size=BATCH, shuffle=True)
                v = train_eval(m, ld, (xf[iva], y[iva]), EPOCHS, LR, icw)[0]['AUC']
                del m, ld; torch.cuda.empty_cache()
            except Exception:
                traceback.print_exc(); v = float('nan')
            scores[lbl].append(v)
    means = {l: np.nanmean(v) for l, v in scores.items()}
    if all(m != m for m in means.values()):
        return cands[0]
    bestl = max(means, key=lambda l: means[l])                 # higher AUC is better
    se = np.nanstd(scores[bestl], ddof=1) / np.sqrt(INNER_K) if INNER_K > 1 else 0.0
    thresh = means[bestl] - (se if se == se else 0.0)
    for lbl, b, f in cands:                                    # smallest within one SE
        if means[lbl] == means[lbl] and means[lbl] >= thresh:
            return (lbl, b, f)
    return (bestl, dict(builds)[bestl])


pooled = []
rows, STORE = ckpt_load()
DONE = {(r['Seed'], r['Fold'], r['Model']) for r in rows}

for seed in range(SEED_START, N_SEEDS):
    torch.manual_seed(seed); np.random.seed(seed)
    V, y = draw(seed)
    print(f'seed {seed}: n={len(V)} sampled, labels {np.bincount(y)}')
    INPUT = build_inputs(V); del V
    cv = (LeaveOneOut() if N <= 10 else
          StratifiedKFold(n_splits=2 if SMOKE else 5, shuffle=True, random_state=seed))
    pool = {}
    for fold, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y)):
        # Weights come from the training split only; computing them on all of y leaks the
        # held-out label, which matters most at n=10 where one subject is 10% of it.
        cw = torch.tensor(len(y[tr]) / (2 * np.bincount(y[tr], minlength=2) + 1e-9),
                          dtype=torch.float32, device=dev)
        for name, builds, key in ARMS:
            if (seed, fold, name) in DONE:
                ky, kp = f'{seed}|{fold}|{name}|y', f'{seed}|{fold}|{name}|p'
                if ky in STORE:          # NaN arms were stored without predictions
                    pool.setdefault(name, ([], []))[0].append(STORE[ky])
                    pool[name][1].append(STORE[kp])
                continue
            base_key, _, std_flag = key.partition(':')
            x = INPUT[base_key]
            if std_flag == 'std': x = standardize(x, tr)
            t = time.time()
            featarm = not base_key.startswith('vol')
            cap, build, f_sel = select_capacity(name, builds, x, y, tr, seed, featarm,
                                                base_key)
            x = trunc(x, f_sel, base_key) if featarm else x
            model = ClassificationWrapper(build(nfeat(f_sel)) if featarm else build(),
                                          NUM_CLASSES).to(dev)
            loader = DataLoader(TensorDataset(x[tr], torch.tensor(y[tr])),
                                batch_size=BATCH, shuffle=True)
            try:
                res, pp = train_eval(model, loader, (x[te], y[te]), EPOCHS, LR, cw)
            except Exception:
                # One broken arm should not cost the other 54 cells. The row is kept as NaN so
                # a dropped arm cannot be mistaken for a missing one.
                traceback.print_exc()
                res = dict(Accuracy=float('nan'), Macro_F1=float('nan'),
                           Balanced_Acc=float('nan'), AUC=float('nan'))
                pp = None
            if pp is not None:                     # pooled across folds, see below
                pool.setdefault(name, ([], []))[0].append(y[te])
                pool[name][1].append(pp[:, 1])
                STORE[f'{seed}|{fold}|{name}|y'] = y[te]
                STORE[f'{seed}|{fold}|{name}|p'] = pp[:, 1]
            res.update(n=N, Seed=seed, Fold=fold, Model=name, Capacity=cap,
                       Params=sum(p.numel() for p in model.parameters()),
                       secs=round(time.time() - t, 1))
            rows.append(res)
            print(f'  n={N} s{seed} f{fold} | {name:28s} | {cap:14s} | Acc {res["Accuracy"]:.3f} '
                  f'F1 {res["Macro_F1"]:.3f} BalAcc {res["Balanced_Acc"]:.3f} '
                  f'AUC {res["AUC"]:.3f} | {res["secs"]:.0f}s', flush=True)
            del model, loader; torch.cuda.empty_cache()
            save(pd.DataFrame(rows), f'hca_sweep_n{N}{SUF}.csv', index=False)
            ckpt_save(STORE)           # written after the row, so the CSV never leads
    # One AUC per seed over every held-out subject. Leave-one-out folds hold a single
    # subject, so a per-fold AUC is undefined; pooling the predictions is what the DBS
    # analysis does and is the only way n=10 yields an AUC at all.
    for name, (ys, ps) in pool.items():
        yy, pv = np.concatenate(ys), np.concatenate(ps)
        pooled.append(dict(n=N, Seed=seed, Model=name, n_pooled=len(yy),
                           AUC_pooled=(roc_auc_score(yy, pv)
                                       if len(np.unique(yy)) > 1 else float('nan'))))
    save(pd.DataFrame(pooled), f'hca_pooled_n{N}{SUF}.csv', index=False)
    del INPUT

MET = ['Accuracy', 'Macro_F1', 'Balanced_Acc', 'AUC']
d = pd.DataFrame(rows)
# Folds share subjects, so they are averaged within a seed; the seed is the replicate
# unit and the spread across seeds covers both the subject draw and initialization.
per_seed = d.groupby(['Model', 'Seed'])[MET].mean()
agg = per_seed.groupby('Model').agg(['mean', 'std'])
print(f'\n=== n={N} ({scheme}, {N_SEEDS} seeds) ===')
print(agg.round(3).sort_values(('AUC', 'mean'), ascending=False).to_string())
save(per_seed.round(3), f'hca_perseed_n{N}{SUF}.csv')
print(f'\nwrote {os.path.join(OUT, f"hca_sweep_n{N}.csv")}')
