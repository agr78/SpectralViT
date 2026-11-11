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
NUM_CLASSES = 1                        # single regression output
# FREQ is the per-block cube truncation, so the feature count is GRID^3 * FREQ^3:
# 512 / 1728 / 4096 for FREQ = 2 / 3 / 4. Overridable to test whether a linear
# head's advantage at large n comes from having 4096 coefficients to weight.
GRID = 4
FREQ = int(os.environ.get('HCA_FREQ', 4))
NFEAT = GRID ** 3 * FREQ ** 3
# The coefficient count is a capacity like any other, and IXI and DBS both select theirs
# by cross-validation. Listing more than one value here puts it in the inner-CV grid.
FREQS = sorted({int(v) for v in os.environ.get('HCA_FREQS', str(FREQ)).split(',')})


def nfeat(f):
    return GRID ** 3 * f ** 3


def trunc(x, f, key='dct'):
    """Low-frequency sub-cube of the FREQ-truncated features.

    The cube truncations nest, so no recomputation is needed, but the f-cube is not a
    prefix of the raveled vector: the block axis has to be restored before slicing. The
    'dct_rank' features are additionally permuted to frequency-major order by PERM, and
    the f-cube is not a prefix of that order either -- (0,0,3) sorts before (1,1,1) by
    |k| yet lies outside the 3-cube -- so those are un-permuted, truncated block-major,
    and re-permuted at the smaller size.
    """
    if f == FREQ:
        return x
    ranked = key.startswith('dct_rank')
    xb = x[:, INV_PERM] if ranked else x
    xb = xb.reshape(len(x), GRID ** 3, FREQ, FREQ, FREQ)[:, :, :f, :f, :f] \
           .reshape(len(x), nfeat(f))
    return xb[:, rank_perm(f)] if ranked else xb
OUT = os.environ.get('HCA_OUT', os.path.expanduser('~/spacelink/aHCP/derivatives/regress'))
N_SEEDS = int(os.environ.get('HCA_SEEDS', 5))   # matches the MNIST sweep
# Resume a partially finished sweep: run seeds [SEED_START, N_SEEDS). draw(seed) is
# independent per seed, so a later seed is reproducible without rerunning the earlier
# ones. Outputs are suffixed when SEED_START > 0 so completed seeds are never overwritten.
SEED_START = int(os.environ.get('HCA_SEED_START', 0))
INNER_K = int(os.environ.get('HCA_INNER_K', 3))   # inner folds for capacity selection
# Results also land in the repo's gitignored results/ so they are readable without
# reaching onto spacelink. Summaries only; volumes and caches stay on spacelink.
MIRROR = os.environ.get('HCA_MIRROR', os.path.expanduser('~/SpectralViT/results/regress'))
SUF = '' if SEED_START == 0 else f'_s{SEED_START}'
if os.environ.get('HCA_SUFFIX'): SUF = os.environ['HCA_SUFFIX']
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
Xall = np.load(PREFIX + '.npy', mmap_mode='r')[:n_valid]
yall = meta['y'][:n_valid]              # median-split group, used only to balance folds
aall = meta['age'][:n_valid].astype(np.float32)        # regression target, in years
print(f'cache {PREFIX}: {n_valid} subjects, {VOL}^3, age {aall.min():.0f}-{aall.max():.0f} '
      f'mean {aall.mean():.1f} sd {aall.std():.1f}')

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
    return np.asarray(Xall[idx], dtype=np.float32), yall[idx], aall[idx]


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
    return dict(F, dct_rank=F['dct'][:, PERM], vol=Vt,
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


def reg_metrics(y_true, y_pred):
    """MAE and RMSE in years; R2 and Pearson need >1 subject so are nan under LOO."""
    e = y_pred - y_true
    out = dict(MAE=float(np.mean(np.abs(e))), RMSE=float(np.sqrt(np.mean(e ** 2))))
    if len(y_true) > 1 and np.std(y_true) > 0:
        out['R2'] = float(1 - np.sum(e ** 2) / np.sum((y_true - y_true.mean()) ** 2))
        out['Pearson'] = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        out['R2'] = out['Pearson'] = float('nan')
    return out


def train_eval(model, loader, val, epochs, lr, tgt_mu, tgt_sd):
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
    crit = nn.L1Loss()                             # matches the reported MAE
    # Fixed budget, no checkpoint selection. The only held-out data is the test fold, so
    # early stopping on it would pick the epoch that best fits what is being reported. A
    # validation slice is unavailable at n=10, where leave-one-out leaves 9 subjects.
    # Running the full budget also lets OneCycleLR complete, as it is designed to.
    for ep in range(1, epochs + 1):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(dev), by.to(dev, dtype=torch.float32)
            opt.zero_grad(); o = model(bx).squeeze(-1); l = crit(o, by)
            if not torch.isfinite(l): continue
            l.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step()
    model.eval()
    with torch.no_grad():
        vx, vy = val
        p = (model(vx.to(dev)) if vx.ndim < 5
             else torch.cat([model(c.to(dev)) for c in torch.split(vx, 2)]))
        pred = p.squeeze(-1).cpu().numpy() * tgt_sd + tgt_mu      # back to years
    return reg_metrics(vy, pred), pred


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


def svit(embed, rank=False, layers=2, learnable=True):
    def f(nf=NFEAT):
        net = SpectralViT(n_inputs=nf, patch_size=nf // GRID ** 3, embed_dim=embed, n_layers=layers,
                          n_heads=2, use_rank_weights=rank, use_pos_embed=False,
                          learnable_rank_weights=learnable)
        if not rank:
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
        (f'SpectralViT hcp ({tag})',
         [(f'embed {e}', svit(e, layers=2)) for e in EMB], key),
        (f'SpectralViT small ({tag})',
         [(f'embed {e}', svit(e, layers=1)) for e in EMB], key),
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
# Same model and basis as the rank arm, with per-feature standardisation fit on the
# training fold. Appended last so every arm above is unaffected.
ARMS += [('SpectralViT rank no-std (Block-DCT)',
          [(f'embed {e}', svit(e, rank=True)) for e in EMB], 'dct_rank')]
# Predict the training mean. Without this floor, differences between trained models at
# small n cannot be distinguished from all of them failing to learn anything.
# Equation 2 as the methods write it: one embedding vector per coefficient rather than a
# shared projection over chunks of nf/GRID^3. At FREQ=4 that is 4,096 tokens instead of
# 64, which is why the arms above chunk; on the classification sweep the same substitution
# gained 0.027-0.088 pooled AUC, so it is worth measuring here. Appended last, so every
# arm above is byte-identical and the completed runs stay valid.
# use_rank_weights stays off, matching hca_sweep.py's phi arm, and that is not a choice
# to omit the hierarchy: with one embedding vector per coefficient the forward pass is
# s_i * omega_i * W_phi_i, and W_phi_i is a free d_e vector, so omega_i is absorbed into
# it. The flag changes only the initialization, from randn*0.02 to (1/i)*randn*0.02. The
# rank weight reweights for real only in the chunked arms, where it multiplies before a
# projection shared across chunks and cannot be absorbed.
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

ARMS += [('MeanBaseline', [('--', None)], 'dct')]
# Run only arms whose name contains one of these substrings, so a changed or added arm
# can be backfilled across every seed without recomputing the ones already on disk.
# Outputs take a suffix when filtering, keeping the completed runs intact.
ONLY = [a for a in os.environ.get('HCA_ARMS', '').split(',') if a]
if ONLY:
    ARMS = [a for a in ARMS if any(o in a[0] for o in ONLY)]
    if not ARMS: sys.exit(f'HCA_ARMS matched no arms: {ONLY}')
if SMOKE: ARMS = ARMS[:2] + ARMS[-3:]

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


RECAL = os.environ.get('HCA_RECAL') == '1'


def recalibrate(oof, pred):
    """Least-squares affine map from out-of-fold predictions to years.

    The spectral arm with one embedding vector per coefficient orders subjects well before
    it predicts their age correctly: at n=25 the per-fold correlation averages 0.82 while
    the MAE exceeds the training-mean floor on a quarter of folds. Two parameters fitted
    on held-out training predictions fix the scale without seeing a test label.
    """
    if oof is None or pred is None:
        return pred
    y, p = oof
    ok = np.isfinite(y) & np.isfinite(p)
    if ok.sum() < 3 or np.std(p[ok]) < 1e-6:
        return pred
    g, c = np.polyfit(p[ok], y[ok], 1)
    return g * np.asarray(pred, np.float32) + c


def select_capacity(builds, x, a, y, tr, seed, featarm=False, base_key='dct'):
    """Pick a capacity by inner cross-validated MAE on the training split alone.

    A single inner holdout made selection arbitrary: the three capacities perform within
    noise of one another, so the top choice was picked only 36-48% of the time even at
    n=1000 where the inner set held 200 subjects. Averaging over INNER_K folds gives a
    spread, and the one-standard-error rule then takes the smallest capacity that is
    statistically indistinguishable from the best, so ties resolve to parsimony instead
    of to noise. Builds are ordered smallest-first.
    """
    # The inner folds already train every candidate on held-out data; keeping those
    # predictions costs nothing and is the only honest basis for a calibration, since a
    # model with one embedding vector per coefficient is perfectly calibrated on its own
    # training targets and recalibrating there would be a no-op.
    expand = featarm and all(b is not None for _, b in builds)
    freqs = FREQS if expand else [FREQ]
    cands = [((lbl if len(freqs) == 1 else f'{lbl}, {nfeat(f)} coef'), b, f)
             for f in freqs for lbl, b in builds]
    if len(cands) == 1:
        return cands[0] + (None,)
    strat = np.bincount(y[tr], minlength=2).min() >= INNER_K
    inner = (StratifiedKFold(n_splits=INNER_K, shuffle=True, random_state=seed) if strat
             else KFold(n_splits=INNER_K, shuffle=True, random_state=seed))
    scores = {lbl: [] for lbl, _, _ in cands}
    oof = {lbl: ([], []) for lbl, _, _ in cands}
    for itr_i, iva_i in inner.split(np.zeros(len(tr)), y[tr]):
        itr, iva = tr[itr_i], tr[iva_i]
        imu, isd = float(a[itr].mean()), float(a[itr].std() + 1e-8)
        for lbl, b, f in cands:
            try:
                xf = trunc(x, f, base_key)
                m = ClassificationWrapper(b(nfeat(f)) if expand else b(),
                                          NUM_CLASSES).to(dev)
                ld = DataLoader(TensorDataset(xf[itr],
                                              torch.tensor((a[itr] - imu) / isd,
                                                           dtype=torch.float32)),
                                batch_size=BATCH, shuffle=True)
                _r, _p = train_eval(m, ld, (xf[iva], a[iva]), EPOCHS, LR, imu, isd)
                v = _r['MAE']
                if _p is not None:
                    oof[lbl][0].append(a[iva]); oof[lbl][1].append(np.asarray(_p, np.float32))
                del m, ld; torch.cuda.empty_cache()
            except Exception:
                traceback.print_exc(); v = float('nan')
            scores[lbl].append(v)
    means = {l: np.nanmean(v) for l, v in scores.items()}
    if all(m != m for m in means.values()):
        return cands[0] + (None,)
    bestl = min(means, key=lambda l: means[l])                 # lower MAE is better
    se = np.nanstd(scores[bestl], ddof=1) / np.sqrt(INNER_K) if INNER_K > 1 else 0.0
    thresh = means[bestl] + (se if se == se else 0.0)
    def _oof(lbl):
        ys, ps = oof[lbl]
        return (np.concatenate(ys), np.concatenate(ps)) if ys else None
    for lbl, b, f in cands:                                    # smallest within one SE
        if means[lbl] == means[lbl] and means[lbl] <= thresh:
            return (lbl, b, f, _oof(lbl))
    best = next(c for c in cands if c[0] == bestl)
    return best + (_oof(bestl),)


rows, pooled = [], []
for seed in range(SEED_START, N_SEEDS):
    torch.manual_seed(seed); np.random.seed(seed)
    V, y, a = draw(seed)
    print(f'seed {seed}: n={len(V)} sampled, age {a.min():.0f}-{a.max():.0f} '
          f'mean {a.mean():.1f}')
    INPUT = build_inputs(V); del V
    cv = (LeaveOneOut() if N <= 10 else
          StratifiedKFold(n_splits=2 if SMOKE else 5, shuffle=True, random_state=seed))
    pool = {}
    for fold, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y)):
        # Weights come from the training split only; computing them on all of y leaks the
        # held-out label, which matters most at n=10 where one subject is 10% of it.
        # Target scaling from the training split only; predictions are mapped back to
        # years before scoring, so MAE and RMSE are in years throughout.
        mu, sd = float(a[tr].mean()), float(a[tr].std() + 1e-8)
        for name, builds, key in ARMS:
            t, model = time.time(), None
            # Resolve the input before selection so the probe sees the same tensor the
            # final fit will; standardisation is fit on tr, which contains the inner folds.
            base_key, _, std_flag = key.partition(':')
            x = INPUT[base_key]
            if std_flag == 'std': x = standardize(x, tr)
            featarm = not base_key.startswith('vol')
            cap, build, f_sel, oof = select_capacity(builds, x, a, y, tr, seed, featarm,
                                                     base_key)
            if build is not None and featarm:
                x = trunc(x, f_sel, base_key)
            if build is None:                      # predict the training mean
                pred = np.full(len(te), mu, dtype=np.float32)
                res, nparam = reg_metrics(a[te], pred), 0
            else:
                model = ClassificationWrapper(
                    build(nfeat(f_sel)) if featarm else build(),
                    NUM_CLASSES).to(dev)
                tgt = torch.tensor((a[tr] - mu) / sd, dtype=torch.float32)
                loader = DataLoader(TensorDataset(x[tr], tgt),
                                    batch_size=BATCH, shuffle=True)
                try:
                    res, pred = train_eval(model, loader, (x[te], a[te]),
                                           EPOCHS, LR, mu, sd)
                    if RECAL:
                        pred = recalibrate(oof, pred)
                        res = reg_metrics(a[te], pred)
                except Exception:
                    # One broken arm should not cost the rest. The row is kept as NaN so
                    # a dropped arm cannot be mistaken for a missing one.
                    traceback.print_exc()
                    res = dict(MAE=float('nan'), RMSE=float('nan'),
                               R2=float('nan'), Pearson=float('nan'))
                    pred = None
                nparam = sum(p.numel() for p in model.parameters())
            if pred is not None:                   # pooled across folds, see below
                pool.setdefault(name, ([], []))[0].append(a[te])
                pool[name][1].append(np.asarray(pred, dtype=np.float32))
            res.update(n=N, Seed=seed, Fold=fold, Model=name, Capacity=cap,
                       Params=nparam,
                       secs=round(time.time() - t, 1))
            rows.append(res)
            print(f'  n={N} s{seed} f{fold} | {name:28s} | {cap:14s} | MAE {res["MAE"]:.2f} '
                  f'RMSE {res["RMSE"]:.2f} R2 {res["R2"]:.3f} | {res["secs"]:.0f}s',
                  flush=True)
            if model is not None: del model
            torch.cuda.empty_cache()
            save(pd.DataFrame(rows), f'hca_reg_n{N}{SUF}.csv', index=False)
    # One AUC per seed over every held-out subject. Leave-one-out folds hold a single
    # subject, so a per-fold AUC is undefined; pooling the predictions is what the DBS
    # analysis does and is the only way n=10 yields an AUC at all.
    for name, (ys, ps) in pool.items():
        yy, pv = np.concatenate(ys), np.concatenate(ps)
        m = reg_metrics(yy, pv)
        pooled.append(dict(n=N, Seed=seed, Model=name, n_pooled=len(yy),
                           **{k + '_pooled': v for k, v in m.items()}))
    save(pd.DataFrame(pooled), f'hca_regpooled_n{N}{SUF}.csv', index=False)
    del INPUT

MET = ['MAE', 'RMSE', 'R2', 'Pearson']
d = pd.DataFrame(rows)
# Folds share subjects, so they are averaged within a seed; the seed is the replicate
# unit and the spread across seeds covers both the subject draw and initialization.
per_seed = d.groupby(['Model', 'Seed'])[MET].mean()
agg = per_seed.groupby('Model').agg(['mean', 'std'])
print(f'\n=== n={N} ({scheme}, {N_SEEDS} seeds) ===')
print(agg.round(3).sort_values(('MAE', 'mean')).to_string())
save(per_seed.round(3), f'hca_regperseed_n{N}{SUF}.csv')
print(f'\nwrote {os.path.join(OUT, f"hca_reg_n{N}.csv")}')
