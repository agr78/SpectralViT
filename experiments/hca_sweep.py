"""Data-efficiency sweep on HCP-Aging: spectral vs spatial tokenization by sample size.

usage: python experiments/hca_sweep.py <n> [prefix] [--smoke]
  --smoke runs 2 epochs and 2 folds to check shapes without spending GPU time"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repository root: shared modules and paths.py
from paths import DATA
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from networks import (SpectralViT, SpatialViT, AttentionUNet, SwinTransformer,
                      LogisticRegression, MultiLayerPerceptron)

N          = int(sys.argv[1])
PREFIX     = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') \
             else f'{DATA}/aHCP/derivatives/hca_vols_96'
SMOKE      = '--smoke' in sys.argv
EPOCHS     = 2 if SMOKE else int(os.environ.get('HCA_EPOCHS', '500'))
LR         = 1e-3
BATCH      = 4
NUM_CLASSES = 2
GRID = 4
FREQ = int(os.environ.get('HCA_FREQ', 4))
NFEAT = GRID ** 3 * FREQ ** 3
FREQS = sorted({int(v) for v in os.environ.get('HCA_FREQS', str(FREQ)).split(',')})


def nfeat(f):
    return GRID ** 3 * f ** 3


def trunc(x, f, key='dct'):
    if f == FREQ:
        return x
    ranked = key.startswith('dct_rank')
    xb = x[:, INV_PERM] if ranked else x
    xb = xb.reshape(len(x), GRID ** 3, FREQ, FREQ, FREQ)[:, :, :f, :f, :f] \
           .reshape(len(x), nfeat(f))
    return xb[:, rank_perm(f)] if ranked else xb
OUT = os.environ.get('HCA_OUT', f'{DATA}/aHCP/derivatives/sweep')
N_SEEDS = int(os.environ.get('HCA_SEEDS', 5))
SEED_START = int(os.environ.get('HCA_SEED_START', 0))
RANKON = os.environ.get('HCA_RANK') == '1'
INNER_K = int(os.environ.get('HCA_INNER_K', 3))
MIRROR = os.environ.get('HCA_MIRROR', os.path.expanduser('~/SpectralViT/results/sweep'))
SUF = '' if SEED_START == 0 else f'_s{SEED_START}'
if os.environ.get('HCA_SUFFIX'): SUF = os.environ['HCA_SUFFIX']
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
        try:
            df.to_csv(os.path.join(MIRROR, name), **kw)
        except OSError:
            pass


dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


meta = np.load(PREFIX + '_meta.npz', allow_pickle=True)
n_valid, VOL = int(meta['n']), int(meta['vol'])
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
    rng = np.random.RandomState(seed)
    idx = []
    for c in np.unique(yall):
        pool = np.where(yall == c)[0]
        take = int(round(N * len(pool) / len(yall)))
        idx += list(rng.choice(pool, size=min(take, len(pool)), replace=False))
    idx = np.array(sorted(idx))[:N]
    return np.asarray(Xall[idx], dtype=np.float32), yall[idx]


def block_spectra(vols, kind):
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
    Vt = torch.tensor(V).unsqueeze(1)
    print(f'features built in {time.time()-t0:.0f}s; {F["dct"].shape}')
    return dict(F, dct_rank=F['dct'][:, PERM], fft_rank=F['fft'][:, PERM], vol=Vt,
                vol32=torch.nn.functional.interpolate(Vt, size=32))


class ClassificationWrapper(nn.Module):
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


CKPT = os.path.join(OUT, f'hca_ckpt_n{N}{SUF}.npz')


def ckpt_load():
    rowf = os.path.join(OUT, f'hca_sweep_n{N}{SUF}.csv')
    if not (os.path.exists(CKPT) and os.path.exists(rowf)):
        return [], {}
    store = {k: np.load(CKPT)[k] for k in np.load(CKPT).files}
    done = pd.read_csv(rowf).to_dict('records')
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
    model.eval()
    with torch.no_grad():
        px, _ = next(iter(loader)); model(px.to(dev))
    model.train()
    wd = getattr(getattr(model, 'base_model', model), '_wd', 0.05)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sch = OneCycleLR(opt, max_lr=lr, steps_per_epoch=len(loader), epochs=epochs)
    crit = nn.CrossEntropyLoss(weight=class_weights)
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


def spectral_hcp():
    net = SpectralViT(n_inputs=NFEAT, patch_size=FREQ ** 3, embed_dim=32, n_layers=2,
                      n_heads=2, use_rank_weights=False, use_pos_embed=False)
    net.use_pos_embed = True
    net.pos_embed = nn.Parameter(torch.randn(64, 1, 32) * 0.02)
    net.mlp_head = nn.Identity()
    return net


def spectral_small():
    net = SpectralViT(n_inputs=NFEAT, patch_size=FREQ ** 3, embed_dim=16, n_layers=1,
                      n_heads=2, use_rank_weights=False, use_pos_embed=False)
    net.use_pos_embed = True
    net.pos_embed = nn.Parameter(torch.randn(64, 1, 16) * 0.02)
    net.mlp_head = nn.Identity()
    return net


def spectral_rank():
    net = SpectralViT(n_inputs=NFEAT, patch_size=FREQ ** 3, embed_dim=32, n_layers=2,
                      n_heads=2, use_rank_weights=True, use_pos_embed=False)
    net.mlp_head = nn.Identity()
    return net


def svit(embed, rank=False, layers=2, learnable=True):
    def f(nf=NFEAT):
        net = SpectralViT(n_inputs=nf, patch_size=nf // GRID ** 3, embed_dim=embed, n_layers=layers,
                          n_heads=2, use_rank_weights=rank, use_pos_embed=False,
                          learnable_rank_weights=learnable)
        if (not rank) or RANKON:
            net.use_pos_embed = True
            net.pos_embed = nn.Parameter(torch.randn(64, 1, embed) * 0.02)
        net.mlp_head = nn.Identity()
        return net
    return f


def _wd(m, w):
    m._wd = w
    return m


WD = [0.005, 0.05, 0.5]
EMB = [16, 32, 64]
HID = [2, 6, 16]
ARMS = []
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
    ('Swin (Micro)',
     [(f'embed {e}', (lambda e=e: SwinTransformer(img_size=VOL, embed_dim=e,
                                                  window_size=6))) for e in EMB], 'vol'),
]
ARMS += [('SpectralViT rank (Block-DCT)',
          [(f'embed {e}', svit(e, rank=True)) for e in EMB], 'dct_rank:std')]
ARMS += [('SpectralViT fixrank (Block-DCT)',
          [(f'embed {e}', svit(e, rank=True, learnable=False)) for e in EMB], 'dct_rank:std')]
ARMS += [('SpectralViT rank no-std (Block-DCT)',
          [(f'embed {e}', svit(e, rank=True)) for e in EMB], 'dct_rank')]
ARMS += [('SpectralViT perm no-rank (Block-DCT)',
          [(f'embed {e}', svit(e, rank=False)) for e in EMB], 'dct_rank:std')]

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
ONLY = [a for a in os.environ.get('HCA_ARMS', '').split(',') if a]
if ONLY:
    ARMS = [a for a in ARMS if any(o in a[0] for o in ONLY)]
    if not ARMS: sys.exit(f'HCA_ARMS matched no arms: {ONLY}')
if SMOKE: ARMS = ARMS[:2] + ARMS[-3:]

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
    mu = x[tr].mean(0, keepdim=True)
    sd = x[tr].std(0, keepdim=True) + 1e-6
    return (x - mu) / sd


def select_capacity(name, builds, x, y, tr, seed, featarm=False, base_key='dct'):
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
    bestl = max(means, key=lambda l: means[l])
    se = np.nanstd(scores[bestl], ddof=1) / np.sqrt(INNER_K) if INNER_K > 1 else 0.0
    thresh = means[bestl] - (se if se == se else 0.0)
    for lbl, b, f in cands:
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
        cw = torch.tensor(len(y[tr]) / (2 * np.bincount(y[tr], minlength=2) + 1e-9),
                          dtype=torch.float32, device=dev)
        for name, builds, key in ARMS:
            if (seed, fold, name) in DONE:
                ky, kp = f'{seed}|{fold}|{name}|y', f'{seed}|{fold}|{name}|p'
                if ky in STORE:
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
                traceback.print_exc()
                res = dict(Accuracy=float('nan'), Macro_F1=float('nan'),
                           Balanced_Acc=float('nan'), AUC=float('nan'))
                pp = None
            if pp is not None:
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
            ckpt_save(STORE)
    for name, (ys, ps) in pool.items():
        yy, pv = np.concatenate(ys), np.concatenate(ps)
        pooled.append(dict(n=N, Seed=seed, Model=name, n_pooled=len(yy),
                           AUC_pooled=(roc_auc_score(yy, pv)
                                       if len(np.unique(yy)) > 1 else float('nan'))))
    save(pd.DataFrame(pooled), f'hca_pooled_n{N}{SUF}.csv', index=False)
    del INPUT

MET = ['Accuracy', 'Macro_F1', 'Balanced_Acc', 'AUC']
d = pd.DataFrame(rows)
per_seed = d.groupby(['Model', 'Seed'])[MET].mean()
agg = per_seed.groupby('Model').agg(['mean', 'std'])
print(f'\n=== n={N} ({scheme}, {N_SEEDS} seeds) ===')
print(agg.round(3).sort_values(('AUC', 'mean'), ascending=False).to_string())
save(per_seed.round(3), f'hca_perseed_n{N}{SUF}.csv')
print(f'\nwrote {os.path.join(OUT, f"hca_sweep_n{N}.csv")}')
