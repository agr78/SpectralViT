"""IXI distribution-shift comparison: spectral ViT vs fine-tuned SFCN, nested selection.

Both models train on uncorrupted images only; Rician noise is applied to the held-out
fold at test time, so the comparison is zero-shot robustness to a shift in the test
distribution. Identical noise realisations go to both models, keeping the comparison
paired.

Two defects in the notebook version are fixed:
  * component count was selected by 5-fold CV over the whole dataset and then reported on
    that same KFold(random_state=0) partition -- selection measured on the test folds.
    Selection now runs inside each outer fold on its training split alone.
  * only the spectral side had a hyperparameter grid; SFCN ran at a fixed fine-tuning
    learning rate. Both sides now get a grid, chosen by the same rule.

Volumes are full size (160x192x160, the UK Biobank geometry the weights expect), so this
needs roughly 12 GB of RAM for the PCA training matrix.

usage: python sfcn_shift.py [--smoke]
env: SFCN_NII_DIR SFCN_CACHE_DIR SFCN_WEIGHTS SFCN_DP_MODEL SFCN_OUT
     SFCN_N SFCN_SEED SFCN_SUFFIX  (one cell of the sample-size curve)
"""
import os, sys, re, json, copy, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

NII   = os.environ.get('SFCN_NII_DIR',   '/checkpoints/data/IXI_extracted')
CACHE = os.environ.get('SFCN_CACHE_DIR', '/checkpoints/data/IXI_cached_npy')
WTS   = os.environ.get('SFCN_WEIGHTS',   '/checkpoints/run_20191008_00_epoch_last.p')
DPM   = os.environ.get('SFCN_DP_MODEL',  '/checkpoints')
OUT   = os.environ.get('SFCN_OUT',       '')   # set below, relative to the script
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DPM); sys.path.insert(0, HERE)
from dp_model.model_files.sfcn import SFCN
from networks import SpectralViT
from util import seed_everything, compute_delong_paired

SMOKE    = '--smoke' in sys.argv
NF       = 2 if SMOKE else 5
INNER_K  = 2 if SMOKE else 3
EP_SPEC  = 3 if SMOKE else int(os.environ.get('SFCN_EP_SPEC', 500))
EP_SFCN  = 2 if SMOKE else int(os.environ.get('SFCN_EP_SFCN', 50))
SEL_SPEC = 3 if SMOKE else int(os.environ.get('SFCN_SEL_SPEC', 200))
SEL_SFCN = 2 if SMOKE else int(os.environ.get('SFCN_SEL_SFCN', 20))
SPEC_LR, BATCH, PATIENCE = 1e-4, 2, 5
NCOMPS   = [16, 32] if SMOKE else [16, 32, 64, 128]
SFCN_LRS = [1e-5, 1e-4] if SMOKE else [1e-5, 1e-4, 1e-3]      # ascending: ties -> gentler
SNRS     = [None, 9, 4, 1, 0.25, 0.111]
SUB      = int(os.environ.get('SFCN_N', 0))        # 0 = every subject
SEED     = int(os.environ.get('SFCN_SEED', 0))     # subsample draw and fold partition
TAGSUF   = os.environ.get('SFCN_SUFFIX', '')
OUT = OUT or os.path.join(HERE, 'results', 'sfcn_shift')
os.makedirs(OUT, exist_ok=True)
seed_everything(0)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def rician(x, snr, rng):
    """Rician noise at a given SNR, then renormalise. Matches the notebook exactly."""
    if snr is None: return x
    xm = x.min(); xp = x - xm
    sigma = np.mean(xp[xp > 0]) / snr
    n1 = rng.normal(0, sigma, x.shape); n2 = rng.normal(0, sigma, x.shape)
    xn = np.sqrt((xp + n1) ** 2 + n2 ** 2)
    return ((xn - np.mean(xn)) / (np.std(xn) + 1e-8)).astype(np.float32)


df = pd.read_csv(os.path.join(NII, 'IXI.csv'))
idc = [c for c in df.columns if 'ID' in c.upper()][0]
sxc = [c for c in df.columns if 'SEX' in c.upper()][0]
# male=1, female=0 -- the pretrained SFCN's convention, inverted from loader.py
look = dict(zip(df[idc].astype(int), df[sxc].map({1: 1, 2: 0})))
paths, labels = [], []
for f in sorted(os.listdir(NII)):
    if not f.endswith('.nii.gz'): continue
    m = re.search(r'(\d+)', f)
    p = os.path.join(CACHE, f.replace('.nii.gz', '.npy'))
    if m and int(m.group(1)) in look and os.path.exists(p):
        paths.append(p); labels.append(look[int(m.group(1))])
paths, Y = np.array(paths), np.array(labels, dtype=np.float32)
if SMOKE: paths, Y = paths[:24], Y[:24]
if SUB and SUB < len(Y):
    # Stratified on sex so every size keeps the cohort's class balance. Subsampling
    # happens before Xf is built, so the flat matrix costs n volumes rather than 566.
    rng = np.random.RandomState(SEED)
    keep = []
    for c in np.unique(Y):
        pool = np.where(Y == c)[0]
        keep += list(rng.choice(pool, size=int(round(SUB * len(pool) / len(Y))), replace=False))
    keep = np.array(sorted(keep))[:SUB]
    paths, Y = paths[keep], Y[keep]
    print(f'subsampled to n={len(Y)} (seed {SEED}), labels {np.bincount(Y.astype(int))}',
          flush=True)
print(f'IXI: {len(paths)} volumes, labels {np.bincount(Y.astype(int))}', flush=True)

V0 = np.load(paths[0])
print(f'volume shape {V0.shape}; loading flat matrix (~{len(paths)*V0.size*4/1e9:.1f} GB)',
      flush=True)
Xf = np.stack([np.load(p).astype(np.float32).ravel() for p in paths])
if np.bincount(Y.astype(int), minlength=2).min() >= NF:
    outer = StratifiedKFold(n_splits=NF, shuffle=True, random_state=SEED)
else:
    outer = KFold(n_splits=NF, shuffle=True, random_state=SEED)


def train_spec(ncomp, Ztr, ytr, Zva, yva, epochs):
    m = SpectralViT(n_inputs=ncomp, embed_dim=16, use_mode_weights=True).to(dev)
    ld = DataLoader(TensorDataset(torch.from_numpy(Ztr).float(),
                                  torch.from_numpy(ytr).float()),
                    batch_size=8, shuffle=True)
    opt = optim.AdamW(m.parameters(), lr=SPEC_LR); crit = nn.BCEWithLogitsLoss()
    best, bw, bad = float('inf'), None, 0
    for ep in range(epochs):
        m.train()
        for bx, by in ld:
            opt.zero_grad(); crit(m(bx.to(dev)), by.to(dev)).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            vl = crit(m(torch.from_numpy(Zva).float().to(dev)),
                      torch.from_numpy(yva).float().to(dev)).item()
        if vl < best: best, bw, bad = vl, copy.deepcopy(m.state_dict()), 0
        else:
            bad += 1
            if bad >= PATIENCE: break
    if bw is not None: m.load_state_dict(bw)
    m.eval(); return m


def train_sfcn(lr, ptr, ytr, pva, yva, epochs):
    m = SFCN(output_dim=2, channel_number=[28, 58, 128, 256, 256, 64])
    sd = torch.load(WTS, map_location='cpu')
    m.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()})
    m.to(dev)
    opt = optim.Adam(m.parameters(), lr=lr); crit = nn.NLLLoss()
    best, bw, bad = float('inf'), None, 0
    for ep in range(epochs):
        m.train()
        for i in range(0, len(ptr), BATCH):
            xb = torch.from_numpy(np.stack([np.load(p) for p in ptr[i:i+BATCH]])
                                  ).float().unsqueeze(1).to(dev)
            yb = torch.from_numpy(ytr[i:i+BATCH]).long().to(dev)
            o = m(xb)
            if isinstance(o, list): o = o[0]
            opt.zero_grad(); crit(o.view(o.size(0), -1), yb).backward(); opt.step()
        m.eval(); vl = 0.0
        with torch.no_grad():
            for i in range(0, len(pva), BATCH):
                xb = torch.from_numpy(np.stack([np.load(p) for p in pva[i:i+BATCH]])
                                      ).float().unsqueeze(1).to(dev)
                yb = torch.from_numpy(yva[i:i+BATCH]).long().to(dev)
                o = m(xb)
                if isinstance(o, list): o = o[0]
                vl += crit(o.view(o.size(0), -1), yb).item() * len(yb)
        vl /= max(1, len(pva))
        if vl < best: best, bw, bad = vl, copy.deepcopy(m.state_dict()), 0
        else:
            bad += 1
            if bad >= PATIENCE: break
    if bw is not None: m.load_state_dict(bw)
    m.eval(); return m


def one_se(scores, cands):
    means = {c: np.nanmean(v) for c, v in scores.items()}
    if all(m != m for m in means.values()): return cands[0]
    best = max(means, key=lambda c: means[c])
    se = np.nanstd(scores[best], ddof=1) / np.sqrt(len(scores[best]))
    th = means[best] - (se if se == se else 0.0)
    for c in cands:
        if means[c] == means[c] and means[c] >= th: return c
    return best


rows = []
for fold, (tr, te) in enumerate(outer.split(np.zeros(len(Y)), Y), 1):
    t0 = time.time()
    inner = StratifiedKFold(n_splits=INNER_K, shuffle=True, random_state=0)
    # PCA rank is bounded by the inner-training sample count; restrict the grid once.
    kmin = int(len(tr) * (INNER_K - 1) / INNER_K) - 1
    cands = [c for c in NCOMPS if c <= kmin] or [max(2, kmin)]
    sc_spec = {c: [] for c in cands}
    sc_sfcn = {l: [] for l in SFCN_LRS}
    for i_i, v_i in inner.split(np.zeros(len(tr)), Y[tr]):
        itr, iva = tr[i_i], tr[v_i]
        pca = PCA(n_components=max(cands), whiten=True, random_state=0).fit(Xf[itr])
        Ztr, Zva = pca.transform(Xf[itr]), pca.transform(Xf[iva])
        for c in cands:
            m = train_spec(c, Ztr[:, :c], Y[itr], Zva[:, :c], Y[iva], SEL_SPEC)
            with torch.no_grad():
                p = torch.sigmoid(m(torch.from_numpy(Zva[:, :c]).float().to(dev))).cpu().numpy()
            sc_spec[c].append(roc_auc_score(Y[iva], p)); del m; torch.cuda.empty_cache()
        for lr in SFCN_LRS:
            m = train_sfcn(lr, paths[itr], Y[itr], paths[iva], Y[iva], SEL_SFCN)
            pr = []
            with torch.no_grad():
                for p_ in paths[iva]:
                    x = torch.from_numpy(np.load(p_)).float().unsqueeze(0).unsqueeze(0).to(dev)
                    o = m(x)
                    if isinstance(o, list): o = o[0]
                    pr.append(torch.exp(o.view(1, -1))[:, 1].item())
            sc_sfcn[lr].append(roc_auc_score(Y[iva], pr)); del m; torch.cuda.empty_cache()
    ncomp, sfcn_lr = one_se(sc_spec, cands), one_se(sc_sfcn, SFCN_LRS)
    print(f'fold {fold}: selected ncomp={ncomp} sfcn_lr={sfcn_lr:g}', flush=True)

    # refit on the full training split, early stopping on a slice of it
    vcut = max(2, int(0.1 * len(tr)))
    itr, iva = tr[vcut:], tr[:vcut]
    pca = PCA(n_components=ncomp, whiten=True, random_state=0).fit(Xf[itr])
    spec = train_spec(ncomp, pca.transform(Xf[itr]), Y[itr],
                      pca.transform(Xf[iva]), Y[iva], EP_SPEC)
    sfcn = train_sfcn(sfcn_lr, paths[itr], Y[itr], paths[iva], Y[iva], EP_SFCN)

    for snr in SNRS:
        rng = np.random.RandomState(1000 * fold + int((snr or 0) * 1000))
        ps, pf = [], []
        with torch.no_grad():
            for p_ in paths[te]:
                img = rician(np.load(p_).astype(np.float32), snr, rng)   # same noise, both
                x = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(dev)
                o = sfcn(x)
                if isinstance(o, list): o = o[0]
                pf.append(torch.exp(o.view(1, -1))[:, 1].item())
                z = torch.tensor(pca.transform(img.reshape(1, -1))).float().to(dev)
                ps.append(torch.sigmoid(spec(z)).item())
        lab = 'clean' if snr is None else f'SNR {snr:g}'
        rows.append(dict(fold=fold, shift_label=lab, ncomp=ncomp, sfcn_lr=sfcn_lr,
                         sfcn_auc=roc_auc_score(Y[te], pf),
                         spec_auc=roc_auc_score(Y[te], ps),
                         sfcn_probs=json.dumps(pf), spec_probs=json.dumps(ps),
                         y=json.dumps(Y[te].tolist())))
        print(f'  [{lab:9s}] SFCN {rows[-1]["sfcn_auc"]:.4f} | Spec {rows[-1]["spec_auc"]:.4f}',
              flush=True)
    del spec, sfcn; torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f'sfcn_shift_folds{TAGSUF}.csv'), index=False)
    print(f'fold {fold} done in {time.time()-t0:.0f}s', flush=True)

d = pd.DataFrame(rows)
summ = []
for lab in ['clean'] + [f'SNR {s:g}' for s in SNRS if s is not None]:
    g = d[d.shift_label == lab]
    y = np.concatenate([np.array(json.loads(v)) for v in g.y])
    pf = np.concatenate([np.array(json.loads(v)) for v in g.sfcn_probs])
    ps = np.concatenate([np.array(json.loads(v)) for v in g.spec_probs])
    z, p = compute_delong_paired(y, pf, ps)
    summ.append(dict(shift=lab, SFCN_AUC=g.sfcn_auc.mean(), SFCN_sd=g.sfcn_auc.std(),
                     Spec_AUC=g.spec_auc.mean(), Spec_sd=g.spec_auc.std(),
                     pooled_SFCN=roc_auc_score(y, pf), pooled_Spec=roc_auc_score(y, ps),
                     DeLong_p=p))
t = pd.DataFrame(summ)
t.to_csv(os.path.join(OUT, f'sfcn_shift_table{TAGSUF}.csv'), index=False)
pd.set_option('display.width', 200)
print('\n' + t.round(4).to_string(index=False))
print(f'\nwrote {OUT}/sfcn_shift_table{TAGSUF}.csv', flush=True)
