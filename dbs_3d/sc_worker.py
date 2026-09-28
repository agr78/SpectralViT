"""One (model, K) cell of the multi-model scorecard, or one permutation chunk.

Cells are independent, so many run concurrently. Each is tiny -- the limit is
kernel-launch latency, not compute -- so throughput comes from concurrency.

phase cell : LOO internal over NS seeds + external, predictions saved
phase perm : one chunk of label permutations with a full LOO refit each

usage: sc_worker.py cell <model> <K> <nseeds>
       sc_worker.py perm <model> <K> <chunk> <nperm>
"""
import sys, os
SPDIR = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(SPDIR)
OUT = os.path.join(REPO, 'docs', 'dbs_3d', 'sc'); os.makedirs(OUT, exist_ok=True)
DATA = os.environ.get('DBS_CLINICAL_DIR', os.path.join(REPO, 'docs'))
CACHE = os.environ.get('DBS_CACHE_DIR', '/checkpoints')
sys.path.insert(0, REPO); sys.path.insert(0, SPDIR); os.chdir(REPO)
import numpy as np, torch, torch.nn as nn, pickle, time, warnings
warnings.filterwarnings('ignore')
from scipy.fft import fftn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from basis_fix import low_modes, eigvals
from rank_model import RankAttn
from networks import AttentionUNet

PHASE = sys.argv[1]
MODEL = sys.argv[2]
K = int(sys.argv[3])
torch.set_num_threads(2)
src = open(os.path.join(SPDIR, 'roi_x_models.py')).read().split('IDX = low_modes')[0]
g = {'__file__': os.path.join(SPDIR, 'roi_x_models.py')}
_argv = sys.argv; sys.argv = [_argv[0]]          # the prep block reads argv for its own seed count
exec(compile(src, 'x', 'exec'), g)
sys.argv = _argv
HA, LA, HB, LB, bA, bB, Ca, Cb = g['HA'], g['LA'], g['HB'], g['LB'], g['bA'], g['bB'], g['Ca'], g['Cb']
H, DZ = 10, 8
dev = torch.device('cuda'); T = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
sig = lambda z: 1/(1+np.exp(-z))

IDX = low_modes(K, (2*H, 2*H, 2*DZ), 'lap')
LAM = np.r_[eigvals(IDX, (2*H, 2*H, 2*DZ))]*2 if False else np.r_[eigvals(IDX, (2*H, 2*H, 2*DZ)),
                                                                   eigvals(IDX, (2*H, 2*H, 2*DZ))]
lg = lambda V: np.stack([np.log1p(np.abs(fftn(V, axes=(1, 2, 3))))[:, a, b, c] for a, b, c in IDX], 1)
XA = np.hstack([lg(HA), lg(LA)]); XB = np.hstack([lg(HB), lg(LB)])          # 'both' ROI
# Spatial ViT is tuned over patch size on the same footing as K for the
# spectral models: K in {8,16,32} maps to patch size {8,5,4}, giving token
# counts of roughly 16, 96 and 200 for the 'both' ROI.
PS = {8: 8, 16: 5, 32: 4}.get(K, 5)
patch = lambda V: np.stack([V[:, i:i+PS, j:j+PS, k:k+PS].reshape(len(V), -1).mean(1)
                            for i in range(0, 2*H-PS+1, PS) for j in range(0, 2*H-PS+1, PS)
                            for k in range(0, 2*DZ-PS+1, PS)], 1)
PA = np.hstack([patch(HA), patch(LA)]); PB = np.hstack([patch(HB), patch(LB)])
# Attention U-Net baseline: the repo class on the raw crops, both nuclei as channels.
# K indexes width so it searches capacity on the same footing as the other models.
# Volumes are z-scored with MSW statistics only.
BC = {8: 4, 16: 7, 32: 12}.get(K, 7)
UA = np.stack([HA, LA], 1); UB = np.stack([HB, LB], 1)
_um, _us = UA.mean(), UA.std()
UA = (UA - _um) / (_us + 1e-8); UB = (UB - _um) / (_us + 1e-8)
sX = StandardScaler().fit(XA); ZA, ZB = sX.transform(XA), sX.transform(XB)
sP = StandardScaler().fit(PA); QA, QB = sP.transform(PA), sP.transform(PB)
sC = StandardScaler().fit(Ca); DA, DB = sC.transform(Ca), sC.transform(Cb)


class TokenViT(nn.Module):
    def __init__(s, n, d=16):
        super().__init__()
        s.tok = nn.Linear(1, d); s.pos = nn.Parameter(torch.zeros(1, n, d))
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 2, d*2, 0.1, batch_first=True), 1)
        s.hd = nn.Linear(d, 1); s.lin = nn.Linear(n, 1)
    def forward(s, x):
        z = s.tok(x.unsqueeze(-1)) + s.pos
        return s.hd(s.enc(z).mean(1)).squeeze(-1) + s.lin(x).squeeze(-1)


def predict(tr, te, y, seed):
    """fit on rows `tr`, predict rows `te`; te may index MSW (int array) or be 'EXT'"""
    ext = isinstance(te, str)
    if MODEL == 'Clinical':
        m = LogisticRegression(C=1.0, max_iter=3000, class_weight='balanced').fit(DA[tr], y[tr])
        return sig(m.decision_function(DB if ext else DA[te]))
    if MODEL == 'Spectral LR':
        m = LogisticRegression(C=1.0, max_iter=3000, class_weight='balanced').fit(ZA[tr], y[tr])
        return sig(m.decision_function(ZB if ext else ZA[te]))
    if MODEL == 'Spectral MLP':
        m = MLPClassifier(hidden_layer_sizes=(32,), max_iter=2000, random_state=seed).fit(ZA[tr], y[tr])
        return m.predict_proba(ZB if ext else ZA[te])[:, 1]
    if MODEL == 'Spatial ViT':
        return tfit(lambda: TokenViT(QA.shape[1]), QA[tr], y[tr].astype(float), QB if ext else QA[te], seed)
    if MODEL == 'Attention UNet':
        return ufit(UA[tr], y[tr].astype(float), UB if ext else UA[te], seed)
    return ofit(ZA[tr], DA[tr], y[tr].astype(float),
                (ZB, DB) if ext else (ZA[te], DA[te]), seed)


def tfit(mk, Xtr, ytr, Xte, seed, ep=200):
    torch.manual_seed(seed); m = mk().to(dev)
    opt = torch.optim.AdamW(m.parameters(), 3e-3, weight_decay=0.1); lf = nn.BCEWithLogitsLoss()
    xt = T(Xtr); yt = torch.tensor(ytr, dtype=torch.float32, device=dev)
    for _ in range(ep):
        opt.zero_grad(); lf(m(xt), yt).backward(); opt.step()
    m.eval()
    with torch.no_grad(): return sig(m(T(Xte)).cpu().numpy())


def ufit(Xtr, ytr, Xte, seed, ep=200):
    torch.manual_seed(seed)
    m = AttentionUNet(in_channels=2, base_channels=BC).to(dev)
    opt = torch.optim.AdamW(m.parameters(), 3e-3, weight_decay=0.1); lf = nn.BCEWithLogitsLoss()
    xt = T(Xtr); yt = torch.tensor(ytr, dtype=torch.float32, device=dev)
    for _ in range(ep):
        opt.zero_grad(); lf(m(xt), yt).backward(); opt.step()
    m.eval()
    with torch.no_grad(): return sig(m(T(Xte)).cpu().numpy())


def ofit(Xtr, Ctr, ytr, te, seed, ep=200):
    torch.manual_seed(seed)
    m = RankAttn(Xtr.shape[1], Ctr.shape[1], np.resize(LAM, Xtr.shape[1]), weight='none').to(dev)
    opt = torch.optim.AdamW(m.parameters(), 3e-3, weight_decay=0.1); lf = nn.BCEWithLogitsLoss()
    xt, ct = T(Xtr), T(Ctr); yt = torch.tensor(ytr, dtype=torch.float32, device=dev)
    for _ in range(ep):
        opt.zero_grad(); lf(m(xt, ct), yt).backward(); opt.step()
    m.eval()
    with torch.no_grad(): return sig(m(T(te[0]), T(te[1])).cpu().numpy())


def loo(y, seed):
    p = np.empty(len(y))
    for i in range(len(y)):
        tr = np.delete(np.arange(len(y)), i)
        p[i] = predict(tr, np.array([i]), y, seed)[0]
    return p


t0 = time.time()
if PHASE == 'cell':
    NS = int(sys.argv[4])
    ii, ee, pbs = [], [], []
    for sd in range(NS):
        pr = loo(bA, sd)
        pb = predict(np.arange(len(bA)), 'EXT', bA, sd)
        ii.append(roc_auc_score(bA, pr)); ee.append(roc_auc_score(bB, pb)); pbs.append(pb)
    ii, ee = np.array(ii), np.array(ee)
    pickle.dump(dict(model=MODEL, K=K, internal=ii, external=ee, preds=np.array(pbs),
                     ens=np.mean(pbs, 0), bA=bA, bB=bB, secs=time.time()-t0),
                open(os.path.join(OUT, f'cell_{MODEL.replace(" ","_")}_{K}.pkl'), 'wb'))
    print(f"{MODEL:20s} K={K:2d}  LOO int {ii.mean():.3f}+-{ii.std()/np.sqrt(NS):.3f}  "
          f"ext {ee.mean():.3f}+-{ee.std():.3f}  ({time.time()-t0:.0f}s)", flush=True)
else:
    cid, NP = int(sys.argv[4]), int(sys.argv[5])
    rng = np.random.RandomState(2000 + cid)
    vals = [roc_auc_score(yp, loo(yp, 0)) for yp in (bA[rng.permutation(len(bA))] for _ in range(NP))]
    pickle.dump(np.array(vals), open(os.path.join(OUT, f'perm_{MODEL.replace(" ","_")}_{K}_{cid}.pkl'), 'wb'))
    print(f"{MODEL:20s} K={K} perm chunk {cid}: {NP} perms, null mean {np.mean(vals):.3f} "
          f"({time.time()-t0:.0f}s)", flush=True)
