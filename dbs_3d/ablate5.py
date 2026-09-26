"""Test 5 on the final configuration: remove one input stream at test time only.

Trains normally, then hands the trained model degraded inputs. `zero` gives a
constant the model can partly ignore; `shuffle` gives another patient's values,
which preserves the marginal distribution. A model that ignores a stream is
unaffected by both.

usage: ablate5.py [nseeds]
"""
import sys, os
SPDIR = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(SPDIR)
ABL_OUT = os.path.join(REPO, 'docs', 'dbs_3d')   # `OUT` is clobbered by the sc_worker exec
DATA = os.environ.get('DBS_CLINICAL_DIR', os.path.join(REPO, 'docs'))
CACHE = os.environ.get('DBS_CACHE_DIR', '/checkpoints')
sys.path.insert(0, REPO); sys.path.insert(0, SPDIR); os.chdir(REPO)
import numpy as np, torch, pickle, time, warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score

NS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
_argv = sys.argv; sys.argv = [_argv[0], 'cell', 'SpectralViT', '16', '1']
exec(open(os.path.join(SPDIR, 'sc_worker.py')).read().split('t0 = time.time()')[0])
sys.argv = _argv

t0 = time.time()
res = {k: [] for k in ['intact', 'img zero', 'img shuffle', 'clin zero', 'clin shuffle']}
for sd in range(NS):
    torch.manual_seed(sd)
    m = RankAttn(ZA.shape[1], DA.shape[1], np.resize(LAM, ZA.shape[1]), weight='none').to(dev)
    opt = torch.optim.AdamW(m.parameters(), 3e-3, weight_decay=0.1)
    lf = nn.BCEWithLogitsLoss()
    xt, ct = T(ZA), T(DA)
    yt = torch.tensor(bA.astype(float), dtype=torch.float32, device=dev)
    for _ in range(200):
        opt.zero_grad(); lf(m(xt, ct), yt).backward(); opt.step()
    m.eval()
    r = np.random.RandomState(sd)
    with torch.no_grad():
        auc = lambda Z, D: roc_auc_score(bB, sig(m(T(Z), T(D)).cpu().numpy()))
        res['intact'].append(auc(ZB, DB))
        res['img zero'].append(auc(np.zeros_like(ZB), DB))
        res['img shuffle'].append(auc(ZB[r.permutation(len(ZB))], DB))
        res['clin zero'].append(auc(ZB, np.zeros_like(DB)))
        res['clin shuffle'].append(auc(ZB, DB[r.permutation(len(DB))]))
    print(f"  seed {sd}: " + "  ".join(f"{k} {res[k][-1]:.3f}" for k in res), flush=True)

res = {k: np.array(v) for k, v in res.items()}
pickle.dump(dict(nseeds=NS, auc=res, bB=bB, secs=time.time() - t0),
            open(os.path.join(ABL_OUT, 'ablate5.pkl'), 'wb'))
print()
for k, v in res.items():
    print(f"{k:14s} {v.mean():.3f} +- {v.std():.3f}   (delta {v.mean()-res['intact'].mean():+.3f})")
print(f"\n({time.time()-t0:.0f}s)")
