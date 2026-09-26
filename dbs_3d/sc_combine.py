"""Combine the multi-model scorecard: cells + permutation chunks -> tests 1-4."""
import sys, os, glob, pickle
SPDIR = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(SPDIR)
SC = os.path.join(REPO, 'docs', 'dbs_3d', 'sc')
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

cells = {}
for p in sorted(glob.glob(os.path.join(SC, 'cell_*.pkl'))):
    r = pickle.load(open(p, 'rb')); cells[(r['model'], r['K'])] = r
if not cells: sys.exit('no cells')
bB = next(iter(cells.values()))['bB']
n = len(bB); n1 = int(bB.sum()); n0 = n - n1
rng = np.random.RandomState(0)

models = sorted({k[0] for k in cells})
print(f"CHH external: {n} patients, {n1} responders / {n0} non-responders\n")
print(f"{'model':20s} {'K':>3} {'LOO int':>16} {'external':>16} {'fw p':>7} {'perm p':>8} {'nperm':>6}")
rows = []
for m in models:
    ks = [k for k in cells if k[0] == m]
    best = max(ks, key=lambda k: np.mean(cells[k]['internal']))
    r = cells[best]
    ii, ee = r['internal'], r['external']

    # TEST 1: family-wise null over that model's own K cells
    R = np.array([rankdata(cells[k]['ens']) for k in ks])
    auc = lambda y: (R @ y - n1*(n1+1)/2)/(n1*n0)
    obs = auc(bB); mx = np.array([auc(bB[rng.permutation(n)]).max() for _ in range(5000)])
    fw = (mx >= obs.max()).mean()

    # TEST 4: internal permutation, from the chunks
    ch = sorted(glob.glob(os.path.join(SC, f'perm_{m.replace(" ","_")}_{best[1]}_*.pkl'))) or sorted(glob.glob(os.path.join(SC, f'perm_{m.replace(" ","_")}_*.pkl')))
    if ch:
        null = np.concatenate([pickle.load(open(c, 'rb')) for c in ch])
        pp = float((null >= ii.mean()).mean()); npm = len(null)
        nullstr = f"{null.mean():.3f}+-{null.std():.3f}"
    else:
        pp, npm, nullstr = np.nan, 0, '-'
    print(f"{m:20s} {best[1]:3d} {ii.mean():7.3f}+-{ii.std()/np.sqrt(len(ii)):.3f} "
          f"{ee.mean():9.3f}+-{ee.std():.3f} {fw:7.3f} {pp:8.3f} {npm:6d}")
    rows.append(dict(model=m, K=best[1], loo_int=ii.mean(), int_se=ii.std()/np.sqrt(len(ii)),
                     ext=ee.mean(), ext_sd=ee.std(), fw_p=fw, perm_p=pp, nperm=npm, null=nullstr))
d = pd.DataFrame(rows); d.to_csv(os.path.join(REPO, 'docs', 'dbs_3d', 'sc_summary.csv'), index=False)
print("\nnull distributions (test 4):")
for _, r in d.iterrows():
    if r['nperm']: print(f"  {r['model']:20s} observed {r['loo_int']:.3f}   null {r['null']}   p = {r['perm_p']:.3f}  (n={int(r['nperm'])})")
