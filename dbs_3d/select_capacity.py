"""Capacity each model's internal cross-validation selects, one `model:K` per line.

usage: select_capacity.py"""
import os, pickle, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(REPO, 'docs', 'dbs_3d', 'dbs_inference_cells.pkl')
if not os.path.exists(CACHE):
    sys.exit(f'no cells at {CACHE}; run dbs_inference.ipynb first')

KGRID = [16, 32]

CELL = pickle.load(open(CACHE, 'rb'))
models = sorted({m for m, _ in CELL})
for m in models:
    ks = [K for K in KGRID if (m, K) in CELL]
    if not ks:
        sys.exit(f'{m}: no cached cell in KGRID={KGRID}; rerun dbs_inference.ipynb')
    sc = [CELL[(m, K)]['per_seed_int'] for K in ks]
    mu = [x.mean() for x in sc]
    b = int(np.argmax(mu))
    se = sc[b].std(ddof=1) / np.sqrt(len(sc[b]))
    best = ks[min(i for i in range(len(ks)) if mu[i] >= mu[b] - se)]
    print(f'{m}:{best}')
