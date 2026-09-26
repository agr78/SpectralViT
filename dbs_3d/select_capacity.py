"""Capacity each model's internal cross-validation selects, one `model:K` per line.

Reads the leave-one-out cells the notebook caches. Hardcoding these values is how the
metrics table and the scorecard drifted apart, so both runners derive them from here.

usage: select_capacity.py
"""
import os, pickle, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(REPO, 'docs', 'dbs_3d', 'dbs_inference_cells.pkl')
if not os.path.exists(CACHE):
    sys.exit(f'no cells at {CACHE}; run dbs_inference.ipynb first')

CELL = pickle.load(open(CACHE, 'rb'))
models, Ks = sorted({m for m, _ in CELL}), sorted({K for _, K in CELL})
for m in models:
    ks = [K for K in Ks if (m, K) in CELL]
    best = max(ks, key=lambda K: CELL[(m, K)]['per_seed_int'].mean())
    print(f'{m}:{best}')
