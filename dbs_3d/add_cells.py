"""Add one model's leave-one-out cells to the notebook cache, leaving existing cells alone.

Refuses to overwrite a cell that is already cached, so a model can be appended to the
comparison without disturbing any published number.

usage: add_cells.py "<model name>" [nseeds]
"""
import sys, os, pickle, time, warnings
warnings.filterwarnings('ignore')
REPO, SPDIR = '/workspace', '/workspace/dbs_3d'
sys.path.insert(0, REPO); sys.path.insert(0, SPDIR); os.chdir(REPO)
import numpy as np
from sklearn.metrics import roc_auc_score

M = sys.argv[1]
NSEED = int(sys.argv[2]) if len(sys.argv) > 2 else 10
KGRID = [8, 16, 32]
CACHE_F = os.path.join(REPO, 'docs', 'dbs_3d', 'dbs_inference_cells.pkl')

SRC = open(os.path.join(SPDIR, 'sc_worker.py')).read().split('t0 = time.time()')[0]
MARK = chr(10) + 'IDX = low_modes('          # line 32 holds the bare string too
HEAD, TAIL = SRC.split(MARK, 1); TAIL = 'IDX = low_modes(' + TAIL
G = {'__file__': os.path.join(SPDIR, 'sc_worker.py')}
_a = sys.argv; sys.argv = [_a[0], 'cell', M, '16', '1']
exec(compile(HEAD, 'h', 'exec'), G); sys.argv = _a
bA, bB = G['bA'], G['bB']

t0 = time.time(); new = {}
for K in KGRID:
    G['MODEL'], G['K'] = M, K
    exec(compile(TAIL, 't', 'exec'), G)
    PR = [G['loo'](bA, sd) for sd in range(NSEED)]
    PB = [G['predict'](np.arange(len(bA)), 'EXT', bA, sd) for sd in range(NSEED)]
    new[(M, K)] = dict(pr=np.mean(PR, 0), pb=np.mean(PB, 0),
                       per_seed_int=np.array([roc_auc_score(bA, p) for p in PR]),
                       per_seed_ext=np.array([roc_auc_score(bB, p) for p in PB]))
    c = new[(M, K)]
    print(f"{M} K={K:2d}  int {roc_auc_score(bA, c['pr']):.3f}  "
          f"ext {roc_auc_score(bB, c['pb']):.3f}  ({time.time()-t0:.0f}s)", flush=True)

CELL = pickle.load(open(CACHE_F, 'rb')) if os.path.exists(CACHE_F) else {}
before = len(CELL)
clash = [k for k in new if k in CELL]
assert not clash, f'refusing to overwrite cached cells: {clash}'
CELL.update(new); pickle.dump(CELL, open(CACHE_F, 'wb'))
print(f'cache {before} -> {len(CELL)} cells (existing untouched)')
