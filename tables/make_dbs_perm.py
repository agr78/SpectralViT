"""Exact permutation p-values for table:dbs, and the paired tests quoted in sec:results.

usage: python tables/make_dbs_perm.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repository root: shared modules and paths.py
import itertools
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, 'results', 'dbs_perm.txt')

_src = open(os.path.join(HERE, 'tables', 'make_table_dbs.py')).read()
_g = {'__name__': 'dbs', '__file__': os.path.join(HERE, 'tables', 'make_table_dbs.py')}
exec(compile(_src.split('def table(grid, rule):')[0], 'dbs', 'exec'), _g)
C, bA, bB, sel, MODELS, lab = _g['C'], _g['bA'], _g['bB'], _g['sel'], _g['MODELS'], _g['lab']

PUB = {'Clinical': 0.188, 'Clinical Transformer': 0.236, 'Spectral LR': 0.074,
       'Spectral MLP': 0.203, 'Spatial ViT': 0.052, 'Attention UNet': 0.729,
       'SpectralViT': 0.040}


def auc(y, p):
    pos, neg = p[y == 1], p[y == 0]
    d = pos[:, None] - neg[None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / (len(pos) * len(neg)))


n = len(bB)
nneg = int((bB == 0).sum())
combos = list(itertools.combinations(range(n), nneg))

L = [f'Exact permutation test on external AUC: all C({n},{nneg}) = {len(combos)} label',
     'assignments enumerated, one-sided.', '',
     '%-26s %-13s %8s %9s' % ('Model', 'Capacity', 'AUC', 'p')]
got = {}
for m in MODELS:
    k = sel(m, [16, 32], '1SE')
    pb = C[(m, k)]['pb']
    obs = auc(bB, pb)
    hits = sum(1 for c in combos
               if auc(np.where(np.isin(np.arange(n), c), 0, 1), pb) >= obs)
    got[m] = hits / len(combos)
    L.append('%-26s %-13s %8.3f %9.3f' % (m, lab(m, k), obs, got[m]))

bad_pair = []

PROP = 'SpectralViT'
PUB_PAIR = {'Clinical': 0.231, 'Clinical Transformer': 0.197, 'Spectral LR': 0.369,
            'Spectral MLP': 0.184, 'Spatial ViT': 0.459, 'Attention UNet': 0.046}
PUB_UNET_ALT = (0.686, 0.294)

sp = C[(PROP, sel(PROP, [16, 32], '1SE'))]['pb']


def paired(pb):
    obs = auc(bB, sp) - auc(bB, pb)
    hits = sum(1 for c in combos
               if auc(np.where(np.isin(np.arange(n), c), 0, 1), sp)
               - auc(np.where(np.isin(np.arange(n), c), 0, 1), pb) >= obs)
    return obs, hits / len(combos)


L += ['', 'Paired against the proposed model (one-sided on the AUC difference):',
      '%-26s %9s %9s' % ('baseline', 'delta', 'p')]
pair = {}
for m in MODELS:
    if m == PROP:
        continue
    d, q = paired(C[(m, sel(m, [16, 32], '1SE'))]['pb'])
    pair[m] = q
    L.append('%-26s %9s %9.3f' % (m, format(d, '+.3f'), q))
alt = [k for k in (16, 32) if C[('Attention UNet', k)]['pb'] is not
       C[('Attention UNet', sel('Attention UNet', [16, 32], '1SE'))]['pb']]
alt_pb = C[('Attention UNet', alt[0])]['pb']
alt_auc, alt_p = auc(bB, alt_pb), paired(alt_pb)[1]
L += ['', f'U-net at the grid\'s other capacity: external {alt_auc:.3f}, paired p={alt_p:.3f}',
      '  (the selected capacity gives 0.392 and p=0.046, so the one significant pairwise',
      '   result turns on a choice made by a below-chance internal criterion)']
for m, v in PUB_PAIR.items():
    if abs(pair[m] - v) > 0.0005:
        bad_pair.append(f'paired {m}: {pair[m]:.3f}, paper says {v:.3f}')
if abs(alt_auc - PUB_UNET_ALT[0]) > 0.0005 or abs(alt_p - PUB_UNET_ALT[1]) > 0.0005:
    bad_pair.append(f'U-net alternate capacity {alt_auc:.3f}/{alt_p:.3f}, '
                    f'paper says {PUB_UNET_ALT[0]:.3f}/{PUB_UNET_ALT[1]:.3f}')

report = '\n'.join(L)
print(report)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, 'w').write(report + '\n')
print(f'\nwrote {OUT}')

bad = [f'{m}: {got[m]:.3f}, table says {v:.3f}' for m, v in PUB.items()
       if abs(got[m] - v) > 0.0005] + bad_pair
if bad:
    raise SystemExit('DOES NOT MATCH table:dbs:\n  ' + '\n  '.join(bad))
print(f'matches table:dbs and sec:results: {len(PUB)} per-model and {len(PUB_PAIR)} paired '
      f'p-values, and the U-net counterfactual capacity')
