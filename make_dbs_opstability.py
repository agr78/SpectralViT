"""How much the external operating point depends on the external cohort.

The threshold transferred to the external cohort is a QUANTILE of the internal score
distribution, mapped back through the external one (see make_table_dbs.py). No external
label is involved, but the external scores are: the threshold applied to one patient is a
percentile of all of them, so the decision rule depends on the test batch.

This measures how much that matters, by refitting the threshold on every leave-one-out
subset of the external cohort and re-scoring everyone against it. It also reports the
operating quantile per model, which shows the transfer is not what produces the proposed
model's specificity -- the spatial baselines reach the same specificity only by calling
almost the whole cohort a non-responder.

usage: python make_dbs_opstability.py
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results', 'dbs_opstability.txt')

# Reuse make_table_dbs.py's loading, calibration and capacity selection rather than
# restating them, so this cannot diverge from the table it qualifies.
_src = open(os.path.join(HERE, 'make_table_dbs.py')).read()
_g = {'__name__': 'dbs', '__file__': os.path.join(HERE, 'make_table_dbs.py')}
exec(compile(_src.split('def table(grid, rule):')[0], 'dbs', 'exec'), _g)
C, bA, bB, cal, sel, MODELS, lab = (_g['C'], _g['bA'], _g['bB'], _g['cal'], _g['sel'],
                                    _g['MODELS'], _g['lab'])

PROP = 'SpectralViT'
PUB_Q = {'SpectralViT': 0.470, 'Spatial ViT': 0.955, 'Attention UNet': 0.970}
PUB_LO, PUB_HI = 0.999341, 0.999346          # threshold range over leave-one-out

L = ['Operating quantile and its stability on the external cohort.', '',
     '%-24s %-12s %7s %12s %9s' % ('Model', 'Capacity', 'q', 'pred.neg', 'spec')]
qs = {}
for m in MODELS:
    k = sel(m, [16, 32], '1SE')
    pr, pb = C[(m, k)]['pr'], C[(m, k)]['pb']
    q = float((pr < cal(bA, pr)).mean())
    qs[m] = q
    pred = (pb >= float(np.quantile(pb, q))).astype(int)
    sp = float((pred[bB == 0] == 0).mean())
    L.append('%-24s %-12s %7.3f %7d/%-4d %9.3f' % (m, lab(m, k), q, (pred == 0).sum(), len(pb), sp))

k = sel(PROP, [16, 32], '1SE')
pr, pb = C[(PROP, k)]['pr'], C[(PROP, k)]['pb']
q = qs[PROP]
thr = [float(np.quantile(np.delete(pb, j), q)) for j in range(len(pb))]
spec = [float(((pb >= t).astype(int)[bB == 0] == 0).mean()) for t in thr]
bal = []
for t in thr:
    p = (pb >= t).astype(int)
    bal.append(((p[bB == 1] == 1).mean() + (p[bB == 0] == 0).mean()) / 2)
L += ['', f'Leave-one-out over the {len(pb)} external patients, {PROP}:',
      f'  threshold        {min(thr):.6f} - {max(thr):.6f}  (full cohort {np.quantile(pb, q):.6f})',
      f'  specificity      {min(spec):.3f} - {max(spec):.3f}',
      f'  balanced acc.    {min(bal):.3f} - {max(bal):.3f}']

report = '\n'.join(L)
print(report)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, 'w').write(report + '\n')
print(f'\nwrote {OUT}')

bad = []
for m, v in PUB_Q.items():
    if abs(qs[m] - v) > 0.0005:
        bad.append(f'{m} operates at q={qs[m]:.3f}, paper says {v:.3f}')
if not (abs(min(thr) - PUB_LO) < 5e-6 and abs(max(thr) - PUB_HI) < 5e-6):
    bad.append(f'threshold range {min(thr):.6f}-{max(thr):.6f}, paper says {PUB_LO}-{PUB_HI}')
if min(spec) != 1.0:
    bad.append(f'specificity drops to {min(spec):.3f} under leave-one-out; the paper says it never does')
if bad:
    raise SystemExit('DOES NOT MATCH sec:dbscap:\n  ' + '\n  '.join(bad))
print('matches sec:dbscap: operating quantiles, threshold range and invariant specificity')
