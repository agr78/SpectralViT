"""Paired comparisons against the proposed model for sec:ixipaired.

The draw is the unit of analysis: each cell is one subsample's mean fold AUC, and every
model is fit on the same draws, so the comparison is paired. We report the mean difference
in AUC, the number of draws in which the proposed model leads, and a two-sided paired
t-test.

The sizes at and above n=100 are untouched by the ten-draw run, so their numbers must
still match the published prose. ASSERT_N566 holds that check: if a refactor silently
changes how a cell is scored, this fails rather than quietly rewriting the paper.

usage: python make_ixi_paired.py
"""
import os

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results', 'ixi_paired.txt')
DIR = os.environ.get('IXI_SWEEP_DIR', os.path.join(HERE, 'results', 'ixi_curve_merged'))
SIZES = [25, 50, 100, 200, 566]
DRAWS = {25: 10, 50: 10, 100: 5, 200: 5, 566: 5}
BASE = [('unet', 'U-net with attention'), ('spat', 'Spatial ViT'), ('swin', 'Swin ViT'),
        ('logreg', 'Logistic regression'), ('mlp', 'Multilayer perceptron')]
# published n=566 margins, proposed minus baseline, to 3 dp
ASSERT_N566 = {'unet': -0.053, 'spat': 0.111, 'swin': 0.205,
               'logreg': -0.035, 'mlp': -0.040}


def draws(tag, n):
    """Mean fold AUC for each subsample draw of one (model, size) cell."""
    out = []
    for s in range(DRAWS[n]):
        f = os.path.join(DIR, f'oof_{tag}_n{n}_s{s}.npz')
        if not os.path.exists(f):
            raise SystemExit(f'missing cell: {f}')
        z = np.load(f)
        cuts = np.cumsum(z['fold_sizes'])[:-1]
        a = [roc_auc_score(y, p) for y, p in
             zip(np.split(z['y'], cuts), np.split(z['probs'], cuts))
             if len(np.unique(y)) > 1]
        out.append(np.mean(a))
    return np.array(out)


L, got566 = [], {}
for n in SIZES:
    spec = draws('spec', n)
    L.append(f'n={n}  ({DRAWS[n]} draws, proposed mean {spec.mean():.3f})')
    L.append('  %-24s %9s %7s %9s' % ('baseline', 'mean diff', 'leads', 'p'))
    for tag, name in BASE:
        d = spec - draws(tag, n)
        p = stats.ttest_rel(spec, draws(tag, n)).pvalue
        L.append('  %-24s %+9.3f %4d/%-2d %9.3f'
                 % (name, d.mean(), int((d > 0).sum()), DRAWS[n], p))
        if n == 566:
            got566[tag] = round(float(d.mean()), 3)
    L.append('')

report = '\n'.join(L)
print(report)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, 'w').write(report + '\n')
print(f'wrote {OUT}')

bad = {k: (v, ASSERT_N566[k]) for k, v in got566.items() if abs(v - ASSERT_N566[k]) > 5e-4}
if bad:
    raise SystemExit(f'n=566 margins moved, got vs published: {bad}')
print('n=566 margins match the published prose')
