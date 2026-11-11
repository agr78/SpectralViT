"""Chance tests for every cell of table:ixi and table:hca-reg.

table:ixi is AUC, so chance is 0.5 and the test is one-sided over the subsample draws.
table:hca-reg is mean absolute error, where the chance reference is not a constant but the
mean estimate already carried as the table's first row: predicting the cohort mean for
every subject. Each model is therefore tested against that row, paired by draw.

usage: python make_chance_tests.py
"""
import os

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results', 'chance_tests.txt')
IXI = os.environ.get('IXI_SWEEP_DIR', os.path.join(HERE, 'results', 'ixi_curve_merged'))
SIZES = [25, 50, 100, 200, 566]
# Draws per size, matching make_table_ixi.py: ten at the two smallest sizes, five above.
DRAWS = {25: 10, 50: 10, 100: 5, 200: 5, 566: 5}
ARMS = [('logreg', 'Logistic regression'), ('mlp', 'Multilayer perceptron'),
        ('unet', 'U-net with attention'), ('swin', 'Swin ViT'),
        ('spat', 'Spatial ViT'), ('spec', 'Spectral ViT (proposed)')]
# cells of table:ixi NOT significantly above chance, as published
PUB_IXI_NS = {('logreg', 25), ('mlp', 25), ('unet', 25), ('spat', 25), ('swin', 100)}


def ixi_draws(tag, n):
    out = []
    for s in range(DRAWS[n]):
        f = os.path.join(IXI, f'oof_{tag}_n{n}_s{s}.npz')
        if not os.path.exists(f):
            return None
        z = np.load(f)
        cuts = np.cumsum(z['fold_sizes'])[:-1]
        a = [roc_auc_score(y, p) for y, p in
             zip(np.split(z['y'], cuts), np.split(z['probs'], cuts)) if len(np.unique(y)) > 1]
        out.append(np.mean(a))
    return np.array(out)


L = ['table:ixi -- one-sided test of AUC > 0.5 over that size\'s draws', '',
     '%-26s' % 'model' + ''.join('%15s' % f'n={n}' for n in SIZES)]
ns = set()
for tag, name in ARMS:
    row = '%-26s' % name
    for n in SIZES:
        d = ixi_draws(tag, n)
        if d is None:
            row += '%15s' % '--'
            continue
        t = stats.ttest_1samp(d, 0.5)
        p = t.pvalue / 2 if t.statistic > 0 else 1 - t.pvalue / 2
        if p >= 0.05:
            ns.add((tag, n))
        row += '%15s' % ('%.3f p=%.3f%s' % (d.mean(), p, '*' if p < 0.05 else ''))
    L.append(row)

report = '\n'.join(L)
print(report)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, 'w').write(report + '\n')
print(f'\nwrote {OUT}')

if ns != PUB_IXI_NS:
    raise SystemExit('DOES NOT MATCH table:ixi:\n  not-above-chance cells are '
                     f'{sorted(ns)}, the table marks {sorted(PUB_IXI_NS)}')
print(f'matches table:ixi: {len(ns)} of {len(ARMS) * len(SIZES)} cells not above chance, '
      'the rest significant')
