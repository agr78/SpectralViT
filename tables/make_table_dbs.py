"""Regenerate table:dbs, the neurostimulation external-cohort comparison.

usage: python tables/make_table_dbs.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repository root: shared modules and paths.py
import os
import pickle

import numpy as np
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                             roc_auc_score)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, 'results', 'table_dbs.txt')
CELLS = os.path.join(HERE, 'docs', 'dbs_3d', 'dbs_inference_cells.pkl')
LABELS = os.path.join(HERE, 'results', 'dbs_labels.npz')

PS = {8: 8, 16: 5, 32: 4}
BC = {8: 4, 16: 7, 32: 12}
MODELS = ['Clinical', 'Clinical Transformer', 'Spectral LR', 'Spectral MLP',
          'Spatial ViT', 'Attention UNet', 'SpectralViT']

PUB = {
    'Clinical':             ('--',          0.539, 0.667, 0.667),
    'Clinical Transformer': ('--',          0.593, 0.333, 0.637),
    'Spectral LR':          ('16 modes',    0.569, 0.667, 0.765),
    'Spectral MLP':         ('16 modes',    0.657, 0.667, 0.657),
    'Spatial ViT':          ('patch 4',     0.529, 1.000, 0.794),
    'Attention UNet':       ('12 channels', 0.529, 1.000, 0.392),
    'SpectralViT':          ('16 modes',    0.794, 1.000, 0.814),
}

C = pickle.load(open(CELLS, 'rb'))
L = np.load(LABELS)
bA, bB = L['bA'], L['bB']


def cal(y, p):
    best, bt = -1, float(np.median(p))
    for t in np.unique(np.concatenate([p, np.linspace(p.min(), p.max(), 300)])):
        b = balanced_accuracy_score(y, (p >= t).astype(int))
        if b > best:
            best, bt = b, float(t)
    return bt


def metrics(y, p, thr):
    yh = (p >= thr).astype(int)
    tn, fp, _fn, _tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    return (balanced_accuracy_score(y, yh),
            tn / (tn + fp) if tn + fp else np.nan,
            f1_score(y, yh, zero_division=0),
            roc_auc_score(y, p))


def sel(m, grid, rule):
    mu = np.array([C[(m, K)]['per_seed_int'].mean() for K in grid])
    b = int(np.argmax(mu))
    if rule == 'argmax':
        return grid[b]
    a = C[(m, grid[b])]['per_seed_int']
    se = a.std(ddof=1) / np.sqrt(len(a))
    return grid[min(i for i in range(len(grid)) if mu[i] >= mu[b] - se)]


def lab(m, k):
    if m.startswith('Clinical'):
        return '--'
    if m == 'Spatial ViT':
        return 'patch %d' % PS[k]
    if m == 'Attention UNet':
        return '%d channels' % BC[k]
    return '%d modes' % k


def table(grid, rule):
    rows = {}
    for m in MODELS:
        k = sel(m, grid, rule)
        c = C[(m, k)]
        pr, pb = c['pr'], c['pb']
        thr = cal(bA, pr)
        q = float((pr < thr).mean())
        tq = float(np.quantile(pb, q))
        rows[m] = (lab(m, k),) + metrics(bB, pb, tq)
    return rows


L_out = []
for title, grid, rule in [('argmax, K in {8,16,32}', [8, 16, 32], 'argmax'),
                          ('PUBLISHED  (one-SE, K in {16,32})', [16, 32], '1SE')]:
    rows = table(grid, rule)
    L_out.append('=== ' + title)
    L_out.append('%-26s %-12s %8s %8s %8s %8s'
                 % ('Model', 'Capacity', 'BalAcc', 'Spec', 'F1', 'AUC'))
    for m in MODELS:
        cap, ba, sp, f1, auc = rows[m]
        L_out.append('%-26s %-12s %8.3f %8.3f %8.3f %8.3f' % (m, cap, ba, sp, f1, auc))
    L_out.append('')
    if rule == '1SE':
        published = rows

report = '\n'.join(L_out)
print(report)
with open(OUT, 'w') as fh:
    fh.write(report + '\n')
print(f'wrote {OUT}')

bad = []
for m, (cap, ba, sp, auc) in PUB.items():
    g = published[m]
    if g[0] != cap:
        bad.append(f'{m}: capacity {g[0]!r} != published {cap!r}')
    for name, got, exp in zip(('BalAcc', 'Spec', 'AUC'),
                              (g[1], g[2], g[4]), (ba, sp, auc)):
        if abs(got - exp) > 0.0006:
            bad.append(f'{m} {name}: {got:.3f} != published {exp:.3f}')
if bad:
    raise SystemExit('DOES NOT MATCH table:dbs:\n  ' + '\n  '.join(bad))
print('matches table:dbs: 7 models x (capacity, balanced accuracy, specificity, AUC); '
      'the p column is asserted by make_dbs_perm.py')
