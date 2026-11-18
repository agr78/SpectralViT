"""Rebuild table:modelnet and assert every published cell.

usage: python make_table_modelnet.py"""
import os

import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.environ.get('MODELNET_DIR',
                   os.path.join(HERE, 'results', 'recovered', 'modelnet_run'))
OUT = os.path.join(HERE, 'results', 'modelnet_table.tex')
SIZES = [40, 80, 200, 400, 800]
ARMS = [('laplacian', 'Laplacian'), ('pca', 'PCA'), ('learned', 'Learned'),
        ('rand', 'Random'), ('vertex', 'Vertices')]
FIXED = {'laplacian', 'rand', 'vertex'}
PUB = {
    'T1': {40: (0.318, 0.298, 0.247, 0.246, 0.221),
           80: (0.387, 0.394, 0.341, 0.327, 0.293),
           200: (0.475, 0.502, 0.444, 0.408, 0.395),
           400: (0.534, 0.567, 0.524, 0.474, 0.472),
           800: (0.581, 0.630, 0.595, 0.555, 0.518)},
    'AUC': {40: (0.754, 0.756, 0.724, 0.731, 0.720),
            80: (0.810, 0.820, 0.794, 0.785, 0.778),
            200: (0.872, 0.892, 0.873, 0.857, 0.855),
            400: (0.920, 0.931, 0.922, 0.909, 0.907),
            800: (0.935, 0.950, 0.946, 0.931, 0.927)}}

d = {}
for tag, _ in ARMS:
    f = os.path.join(D, f'modelnet_{tag}.csv')
    if not os.path.exists(f):
        raise SystemExit(f'missing {f}')
    d[tag] = pd.read_csv(f)

rows, bad = [], []
for metric in ('T1', 'AUC'):
    rows.append(r'\multicolumn{6}{@{}l}{\emph{%s}} \\' %
                ('Top-1 accuracy' if metric == 'T1' else 'AUC'))
    for n in SIZES:
        per_seed = {t: d[t][d[t].n == n].sort_values('seed')[metric].to_numpy()
                    for t, _ in ARMS}
        mu = {t: v.mean() for t, v in per_seed.items()}
        top = max(FIXED, key=lambda t: mu[t])
        keep = {t for t in FIXED
                if t == top or stats.ttest_rel(per_seed[top], per_seed[t]).pvalue >= 0.05}
        cells = []
        for i, (t, _) in enumerate(ARMS):
            cells.append((r'$\bf{%.3f}$' if t in keep else '$%.3f$') % mu[t])
            if f'{mu[t]:.3f}' != f'{PUB[metric][n][i]:.3f}':
                bad.append(f'{metric} n={n} {t}: {mu[t]:.3f} '
                           f"against the table's {PUB[metric][n][i]:.3f}")
        rows.append('$%s$ & %s \\\\' % (f'{n:,}'.replace(',', '{,}'), ' & '.join(cells)))
    if metric == 'T1':
        rows.append(r'\midrule')

table = '\n'.join('    ' + r for r in rows)
print(table)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, 'w').write(table + '\n')
print(f'\nwrote {OUT}')
if bad:
    raise SystemExit('DOES NOT MATCH table:modelnet:\n  ' + '\n  '.join(bad))
print(f'matches table:modelnet: all {len(SIZES) * len(ARMS) * 2} cells')
