"""Rebuild table:mnist and assert every published cell.

usage: python tables/make_table_mnist.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repository root: shared modules and paths.py
import os

import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN = os.environ.get('MNIST_CLEAN_DIR', os.path.join(HERE, 'results', 'mnist_mode'))
AUG = os.environ.get('MNIST_AUG_DIR',
                     os.path.join(HERE, 'results', 'mnist_aug_cleanbasis', 'aug'))
OUT = os.path.join(HERE, 'results', 'mnist_table.tex')
SIZES = [50, 200, 1000, 5000]
COLS = [('spat', 'T1'), ('spec', 'T1'), ('spat', 'AUC'), ('spec', 'AUC')]
PUB = {('clean', 50): (0.296, 0.608, 0.699, 0.899),
       ('clean', 200): (0.435, 0.773, 0.810, 0.960),
       ('clean', 1000): (0.766, 0.873, 0.965, 0.985),
       ('clean', 5000): (0.943, 0.918, 0.998, 0.994),
       ('aug', 50): (0.362, 0.585, 0.758, 0.879),
       ('aug', 200): (0.641, 0.751, 0.926, 0.956),
       ('aug', 1000): (0.933, 0.866, 0.996, 0.986),
       ('aug', 5000): (0.976, 0.913, 1.000, 0.994)}


def block(d):
    f = pd.read_csv(os.path.join(d, 'mnist_perseed.csv'))
    mean, per_seed = {}, {}
    for n in SIZES:
        g = f[f.n == n].sort_values('seed')
        mean[n] = tuple(g[f'{a}_{m}'].mean() for a, m in COLS)
        per_seed[n] = tuple(g[f'{a}_{m}'].values for a, m in COLS)
    return mean, per_seed


def bolded(v, seeds):
    out = set()
    for lo, hi in ((0, 1), (2, 3)):
        win = lo if v[lo] > v[hi] else hi
        other = hi if win == lo else lo
        out.add(win)
        a, b = seeds[win], seeds[other]
        if len(a) == len(b) > 1 and stats.ttest_rel(a, b).pvalue >= 0.05:
            out.add(other)
    return out


rows, bad = [], []
for tag, d in (('clean', CLEAN), ('aug', AUG)):
    got, seeds = block(d)
    rows.append(r'\multicolumn{5}{@{}l}{\emph{%s}} \\' %
                ('No augmentation' if tag == 'clean' else 'Augmented'))
    for n in SIZES:
        v = got[n]
        best = bolded(v, seeds[n])
        cells = [(r'$\bf{%.3f}$' if i in best else '$%.3f$') % x for i, x in enumerate(v)]
        rows.append('$%s$ & %s \\\\' % (f'{n:,}'.replace(',', '{,}'), ' & '.join(cells)))
        for i, x in enumerate(v):
            if f'{x:.3f}' != f'{PUB[(tag, n)][i]:.3f}':
                bad.append(f'{tag} n={n} {COLS[i][0]}_{COLS[i][1]}: '
                           f'{x:.3f} against the table\'s {PUB[(tag, n)][i]:.3f}')
    if tag == 'clean':
        rows.append(r'\addlinespace')

table = '\n'.join('    ' + r for r in rows)
print(table)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, 'w').write(table + '\n')
print(f'\nwrote {OUT}')
if bad:
    raise SystemExit('DOES NOT MATCH table:mnist:\n  ' + '\n  '.join(bad))
print(f'matches table:mnist: all {len(SIZES) * len(COLS) * 2} cells')
