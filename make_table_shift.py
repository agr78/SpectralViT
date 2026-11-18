"""Rebuild table:shift and assert every published cell.

usage: python make_table_shift.py"""
import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.environ.get('SHIFT_DIR', os.path.join(HERE, 'results', 'sfcn_shift'))
OUT = os.path.join(HERE, 'results', 'shift_table.tex')
ROWS = [('clean', 'Uncorrupted'), ('SNR 9', 'SNR $9$'), ('SNR 4', 'SNR $4$'),
        ('SNR 1', 'SNR $1$'), ('SNR 0.25', r'SNR $\frac{1}{4}$'),
        ('SNR 0.111', r'SNR $\frac{1}{9}$')]
PUB = {
    'clean':     ('0.991', '0.013', '0.931', '0.007', '-0.060', '0.0016'),
    'SNR 9':     ('0.991', '0.015', '0.932', '0.009', '-0.059', '0.0043'),
    'SNR 4':     ('0.986', '0.021', '0.930', '0.014', '-0.056', '0.021'),
    'SNR 1':     ('0.479', '0.242', '0.883', '0.008', '+0.404', '0.020'),
    'SNR 0.25':  ('0.308', '0.124', '0.795', '0.054', '+0.487', '0.0010'),
    'SNR 0.111': ('0.454', '0.129', '0.747', '0.053', '+0.293', '0.015')}
PUB_WINS = {'clean': 0, 'SNR 9': 0, 'SNR 4': 1, 'SNR 1': 5,
            'SNR 0.25': 5, 'SNR 0.111': 5}

f = os.path.join(D, 'sfcn_shift_folds.csv')
if not os.path.exists(f):
    raise SystemExit(f'missing {f}')
d = pd.read_csv(f, usecols=['fold', 'shift_label', 'sfcn_auc', 'spec_auc'])


def at(x, pub, sign=False):
    k = len(pub.split('.')[1])
    return f'{x:+.{k}f}' if sign else f'{x:.{k}f}'


rows, bad = [], []
for lab, name in ROWS:
    g = d[d.shift_label == lab].sort_values('fold')
    if len(g) != 5:
        raise SystemExit(f'{lab}: {len(g)} folds, expected 5')
    a, b = g.sfcn_auc.to_numpy(), g.spec_auc.to_numpy()
    got = (a.mean(), a.std(ddof=1), b.mean(), b.std(ddof=1), b.mean() - a.mean(),
           stats.ttest_rel(b, a).pvalue)
    pub = PUB[lab]
    cells = [at(got[i], pub[i], sign=(i == 4)) for i in range(6)]
    for i, (c, p) in enumerate(zip(cells, pub)):
        if c != p:
            col = ('SFCN AUC', 'SFCN sd', 'spectral AUC', 'spectral sd',
                   'delta AUC', 'p')[i]
            bad.append(f'{lab} {col}: {c} against the table\'s {p}')
    wins = int((b > a).sum())
    if wins != PUB_WINS[lab]:
        bad.append(f'{lab} folds won by the spectral ViT: {wins} '
                   f"against the caption's {PUB_WINS[lab]}")
    pair = [r'$%s \pm %s$' % (cells[0], cells[1]),
            r'$%s \pm %s$' % (cells[2], cells[3])]
    w = 0 if a.mean() > b.mean() else 1
    pair[w] = r'$\mathbf{%s \pm %s}$' % (cells[2 * w], cells[2 * w + 1])
    rows.append('%-18s & %-26s & %-26s & $%s$ & $%s$ \\\\'
                % (name, pair[0], pair[1], cells[4], cells[5]))

table = '\n'.join('    ' + r for r in rows)
print(table)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, 'w').write(table + '\n')
print(f'\nwrote {OUT}')
if bad:
    raise SystemExit('DOES NOT MATCH table:shift:\n  ' + '\n  '.join(bad))
print(f'matches table:shift: all {len(ROWS) * 7} cells')
