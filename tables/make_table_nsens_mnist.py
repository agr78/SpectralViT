"""Rebuild table:nsens-mnist and assert every published cell.

usage: python tables/make_table_nsens_mnist.py [out.tex]"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repository root: shared modules and paths.py
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'results',
                                                         'nsens_mnist_table.tex')
D = os.environ.get('MNIST_COMP_DIR', os.path.join(HERE, 'results', 'recovered',
                                                  'mnist_comp'))
SIZES = [50, 200, 1000, 5000]
SEEDS = 5
REPORTED = 16
LABEL = 'table:nsens-mnist and its sec:nsens prose'
ROWS = [(8, 'c8'), (16, 'c16'), (32, 'c32'), (64, 'c64')]
PUB = {8: (0.574, 0.721, 0.798, 0.852),
       16: (0.608, 0.773, 0.873, 0.918),
       32: (0.583, 0.791, 0.884, 0.927),
       64: (0.584, 0.780, 0.885, 0.927)}
PUB_SPAT = (0.296, 0.435, 0.766, 0.943)
PUB_MOVE = {50: 0.025, 5000: 0.009}

spec, spat, bad = {}, {}, []
for comp, sub in ROWS:
    f = os.path.join(D, sub, 'mnist_perseed.csv')
    if not os.path.exists(f):
        raise SystemExit(f'missing {f}')
    d = pd.read_csv(f)
    for n in SIZES:
        v = d[d.n == n].sort_values('seed')
        if len(v) != SEEDS:
            raise SystemExit(f'{sub} n={n}: {len(v)} seeds, expected {SEEDS}')
        spec[(comp, n)] = float(v.spec_T1.mean())
        spat[(comp, n)] = float(v.spat_T1.mean())

for comp, _ in ROWS:
    for i, n in enumerate(SIZES):
        if f'{spec[(comp, n)]:.3f}' != f'{PUB[comp][i]:.3f}':
            bad.append(f'{comp} components n={n}: {spec[(comp, n)]:.3f} '
                       f"against the table's {PUB[comp][i]:.3f}")
for i, n in enumerate(SIZES):
    if f'{spat[(REPORTED, n)]:.3f}' != f'{PUB_SPAT[i]:.3f}':
        bad.append(f'caption spatial n={n}: {spat[(REPORTED, n)]:.3f} '
                   f"against the caption's {PUB_SPAT[i]:.3f}")
for n, pub in PUB_MOVE.items():
    mv = abs(spec[(32, n)] - spec[(REPORTED, n)])
    alt = abs(spec[(64, n)] - spec[(REPORTED, n)])
    if f'{mv:.3f}' != f'{pub:.3f}':
        bad.append(f'prose 16 to 32 components at n={n}: {mv:.3f} against the '
                   f"text's {pub:.3f} (16 to 64 there is {alt:.3f})")

best = {n: f'{max(spec[(c, n)] for c, _ in ROWS):.3f}' for n in SIZES}
L = [r'\begin{table}[t]',
     '  \\caption{Digit classification on MNIST, top-1 accuracy over $' + str(SEEDS) +
     '$ seeds, as the retained principal component count varies. Bold marks the best '
     'count at each size; the reported setting is $' + str(REPORTED) + '$. The spatial '
     'baseline, unaffected by this choice, attains $' +
     '$, $'.join(f'{spat[(REPORTED, n)]:.3f}' for n in SIZES[:-1]) + '$ and $' +
     f'{spat[(REPORTED, SIZES[-1])]:.3f}' + '$.}',
     r'  \label{table:nsens-mnist}', r'  \centering', r'  \begin{small}',
     r'  \begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lcccc@{}}',
     r'    \toprule',
     r'    Components & ' + ' & '.join(
         f'$n{{=}}{n:,}$'.replace(',', '{,}') for n in SIZES) + r' \\',
     r'    \midrule']
for comp, _ in ROWS:
    lab = f'$\\bf{{{comp}}}$' if comp == REPORTED else f'${comp}$'
    out = []
    for n in SIZES:
        s = f'{spec[(comp, n)]:.3f}'
        out.append(f'$\\bf{{{s}}}$' if s == best[n] else f'${s}$')
    L.append(f'    {lab:12s} & ' + ' & '.join(f'{c:12s}' for c in out).rstrip() + r' \\')
L += [r'    \bottomrule', r'  \end{tabular*}', r'  \end{small}', r'\end{table}']

os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
print('\nspatial top-1 by component count, which the caption calls unaffected:')
for n in SIZES:
    v = [spat[(c, n)] for c, _ in ROWS]
    print(f'  n={n:5d}: ' + ' '.join(f'{x:.3f}' for x in v) +
          f'  (spread {max(v) - min(v):.3f})')
print(f'\nwrote {OUT}')
if bad:
    raise SystemExit(f'DOES NOT MATCH {LABEL}:\n  ' + '\n  '.join(bad))
print(f'matches {LABEL}: all '
      f'{len(ROWS) * len(SIZES) + len(SIZES) + len(PUB_MOVE)} cells')
