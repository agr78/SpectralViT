"""What the magnitude basis costs on age estimation: Block-|FFT| against Block-DCT.

usage: python tables/make_basis_cost.py [results_dir]"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repository root: shared modules and paths.py
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, 'results', 'basis_cost.txt')

_src = open(os.path.join(HERE, 'tables', 'make_table_hca_reg.py')).read()
exec(_src.split('data = {n: load(n) for n in SIZES}')[0])

FAM = ['LogisticRegression', 'MLP', 'SpectralViT small', 'SpectralViT hcp']
ROW = {'LogisticRegression': 'Logistic regression',
       'MLP': 'Multilayer perceptron',
       'SpectralViT small': 'Spectral ViT, 1 layer',
       'SpectralViT hcp': 'Spectral ViT, 2 layers'}

PUB_CELLS, PUB_COST, PUB_SIG, PUB_MAX = 20, 19, 11, 1.58

PUB = {
    'LogisticRegression': {
        25: ('-0.19', '0.757', False), 50: ('+0.81', '0.019', True),
        100: ('+0.37', '0.042', True), 200: ('+0.08', '0.603', False),
        1000: ('+0.09', '0.312', False)},
    'MLP': {
        25: ('+0.88', '0.007', True), 50: ('+0.60', '0.003', True),
        100: ('+0.33', '0.306', False), 200: ('+0.19', '0.162', False),
        1000: ('+0.29', '0.133', False)},
    'SpectralViT small': {
        25: ('+1.18', '0.072', False), 50: ('+1.37', '0.052', False),
        100: ('+1.58', '<0.001', True), 200: ('+0.99', '0.006', True),
        1000: ('+0.80', '0.002', True)},
    'SpectralViT hcp': {
        25: ('+1.41', '0.076', False), 50: ('+1.56', '0.031', True),
        100: ('+1.16', '<0.001', True), 200: ('+1.12', '0.047', True),
        1000: ('+0.52', '0.001', True)},
}

data = {n: load(n) for n in SIZES}
missing = [n for n in SIZES if data[n] is None]
if missing:
    raise SystemExit(f'no results for n={missing} under {D}')

L = ['Block-|FFT| minus Block-DCT, mean absolute error in years.',
     'Positive means the magnitude basis costs error. Per-draw unit of analysis,',
     'paired t-test over draws, * marks p<0.05.', '',
     '%-20s' % 'family' + ''.join('%20s' % f'n={n}' for n in SIZES)]

cells = cost = sig = enc = 0
worst = ('', -np.inf)
bad = []
for fam in FAM:
    row = '%-20s' % fam
    for n in SIZES:
        d = data[n]
        a, b = f'{fam} (Block-DCT)', f'{fam} (Block-|FFT|)'
        piv = d.pivot_table(index=['Seed', 'Fold'], columns='Model', values='MAE')
        want = PUB[fam][n]
        if a not in piv or b not in piv:
            row += '%20s' % '--'
            bad.append(f'{ROW[fam]} n={n}: no arm pair in the results, '
                       f"the table prints {want[0]} ({want[1]})")
            continue
        per_draw = (piv[b] - piv[a]).groupby(level='Seed').mean()
        m = per_draw.mean()
        p = stats.ttest_1samp(per_draw, 0).pvalue
        cells += 1
        cost += m > 0
        sig += m > 0 and p < 0.05
        enc += fam.startswith('SpectralViT') and m > 0 and p < 0.05
        if m > worst[1]:
            worst = (f'{fam} at n={n}', m)
        row += '%20s' % ('%+.2f p=%.3f%s' % (m, p, '*' if p < 0.05 else ' '))
        got = (f'{m:+.2f}', '<0.001' if p < 0.001 else f'{p:.3f}', p < 0.05)
        if got != want:
            bad.append("%s n=%d: %s (%s)%s against the table's %s (%s)%s"
                       % (ROW[fam], n, got[0], got[1], ' bold' if got[2] else '',
                          want[0], want[1], ' bold' if want[2] else ''))
    L.append(row)

draws = ', '.join(f'n={n} {data[n].Seed.nunique()}' for n in SIZES)
L += ['', f'  draws: {draws}',
      f'  cells                {cells}',
      f'  magnitude costs in   {cost}',
      f'  significantly in     {sig} (encoders {enc}, heads {sig - enc})',
      f'  largest cost         {worst[1]:.2f} years ({worst[0]})']

report = '\n'.join(L)
print(report)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w') as fh:
    fh.write(report + '\n')
print(f'\nwrote {OUT}')

if bad:
    raise SystemExit('DOES NOT MATCH table:magcost:\n  ' + '\n  '.join(bad))
print(f'matches table:magcost: all {len(PUB) * len(SIZES)} cells '
      '(difference, p-value and bold mark)')

prose = []
if cells != PUB_CELLS:
    prose.append(f'{cells} cells, paper says {PUB_CELLS}')
if cost != PUB_COST:
    prose.append(f'costs in {cost}, paper says {PUB_COST}')
if sig != PUB_SIG:
    prose.append(f'significant in {sig}, paper says {PUB_SIG}')
if f'{worst[1]:.2f}' != f'{PUB_MAX:.2f}':
    prose.append(f'largest cost {worst[1]:.2f}, paper says {PUB_MAX:.2f}')
if prose:
    raise SystemExit('DOES NOT MATCH the basis sentence of sec:magcost:\n  ' + '\n  '.join(prose))
print('matches the basis sentence of sec:magcost: cells, direction, significance and largest cost')
