"""What the magnitude basis costs on age estimation: Block-|FFT| against Block-DCT.

This is the number behind the basis sentence of sec/4_results.tex -- that discarding
phase for translation invariance costs error in nearly every model-by-size cell. It was
computed inline in the session that wrote that sentence and nothing in the repo rebuilt
it, so the published figures had drifted: they were taken while n=25 and n=50 still had
five draws apiece rather than ten, which undercounted the significant cells (6 against
11) and overstated the largest cost (1.83 years against 1.58).

The loader is taken from make_table_hca_reg.py rather than restated, so this cannot read
a different set of files than the table it qualifies. Same budget (500 epochs), same
draws (10 at n=25 and n=50, 5 above), same per-draw unit of analysis.

Four families are run in both bases, which is where the 20 cells come from. The proposed
arm itself (SpectralViT rank) has no Block-|FFT| counterpart and is not among them --
'SpectralViT hcp' and 'SpectralViT small' are the two spectral capacities that are.

usage: python make_basis_cost.py [results_dir]
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results', 'basis_cost.txt')

# D, ROWS, SIZES and load() come from the table builder; stop before it uses them.
_src = open(os.path.join(HERE, 'make_table_hca_reg.py')).read()
exec(_src.split('data = {n: load(n) for n in SIZES}')[0])

FAM = ['LogisticRegression', 'MLP', 'SpectralViT hcp', 'SpectralViT small']

# As published, after this script corrected them. A change fails loudly below.
PUB_CELLS, PUB_COST, PUB_SIG, PUB_MAX = 20, 19, 11, 1.58

data = {n: load(n) for n in SIZES}
missing = [n for n in SIZES if data[n] is None]
if missing:
    raise SystemExit(f'no results for n={missing} under {D}')

L = ['Block-|FFT| minus Block-DCT, mean absolute error in years.',
     'Positive means the magnitude basis costs error. Per-draw unit of analysis,',
     'paired t-test over draws, * marks p<0.05.', '',
     '%-20s' % 'family' + ''.join('%20s' % f'n={n}' for n in SIZES)]

cells = cost = sig = 0
worst = ('', -np.inf)
for fam in FAM:
    row = '%-20s' % fam
    for n in SIZES:
        d = data[n]
        a, b = f'{fam} (Block-DCT)', f'{fam} (Block-|FFT|)'
        piv = d.pivot_table(index=['Seed', 'Fold'], columns='Model', values='MAE')
        if a not in piv or b not in piv:
            row += '%20s' % '--'
            continue
        per_draw = (piv[b] - piv[a]).groupby(level='Seed').mean()
        m = per_draw.mean()
        p = stats.ttest_1samp(per_draw, 0).pvalue
        cells += 1
        cost += m > 0
        sig += m > 0 and p < 0.05
        if m > worst[1]:
            worst = (f'{fam} at n={n}', m)
        row += '%20s' % ('%+.2f p=%.3f%s' % (m, p, '*' if p < 0.05 else ' '))
    L.append(row)

draws = ', '.join(f'n={n} {data[n].Seed.nunique()}' for n in SIZES)
L += ['', f'  draws: {draws}',
      f'  cells                {cells}',
      f'  magnitude costs in   {cost}',
      f'  significantly in     {sig}',
      f'  largest cost         {worst[1]:.2f} years ({worst[0]})']

report = '\n'.join(L)
print(report)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w') as fh:
    fh.write(report + '\n')
print(f'\nwrote {OUT}')

bad = []
if cells != PUB_CELLS:
    bad.append(f'{cells} cells, paper says {PUB_CELLS}')
if cost != PUB_COST:
    bad.append(f'costs in {cost}, paper says {PUB_COST}')
if sig != PUB_SIG:
    bad.append(f'significant in {sig}, paper says {PUB_SIG}')
if abs(worst[1] - PUB_MAX) > 0.005:
    bad.append(f'largest cost {worst[1]:.2f}, paper says {PUB_MAX:.2f}')
if bad:
    raise SystemExit('DOES NOT MATCH the basis sentence of sec:results:\n  ' + '\n  '.join(bad))
print('matches the basis sentence of sec:results: cells, direction, significance and largest cost')
