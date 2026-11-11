"""Build table:nsens, the HCP-Aging coefficient-count sensitivity table.

Every cell is SpectralViT rank (Block-DCT), pooled withheld-fold AUC, mean over draws.
The coefficient count is set per run, not per arm, so each row is a separate sweep:

    512    hca_pooled_n{N}_f2.csv    HCA_FREQS=2
    1,728  hca_pooled_n{N}_f3.csv    HCA_FREQS=3
    4,096  hca_pooled_n{N}.csv       the default, also table:hca-cls's spectral row
    sel    hca_pooled_n{N}_cv.csv    HCA_FREQS=2,3,4, chosen by inner CV

Seeds 0-4 are in those files; seeds 5-9 at n=25 and n=50 are in per-seed companions
written by run_hca_draws10.sbatch / run_hca_cadiz.sh, and are unioned here. Draws are
counted PER MODEL, not per file: the backfill deliberately raised only the arms any
reported number depends on, so a file can hold ten draws of one arm and five of another.
A cell whose draw count is not the expected one is refused rather than printed.

n=100 was not backfilled, so it must still reproduce the published row -- that is the
guard. If it moves, the loader changed and the table should not be rebuilt.

usage: python make_table_nsens.py [out.tex]
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'results', 'nsens_table.tex')
D = os.environ.get('HCA_SWEEP_DIR', os.path.join(HERE, 'results', 'sweep'))

MODEL = 'SpectralViT rank (Block-DCT)'
SIZES = [25, 50, 100]
DRAWS = {25: 10, 50: 10, 100: 5}
# NSENS_DRAWS=5 reproduces the table as published, before the n=25/50 backfill. Kept so
# the loader can be checked against a known-good answer rather than only against data it
# has never been run on.
if os.environ.get('NSENS_DRAWS'):
    DRAWS = {n: int(os.environ['NSENS_DRAWS']) for n in SIZES}
    DRAWS[100] = 5
ROWS = [('$512$', '_f2'), ('$1{,}728$', '_f3'), ('$\\bf{4{,}096}$', ''),
        ('Selected per fold', '_cv')]
# published n=100 column, which this run does not touch
PUB_N100 = {'_f2': 0.889, '_f3': 0.930, '': 0.928, '_cv': 0.901}


def draws(suf, n):
    """Pooled AUC per draw for one (coefficient setting, size), unioned over seed files."""
    files = [os.path.join(D, f'hca_pooled_n{n}{suf}.csv')]
    files += [os.path.join(D, f'hca_pooled_n{n}{suf}_s{s}.csv') for s in range(5, 10)]
    got = []
    for f in files:
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f)
        got.append(d[d.Model == MODEL][['Seed', 'AUC_pooled']])
    if not got:
        raise SystemExit(f'no file for suffix {suf!r} at n={n} under {D}')
    v = pd.concat(got, ignore_index=True).drop_duplicates('Seed').sort_values('Seed')
    return v.Seed.to_numpy(), v.AUC_pooled.to_numpy()


cells, counts, bad = {}, {}, []
for _, suf in ROWS:
    for n in SIZES:
        seeds, a = draws(suf, n)
        counts[(suf, n)] = len(a)
        if len(a) != DRAWS[n]:
            bad.append(f'{suf or "4096"} n={n}: {len(a)} draws (seeds {list(seeds)}), '
                       f'expected {DRAWS[n]}')
        cells[(suf, n)] = (float(np.mean(a)), float(np.std(a, ddof=1)))

if bad:
    raise SystemExit('wrong draw count, refusing to build:\n  ' + '\n  '.join(bad))

moved = {s: (round(cells[(s, 100)][0], 3), v) for s, v in PUB_N100.items()
         if abs(cells[(s, 100)][0] - v) > 0.0006}
if moved:
    raise SystemExit(f'n=100 was not rerun but moved, got vs published: {moved}')

best = {n: max(cells[(s, n)][0] for _, s in ROWS) for n in SIZES}
sd = [v[1] for v in cells.values()]
L = [r'\begin{table}[t]',
     '  \\caption{Age at the median split on HCP-Aging, pooled AUC over independent '
     'subsamples, of which there are $' + str(DRAWS[25]) + '$ at $n=25$ and $n=50$ and $'
     + str(DRAWS[100]) + '$ at $n=100$, as the retained coefficient count varies. Standard '
     'deviations range from $' + f'{min(sd):.3f}' + '$ to $' + f'{max(sd):.3f}' + '$ and '
     'are largest at the smallest sample. The final row selects the count inside each outer '
     'fold instead of fixing it. Bold marks the setting reported elsewhere.}',
     r'  \label{table:nsens}', r'  \centering', r'  \begin{small}',
     r'  \begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lccc@{}}',
     r'    \toprule',
     r'    Coefficients & ' + ' & '.join(f'$n{{=}}{n}$' for n in SIZES) + r' \\',
     r'    \midrule']
for i, (lab, suf) in enumerate(ROWS):
    if lab.startswith('Selected'):
        L.append(r'    \midrule')
    out = []
    for n in SIZES:
        m, _ = cells[(suf, n)]
        out.append(f'$\\bf{{{m:.3f}}}$' if abs(m - best[n]) < 1e-12 else f'${m:.3f}$')
    L.append(f'    {lab:17s} & ' + ' & '.join(f'{c:12s}' for c in out).rstrip() + r' \\')
L += [r'    \bottomrule', r'  \end{tabular*}', r'  \end{small}', r'\end{table}']

os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
print('\ndraws per cell: ' + ', '.join(
    f'{s or "4096"}@{n}={counts[(s, n)]}' for _, s in ROWS for n in SIZES))
print(f'n=100 matches the published column\nwrote {OUT}')
