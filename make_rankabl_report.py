"""Regenerate the rank weight ablation figures quoted in sec:rankabl.

Three arms share the basis, tokenization and encoder and differ only in the rank weights:
omitted (hcp), imposed at 1/i (fixrank), and imposed then trained (rank).

The source file per sample size is pinned below rather than globbed. Globbing was how the
published numbers went wrong: results/regress holds two epoch budgets for the same sizes,
and the quoted figures turned out to come from the earlier one while table:hca-reg used
the 500-epoch runs, so the ablation and the main table disagreed -- 0.97 against 2.28
years at n=1000. The selection here is the one that reproduces the table's proposed row
(7.64, 7.05, 6.20, 5.88, 5.23), which is asserted below so drift fails loudly.

Per combine_hca_10seed.py: n=25 and n=50 have ten seeds in the combined _ep500_10s file,
while n=100, n=200 and n=1000 have five and live in per-seed files.

usage: python make_rankabl_report.py
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results', 'regress')
OUT = os.path.join(HERE, 'results', 'rankabl_report.txt')

SRC = {
    25:   [os.path.join(R, 'hca_reg_n25_ep500_10s.csv')],
    50:   [os.path.join(R, 'hca_reg_n50_ep500_10s.csv')],
    100:  [os.path.join(R, f'hca_reg_n100_ep500_s{s}.csv') for s in range(5)],
    200:  [os.path.join(R, f'hca_reg_n200_ep500_s{s}.csv') for s in range(5)],
    1000: [os.path.join(R, f'hca_reg_n1000_ep500_s{s}.csv') for s in range(5)],
}
ARMS = {'omit': 'SpectralViT hcp (Block-DCT)',
        'fixed': 'SpectralViT fixrank (Block-DCT)',
        'trained': 'SpectralViT rank (Block-DCT)',
        'mean': 'MeanBaseline'}
SIZES = [25, 50, 100, 200, 1000]
# table:hca-reg's proposed row. The trained arm is that row, so a mismatch means the
# source selection has drifted away from what the main table reports.
TABLE_ROW = [7.64, 7.05, 6.20, 5.88, 5.23]


def size(paths):
    """Mean MAE per arm over seeds, each seed summarized by its mean over folds."""
    cell = {}
    for p in paths:
        d = pd.read_csv(p)
        d = d[d.MAE.notna()]
        # One entry per (arm, seed): the same seed appears in more than one file for some
        # sizes, and pooling folds across files would silently reweight it.
        for (model, seed), g in d.groupby(['Model', 'Seed']):
            cell[(model, int(seed))] = g.MAE.mean()
    out = {k: float(np.mean([v for (m, _), v in cell.items() if m == a]))
           for k, a in ARMS.items()}
    out['nseed'] = len({s for (m, s) in cell if m == ARMS['trained']})
    return out


res = {n: size(p) for n, p in SRC.items()}

L = ['Rank weight ablation on HCP-Aging brain age, 500-epoch budget.', '']
L.append(f"{'n':>6} {'seeds':>5} {'omit':>7} {'fixed':>7} {'trained':>8} {'mean':>7}"
         f" {'cost':>7} {'gain':>7}")
for n in SIZES:
    r = res[n]
    L.append(f"{n:>6} {r['nseed']:>5} {r['omit']:7.2f} {r['fixed']:7.2f} {r['trained']:8.2f}"
             f" {r['mean']:7.2f} {r['fixed'] - r['omit']:7.2f} {r['omit'] - r['trained']:7.2f}")

cost = [res[n]['fixed'] - res[n]['omit'] for n in SIZES]
gain = [res[n]['omit'] - res[n]['trained'] for n in SIZES]
L += ['',
      'cost of imposing 1/i rather than omitting it (sec:rankabl sentence 3):',
      '  ' + ', '.join(f'{c:.2f}' for c in cost),
      f'gain from training it instead: between {min(gain):.2f} and {max(gain):.2f}',
      f"at n=25 the fixed profile reaches {res[25]['fixed']:.2f} against the training-mean"
      f" baseline's {res[25]['mean']:.2f}"]

trained = [res[n]['trained'] for n in SIZES]
assert np.allclose(trained, TABLE_ROW, atol=0.005), \
    f'trained arm {[round(t, 2) for t in trained]} != table:hca-reg row {TABLE_ROW}'
assert all(c > 0 for c in cost), f'imposing 1/i no longer costs at every size: {cost}'
L.append('')
L.append('checks: trained arm matches table:hca-reg; imposing 1/i costs at every size.')

report = '\n'.join(L)
print(report)
with open(OUT, 'w') as fh:
    fh.write(report + '\n')
print(f'\nwrote {OUT}')
