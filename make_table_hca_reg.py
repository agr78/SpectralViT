"""Build table:hca-reg from the fixed-epoch HCA regression results.

Reads per-seed files in preference to the combined ones. That is not a style choice: a
long-running serial driver still writes the un-suffixed hca_reg_n{N}_ep500.csv from
scratch as it reaches each size, so that file can be a partial rewrite at any moment
(it was 39 rows for n=200 on Oct 18). The per-seed files one task each are append-only
and never revisited, so they are the durable record. Seeds 0-4 of n=25 and n=50 have no
per-seed files -- a single process produced them -- so those fall back to the combined
file, which the driver has long since moved past.

usage: python make_table_hca_reg.py [results_dir]
"""
import os, sys
import numpy as np
import pandas as pd
from scipy import stats

# Repo-relative, not ~-relative: $HOME differs between the cluster account and the
# container the analysis sometimes runs in, and the ~ form silently resolved to a
# nonexistent tree there.
D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'results', 'regress')
ROWS, SIZES = 75, [25, 50, 100, 200, 1000]
PROP = 'SpectralViT rank (Block-DCT)'
LAB = [('Spectral ViT (proposed)', PROP),
       ('U-net with attention', 'AttnUNet (Micro)'),
       ('Logistic regression', 'LogisticRegression (Block-DCT)'),
       ('Multilayer perceptron', 'MLP (Block-DCT)'),
       ('Spatial ViT', 'SpatialViT (Micro)'),
       ('Swin ViT', 'Swin (Micro)'),
       ('Training-mean baseline', 'MeanBaseline')]


def load(n, want=int(os.environ.get("HCA_WANT", 10))):
    """Every complete seed of one size, per-seed files first."""
    frames = []
    for s in range(want):
        f = os.path.join(D, f'hca_reg_n{n}_ep500_s{s}.csv')
        if os.path.exists(f):
            d = pd.read_csv(f)
            if len(d) == ROWS:
                frames.append(d.assign(Seed=s))
            continue
        if s < 5:
            pooled = os.path.join(D, f'hca_reg_n{n}_ep500.csv')
            if os.path.exists(pooled):
                d = pd.read_csv(pooled)
                d = d[d.Seed == s]
                if len(d) == ROWS:
                    frames.append(d.copy())
    return pd.concat(frames, ignore_index=True) if frames else None


data = {n: load(n) for n in SIZES}
have = {n: (0 if data[n] is None else data[n].Seed.nunique()) for n in SIZES}
cols = [n for n in SIZES if have[n] >= 5]
print('seeds available: ' + ', '.join(f'n={n} {have[n]}' for n in SIZES))
print()


def cell(n, m):
    sm = data[n][data[n].Model == m].groupby('Seed').MAE.mean()
    return sm.mean(), sm.std(ddof=1)


print('mean +/- sd of per-seed MAE, years (fixed-epoch budget, 500 epochs)')
print('%-24s' % 'model' + ''.join('%18s' % f'n={n} ({have[n]}s)' for n in cols))
best = {n: min((cell(n, m)[0], m) for _, m in LAB if m != 'MeanBaseline')[1] for n in cols}
for lab, m in LAB:
    row = '%-24s' % lab
    for n in cols:
        mu, sd = cell(n, m)
        row += '%18s' % ('%.2f +/- %.2f%s' % (mu, sd, '*' if m == best[n] else ' '))
    print(row)
print('* best in column')
print()
print('paired t-test on per-seed MAE, proposed vs each baseline (delta>0 favors proposed)')
print('%-24s' % 'baseline' + ''.join('%22s' % f'n={n}' for n in cols))
for lab, m in LAB:
    if m in (PROP, 'MeanBaseline'):
        continue
    row = '%-24s' % lab
    for n in cols:
        piv = data[n].pivot_table(index=['Seed', 'Fold'], columns='Model', values='MAE')
        sm = (piv[m] - piv[PROP]).groupby(level='Seed').mean()
        row += '%22s' % ('%+.2f %d/%d p=%.3f' % (sm.mean(), (sm > 0).sum(), len(sm),
                                                 stats.ttest_1samp(sm, 0).pvalue))
    print(row)
print()
print('parameters (mean over folds, n=25):')
for lab, m in LAB:
    p = data[25][data[25].Model == m].Params.mean()
    print('  %-24s %10s' % (lab, '--' if not np.isfinite(p) or p == 0 else f'{int(round(p)):,}'))
