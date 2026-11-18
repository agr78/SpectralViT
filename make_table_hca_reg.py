"""Build and assert table:hca-reg from the fixed-epoch HCA regression results.

usage: python make_table_hca_reg.py [results_dir]"""
import os, sys
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'results', 'regress')
ROWS, SIZES = 75, [25, 50, 100, 200, 1000]
PROP = 'SpectralViT rank (Block-DCT)'
LAB = [('Spectral ViT (proposed)', PROP),
       ('U-net with attention', 'AttnUNet (Micro)'),
       ('Logistic regression', 'LogisticRegression (Block-DCT)'),
       ('Multilayer perceptron', 'MLP (Block-DCT)'),
       ('Spatial ViT', 'SpatialViT (Micro)'),
       ('Swin ViT', 'Swin (Micro)'),
       ('Training-mean baseline', 'MeanBaseline')]
PUB = {
    PROP: (10149, {25: (7.64, 1.14), 50: (7.05, 0.84), 100: (6.20, 0.08),
                   200: (5.88, 0.22), 1000: (5.23, 0.16)}),
    'AttnUNet (Micro)': (18387, {25: (9.04, 1.04), 50: (7.95, 0.73), 100: (6.63, 0.23),
                                 200: (6.11, 0.28), 1000: (5.20, 0.10)}),
    'LogisticRegression (Block-DCT)': (4097, {25: (11.07, 1.22), 50: (8.43, 0.75),
                                              100: (6.51, 0.24), 200: (5.92, 0.30),
                                              1000: (5.20, 0.11)}),
    'MLP (Block-DCT)': (43768, {25: (9.05, 0.96), 50: (7.96, 0.89), 100: (6.64, 0.36),
                                200: (5.99, 0.32), 1000: (4.92, 0.31)}),
    'SpatialViT (Micro)': (65554, {25: (10.86, 1.92), 50: (9.39, 0.84), 100: (7.95, 1.38),
                                   200: (6.30, 0.25), 1000: (5.05, 0.23)}),
    'Swin (Micro)': (13682, {25: (10.68, 1.69), 50: (10.41, 1.04), 100: (9.88, 0.93),
                             200: (9.04, 1.17), 1000: (7.40, 0.17)}),
    'MeanBaseline': (None, {25: (12.96, 0.67), 50: (13.49, 0.88), 100: (13.79, 1.11),
                            200: (14.09, 1.04), 1000: (13.84, 0.06)}),
}
BOLD = {25: PROP, 50: PROP, 100: PROP, 200: PROP, 1000: 'MLP (Block-DCT)'}
BOLD_PARAMS = 'LogisticRegression (Block-DCT)'
DRAWS = {25: 10, 50: 10, 100: 5, 200: 5, 1000: 5}
PMAX = 0.005


def load(n, want=int(os.environ.get("HCA_WANT", 10))):
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

bad = [f'n={n} draws: {have[n]} against the caption\'s {DRAWS[n]}'
       for n in SIZES if have[n] != DRAWS[n]]


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
        pm, ps = PUB[m][1][n]
        if (f'{mu:.2f}', f'{sd:.2f}') != (f'{pm:.2f}', f'{ps:.2f}'):
            bad.append(f'{m} n={n}: {mu:.2f} +/- {sd:.2f} '
                       f"against the table's {pm:.2f} +/- {ps:.2f}")
    print(row)
print('* best in column')
bad += [f"n={n} bold: {best[n]} against the table's {BOLD[n]}"
        for n in cols if best[n] != BOLD[n]]
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
worst = 0.0
for n in cols:
    piv = data[n].pivot_table(index=['Seed', 'Fold'], columns='Model', values='MAE')
    for _, m in LAB:
        if m != 'MeanBaseline':
            sm = (piv['MeanBaseline'] - piv[m]).groupby(level='Seed').mean()
            worst = max(worst, stats.ttest_1samp(sm, 0, alternative='greater').pvalue)
print('improvement over the training-mean baseline, one-sided paired t over draws:')
print('  largest p over %d cells: %.3f' % (len(cols) * (len(LAB) - 1), worst))
if float(f'{worst:.3f}') > PMAX:
    bad.append(f"mean-estimate p: {worst:.3f} against the caption's <= {PMAX:.3f}")
print()
print('parameters (mean over folds and draws, n=25):')
if data[25] is None:
    raise SystemExit('DOES NOT MATCH table:hca-reg:\n  ' + '\n  '.join(
        bad + ['parameters: no n=25 results under ' + D]))
counted = {}
for lab, m in LAB:
    p = data[25][data[25].Model == m].Params.mean()
    counted[m] = p
    got = '--' if not np.isfinite(p) or p == 0 else f'{int(round(p)):,}'
    print('  %-24s %10s' % (lab, got))
    pub = PUB[m][0]
    want = '--' if pub is None else f'{pub:,}'
    if got != want:
        bad.append(f"{m} params: {got} against the table's {want}")
smallest = min((v, m) for m, v in counted.items() if np.isfinite(v) and v > 0)[1]
if smallest != BOLD_PARAMS:
    bad.append(f"params bold: {smallest} against the table's {BOLD_PARAMS}")

if bad:
    raise SystemExit('DOES NOT MATCH table:hca-reg:\n  ' + '\n  '.join(bad))
print()
print(f'matches table:hca-reg: all {len(PUB) * (len(SIZES) + 1)} cells')
