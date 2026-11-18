"""Regenerate table:medmnist, the MedMNIST3D sample-size curve.

usage: python make_medmnist_curve.py"""
import os

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_rel

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results', 'medmnist_curve.txt')
NS = [25, 50, 100, 200, 400, 800]
DS = ['organmnist3d', 'nodulemnist3d', 'adrenalmnist3d',
      'fracturemnist3d', 'vesselmnist3d', 'synapsemnist3d']
NICE = dict(zip(DS, ['Organ', 'Nodule', 'Adrenal', 'Fracture', 'Vessel', 'Synapse']))

PUB = {
    'Organ':    [+.068, +.039, +.014, +.001, -.003, -.003],
    'Nodule':   [+.156, +.111, +.100, +.077, +.019, +.015],
    'Adrenal':  [+.033, +.057, +.048, +.049, +.014, +.057],
    'Fracture': [+.025, -.005, +.015, -.001, +.003, +.014],
    'Vessel':   [+.067, +.040, +.057, +.031, +.016, -.010],
    'Synapse':  [+.013, -.016, -.049, -.027, -.036, -.038],
}
PUB_POOLED = [+.060, +.038, +.031, +.022, +.002, +.006]
PUB_WON = ['6/6', '4/6', '4/6', '3/6', '0/6', '1/6']

SEL = {}
for n in NS:
    f = os.path.join(HERE, 'results', f'ladder_report_n{n}.tsv')
    if not os.path.exists(f):
        continue
    for line in open(f):
        p = line.rstrip('\n').split('\t')
        SEL[(n, p[0])] = (p[3], int(p[1]), p[2])


def cell(n, ds):
    if (n, ds) not in SEL:
        return None
    cfg, patch, tag = SEL[(n, ds)]
    f = os.path.join(HERE, 'results', 'ladder_100', tag, ds, 'medmnist_perseed.csv')
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f).sort_values('seed')
    return cfg, patch, d.spec_AUC.values, d.spat_AUC.values


L = ['MedMNIST3D sample-size curve: AUC delta (spectral - spatial).', '']
L.append(f"{'cohort':<10}" + ''.join(f'{"n="+str(n):>12}' for n in NS))
got, bad = {}, []
for ds in DS:
    row, line = [], f'{NICE[ds]:<10}'
    for n in NS:
        c = cell(n, ds)
        if c is None:
            row.append(None)
            line += f'{"--":>12}'
            continue
        _, _, v, b = c
        d = v - b
        p = ttest_rel(v, b).pvalue
        row.append(d.mean())
        mark = '*' if (d.mean() > 0 and p < 0.05) else ('d' if (d.mean() < 0 and p < 0.05) else ' ')
        line += f'{d.mean():+11.3f}{mark}'
    got[NICE[ds]] = row
    L.append(line)

pooled, won, pooled_p = [], [], []
for n in NS:
    V, B, w, tot, per_cohort = [], [], 0, 0, []
    for ds in DS:
        c = cell(n, ds)
        if c is None:
            continue
        _, _, v, b = c
        V += list(v)
        B += list(b)
        tot += 1
        if (v - b).mean() > 0 and ttest_rel(v, b).pvalue < 0.05:
            w += 1
        per_cohort.append(float((v - b).mean()))
    if not V:
        pooled.append(None)
        won.append('--')
        continue
    pooled.append(float((np.array(V) - np.array(B)).mean()))
    pooled_p.append(float(stats.ttest_1samp(np.array(per_cohort), 0).pvalue)
                    if len(per_cohort) > 1 else float('nan'))
    won.append(f'{w}/{tot}')
L.append(f"{'Pooled':<10}" + ''.join(f'{p:+11.3f} ' if p is not None else f'{"--":>12}' for p in pooled))
L.append(f"{'Won':<10}" + ''.join(f'{x:>12}' for x in won))
L += ['', '* significant win, d significant loss, paired t-test p < 0.05']

report = '\n'.join(L)
print(report)
with open(OUT, 'w') as fh:
    fh.write(report + '\n')

for name, pub in PUB.items():
    for n, (g, p) in enumerate(zip(got[name], pub)):
        if g is None:
            bad.append(f'{name} n={NS[n]}: missing')
        elif abs(g - p) > 0.0006:
            bad.append(f'{name} n={NS[n]}: {g:+.3f} != published {p:+.3f}')
for n, (g, p) in enumerate(zip(pooled, PUB_POOLED)):
    if g is None or abs(g - p) > 0.0006:
        bad.append(f'Pooled n={NS[n]}: {g} != published {p:+.3f}')
if won != PUB_WON:
    bad.append(f'cohorts won {won} != published {PUB_WON}')

print(f'\nwrote {OUT}')
PUB_POOLED_P = {25: 0.036, 400: 0.79, 800: 0.67}
badp = [f'n={n}: pooled p {pooled_p[NS.index(n)]:.3f} against the text\'s {v}'
        for n, v in PUB_POOLED_P.items()
        if round(pooled_p[NS.index(n)], 3 if n == 25 else 2) != v]
if badp:
    raise SystemExit('DOES NOT MATCH sec:medmnist:\n  ' + '\n  '.join(badp))
print('matches sec:medmnist pooled p (cohort as unit): '
      + ', '.join(f'n={n} {pooled_p[NS.index(n)]:.3f}' for n in PUB_POOLED_P))
if bad:
    raise SystemExit('DOES NOT MATCH table:medmnist:\n  ' + '\n  '.join(bad))
print('matches table:medmnist exactly (deltas, pooled row and cohorts-won row)')
