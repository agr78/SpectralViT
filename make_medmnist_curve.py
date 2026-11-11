"""Regenerate table:medmnist, the MedMNIST3D sample-size curve.

The table was originally computed by a script written to the session scratchpad, which is
wiped on restart, so nothing in the repo rebuilt it and its source could not be found by
looking under results/medmnist* -- the data lives under results/ladder_* instead. This is
that script, recovered and pinned.

Two directories feed it, both produced by jobs/run_ladder_sweep.sh:

  results/ladder_report_n<N>.tsv        the config select_ladder.py picked at size N,
                                        one line per cohort: ds, patch, tag, cfg
  results/ladder_100/<tag>/<ds>/        the reporting run for that pick

Sizes are 25..800 with 40, 20, 10, 5, 2 and 1 draws, so draws x n is held near 1000 at
every rung. Reporting draws are seeds 5-104; the selector uses seeds 105+ and is disjoint.

Bold in the table marks a significant win, a dagger a significant loss, both at p < 0.05
on a paired t-test over the draws of that cell.

usage: python make_medmnist_curve.py
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results', 'medmnist_curve.txt')
NS = [25, 50, 100, 200, 400, 800]
DS = ['organmnist3d', 'nodulemnist3d', 'adrenalmnist3d',
      'fracturemnist3d', 'vesselmnist3d', 'synapsemnist3d']
NICE = dict(zip(DS, ['Organ', 'Nodule', 'Adrenal', 'Fracture', 'Vessel', 'Synapse']))

# table:medmnist as published, in units of AUC. Asserted at the end so that a change in
# either the selection or the reporting runs fails loudly instead of printing a new table.
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
        SEL[(n, p[0])] = (p[3], int(p[1]), p[2])      # cfg, patch, tag


def cell(n, ds):
    """(cfg, patch, spectral AUCs, spatial AUCs) for one (size, cohort), or None."""
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

pooled, won = [], []
for n in NS:
    V, B, w, tot = [], [], 0, 0
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
    if not V:
        pooled.append(None)
        won.append('--')
        continue
    pooled.append(float((np.array(V) - np.array(B)).mean()))
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
if bad:
    raise SystemExit('DOES NOT MATCH table:medmnist:\n  ' + '\n  '.join(bad))
print('matches table:medmnist exactly (deltas, pooled row and cohorts-won row)')
