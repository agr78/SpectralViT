"""Run-to-run reproducibility of hca_sweep.py, measured from a deliberate repeat pass.

Two independent runs of the SAME (n, seed, fold, arm) should agree exactly if training is
deterministic given the seed. They do not: 53 rank-arm cells duplicated by accident at
n=1000 differed by a mean of 0.0097 AUC and a max of 0.0515. That sample is confined to the
three rank arms, which appear only in the supplement's ablation, so it cannot bound the
error on the arms the main tables report. This script measures the main arms directly.

The cause is not rank-specific: 'fixrank' holds its weights as a buffer and adds no
parameters, yet reproduces worst (mean 0.0143), which rules out an RNG-stream explanation.
The two runs landed on different nodes, so GPU-level reduction order is the leading
candidate -- and that would apply to every arm.

usage: python repro_check.py 25          # compares _rep_s* against _ep500_s* at n=25
"""
import glob, sys
import numpy as np
import pandas as pd

N = int(sys.argv[1]) if len(sys.argv) > 1 else 25
KEY = ['Seed', 'Fold', 'Model']


def load(pat, tag):
    d = [pd.read_csv(f) for f in sorted(glob.glob(f'results/sweep/hca_sweep_n{N}{pat}.csv'))]
    if not d:
        sys.exit(f'no files matching results/sweep/hca_sweep_n{N}{pat}.csv')
    d = pd.concat(d).drop_duplicates(KEY)
    print(f'  {tag}: {len(d)} cells, {d.Model.nunique()} arms')
    return d


a, b = load('_rep_s*', 'repeat  '), load('_ep500_s*', 'original')
m = a.merge(b, on=KEY, suffixes=('_a', '_b'))
if not len(m):
    sys.exit('no overlapping cells')
print(f'\n  {len(m)} cells computed twice\n')

for metric in ['AUC', 'Accuracy', 'Balanced_Acc', 'Macro_F1']:
    if f'{metric}_a' not in m:
        continue
    d = (m[f'{metric}_a'] - m[f'{metric}_b']).abs()
    print('  %-13s mean %.4f  median %.4f  p90 %.4f  max %.4f   >0.01: %3d  >0.02: %3d'
          % (metric, d.mean(), d.median(), d.quantile(.9), d.max(),
             (d > 0.01).sum(), (d > 0.02).sum()))

print('\n  per arm, |AUC difference| between the two runs:\n')
print('  %-36s %8s %8s %8s' % ('arm', 'mean', 'max', 'n'))
g = m.assign(d=(m.AUC_a - m.AUC_b).abs()).groupby('Model').d
for arm, s in sorted(g, key=lambda t: -t[1].mean()):
    print('  %-36s %8.4f %8.4f %8d' % (arm, s.mean(), s.max(), len(s)))

print('\n  What this bounds: a reported contrast smaller than the mean above is not')
print('  separable from rerunning the same code, however small its paired p-value --')
print('  the pairing treats each cell as fixed and never sees this term.')
