"""Did pooling every basis and selecting inside the pool beat choosing one basis?

Compared against dct_energy, the best single fixed arm (+0.031), on the same 30
(dataset, seed) cells and the same cached spatial predictions, so every comparison is
paired. SELK was fixed at 312 before any of these ran -- dct_energy's feature count --
so a win cannot come from the pool simply carrying more features.
"""
import glob, os, re, sys
import numpy as np, pandas as pd
from scipy.stats import ttest_rel

DS = ['organmnist3d', 'nodulemnist3d', 'adrenalmnist3d',
      'fracturemnist3d', 'vesselmnist3d', 'synapsemnist3d']


def load(root, arm):
    r = {}
    for f in glob.glob(f'{root}/{arm}/*/medmnist_perseed.csv'):
        d = pd.read_csv(f)
        d = d[d.seed < 5]
        if len(d) == 5:
            r[f.split('/')[-2]] = d.spec_AUC.values - d.spat_AUC.values
    return r


def flat(r):
    return np.concatenate([r[k] for k in DS if k in r])


ref = load('results/medmnist_n25_paired', 'dct_energy')
pe = load('results/medmnist_n25_paired', 'pca_energy')
print('reference arms (6 datasets x 5 seeds)')
print('  dct_energy  %+.4f   pca_energy %+.4f' % (flat(ref).mean(), flat(pe).mean()))

rows = []
for sel in ['none', 'group', 'mrmr', 'stab', 'fscore', 'mi', 'pcared']:
    r = load('results/medmnist_pool', sel)
    if len(r) < 6:
        print('  %-7s incomplete (%d/6 datasets)' % (sel, len(r)))
        continue
    x, y = flat(ref), flat(r)
    t, p = ttest_rel(y, x)
    rows.append(dict(sel=sel, delta=y.mean(),
                     se=y.std(ddof=1) / np.sqrt(len(y)),
                     wins=sum(1 for k in DS if r[k].mean() > 0),
                     vs_ref=(y - x).mean(),
                     vs_se=(y - x).std(ddof=1) / np.sqrt(len(y)), p=p,
                     per={k: r[k].mean() for k in DS}))
if not rows:
    sys.exit('no complete selector runs yet')
df = pd.DataFrame(rows).sort_values('delta', ascending=False)
print('\npooled tokens, selection inside the pool')
print('  %-8s %8s %8s %5s  %8s %8s %7s' %
      ('select', 'delta', 'se', 'wins', 'vs dct_e', 'se', 'p'))
for _, r in df.iterrows():
    print('  %-8s %+8.4f %8.4f  %d/6  %+8.4f %8.4f  %.3f' %
          (r.sel, r.delta, r.se, r.wins, r.vs_ref, r.vs_se, r.p))

print('\nper dataset')
print('  %-16s %9s %9s %9s' % ('', 'dct_energy', df.iloc[0].sel, 'none'))
nn = next((r for _, r in df.iterrows() if r.sel == 'none'), None)
for k in DS:
    print('  %-16s %+9.3f %+9.3f %+9.3f' %
          (k, ref[k].mean(), df.iloc[0]['per'][k],
           nn['per'][k] if nn is not None else np.nan))

# What the group selector actually kept, over every (dataset, seed) it ran.
log = open('results/medmnist_pool_run.log').read()
keeps = re.findall(r'pool \d+ -> \d+ via group \| (.+)', log)
if keeps:
    blk = {}
    for line in keeps:
        for nm, a, b in re.findall(r'(\w+) (\d+)/(\d+)', line):
            blk.setdefault(nm, []).append(int(a) > 0)
    print('\nhow often the group selector kept each block (%d cells)' % len(keeps))
    for nm, v in sorted(blk.items(), key=lambda kv: -np.mean(kv[1])):
        print('  %-9s %.2f' % (nm, np.mean(v)))
    print('  (band energy helps wherever added, squares hurt wherever added --')
    print('   a selector that works should sit near 1.00 on energy, 0.00 on square)')
