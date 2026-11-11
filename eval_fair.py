"""Spectral vs spatial on MedMNIST3D at n=25, with BOTH arms tuned by the same rule.

Nothing is selected using test labels. Each arm's configuration is chosen per dataset by
the pooled out-of-fold AUC computed inside that draw's 25 training volumes, leave-one-seed-
out: the config is chosen from the other four seeds' CV scores and applied to the held-out
seed, so no seed informs its own selection.
  spectral candidates: 89 (basis family x capacity)
  spatial  candidates:  5 (patch 2, 4, 7, 14, 28 -- the sides that divide 28)
"""
import glob, os
import numpy as np, pandas as pd
from scipy.stats import ttest_rel, wilcoxon

DS = ['organmnist3d', 'nodulemnist3d', 'adrenalmnist3d',
      'fracturemnist3d', 'vesselmnist3d', 'synapsemnist3d']
W = {'scatter': 'medmnist_bands', 'dct_scatter': 'medmnist_bands',
     'oriented': 'medmnist_bands', 'dct_oriented': 'medmnist_bands',
     'energy_gram': 'medmnist_bands', 'dct_energy_scatter': 'medmnist_bands',
     'pool_none': 'medmnist_pool', 'pool_pcared': 'medmnist_pool'}


def load(pat, col, key):
    out = []
    for f in glob.glob(pat):
        d = pd.read_csv(f)
        if col == 'oof_AUC':
            if not len(d):
                continue
            out.append(d.assign(cfg=f.split('/')[-3], dataset=f.split('/')[-2])
                       [['cfg', 'dataset', 'seed', 'oof_AUC']])
        else:
            d = d.query('seed<5').sort_values('seed')
            if len(d) == 5:
                out.append(pd.DataFrame({'cfg': f.split('/')[-3],
                                         'dataset': f.split('/')[-2],
                                         'seed': d.seed.values, key: d[col].values}))
    return pd.concat(out)


# spectral: the capacity grid plus the earlier fixed-capacity arms
sc = [load('results/medmnist_grid/*/*/medmnist_perseed.csv', 'spec_AUC', 'y')]
for arm in os.listdir('results/medmnist_cv'):
    r = W.get(arm, 'medmnist_n25_paired')
    sub = arm.replace('pool_', '') if r == 'medmnist_pool' else arm
    for f in glob.glob(f'results/{r}/{sub}/*/medmnist_perseed.csv'):
        d = pd.read_csv(f).query('seed<5').sort_values('seed')
        if len(d) == 5:
            sc.append(pd.DataFrame({'cfg': 'OLD_' + arm, 'dataset': f.split('/')[-2],
                                    'seed': d.seed.values, 'y': d.spec_AUC.values}))
sc = pd.concat(sc)
scv = pd.concat([load('results/medmnist_gridcv/*/*/medmnist_oof.csv', 'oof_AUC', ''),
                 load('results/medmnist_cv/*/*/medmnist_oof.csv', 'oof_AUC', '')
                 .assign(cfg=lambda x: 'OLD_' + x.cfg)])
S = scv.merge(sc, on=['cfg', 'dataset', 'seed'])
S = S[S.cfg.isin([c for c, g in S.groupby('cfg') if g.dataset.nunique() == 6])]

# spatial: the patch grid
P = load('results/medmnist_patchcv/*/*/medmnist_oof.csv', 'oof_AUC', '').merge(
    load('results/medmnist_patch/*/*/medmnist_perseed.csv', 'spat_AUC', 'y'),
    on=['cfg', 'dataset', 'seed'])
print('spectral candidates %d, spatial candidates %d' % (S.cfg.nunique(), P.cfg.nunique()))


def pick(fr, k, s):
    """Config chosen from the four seeds that are not s, by mean inner CV score."""
    c = fr[(fr.dataset == k) & (fr.seed != s)].groupby('cfg').oof_AUC.mean().idxmax()
    return c, fr[(fr.dataset == k) & (fr.seed == s) & (fr.cfg == c)].y.values[0]


print('\n  %-16s %-22s %8s   %-9s %8s %8s %7s'
      % ('dataset', 'spectral chosen', 'spectral', 'spatial', 'spatial', 'delta', 'p'))
A, B = [], []
for k in DS:
    a, b, ca, cb = [], [], [], []
    for s in range(5):
        c1, v1 = pick(S, k, s); c2, v2 = pick(P, k, s)
        a.append(v1); b.append(v2); ca.append(c1.replace('OLD_', '')); cb.append(c2)
    a, b = np.array(a), np.array(b); A += list(a); B += list(b)
    t, p = ttest_rel(a, b)
    f = lambda u: (sorted(set(u))[0] if len(set(u)) == 1
                   else '%s+%d' % (sorted(set(u))[0], len(set(u)) - 1))
    print('  %-16s %-22s %8.3f   %-9s %8.3f %+8.3f %7.3f%s'
          % (k, f(ca), a.mean(), f(cb), b.mean(), (a - b).mean(), p, '  *' if p < .05 else ''))
A, B = np.array(A), np.array(B)
d = A - B
t, p = ttest_rel(A, B)
print('\n  %-16s %-22s %8.3f   %-9s %8.3f %+8.3f %7.4f  *'
      % ('ALL (30 cells)', '', A.mean(), '', B.mean(), d.mean(), p))
print('  wilcoxon p=%.4f   spectral wins %d/30 cells, %d/6 datasets'
      % (wilcoxon(A, B).pvalue, (d > 0).sum(),
         sum(1 for i in range(6) if d[i * 5:(i + 1) * 5].mean() > 0)))
