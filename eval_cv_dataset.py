"""Does cross-validation choose the right basis for each DATASET, rather than each draw?

Per-cell selection failed because the inner score is not reproducible across its own
folds. Averaging the score over the five seeds of a dataset cuts that noise by sqrt(5),
and one basis per dataset is what would actually be shipped, so it is the question that
matters. Two versions are reported: the pooled one, which averages all five seeds and so
uses each seed's own CV score, and a leave-one-seed-out one, which chooses the basis from
four seeds and evaluates it on the fifth, so no seed contributes to its own selection.
"""
import glob, os
import numpy as np, pandas as pd
from scipy.stats import spearmanr, ttest_rel

DS = ['organmnist3d', 'nodulemnist3d', 'adrenalmnist3d',
      'fracturemnist3d', 'vesselmnist3d', 'synapsemnist3d']
WHERE = {'scatter': 'medmnist_bands', 'dct_scatter': 'medmnist_bands',
         'oriented': 'medmnist_bands', 'dct_oriented': 'medmnist_bands',
         'energy_gram': 'medmnist_bands', 'dct_energy_scatter': 'medmnist_bands',
         'pool_none': 'medmnist_pool', 'pool_pcared': 'medmnist_pool'}

te = []
for arm in os.listdir('results/medmnist_cv'):
    root = WHERE.get(arm, 'medmnist_n25_paired')
    sub = arm.replace('pool_', '') if root == 'medmnist_pool' else arm
    for f in glob.glob(f'results/{root}/{sub}/*/medmnist_perseed.csv'):
        d = pd.read_csv(f)
        d = d[d.seed < 5]
        if len(d) == 5:
            te.append(pd.DataFrame(dict(dataset=f.split('/')[-2], seed=d.seed.values,
                                        arm=arm,
                                        delta=d.spec_AUC.values - d.spat_AUC.values)))
cv = []
for f in glob.glob('results/medmnist_cv/*/*/medmnist_oof.csv'):
    d = pd.read_csv(f)
    if len(d):
        cv.append(d.assign(arm=f.split('/')[-3], dataset=f.split('/')[-2])
                  [['dataset', 'seed', 'arm', 'oof_AUC']])
m = pd.concat(cv).merge(pd.concat(te), on=['dataset', 'seed', 'arm'])
print('%d bases, %d cells' % (m.arm.nunique(), len(m)))

print('\n--- does the dataset-level CV ranking agree with the dataset-level truth? ---')
rs = []
for k in DS:
    g = m[m.dataset == k].groupby('arm').agg(cvs=('oof_AUC', 'mean'),
                                             tru=('delta', 'mean'))
    r = spearmanr(g.cvs, g.tru)[0]
    rs.append(r)
    pick, best = g.cvs.idxmax(), g.tru.idxmax()
    print('  %-16s rho=%+.2f   CV picks %-18s (%+.3f)  truth %-14s (%+.3f)'
          % (k, r, pick, g.tru[pick], best, g.tru[best]))
print('  mean rho over datasets %+.3f   (per-cell rho was +0.218)' % np.mean(rs))

print('\n--- what you get, averaged over the 30 cells ---')
pool_sel = {k: m[m.dataset == k].groupby('arm').oof_AUC.mean().idxmax() for k in DS}
rows = {}
rows['CV per dataset (pooled)'] = np.mean(
    [m[(m.dataset == k) & (m.arm == pool_sel[k])].delta.mean() for k in DS])
lo = []
for k in DS:
    for s in range(5):
        tr = m[(m.dataset == k) & (m.seed != s)]
        a = tr.groupby('arm').oof_AUC.mean().idxmax()
        lo.append(m[(m.dataset == k) & (m.seed == s) & (m.arm == a)].delta.values[0])
rows['CV per dataset (leave-one-seed-out)'] = np.mean(lo)
rows['CV per cell'] = m.loc[m.groupby(['dataset', 'seed']).oof_AUC.idxmax()].delta.mean()
for a in ['dct_energy', 'pool_pcared']:
    rows['fixed ' + a] = m[m.arm == a].delta.mean()
rows['oracle per dataset'] = np.mean(
    [m[m.dataset == k].groupby('arm').delta.mean().max() for k in DS])
rows['oracle per cell'] = m.groupby(['dataset', 'seed']).delta.max().mean()
for k, v in rows.items():
    print('  %-38s %+.4f' % (k, v))

print('\n--- per dataset: pooled-CV pick vs the two fixed bases vs the truth ---')
print('  %-16s %-18s %8s %9s %9s %8s' % ('dataset', 'CV pick', 'CV pick',
                                          'dct_energy', 'pcared', 'best'))
for k in DS:
    g = m[m.dataset == k].groupby('arm').delta.mean()
    print('  %-16s %-18s %+8.3f %+9.3f %+9.3f %+8.3f'
          % (k, pool_sel[k], g[pool_sel[k]], g['dct_energy'], g['pool_pcared'], g.max()))

s = np.array([m[(m.dataset == k) & (m.arm == pool_sel[k])].delta.values
              for k in DS]).ravel()
for a in ['dct_energy', 'pool_pcared']:
    f = np.array([m[(m.dataset == k) & (m.arm == a)].sort_values('seed').delta.values
                  for k in DS]).ravel()
    t, p = ttest_rel(s, f)
    print('\n  pooled CV minus fixed %-12s %+.4f +- %.4f  t=%+.2f  p=%.3f'
          % (a, (s - f).mean(), (s - f).std(ddof=1) / np.sqrt(30), t, p))
