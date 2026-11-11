"""Did cross-validating every basis inside the 25 training volumes select the right one?

Selection score is the pooled out-of-fold AUC over all 25 training volumes, computed by
medmnist_cv.py with the same draw, basis construction and encoder as the reporting run.
The MedMNIST test split is never read during selection; the test numbers below are the
already-measured paired results for whichever arm the CV picked.
"""
import glob, os
import numpy as np, pandas as pd
from scipy.stats import ttest_rel, spearmanr

DS = ['organmnist3d', 'nodulemnist3d', 'adrenalmnist3d',
      'fracturemnist3d', 'vesselmnist3d', 'synapsemnist3d']
WHERE = {'scatter': 'medmnist_bands', 'dct_scatter': 'medmnist_bands',
         'oriented': 'medmnist_bands', 'dct_oriented': 'medmnist_bands',
         'energy_gram': 'medmnist_bands', 'dct_energy_scatter': 'medmnist_bands',
         'pool_none': 'medmnist_pool', 'pool_pcared': 'medmnist_pool'}


def test_rows():
    out = []
    for arm in os.listdir('results/medmnist_cv'):
        root = WHERE.get(arm, 'medmnist_n25_paired')
        sub = arm.replace('pool_', '') if root == 'medmnist_pool' else arm
        for f in glob.glob(f'results/{root}/{sub}/*/medmnist_perseed.csv'):
            d = pd.read_csv(f)
            d = d[d.seed < 5]
            if len(d) != 5:
                continue
            out.append(pd.DataFrame(dict(dataset=f.split('/')[-2], seed=d.seed.values,
                                         arm=arm, spec=d.spec_AUC.values,
                                         delta=d.spec_AUC.values - d.spat_AUC.values)))
    return pd.concat(out)


def cv_rows():
    out = []
    for f in glob.glob('results/medmnist_cv/*/*/medmnist_oof.csv'):
        d = pd.read_csv(f)
        if not len(d):
            continue
        d['arm'] = f.split('/')[-3]
        d['dataset'] = f.split('/')[-2]
        out.append(d[['dataset', 'seed', 'arm', 'oof_AUC']])
    return pd.concat(out)


cv, te = cv_rows(), test_rows()
m = cv.merge(te, on=['dataset', 'seed', 'arm'])
print('arms with both CV and test results: %d   cells: %d'
      % (m.arm.nunique(), len(m)))
miss = sorted(set(cv.arm) - set(m.arm))
if miss:
    print('  no test counterpart (excluded): ' + ', '.join(miss))

rho = [spearmanr(g.oof_AUC, g.spec)[0] for _, g in m.groupby(['dataset', 'seed'])
       if g.oof_AUC.nunique() > 2]
print('\nwithin-cell rank agreement between the CV score and the test score')
print('  mean Spearman over %d cells: %+.3f   (median %+.3f, %d of %d positive)'
      % (len(rho), np.mean(rho), np.median(rho),
         sum(1 for r in rho if r > 0), len(rho)))

sel = m.loc[m.groupby(['dataset', 'seed']).oof_AUC.idxmax()]
fix = m[m.arm == 'dct_energy'].set_index(['dataset', 'seed']).delta
best = m.groupby('arm').delta.mean().idxmax()
fixb = m[m.arm == best].set_index(['dataset', 'seed']).delta
s = sel.set_index(['dataset', 'seed']).delta.loc[fix.index]
print('\nmean test delta of the CV-selected basis, over the 30 cells')
for nm, v in [('CV-selected', s), ('fixed dct_energy', fix),
              ('fixed %s' % best, fixb),
              ('oracle', m.groupby(['dataset', 'seed']).delta.max().loc[fix.index])]:
    print('  %-22s %+.4f' % (nm, v.mean()))
for nm, f in [('dct_energy', fix), (best, fixb)]:
    t, p = ttest_rel(s, f)
    print('  CV minus fixed %-12s %+.4f +- %.4f  t=%+.2f  p=%.3f'
          % (nm, (s - f).mean(), (s - f).std(ddof=1) / np.sqrt(len(f)), t, p))

print('\nper dataset: CV-selected vs fixed dct_energy, and what CV picked')
for k in DS:
    kk = [c for c in fix.index if c[0] == k]
    picks = sel[sel.dataset == k].arm.value_counts()
    print('  %-16s %+.3f vs %+.3f   %s'
          % (k, s.loc[kk].mean(), fix.loc[kk].mean(),
             ', '.join(f'{a}x{n}' for a, n in picks.items())))

print('\nthe synapse test: squares are the only arms that beat spatial there.')
syn = m[m.dataset == 'synapsemnist3d']
sq = syn[syn.arm.isin(['square', 'pca_sq_energy'])]
print('  squared arms, mean CV score %.4f vs all other arms %.4f'
      % (sq.oof_AUC.mean(), syn[~syn.arm.isin(['square', 'pca_sq_energy'])].oof_AUC.mean()))
print('  CV rank of the squared arms within each synapse cell (1 = picked):')
for (d_, sd), g in syn.groupby(['dataset', 'seed']):
    g = g.sort_values('oof_AUC', ascending=False).reset_index(drop=True)
    r = [i + 1 for i, a in enumerate(g.arm) if a in ('square', 'pca_sq_energy')]
    print('    seed %d: ranks %s of %d   (picked %s)' % (sd, r, len(g), g.arm[0]))
