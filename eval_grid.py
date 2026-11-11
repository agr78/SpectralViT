"""Basis x capacity selection on MedMNIST3D: does a wider grid help or hurt?

61 configs (basis family x its own capacity parameter) x 6 datasets x 5 seeds, with the
inner CV score from medmnist_cv.py computed inside the 25 training volumes only. Compared
against the 28-arm grid, where dataset-level CV reached +0.0498 leave-one-seed-out.
"""
import glob, os, re
import numpy as np, pandas as pd
from scipy.stats import ttest_rel, spearmanr

DS = ['organmnist3d', 'nodulemnist3d', 'adrenalmnist3d',
      'fracturemnist3d', 'vesselmnist3d', 'synapsemnist3d']
rows = []
for f in glob.glob('results/medmnist_grid/*/*/medmnist_perseed.csv'):
    d = pd.read_csv(f).query('seed<5').sort_values('seed')
    if len(d) != 5:
        continue
    rows.append(pd.DataFrame(dict(cfg=f.split('/')[-3], dataset=f.split('/')[-2],
                                  seed=d.seed.values, comp=int(d.comp.iloc[0]),
                                  spec=d.spec_AUC.values, spat=d.spat_AUC.values)))
te = pd.concat(rows)
cv = []
for f in glob.glob('results/medmnist_gridcv/*/*/medmnist_oof.csv'):
    d = pd.read_csv(f)
    if len(d):
        cv.append(d.assign(cfg=f.split('/')[-3], dataset=f.split('/')[-2])
                  [['cfg', 'dataset', 'seed', 'oof_AUC']])
m = pd.concat(cv).merge(te, on=['cfg', 'dataset', 'seed'])
m['delta'] = m.spec - m.spat
print('%d configs x %d datasets x 5 seeds = %d cells' % (m.cfg.nunique(), m.dataset.nunique(), len(m)))

fam = lambda c: re.sub(r'_(c|b|f|s|j|k|d|r|o)\d+', '', c)
m['family'] = m.cfg.map(fam)
print('\n--- best capacity within each family, by mean delta over 30 cells ---')
g = m.groupby('cfg').agg(d=('delta', 'mean'), f=('family', 'first'), n=('comp', 'first'))
for f0, gg in g.groupby('f'):
    gg = gg.sort_values('d', ascending=False)
    sp = gg.d.max() - gg.d.min()
    print('  %-16s best %-22s %+.4f   worst %-22s %+.4f   spread %.4f'
          % (f0, gg.index[0], gg.d.iloc[0], gg.index[-1], gg.d.iloc[-1], sp))

print('\n--- top 10 configs overall ---')
for c, r in g.sort_values('d', ascending=False).head(10).iterrows():
    print('  %-24s %4d feat  %+.4f' % (c, r.n, r.d))

idx = [(k, s) for k in DS for s in range(5)]
val = lambda c, k, s: m[(m.dataset == k) & (m.seed == s) & (m.cfg == c)].delta.values[0]
pooled = np.array([val(m[m.dataset == k].groupby('cfg').oof_AUC.mean().idxmax(), k, s)
                   for k, s in idx])
loso = np.array([val(m[(m.dataset == k) & (m.seed != s)].groupby('cfg').oof_AUC.mean().idxmax(), k, s)
                 for k, s in idx])
best = g.d.idxmax()
fx = np.array([val(best, k, s) for k, s in idx])
orc_ds = np.mean([m[m.dataset == k].groupby('cfg').delta.mean().max() for k in DS])
print('\n--- selection on the 61-config grid (30 cells) ---')
print('  %-34s %+.4f' % ('CV per dataset, pooled', pooled.mean()))
print('  %-34s %+.4f' % ('CV per dataset, leave-one-seed-out', loso.mean()))
print('  %-34s %+.4f  (%s)' % ('best fixed config', fx.mean(), best))
print('  %-34s %+.4f' % ('oracle per dataset', orc_ds))
print('  %-34s %+.4f' % ('oracle per cell', m.groupby(['dataset', 'seed']).delta.max().mean()))
for nm, v in [('pooled', pooled), ('LOSO', loso)]:
    t, p = ttest_rel(v, fx)
    print('  %s minus best fixed  %+.4f +- %.4f  p=%.4f'
          % (nm, (v - fx).mean(), (v - fx).std(ddof=1) / np.sqrt(30), p))
print('  reference: 28-arm grid gave LOSO +0.0498, oracle/dataset +0.0691')

print('\n--- per dataset, leave-one-seed-out, paired spectral vs spatial ---')
print('  %-16s %-26s %8s %8s %8s %7s'
      % ('dataset', 'config(s) chosen', 'spec', 'spat', 'delta', 'p'))
nw = 0
for k in DS:
    sp, st, pk = [], [], []
    for s in range(5):
        c = m[(m.dataset == k) & (m.seed != s)].groupby('cfg').oof_AUC.mean().idxmax()
        r = m[(m.dataset == k) & (m.seed == s) & (m.cfg == c)].iloc[0]
        sp.append(r.spec); st.append(r.spat); pk.append(c)
    sp, st = np.array(sp), np.array(st)
    t, p = ttest_rel(sp, st)
    nw += (sp - st).mean() > 0
    u = sorted(set(pk))
    lbl = u[0] if len(u) == 1 else '%s +%d' % (u[0], len(u) - 1)
    print('  %-16s %-26s %8.3f %8.3f %+8.3f %7.3f' % (k, lbl, sp.mean(), st.mean(), (sp - st).mean(), p))
print('  %d of 6 datasets positive' % nw)

print('\n--- does square\'s synapse edge survive the capacity sweep? ---')
for c in sorted(x for x in m.cfg.unique() if x.startswith(('square', 'pca_sq'))):
    d = m[(m.cfg == c) & (m.dataset == 'synapsemnist3d')]
    t, p = ttest_rel(d.spec, d.spat)
    print('  %-22s synapse %+.4f +- %.4f  p=%.3f   overall %+.4f'
          % (c, d.delta.mean(), d.delta.std(ddof=1) / np.sqrt(5), p, m[m.cfg == c].delta.mean()))
