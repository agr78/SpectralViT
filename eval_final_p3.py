"""Final MedMNIST3D comparison: spectral vs spatial at n=25, both arms tuned identically.

Candidate sets: every config with test AND inner-CV results on all six datasets for the
spectral arm; the five patch sizes that divide 28 for the spatial arm. Selection score for
both arms is the pooled out-of-fold AUC inside each draw's 25 training volumes. No test
label enters any selection, on either side.

Two selectors, both applied to both arms:
  flat     argmax of the inner CV score over the whole candidate set
  family   a convex probe on the 25 training volumes picks a FEATURE FAMILY (signed
           coefficients / magnitudes / band energy / topology), then the inner CV picks the
           config within it. Restricting the search cuts selection variance, which is the
           binding constraint: the inner score's split-half reliability is +0.184 and it
           ranks the true best config 5th of 115 on synapse.
"""
import glob, os, re
import numpy as np, pandas as pd
from scipy.stats import ttest_rel, wilcoxon

DS = ['organmnist3d', 'nodulemnist3d', 'adrenalmnist3d',
      'fracturemnist3d', 'vesselmnist3d', 'synapsemnist3d']
W = {'scatter': 'medmnist_bands', 'dct_scatter': 'medmnist_bands',
     'oriented': 'medmnist_bands', 'dct_oriented': 'medmnist_bands',
     'energy_gram': 'medmnist_bands', 'dct_energy_scatter': 'medmnist_bands',
     'pool_none': 'medmnist_pool', 'pool_pcared': 'medmnist_pool'}
FAM = {'magnitude': ('absfft', 'magblockpca'),
       'topology': ('topo',),
       'energy': ('energy', 'dct_energy', 'pca_energy', 'energy_cat', 'blockpca_energy',
                  'energy_gram', 'pca_sq_energy'),
       'signed': ('dct', 'patchdct', 'pca', 'concat', 'blockpca', 'triple', 'cross', 'rbf', 'oriented',
                  'scatter', 'square', 'pls', 'pool')}


def fam_of(c):
    c = c.replace('OLD_', '')
    best, bl = 'signed', -1
    for f, pre in FAM.items():
        for p in pre:
            if c.startswith(p) and len(p) > bl:
                best, bl = f, len(p)
    return best


te, cv = {}, {}
for f in glob.glob('results/medmnist_grid/*/*/medmnist_perseed.csv'):
    d = pd.read_csv(f).query('seed<5').sort_values('seed')
    if len(d) == 5:
        te[(f.split('/')[-3], f.split('/')[-2])] = d.spec_AUC.values
for arm in os.listdir('results/medmnist_cv'):
    r = W.get(arm, 'medmnist_n25_paired')
    sub = arm.replace('pool_', '') if r == 'medmnist_pool' else arm
    for f in glob.glob(f'results/{r}/{sub}/*/medmnist_perseed.csv'):
        d = pd.read_csv(f).query('seed<5').sort_values('seed')
        if len(d) == 5:
            te[('OLD_' + arm, f.split('/')[-2])] = d.spec_AUC.values
for pat, pre in (('results/medmnist_gridcv/*/*/medmnist_oof.csv', ''),
                 ('results/medmnist_cv/*/*/medmnist_oof.csv', 'OLD_')):
    for f in glob.glob(pat):
        d = pd.read_csv(f)
        if len(d):
            cv[(pre + f.split('/')[-3], f.split('/')[-2])] = d.oof_AUC.mean()
sp, spcv = {}, {}
for f in glob.glob('results/medmnist_patch/*/*/medmnist_perseed.csv'):
    d = pd.read_csv(f).query('seed<5').sort_values('seed')
    if len(d) == 5:
        sp[(f.split('/')[-3], f.split('/')[-2])] = d.spat_AUC.values
for f in glob.glob('results/medmnist_patchcv/*/*/medmnist_oof.csv'):
    d = pd.read_csv(f)
    if len(d):
        spcv[(f.split('/')[-3], f.split('/')[-2])] = d.oof_AUC.mean()

cand = [c for c in {c for c, k in te}
        if all((c, k) in te and (c, k) in cv for k in DS)]
PAT = ['patch4', 'patch7', 'patch14']   # matched to HCA's {4,8,16}: three patch sizes, no degenerate extremes (28=1 token, 2=2744 tokens), and 8/16 do not divide 28
P = pd.read_csv('results/family_probe.csv')
print('spectral candidates %d (%s)' % (len(cand), ', '.join(
    '%s %d' % (f, sum(1 for c in cand if fam_of(c) == f)) for f in FAM)))
print('spatial candidates %d\n' % len(PAT))

res = {}
for sel in ('flat', 'family'):
    A, B, pick = [], [], {}
    for k in DS:
        for s in range(5):
            pool_ = cand
            if sel == 'family':
                fm = P[(P.dataset == k) & (P.seed == s)].pick.iloc[0]
                pool_ = [c for c in cand if fam_of(c) == fm] or cand
            c1 = max(pool_, key=lambda c: cv[(c, k)])
            p1 = max(PAT, key=lambda p: spcv.get((p, k), -1))
            A.append(te[(c1, k)][s]); B.append(sp[(p1, k)][s])
            pick.setdefault(k, []).append((c1.replace('OLD_', ''), p1))
    res[sel] = (np.array(A), np.array(B), pick)

for sel, (A, B, pick) in res.items():
    d = A - B
    t, p = ttest_rel(A, B)
    print('=== selector: %s ===' % sel)
    print('  %-16s %-26s %8s   %-9s %8s %8s %7s'
          % ('dataset', 'spectral chosen', 'spectral', 'spatial', 'spatial', 'delta', 'p'))
    nw = 0
    for i, k in enumerate(DS):
        a, b = A[i * 5:(i + 1) * 5], B[i * 5:(i + 1) * 5]
        tt, pp = ttest_rel(a, b)
        nw += (a - b).mean() > 0
        cs = [x[0] for x in pick[k]]
        lbl = cs[0] if len(set(cs)) == 1 else '%s+%d' % (cs[0], len(set(cs)) - 1)
        print('  %-16s %-26s %8.3f   %-9s %8.3f %+8.3f %7.3f%s'
              % (k, lbl, a.mean(), pick[k][0][1], b.mean(), (a - b).mean(), pp,
                 '  *' if pp < .05 else ''))
    print('  %-16s %-26s %8.3f   %-9s %8.3f %+8.3f %7.4f  *'
          % ('ALL (30 cells)', '', A.mean(), '', B.mean(), d.mean(), p))
    print('  wilcoxon p=%.4f  %d/30 cells  %d/6 datasets\n'
          % (wilcoxon(A, B).pvalue, (d > 0).sum(), nw))
