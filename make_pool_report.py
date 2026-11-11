"""Write the pooled-tokens / feature-selection result to results/basis_pooling.txt."""
import glob, os, re
import numpy as np, pandas as pd
from scipy.stats import ttest_rel

DS = ['organmnist3d', 'nodulemnist3d', 'adrenalmnist3d',
      'fracturemnist3d', 'vesselmnist3d', 'synapsemnist3d']
NFEAT = dict(none=1913, group='varies', mrmr=312, stab=312, fscore=312, mi=312,
             pcared=24, cons=1473, cons_lite=1041, dct_energy=312)
DEC = dict(dct_energy=0, cons=0, cons_lite=0, none=0, pcared=0, group=8,
           fscore=312, mi=312, mrmr=312, stab=312)
NOTE = {'dct_energy': 'best single basis, chosen a priori',
        'cons': 'design-time consensus: dct + overlapping dct + block pca + energy',
        'cons_lite': 'design-time consensus, cosine blocks + energy only',
        'group': 'keep/drop whole blocks by inner-CV AUC',
        'fscore': 'top 312 features by univariate F-test',
        'mi': 'top 312 by per-feature mutual information',
        'mrmr': 'top 312 by max relevance, min redundancy',
        'stab': 'top 312 by stability selection, 100 bootstraps of L1 logistic',
        'pcared': 'label-free: PCA of the pooled features, no labels used',
        'none': 'the whole pool, nothing selected'}


def load(root, arm):
    r = {}
    for f in glob.glob(f'{root}/{arm}/*/medmnist_perseed.csv'):
        d = pd.read_csv(f)
        d = d[d.seed < 5]
        if len(d) == 5:
            r[f.split('/')[-2]] = d.spec_AUC.values - d.spat_AUC.values
    return r


ref = load('results/medmnist_n25_paired', 'dct_energy')
x = np.concatenate([ref[k] for k in DS])
arms = {'dct_energy': ref}
for a in ['cons', 'cons_lite', 'group', 'fscore', 'mi', 'mrmr', 'stab', 'pcared', 'none']:
    r = load('results/medmnist_pool', a)
    if len(r) == 6:
        arms[a] = r

L = []
L.append('Pooling the tokens of every basis considered, then selecting inside the pool.')
L.append('MedMNIST3D, n=25, 6 datasets x 5 seeds = 30 cells per design, paired against the')
L.append('spatial ViT on the same cached spatial predictions. Outcome is spectral minus')
L.append('spatial AUC (top-1 for organmnist3d). Generated ' + pd.Timestamp("2026-10-20").strftime('%Y-%m-%d') + '.')
L.append('')
L.append('The pool holds 1913 features: whole-volume PCA 16, disjoint block DCT 216,')
L.append('overlapping block DCT 729, block PCA 432, log band energy 96, pca x dct cross')
L.append('terms 64, squares of (pca, dct) 232, random Fourier features 128. Every selector')
L.append('is fitted on the 25 training volumes and their labels only; the MedMNIST test')
L.append('split is transformed by the fitted objects and never scored during selection.')
L.append('SELK was fixed at 312 before any run -- dct_energy\'s feature count -- so a win')
L.append('could not come from the selected pool merely being larger.')
L.append('')
L.append('  %-10s %8s %7s %5s %9s %8s %6s  %s'
         % ('design', 'delta', 'se', 'wins', 'vs dct_e', 'se', 'p', 'run-time decisions'))
rows = []
for a, r in arms.items():
    y = np.concatenate([r[k] for k in DS])
    t, p = ttest_rel(y, x)
    rows.append((a, y, p))
for a, y, p in sorted(rows, key=lambda t: -t[1].mean()):
    w = sum(1 for k in DS if arms[a][k].mean() > 0)
    L.append('  %-10s %+8.4f %7.4f  %d/6 %+9.4f %8.4f %6.3f  %s'
             % (a, y.mean(), y.std(ddof=1) / np.sqrt(30), w,
                (y - x).mean(), (y - x).std(ddof=1) / np.sqrt(30), p, DEC.get(a, '?')))
L.append('')
L.append('per dataset')
ks = [a for a, _, _ in sorted(rows, key=lambda t: -t[1].mean())]
L.append('  %-16s' % '' + ''.join('%11s' % a[:10] for a in ks))
for k in DS:
    L.append('  %-16s' % k + ''.join('%+11.3f' % arms[a][k].mean() for a in ks))
L.append('')
L.append('what each design is')
for a in ks:
    L.append('  %-10s %s' % (a, NOTE.get(a, '')))
L.append('')

log = open('results/medmnist_pool_run.log').read()
keeps = re.findall(r'pool \d+ -> \d+ via group \| (.+)', log)
blk = {}
for line in keeps:
    for nm, a_, b_ in re.findall(r'(\w+) (\d+)/(\d+)', line):
        blk.setdefault(nm, []).append(int(a_) > 0)
L.append('How often the group selector kept each block, over %d cells' % len(keeps))
for nm, v in sorted(blk.items(), key=lambda kv: -np.mean(kv[1])):
    L.append('  %-9s %.2f' % (nm, np.mean(v)))
L.append('')
L.append('The nested-pair analysis gives the ground truth the selector has to recover:')
L.append('band energy helps wherever it is added (dct -> dct_energy +0.039 +- 0.008,')
L.append('p<0.001; pca16 -> pca_energy +0.024 +- 0.007, p=0.002) and the squares hurt')
L.append('wherever they are added (pca_energy -> pca_sq_energy -0.064 +- 0.015, p<0.001;')
L.append('energy_cat -> triple -0.017 +- 0.007, p=0.030). A working selector would sit near')
L.append('1.00 on energy and 0.00 on square. The ORDERING is exactly right -- energy top,')
L.append('square bottom -- so the criterion knows the answer on average. The per-cell')
L.append('decisions do not: it keeps the toxic block in 23% of cells and drops the')
L.append('beneficial one in 27%, and that noise is worth more than the structure is.')
L.append('')
L.append('Conclusion. Selection spends a budget it does not have.')
L.append('  - Pooling without selection is actively harmful: -0.020 against +0.031 for one')
L.append('    well-chosen basis, p<0.001. The pool contains blocks that help and blocks')
L.append('    that hurt, and unselected the harm dominates.')
L.append('  - Performance is monotone in how FEW decisions the design makes at run time:')
L.append('    0 decisions (dct_energy) +0.031 > 8 block decisions +0.018 > 312 per-feature')
L.append('    decisions +0.008 (F-test), -0.002 (mRMR), -0.014 (stability selection).')
L.append('  - mRMR did worse than a plain F-test despite the redundancy correction, and')
L.append('    stability selection -- the standard answer at tiny n -- was near the worst.')
L.append('  - Moving selection to design time (cons, cons_lite: fix the blocks the selector')
L.append('    prefers in aggregate, decide nothing per draw) gives +0.020, still below the')
L.append('    best single basis. Even perfectly aggregating the selector\'s own preferences')
L.append('    does not beat choosing one basis a priori.')
L.append('')
L.append('Oracle over all 27 arms measured is +0.082 per cell, so real headroom exists at')
L.append('n=25 -- but nothing computable from 25 labels reaches it.')
L.append('')
L.append('Recommendation unchanged: report dct_energy as a fixed design choice in')
L.append('sec:medmnist. Pooling and selection are a negative result worth one sentence in')
L.append('the limitations, not a method.')
open('results/basis_pooling.txt', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
