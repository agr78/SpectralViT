"""Emit the reporting job list for one sample size of the ladder sweep.

Reads results/ladder_cv/n<N>/<ds>/<cfg>/{d,r}<seed0>/medmnist_cv.csv, averages the pooled
out-of-fold AUC over every draw block, and picks the argmax separately for the spectral
ladder and for the spatial patch grid. Blocks named r<seed0> are short blocks carrying the
remainder when the draw count is not a multiple of five; they are weighted by their own
number of draws so a 2-draw rung is not dominated by a 5-draw block.
Usage: python select_ladder.py 100
"""
import glob, os, sys, pandas as pd
N = int(sys.argv[1])
ROOT = f'/workspace/results/ladder_cv/n{N}'
G = pd.read_csv('/workspace/results/medmnist_grid_configs.tsv', sep='\t', header=None,
                names=['cfg', 'env'])
ENV = dict(zip(G.cfg, G.env))
NPZ = {'organmnist3d': '/agent-home/medmnist/organmnist3d.npz'}
rows = []
for f in glob.glob(f'{ROOT}/*/*/[dr]*/medmnist_cv.csv'):
    p = f.split('/')
    t = pd.read_csv(f)
    c = [x for x in t.columns if 'auc' in x.lower()]
    if not c:
        continue
    rows.append(dict(ds=p[-4], cfg=p[-3], blk=p[-2], cv=t[c[0]].mean(), w=len(t)))
if not rows:
    sys.exit(f'no CV cells under {ROOT}')
C = pd.DataFrame(rows).drop_duplicates(['ds', 'cfg', 'blk'])
for ds, q in C.groupby('ds'):
    m = q.groupby('cfg').apply(lambda g: (g.cv * g.w).sum() / g.w.sum())
    spec = m[[i for i in m.index if not i.startswith('spatial_')]]
    spat = m[[i for i in m.index if i.startswith('spatial_')]]
    if spec.empty or spat.empty:
        continue
    cfg = spec.idxmax()
    patch = int(spat.idxmax().split('_p')[1])
    print('\t'.join([ds, str(patch), f'{cfg}_P{patch}_N{N}', cfg, ENV[cfg],
                     NPZ.get(ds, f'/workspace/data/medmnist/{ds}.npz')]))
