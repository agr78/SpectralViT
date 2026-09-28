"""Permutation p at nested sample sizes, to show the estimate settling.

Chunks are ordered, so the first n permutations of a long run are exactly a short run.
Reports r/n and the conventional (r+1)/(n+1), which is the defensible one to quote for a
Monte Carlo test, with the binomial standard error.

usage: perm_convergence.py [model] [K]
"""
import os, sys, glob, pickle
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SC = os.path.join(REPO, 'docs', 'dbs_3d', 'sc')
M = sys.argv[1] if len(sys.argv) > 1 else 'SpectralViT'
K = int(sys.argv[2]) if len(sys.argv) > 2 else 16

cells = pickle.load(open(os.path.join(REPO, 'docs', 'dbs_3d', 'dbs_inference_cells.pkl'), 'rb'))
obs = cells[(M, K)]['per_seed_int'].mean()

tag = M.replace(' ', '_')
chunks = []
for c in range(1000):
    f = os.path.join(SC, f'perm_{tag}_{K}_{c}.pkl')
    if not os.path.exists(f): break
    chunks.append(pickle.load(open(f, 'rb')))
if not chunks:
    sys.exit(f'no permutation chunks for {M} K={K}')
null = np.concatenate(chunks)

print(f'{M} K={K}   observed internal LOO AUC {obs:.3f}')
print(f'{"n":>6} {"r":>4} {"r/n":>7} {"(r+1)/(n+1)":>12} {"SE":>7}   null mean +- sd')
for n in [10, 100, 1000]:
    if n > len(null): continue
    v = null[:n]; r = int((v >= obs).sum())
    p, pc = r / n, (r + 1) / (n + 1)
    print(f'{n:6d} {r:4d} {p:7.3f} {pc:12.3f} {np.sqrt(p*(1-p)/n):7.3f}   '
          f'{v.mean():.3f} +- {v.std():.3f}')
print(f'\ntotal permutations available: {len(null)}')
