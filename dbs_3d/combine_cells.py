"""Combine the per-cell worker outputs and run the five-test battery.

Descriptive marginals use all 24 cells; the battery uses only the four
pre-specified cells (lambda-sorted ordering, omega in {none, exp(-tau*lambda)}),
so the family-wise null covers the space anyone would commit to in advance.
"""
import sys, os, glob, pickle
SPDIR = os.path.dirname(os.path.abspath(__file__))
REPO  = os.path.dirname(SPDIR)
OUT   = os.path.join(REPO, 'docs', 'dbs_3d')
sys.path.insert(0, REPO); sys.path.insert(0, SPDIR); os.chdir(REPO)
import numpy as np, pandas as pd
from scipy.stats import rankdata, pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

cells = {}
for p in sorted([p for p in glob.glob(os.path.join(OUT, '*.pkl')) + glob.glob(os.path.join(OUT, 'cells', '*.pkl')) if os.path.basename(p).count('_') >= 2 and not p.endswith('_preds.pkl')]):
    r = pickle.load(open(p, 'rb'))
    cells[(r['ordering'], r['K'], r['weight'])] = r
if not cells:
    sys.exit("no cell outputs found")
bB = next(iter(cells.values()))['bB']
n = len(bB); n1 = int(bB.sum()); n0 = n - n1
rng = np.random.RandomState(0)

dfa = pd.DataFrame([{k: r[k] for k in
                     ('ordering', 'K', 'weight', 'int_mean', 'int_se', 'int_sd',
                      'ext_mean', 'ext_sd', 'ens_auc')} for r in cells.values()])
print(f"=== DESCRIPTIVE: all {len(dfa)} cells (not used to select) ===")
print(dfa.sort_values('int_mean', ascending=False).round(3).to_string(index=False))

print("\n=== marginal effect of the RANK PRIOR (averaged over orderings and K) ===")
print(dfa.groupby('weight').agg(int_mean=('int_mean', 'mean'), int_best=('int_mean', 'max'),
                                ext_mean=('ext_mean', 'mean'), ext_best=('ext_mean', 'max')
                                ).round(3).to_string())
print("\n=== marginal effect of MODE ORDERING (averaged over weights and K) ===")
print(dfa.groupby('ordering').agg(int_mean=('int_mean', 'mean'), int_best=('int_mean', 'max'),
                                  ext_mean=('ext_mean', 'mean'), ext_best=('ext_mean', 'max')
                                  ).round(3).to_string())
print("\n=== WITH vs WITHOUT the prior, within each ordering ===")
piv = dfa.pivot_table(index='ordering', columns='weight', values='int_mean').round(3)
print("internal:"); print(piv.to_string())
piv = dfa.pivot_table(index='ordering', columns='weight', values='ext_mean').round(3)
print("external:"); print(piv.to_string())
print(f"\ncorr(internal, external) over all cells = "
      f"{pearsonr(dfa.int_mean, dfa.ext_mean)[0]:+.3f}  Spearman {spearmanr(dfa.int_mean, dfa.ext_mean).statistic:+.3f}")

# ---------------- the pre-specified battery ----------------
PRE = [k for k in cells if k[0] == 'lap' and k[2] in ('none', 'lap')]
df = pd.DataFrame([{k: cells[c][k] for k in
                    ('ordering', 'K', 'weight', 'int_mean', 'int_se', 'ext_mean', 'ext_sd', 'ens_auc')}
                   for c in PRE]).sort_values('int_mean', ascending=False)
print(f"\n\n{'='*74}\n  PRE-SPECIFIED SPACE ({len(PRE)} cells): lambda-sorted, omega in (none, exp(-tau*lambda))\n{'='*74}")
print(df.round(3).to_string(index=False))

R = np.array([rankdata(cells[c]['ens']) for c in PRE])
auc = lambda y: (R @ y - n1 * (n1 + 1) / 2) / (n1 * n0)
obs = auc(bB)
mx = np.array([auc(bB[rng.permutation(n)]).max() for _ in range(20000)])
i = int(obs.argmax())
print(f"\nTEST 1 FAMILY-WISE NULL ({len(PRE)} cells)")
print(f"  best external {obs[i]:.3f}  ({PRE[i]})")
print(f"  chance best-of-{len(PRE)} {mx.mean():.3f}  (95th {np.percentile(mx,95):.3f})")
print(f"  family-wise p = {(mx >= obs[i]).mean():.4f}")

b = df.iloc[0]; key = (b['ordering'], int(b['K']), b['weight']); r = cells[key]
pe = r['ens']; ae = roc_auc_score(bB, pe)
nl = np.array([roc_auc_score(bB[rng.permutation(n)], pe) for _ in range(20000)])
bs = [roc_auc_score(bB[ix], pe[ix]) for ix in rng.randint(0, n, (20000, n)) if len(np.unique(bB[ix])) > 1]
print(f"\nTEST 2 PRE-SPECIFIED CELL (best seed-averaged internal)")
print(f"  cell {key}   internal {b['int_mean']:.3f} +- {b['int_se']:.3f}")
print(f"  external per-seed {b['ext_mean']:.3f} +- {b['ext_sd']:.3f}")
print(f"  seed-ensembled external {ae:.3f}   p = {(nl >= ae).mean():.4f}")
print(f"  bootstrap 95% CI [{np.percentile(bs,2.5):.3f}, {np.percentile(bs,97.5):.3f}]")
if len(df) > 1:
    d = df.iloc[0]['int_mean'] - df.iloc[1]['int_mean']
    se = np.hypot(df.iloc[0]['int_se'], df.iloc[1]['int_se'])
    print(f"  margin over 2nd: {d:+.3f} +- {se:.3f} -> {'separated' if d > 2*se else 'NOT separated'}")

ee = r['external']
print(f"\nTEST 3 SEED STABILITY ({r['nseeds']} inits)")
print(f"  external {ee.mean():.3f} +- {ee.std():.3f}   min {ee.min():.3f}  max {ee.max():.3f}"
      f"   below 0.5 in {(ee < 0.5).sum()}/{len(ee)}")
print(f"  internal {r['internal'].mean():.3f} +- {r['internal'].std():.3f}")

print("\nTEST 4 / TEST 5: run permutation and ablation on the selected cell with")
print(f"  python dbs_3d/final_tests.py {key[0]} {key[1]} {key[2]}")
pd.DataFrame(dfa).to_csv(os.path.join(OUT, 'cells_summary.csv'), index=False)
