"""Radiomics baseline on IXI, scored on the folds and draws of ixi_nested.py.

usage: python experiments/eval_radiomics.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repository root: shared modules and paths.py
import os

import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.environ.get('RADIOMICS_OUT', os.path.join(HERE, 'results', 'radiomics_ixi.npz'))
OUT = os.environ.get('RADIOMICS_RESULT', os.path.join(HERE, 'results', 'radiomics_ixi.txt'))
SIZES = [int(v) for v in os.environ.get('IXI_SIZES', '25,50,100,200,566').split(',')]
DRAWS = {25: 10, 50: 10, 100: 5, 200: 5, 566: 5}
FOLDS, INNER = 5, 3
CS = [0.01, 0.1, 1.0]
TREES = [100, 300]

d = np.load(CACHE)
F, Y = d['F'], d['y'].astype(int)
keep = F.std(0) > 1e-6
F = F[:, keep]
print(f'{CACHE}: {F.shape[0]} volumes x {F.shape[1]} features '
      f'({int((~keep).sum())} near-constant dropped)')


def one_se(scores, cands):
    mu = {c: np.mean(v) for c, v in scores.items()}
    best = max(mu, key=mu.get)
    se = np.std(scores[best], ddof=1) / max(np.sqrt(len(scores[best])), 1)
    ok = [c for c in cands if mu[c] >= mu[best] - se]
    return ok[0] if ok else best


def fit_eval(model_of, grid, Xtr, ytr, Xte, yte):
    inner = StratifiedKFold(INNER, shuffle=True, random_state=0)
    scores = {c: [] for c in grid}
    for i_tr, i_va in inner.split(Xtr, ytr):
        sc = StandardScaler().fit(Xtr[i_tr])
        for c in grid:
            m = model_of(c).fit(sc.transform(Xtr[i_tr]), ytr[i_tr])
            p = m.predict_proba(sc.transform(Xtr[i_va]))[:, 1]
            scores[c].append(roc_auc_score(ytr[i_va], p) if len(set(ytr[i_va])) > 1 else 0.5)
    c = one_se(scores, grid)
    sc = StandardScaler().fit(Xtr)
    m = model_of(c).fit(sc.transform(Xtr), ytr)
    return roc_auc_score(yte, m.predict_proba(sc.transform(Xte))[:, 1])


ARMS = [('radiomics + linear', lambda c: LogisticRegression(C=c, max_iter=5000), CS),
        ('radiomics + forest',
         lambda c: RandomForestClassifier(n_estimators=c, random_state=0, n_jobs=-1), TREES)]
L = ['Radiomics baseline on IXI sex classification.',
     f'{F.shape[1]} features, folds and draws matching ixi_nested.py, one-SE inner selection.',
     '', '%-22s' % 'arm' + ''.join('%14s' % f'n={n}' for n in SIZES)]
rows = {}
for name, build, grid in ARMS:
    cells = []
    for n in SIZES:
        per_draw = []
        for s in range(DRAWS[n]):
            rng = np.random.RandomState(1000 * s + n)
            idx = (np.arange(len(Y)) if n >= len(Y) else
                   np.concatenate([rng.choice(np.where(Y == c)[0], n // 2, replace=False)
                                   for c in (0, 1)]))
            outer = StratifiedKFold(FOLDS, shuffle=True, random_state=s)
            fold_auc = [fit_eval(build, grid, F[idx][tr], Y[idx][tr], F[idx][te], Y[idx][te])
                        for tr, te in outer.split(F[idx], Y[idx])]
            per_draw.append(np.mean(fold_auc))
        rows[(name, n)] = np.array(per_draw)
        cells.append('%.3f $\\pm$ %.3f' % (np.mean(per_draw), np.std(per_draw)))
        print(f'  {name} n={n}: {np.mean(per_draw):.3f}', flush=True)
    L.append('%-22s' % name + ''.join('%14s' % c for c in cells))
report = '\n'.join(L)
print('\n' + report)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, 'w').write(report + '\n')
print(f'\nwrote {OUT}')
