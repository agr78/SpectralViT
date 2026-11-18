"""Build table:faces, the LFW identity-classification table.

usage: python make_table_faces.py [out.tex]"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'results', 'faces_table.tex')
D = os.environ.get('FACES_DIR', os.path.join(HERE, 'results', 'faces_knn'))
SIZES = [62, 124, 186, 310, 620]
ARMS = [('spat', 'Spatial'), ('knn', 'Eigenfaces'), ('mlp', 'MLP'), ('spec', 'Spectral')]
PUB = {
    62: ('0.034', '0.059', '0.061', '0.081', '0.547', '0.522', '0.612', '0.647'),
    124: ('0.044', '0.077', '0.098', '0.123', '0.568', '0.531', '0.675', '0.698'),
    186: ('0.049', '0.081', '0.122', '0.150', '0.582', '0.533', '0.712', '0.739'),
    310: ('0.060', '0.101', '0.159', '0.191', '0.613', '0.543', '0.764', '0.772'),
    620: ('0.100', '0.131', '0.222', '0.246', '0.691', '0.559', '0.813', '0.814'),
}
d = pd.read_csv(os.path.join(D, 'faces_perseed.csv'))
missing = [n for n in SIZES if (d.n == n).sum() != d.seed.nunique()]
if missing:
    raise SystemExit(f'incomplete sizes {missing}: every size needs {d.seed.nunique()} seeds')

cells, drift = {}, []
for tag, _ in ARMS:
    for n in SIZES:
        g = d[d.n == n]
        cells[(tag, n, 'T1')] = g[f'{tag}_T1'].to_numpy()
        cells[(tag, n, 'AUC')] = g[f'{tag}_AUC'].to_numpy()
for n in SIZES:
    for j, (tag, _) in enumerate(ARMS):
        for k, metric in enumerate(('T1', 'AUC')):
            got = f"{cells[(tag, n, metric)].mean():.3f}"
            want = PUB[n][k * len(ARMS) + j]
            if got != want:
                drift.append(f'n={n} {tag} {metric}: {got} against the table\'s {want}')
if drift:
    raise SystemExit('DOES NOT MATCH table:faces:\n  ' + '\n  '.join(drift))
print(f'matches table:faces: all {len(SIZES) * len(ARMS) * 2} cells')

L = [r'\begin{table*}[t]',
     '  \\caption{Identity classification on Labeled Faces in the Wild by training set size, '
     f'over ${d.seed.nunique()}$ seeds, evaluated on a fixed balanced $620$ image test set of '
     '$62$ identities. Eigenfaces, the multilayer perceptron and the spectral ViT all receive '
     'the same $16$ principal components and differ only in the head, so the margin over the '
     'spatial ViT is the representation and the margin over the other two is the encoder. '
     'Eigenfaces is nearest neighbours in that subspace, with $k$ selected by inner '
     'cross-validation and resolving to $1$ at every size. Bold marks the best value in each '
     'column and any value a paired test over seeds does not separate from it.}',
     r'  \label{table:faces}', r'  \centering', r'  \begin{small}',
     r'  \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l' + 'c' * 8 + r'@{}}',
     r'    \toprule',
     r'    & \multicolumn{4}{c}{Top-1 accuracy} & \multicolumn{4}{c}{AUC} \\',
     r'    \cmidrule(lr){2-5}\cmidrule(lr){6-9}',
     r'    $n$ & ' + ' & '.join(nm for _, nm in ARMS) + ' & ' +
     ' & '.join(nm for _, nm in ARMS) + r' \\',
     r'    \midrule']
for n in SIZES:
    out = []
    for metric in ('T1', 'AUC'):
        top = max((t for t, _ in ARMS), key=lambda t: cells[(t, n, metric)].mean())
        keep = {t for t, _ in ARMS
                if t == top or stats.ttest_rel(cells[(top, n, metric)],
                                               cells[(t, n, metric)]).pvalue >= 0.05}
        for tag, _ in ARMS:
            v = cells[(tag, n, metric)].mean()
            out.append(f'$\\bf{{{v:.3f}}}$' if tag in keep else f'${v:.3f}$')
    L.append(f'    ${n}$ & ' + ' & '.join(out) + r' \\')
L += [r'    \bottomrule', r'  \end{tabular*}', r'  \end{small}', r'\end{table*}']

os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
print('\npaired tests, spectral against each head on the same components:')
for tag, nm in ARMS[:3]:
    for n in SIZES:
        a, b = cells[('spec', n, 'T1')], cells[(tag, n, 'T1')]
        print(f'  vs {nm:11s} n={n:4d}  {a.mean() - b.mean():+.3f}  '
              f'{int((a > b).sum())}/{len(a)}  p={stats.ttest_rel(a, b).pvalue:.4f}')
print(f'wrote {OUT}')
