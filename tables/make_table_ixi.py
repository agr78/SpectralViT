"""Rebuild table:ixi and assert every published cell.

usage: python tables/make_table_ixi.py [out.tex]"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repository root: shared modules and paths.py
import os, re, sys
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from networks import SpectralViT, SpatialViT, SwinTransformer, AttentionUNet, \
                     LogisticRegression, MultiLayerPerceptron

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'results', 'ixi_table.tex')
DIR = os.environ.get('IXI_SWEEP_DIR', os.path.join(HERE, 'results', 'ixi_curve_merged'))
SIZES, VOL = [25, 50, 100, 200, 566], 96
DRAWS = {25: 10, 50: 10, 100: 5, 200: 5, 566: 5}

ARMS = [
    ('spec',   'Spectral ViT (proposed)',
     lambda v: SpectralViT(n_inputs=v[0], embed_dim=16, use_mode_weights=True)),
    ('unet',   'U-net with attention',
     lambda v: AttentionUNet(in_channels=1, base_channels=v[0])),
    ('spat',   'Spatial ViT',
     lambda v: SpatialViT(vol_size=VOL, patch_size=v[0], embed_dim=128, n_layers=2,
                          n_heads=4)),
    ('swin',   'Swin ViT',
     lambda v: SwinTransformer(img_size=VOL, embed_dim=v[0], window_size=6)),
    ('logreg', 'Logistic regression', lambda v: LogisticRegression(in_features=v[0])),
    ('mlp',    'Multilayer perceptron',
     lambda v: MultiLayerPerceptron(input_dim=v[0], hidden_dim=v[1])),
]


BUILD = {t: b for t, _, b in ARMS}

PUB = {
    'spec':   (11374, {25: (0.697, 0.149), 50: (0.685, 0.052), 100: (0.748, 0.048),
                       200: (0.776, 0.026), 566: (0.845, 0.010)}),
    'unet':   (74052, {25: (0.540, 0.151), 50: (0.621, 0.053), 100: (0.670, 0.073),
                       200: (0.720, 0.039), 566: (0.898, 0.007)}),
    'spat':  (616101, {25: (0.508, 0.172), 50: (0.619, 0.078), 100: (0.672, 0.051),
                       200: (0.697, 0.033), 566: (0.734, 0.033)}),
    'swin':   (11470, {25: (0.593, 0.106), 50: (0.583, 0.077), 100: (0.554, 0.083),
                       200: (0.593, 0.033), 566: (0.640, 0.027)}),
    'logreg':    (96, {25: (0.550, 0.105), 50: (0.577, 0.058), 100: (0.694, 0.035),
                       200: (0.799, 0.021), 566: (0.880, 0.006)}),
    'mlp':     (1153, {25: (0.563, 0.159), 50: (0.583, 0.063), 100: (0.749, 0.068),
                       200: (0.798, 0.020), 566: (0.885, 0.006)}),
}
PUB_BOLD = {('spec', 25), ('spec', 50), ('mlp', 100), ('logreg', 200), ('unet', 566)}
PUB_BOLD_PARAM = 'logreg'


def cell(tag, n, s):
    f = os.path.join(DIR, f'oof_{tag}_n{n}_s{s}.npz')
    if not os.path.exists(f):
        return None
    z = np.load(f)
    cuts = np.cumsum(z['fold_sizes'])[:-1]
    aucs = [roc_auc_score(a, b) for a, b in
            zip(np.split(z['y'], cuts), np.split(z['probs'], cuts))
            if len(np.unique(a)) > 1]
    if not aucs:
        return None
    pars = []
    for c in z['caps']:
        v = [int(x) for x in re.findall(r'\d+', str(c).replace('0.', '.'))]
        try:
            pars.append(sum(q.numel() for q in BUILD[tag](v).parameters()))
        except Exception:
            pars.append(int(z['nparam']))
    return float(np.mean(aucs)), float(np.mean(pars))


rows, widest = [], 0
for tag, name, _ in ARMS:
    aucs, params = {}, {}
    for n in SIZES:
        k = DRAWS[n]
        got = [cell(tag, n, s) for s in range(k)]
        got = [g for g in got if g is not None]
        if got:
            params[n] = float(np.mean([p for _, p in got]))
        if len(got) == k:
            aucs[n] = (float(np.mean([a for a, _ in got])),
                       float(np.std([a for a, _ in got], ddof=1)))
    if not params:
        continue
    rows.append(dict(tag=tag, name=name, params=params, aucs=aucs))
    widest = max([n for n in SIZES if n in aucs] + [widest]) if aucs else widest

for r in rows:
    r['param'] = r['params'].get(widest, max(r['params'].values()))

best = {n: max((r['aucs'][n][0] for r in rows if n in r['aucs']), default=None)
        for n in SIZES}
L = [r'\begin{table*}[t]',
     '  \\caption{Sex classification on IXI by training set size. Capacity is reselected '
     'inside every outer fold by $3$-fold cross-validation on that fold\'s training split '
     'alone; parameter counts are the mean over the folds at $n=' + str(widest) + '$, '
     'since capacity varies with sample size. Entries are mean '
     'fold AUC $\\pm$ standard deviation over independent subsample draws, of which there '
     'are $' + str(DRAWS[SIZES[0]]) + '$ at $n=25$ and $n=50$ and $' + str(DRAWS[SIZES[-1]]) + '$ '
     'at the larger sizes, with the best in each column in bold. Paired comparisons '
     'against the proposed model are reported in the supplementary material.}',
     r'  \label{table:ixi}', r'  \centering', r'  \begin{small}',
     r'  \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lr' + 'c' * len(SIZES) + r'@{}}',
     r'    \toprule',
     r'    Model & Param. & ' +
     ' & '.join(f'$n{{=}}{n}$' for n in SIZES) + r' \\',
     r'    \midrule']
least = min(rows, key=lambda r: r['param'])['tag']
bold = set()
for r in rows:
    cells = []
    for n in SIZES:
        if n not in r['aucs']:
            cells.append('--'); continue
        m, sd = r['aucs'][n]
        txt = f'{m:.3f} \\pm {sd:.3f}'
        if best[n] is not None and abs(m - best[n]) < 1e-12:
            bold.add((r['tag'], n)); cells.append(f'$\\bf{{{txt}}}$')
        else:
            cells.append(f'${txt}$')
    nm = (r'\bf{' + r['name'] + '}') if 'proposed' in r['name'] else r['name']
    par = f"{r['param']:,.0f}"
    par = f'$\\bf{{{par}}}$' if r['tag'] == least else par
    L.append(f"    {nm} & {par} & " + ' & '.join(cells) + r' \\')
L += [r'    \bottomrule', r'  \end{tabular*}', r'  \end{small}', r'\end{table*}']

os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
print(f'\nwrote {OUT}')

bad, have = [], {r['tag']: r for r in rows}
for tag, (pub_par, pub_auc) in PUB.items():
    r = have.get(tag)
    if r is None:
        bad.append(f'{tag}: no cell written, the table prints a full row')
        continue
    if f"{r['param']:,.0f}" != f'{pub_par:,d}':
        bad.append(f"{tag} param: {r['param']:,.0f} against the table's {pub_par:,d}")
    for n in SIZES:
        pm, ps = pub_auc[n]
        if n not in r['aucs']:
            bad.append(f"{tag} n={n}: blank, the table prints {pm:.3f} +- {ps:.3f}")
            continue
        m, sd = r['aucs'][n]
        if f'{m:.3f}' != f'{pm:.3f}' or f'{sd:.3f}' != f'{ps:.3f}':
            bad.append(f'{tag} n={n}: {m:.3f} +- {sd:.3f} '
                       f"against the table's {pm:.3f} +- {ps:.3f}")
if bold != PUB_BOLD:
    bad.append(f'bold AUC cells are {sorted(bold)}, the table bolds {sorted(PUB_BOLD)}')
if least != PUB_BOLD_PARAM:
    bad.append(f'fewest parameters is {least}, the table bolds {PUB_BOLD_PARAM}')
if bad:
    raise SystemExit('DOES NOT MATCH table:ixi:\n  ' + '\n  '.join(bad))
print(f'matches table:ixi: all {len(PUB) * (len(SIZES) + 1)} cells')
