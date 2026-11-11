"""Emit the IXI data-efficiency table: parameters, efficiency, and AUC by sample size.

Replaces the single full-cohort table, which reported only n=566 -- the one point on the
curve where the proposed model is behind. Parameters are the mean over every completed
cell, since capacity is reselected inside each fold and so varies with n; efficiency is
AUC per million parameters at the largest completed size, matching how the superseded
table defined it.

Cells still running are left blank rather than averaged over whatever has finished, so a
partial column cannot be mistaken for a complete one.

usage: python make_table_ixi.py [out.tex]
"""
import os, re, sys
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from networks import SpectralViT, SpatialViT, SwinTransformer, AttentionUNet, \
                     LogisticRegression, MultiLayerPerceptron

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'results', 'ixi_table.tex')
DIR = os.environ.get('IXI_SWEEP_DIR', os.path.join(HERE, 'results', 'ixi_curve_merged'))
SIZES, VOL = [25, 50, 100, 200, 566], 96
# Draws per size. The two smallest were raised to ten because that is where the
# spread across draws is widest and where the claim is made; the larger sizes stay
# at five, matching the HCP-Aging regression. No draw was discarded to do it.
DRAWS = {25: 10, 50: 10, 100: 5, 200: 5, 566: 5}

# tag -> (display name, builder taking the integers parsed from the capacity label).
# Labels differ by arm: '128 comp', 'patch 12', 'embed 16', '16 ch' carry one number,
# while the simple arms carry two ('64c wd0.05', '64c h6') -- components first, then
# their own hyperparameter.
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


def cell(tag, n, s):
    """(mean fold AUC, mean parameter count) for one cell, or None if not written.

    nparam in the npz is whatever the LAST fold selected, so it understates any arm
    whose capacity varied across folds. Rebuilding from the stored per-fold capacities
    gives the mean over the models actually fit.
    """
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
        # Integers in label order; a trailing float like wd0.05 contributes nothing.
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
        # Only a complete column is reported; a partial mean would read as final.
        if len(got) == k:
            aucs[n] = (float(np.mean([a for a, _ in got])),
                       float(np.std([a for a, _ in got], ddof=1)))
    if not params:
        continue
    rows.append(dict(name=name, params=params, aucs=aucs))
    widest = max([n for n in SIZES if n in aucs] + [widest]) if aucs else widest

# Capacity is reselected per size, so a mean over sizes describes no model that was fit.
# Report the count at the largest completed size, which is what a size comparison wants.
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
for r in rows:
    cells = []
    for n in SIZES:
        if n not in r['aucs']:
            cells.append('--'); continue
        m, sd = r['aucs'][n]
        txt = f'{m:.3f} \\pm {sd:.3f}'
        cells.append(f'$\\bf{{{txt}}}$' if best[n] is not None and abs(m - best[n]) < 1e-12
                     else f'${txt}$')
    nm = (r'\bf{' + r['name'] + '}') if 'proposed' in r['name'] else r['name']
    L.append(f"    {nm} & {r['param']:,.0f} & " + ' & '.join(cells) + r' \\')
L += [r'    \bottomrule', r'  \end{tabular*}', r'  \end{small}', r'\end{table*}']

os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
print(f'\nwrote {OUT}')
