"""Build table:cifar, the CIFAR-10/100 sample-size table.

Adds the coefficient-input control that was computed with every one of these runs and never
reported: FeatureMLP, parameter-matched to the spectral arm and trained in the same loop
under one shared AdamW with summed losses, so the optimizer, batches and step count are
identical and the margin cannot be a tuning artifact.

The point of the column is attribution. The spatial arm changes the basis; the MLP keeps the
basis and changes only the head. So spectral-minus-spatial is the representation and
spectral-minus-MLP is the encoder, and on these two datasets they point opposite ways: the
patch tokenizer is competitive or better on the PCA basis, while the MLP on the same
components is far behind everywhere.

Four blocks, two sources. The PCA blocks are the fixed-basis runs; the overlapping block-DCT
blocks are the stride-2 tokenizer of sec:corr. PUB pins the published spatial and spectral
AUCs so a changed run fails loudly instead of printing a new table.

usage: python make_table_cifar.py [out.tex]
"""
import os
import sys

import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'results', 'cifar_table.tex')
R = os.path.join(HERE, 'results')
TOL = 0.0015

# The stride-2 patch tokenizer, run only for the overlap blocks: it is the control that
# separates the fixed transform from the overlap, and there is no overlap to match on the
# PCA blocks. See run note in cifar_sweep.py (CIFAR_SPAT_STRIDE).
S2 = {'CIFAR-10, overlapping block DCT': 'cifar_ovstride2/cifar10_perseed.csv',
      'CIFAR-100, overlapping block DCT': 'cifar_ovstride2/cifar100_perseed.csv'}

BLOCKS = [
    ('CIFAR-10, PCA basis', 'recovered/cifar_fixed/c10/cifar10_perseed.csv',
     {25: (0.603, 0.601), 50: (0.640, 0.632), 200: (0.691, 0.666), 1000: (0.764, 0.718)}),
    ('CIFAR-10, overlapping block DCT', 'cifar_overlap/cifar10_perseed.csv',
     {25: (0.602, 0.618), 50: (0.640, 0.677), 200: (0.686, 0.723), 1000: (0.772, 0.779)}),
    ('CIFAR-100, PCA basis', 'recovered/cifar_fixed/c100/cifar100_perseed.csv',
     {100: (0.588, 0.588), 200: (0.616, 0.616), 300: (0.635, 0.632), 500: (0.664, 0.652),
      2000: (0.760, 0.713)}),
    ('CIFAR-100, overlapping block DCT', 'cifar_overlap/cifar100_perseed.csv',
     {100: (0.586, 0.600), 200: (0.621, 0.631), 300: (0.637, 0.650), 500: (0.660, 0.679),
      2000: (0.761, 0.765)}),
]

L = [r'\begin{table*}[t]',
     '  \\caption{Classification on CIFAR-10 and CIFAR-100 by training set size, AUC over $5$ '
     'seeds on the full $10{,}000$ image test set. Protocol follows Table~\\ref{table:mnist}. '
     'Each block is its own paired run, so the spatial column is the patch tokenizer trained '
     'alongside that block\'s spectral model. On the overlapping blocks the patch tokenizer is '
     'additionally run at the same stride two, which is the control that separates the fixed '
     'transform from the overlap: at stride four the two differ in both, and the critically '
     'sampled transform is a per-patch invertible map a learned projection absorbs. The '
     'perceptron receives the same coefficients as the spectral ViT and is parameter matched to '
     'it. Under the PCA basis these images are not in correspondence, which bounds the advantage '
     'as discussed above. Bold marks the best value in each row and any value not significantly '
     'different from it; $p$ columns are paired $t$-tests over seeds against each spatial arm.}',
     r'  \label{table:cifar}', r'  \centering', r'  \begin{small}',
     r'  \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccc@{}}',
     r'    \toprule',
     r'    & \multicolumn{2}{c}{Spatial} & & & \multicolumn{2}{c}{$p$ vs spatial} \\',
     r'    \cmidrule(lr){2-3}\cmidrule(lr){6-7}',
     r'    $n$ & stride 4 & stride 2 & MLP & Spectral & stride 4 & stride 2 \\']

def ptex(p):
    return '$<0.001$' if p < 0.001 else '$' + f'{p:.3f}'.rstrip('0') + '$'

drift = []
for title, rel, pub in BLOCKS:
    d = pd.read_csv(os.path.join(R, rel))
    s2 = pd.read_csv(os.path.join(R, S2[title])) if title in S2 else None
    if s2 is not None:
        short = [n for n in pub if (s2.n == n).sum() != 5]
        if short:
            raise SystemExit(f'{title}: stride-2 control incomplete at n={short}')
    L.append(r'    \midrule')
    L.append(r'    \multicolumn{7}{@{}l}{\emph{' + title + r'}} \\')
    for n in sorted(pub):
        g = d[d.n == n]
        want_s, want_c = pub[n]
        if abs(g.spat_AUC.mean() - want_s) > TOL or abs(g.spec_AUC.mean() - want_c) > TOL:
            drift.append(f'{title} n={n}: {g.spat_AUC.mean():.3f}/{g.spec_AUC.mean():.3f} '
                         f'vs published {want_s}/{want_c}')
        arms = {'s4': g.spat_AUC, 'mlp': g.mlp_AUC, 'spec': g.spec_AUC}
        if s2 is not None:
            arms['s2'] = s2[s2.n == n].spat_AUC
        # Bold the best and anything a paired test cannot separate from it. At the largest
        # sizes the three tokenizers are genuinely indistinguishable, and bolding one of
        # them alone would assert a difference the test does not support.
        best = max(arms, key=lambda k: arms[k].mean())
        bold = {k for k, v in arms.items()
                if k == best or stats.ttest_rel(arms[best], v).pvalue >= 0.05}
        def cell(k):
            if k not in arms:
                return '--'
            v = arms[k].mean()
            return f'$\\bf{{{v:.3f}}}$' if k in bold else f'${v:.3f}$'
        p4 = ptex(stats.ttest_rel(g.spec_AUC, g.spat_AUC).pvalue)
        p2 = ptex(stats.ttest_rel(arms['spec'], arms['s2']).pvalue) if s2 is not None else '--'
        nn = f'{n:,}'.replace(',', '{,}')
        L.append(f'    ${nn}$ & {cell("s4")} & {cell("s2")} & {cell("mlp")} & {cell("spec")} '
                 f'& {p4} & {p2} ' + r'\\')
L += [r'    \bottomrule', r'  \end{tabular*}', r'  \end{small}', r'\end{table*}']

if drift:
    raise SystemExit('does not reproduce the published columns:\n  ' + '\n  '.join(drift))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
print('\nspectral minus MLP, the encoder margin:')
for title, rel, pub in BLOCKS:
    d = pd.read_csv(os.path.join(R, rel))
    for n in sorted(pub):
        g = d[d.n == n]
        p = stats.ttest_rel(g.spec_AUC, g.mlp_AUC).pvalue
        print(f'  {title:34s} n={n:5d}  {g.spec_AUC.mean()-g.mlp_AUC.mean():+.3f}  '
              f'{int((g.spec_AUC>g.mlp_AUC).sum())}/{len(g)}  p={p:.4f}')
print(f'\nreproduces the published spatial and spectral columns within {TOL}\nwrote {OUT}')
