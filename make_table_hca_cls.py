"""Build table:hca-cls, the HCP-Aging median-split classification table.

Pooled AUC, not the mean of per-fold AUCs: the out-of-fold predictions of a subsample are
concatenated and scored once, which at n=25 differs from the fold mean by about 0.03. The
pooled value per (model, seed) is what hca_sweep.py writes to hca_pooled_n{N}.csv, and
that file -- not the hca_ckpt_*.npz resume checkpoints, which hold the raw predictions and
have not survived -- is the durable record.

Reads the un-suffixed files plus the per-seed companions (hca_pooled_n{N}_s{5..9}.csv)
written by run_hca_draws10.sbatch / run_hca_cadiz.sh, which raise n=25 to ten subsamples
to match table:hca-reg and table:ixi at the same size.

Draws are counted PER MODEL, not per file. The backfill deliberately raised only the arms
a reported number depends on, so at n=50 a file holds ten draws of five arms and five of
the rest; n=100 and n=1000 were not rerun at all. A cell is refused rather than printed if
its draw count is not the expected one, so a mixed cell cannot reach the paper.

usage: python make_table_hca_cls.py [results_dir]
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'results', 'sweep')
OUT = os.path.join(HERE, 'results', 'table_hca_cls.txt')
# n=50 and n=200 were absent from the published table although the sweep had run them,
# which left sec:hcacls asserting that a linear head is 'equal or better from n=50
# upward' with no n=50 column to check it against. Both are added here, matching
# table:hca-reg's sizes. n=200 needs no new compute: it joins n=100 and n=1000 at five
# draws under the same rule table:ixi and table:hca-reg use.
SIZES = [25, 50, 100, 200, 1000]
DRAWS = {25: 10, 50: 10, 100: 5, 200: 5, 1000: 5}
# HCACLS_DRAWS=5 reproduces the table as published, before the n=25 backfill, so the
# loader can be checked against a known-good answer.
if os.environ.get('HCACLS_DRAWS'):
    DRAWS = {n: 5 for n in SIZES}
    DRAWS[25] = DRAWS[50] = int(os.environ['HCACLS_DRAWS'])

LAB = [('Logistic regression', 'LogisticRegression (Block-DCT)'),
       ('Multilayer perceptron', 'MLP (Block-DCT)'),
       ('U-net with attention', 'AttnUNet (Micro)'),
       ('Swin ViT', 'Swin (Micro)'),
       ('Spatial ViT', 'SpatialViT (Micro)'),
       ('Spectral ViT (proposed)', 'SpectralViT rank (Block-DCT)')]

# table:hca-cls as published. n=25 is rebuilt from ten draws and will move, so only the
# two sizes this backfill does not touch are held as a guard -- the same role n=566 plays
# in make_ixi_paired.py and n=100 in make_table_nsens.py.
# n=25 and n=50 are rebuilt from ten draws and will move. n=200 was never published but
# its data is untouched, so its five-draw values are pinned here as a regression guard
# the same way the published columns are.
GUARD_SIZES = {100, 200, 1000}
PUB = {'LogisticRegression (Block-DCT)': {25: 0.897, 50: None, 100: 0.935, 200: 0.937, 1000: 0.966},
       'MLP (Block-DCT)': {25: 0.882, 50: None, 100: 0.922, 200: 0.921, 1000: 0.951},
       'AttnUNet (Micro)': {25: 0.84, 50: None, 100: 0.906, 200: 0.935, 1000: 0.944},
       'Swin (Micro)': {25: 0.803, 50: None, 100: 0.79, 200: 0.837, 1000: 0.875},
       'SpatialViT (Micro)': {25: 0.805, 50: None, 100: 0.766, 200: 0.849, 1000: 0.912},
       'SpectralViT rank (Block-DCT)': {25: 0.904, 50: None, 100: 0.928, 200: 0.927, 1000: 0.947},
}

data = {}
for n in SIZES:
    files = [os.path.join(D, f'hca_pooled_n{n}.csv')]
    files += [os.path.join(D, f'hca_pooled_n{n}_s{s}.csv') for s in range(5, 10)]
    # The three volume arms at n=50 were backfilled separately (run_hca_n50vol.sh),
    # under their own suffix so they could not race the five feature arms the SLURM
    # array was writing to _s{5..9} at the same size.
    files += [os.path.join(D, f'hca_pooled_n{n}_vol_s{s}.csv') for s in range(5, 10)]
    got = [pd.read_csv(f) for f in files if os.path.exists(f)]
    if not got:
        raise SystemExit(f'no file for n={n} under {D}')
    # One row per (Model, Seed): a seed present in both the base file and a companion is
    # the same draw, not two.
    data[n] = pd.concat(got, ignore_index=True).drop_duplicates(['Model', 'Seed'])

L = ['Pooled withheld-fold AUC on the HCP-Aging median split, mean over subsamples.',
     'Parenthesized values are the standard deviation over subsamples, which the caption',
     'promises but the published table does not print.', '',
     '%-24s' % 'model' + ''.join('%20s' % f'n={n}' for n in SIZES)]

bad = []
for lab, m in LAB:
    row = '%-24s' % lab
    for n in SIZES:
        v = data[n][data[n].Model == m].AUC_pooled
        if not len(v):
            row += '%20s' % '--'
            bad.append(f'{m} absent at n={n}')
            continue
        row += '%20s' % ('%.3f (%.3f) %d' % (v.mean(), v.std(ddof=1), len(v)))
        if len(v) != DRAWS[n]:
            bad.append(f'{lab} n={n}: {len(v)} draws, expected {DRAWS[n]}')
        elif n in GUARD_SIZES and abs(v.mean() - PUB[m][n]) > 0.0005:
            bad.append(f'{lab} n={n}: {v.mean():.4f}, table says {PUB[m][n]:.3f}')
    L.append(row)
L.append('')
L.append('  cells are  mean (sd) subsamples')

# ---------------------------------------------------------------- LaTeX
# n=50 and n=200 were absent from the published table although the sweep had run both.
# Their absence left sec:hcacls asserting that a linear head is "equal or better from n=50
# upward" with no column a reader could check it against.
PARAM = {'Logistic regression': '$\\bf{4,101}$', 'Multilayer perceptron': '23,019',
         'U-net with attention': '24,213', 'Swin ViT': '23,799',
         'Spatial ViT': '89,492', 'Spectral ViT (proposed)': '10,715'}
T = [r'\begin{table*}[tb]',
     '  \\caption{The HCP-Aging cohort with age thresholded at the median. AUC computed on '
     'the pooled withheld folds of each subsample, $\\pm$ standard deviation over independent '
     'subsamples, of which there are $' + str(DRAWS[25]) + '$ at $n=25$ and $n=50$ and $'
     + str(DRAWS[100]) + '$ at the larger sizes. Coefficient-input models are shown on the '
     'block cosine basis; the Fourier magnitude variants of the same models are reported in '
     'the text. Bold marks the best value in each column.}',
     r'  \label{table:hca-cls}', r'  \centering', r'  \begin{small}',
     r'  \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lr' + 'c' * len(SIZES) + r'@{}}',
     r'    \toprule',
     r'    Model & Param. & ' + ' & '.join(f'$n{{=}}{n}$' for n in SIZES) + r' \\',
     r'    \midrule']
best = {n: max(data[n][data[n].Model == m].AUC_pooled.mean() for _, m in LAB) for n in SIZES}
for lab, m in LAB:
    cells = []
    for n in SIZES:
        v = data[n][data[n].Model == m].AUC_pooled
        txt = f'{v.mean():.3f} \\pm {v.std(ddof=1):.3f}'
        cells.append(f'$\\bf{{{txt}}}$' if abs(v.mean() - best[n]) < 1e-12 else f'${txt}$')
    nm = (r'\bf{' + lab + '}') if 'proposed' in lab else lab
    T.append(f'    {nm:28s} & {PARAM[lab]} & ' + ' & '.join(cells) + r'  \\')
T += [r'    \bottomrule', r'  \end{tabular*}', r'  \end{small}', r'\end{table*}']
open(os.path.join(HERE, 'results', 'hca_cls_table.tex'), 'w').write('\n'.join(T) + '\n')
print('\n'.join(T))
print()

report = '\n'.join(L)
print(report)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w') as fh:
    fh.write(report + '\n')
print(f'\nwrote {OUT}')

if bad:
    raise SystemExit('DOES NOT MATCH table:hca-cls:\n  ' + '\n  '.join(bad))
print(f'draw counts as expected; n={sorted(GUARD_SIZES)} match the published table')
