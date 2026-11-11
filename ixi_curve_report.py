"""Report the IXI data-efficiency curve from whatever sweep cells have landed so far.

Reads the suffixed out-of-fold files the array writes and prints mean AUC by sample size,
skipping cells that are not yet written. Mirrors the summary into results/ so it can be
read without pasting.

usage: python ixi_curve_report.py
"""
import os
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

def _sweep_dir():
    """Prefer the in-repo mirror, which is readable everywhere; else the scratch copy."""
    env = os.environ.get('IXI_SWEEP_DIR')
    if env:
        return env
    mirror = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'ixi_curve')
    if os.path.isdir(mirror) and any(f.startswith('oof_') for f in os.listdir(mirror)):
        return mirror
    return os.path.expanduser('~/spacelink/ixi_nested')


DIR = _sweep_dir()
# Overridable so a run against a scratch directory cannot clobber the real report.
OUT = os.environ.get('IXI_REPORT_OUT') or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'results', 'ixi_curve_progress.txt')
SIZES = [25, 50, 100, 200, 566]
ARMS = [('spec', 'Spectral ViT'), ('unet', 'AttnU-net'), ('spat', 'Spatial ViT'),
        ('swin', 'Swin ViT'), ('logreg', 'Logistic regression'),
        ('mlp', 'Multilayer perceptron')]


def cell(tag, n, s):
    """Mean fold AUC for one (arm, size, seed), or None if the cell is not written."""
    f = os.path.join(DIR, f'oof_{tag}_n{n}_s{s}.npz')
    if not os.path.exists(f):
        return None
    z = np.load(f)
    cuts = np.cumsum(z['fold_sizes'])[:-1]
    a = [roc_auc_score(y, p) for y, p in
         zip(np.split(z['y'], cuts), np.split(z['probs'], cuts)) if len(np.unique(y)) > 1]
    return float(np.mean(a)) if a else None


lines = [f'IXI data-efficiency sweep, reading {DIR}',
         'Mean fold AUC per cell, averaged over the seeds that have finished.',
         'A cell is written only when all five arms of that (n, seed) complete.', '',
         ' ' * 22 + ' ' + ' '.join(f'{"n=" + str(n):>12s}' for n in SIZES)]
# Keyed by seed, not a bare list: arms within a cell finish in order, so one arm can
# be a seed ahead of another and comparing their first k entries would silently pair
# different subsamples.
grid = {}
for tag, name in ARMS:
    row = []
    for n in SIZES:
        vals = {s: v for s in range(5) if (v := cell(tag, n, s)) is not None}
        grid[(tag, n)] = vals
        row.append(f'{np.mean(list(vals.values())):.3f} ({len(vals)}/5)'
                   if vals else f'{"--":>9s}')
    lines.append(f'{name:22s} ' + ' '.join(f'{c:>12s}' for c in row))

lines += ['', 'Spectral ViT minus each baseline, paired on the seeds both have finished:']
for tag, name in ARMS[1:]:
    row = []
    for n in SIZES:
        a, b = grid[('spec', n)], grid[(tag, n)]
        shared = sorted(set(a) & set(b))
        if shared:
            d = np.array([a[s] - b[s] for s in shared])
            wins = int((d > 0).sum())
            row.append(f'{d.mean():+.3f} {wins}/{len(shared)}')
        else:
            row.append(f'{"--":>9s}')
    lines.append(f'  vs {name:19s} ' + ' '.join(f'{c:>12s}' for c in row))

lines += ['', 'Paired t over subsample draws, once a size has all 5 seeds in both arms:']
for tag, name in ARMS[1:]:
    row = []
    for n in SIZES:
        a, b = grid[('spec', n)], grid[(tag, n)]
        shared = sorted(set(a) & set(b))
        if len(shared) >= 3:
            d = [a[s] - b[s] for s in shared]
            p_ = stats.ttest_rel([a[s] for s in shared], [b[s] for s in shared]).pvalue
            row.append(f'p={p_:.4f} ({len(shared)})' if np.std(d) > 0 else f'{"--":>9s}')
        else:
            row.append(f'{"--":>9s}')
    lines.append(f'  vs {name:19s} ' + ' '.join(f'{c:>14s}' for c in row))

# Counted per group: the two simple baselines were added after the first cells ran, so
# a single all-arms count would report zero while nine volume-arm cells are finished.
VOL_ARMS = ['spec', 'spat', 'swin', 'unet']
SIMPLE = ['logreg', 'mlp']


def ncomplete(tags):
    return sum(1 for n in SIZES for s in range(5)
               if all(cell(t, n, s) is not None for t in tags))


lines += ['', f'cells complete, network arms: {ncomplete(VOL_ARMS)}/25',
          f'cells complete, simple baselines: {ncomplete(SIMPLE)}/25']
missing = [i for i in range(25)
           if any(cell(t, SIZES[i // 5], i % 5) is None for t in SIMPLE)
           and all(cell(t, SIZES[i // 5], i % 5) is not None for t in VOL_ARMS)]
if missing:
    lines += ['', 'cells with networks but no simple baselines (top-up needed):',
              '  ' + ','.join(map(str, missing))]
txt = '\n'.join(lines)
print(txt)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT + '.tmp', 'w').write(txt + '\n')
os.replace(OUT + '.tmp', OUT)
print(f'\nwrote {OUT}')
