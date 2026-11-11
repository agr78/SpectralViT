"""Combine the fixed-epoch HCA regression CSVs at ten seeds per sample size.

Supersedes combine_hca_ep500.py, which assumed five seeds and that every seed had its
own per-seed file. Neither holds now:

  * seeds 5-9 come from slurm/hca_regress_ep500_seeds59.sbatch, one file per (size, seed);
  * seeds 0-4 of n=25 and n=50 were produced by a single serial driver that wrote the
    combined hca_reg_n{N}_ep500.csv directly and left no per-seed files behind, so that
    combined file is their only record;
  * n=1000 stays at five seeds -- its seed-level effect size is 0.19, so no feasible
    replicate count separates it and 63 h/seed is not worth spending on an interval.

Nothing is overwritten. Output goes to hca_reg_n{N}_ep500_10s.csv alongside the five-seed
files, which stay exactly as they are until the ten-seed numbers have been looked at.

A size is written only when every expected seed is present AND complete: 15 arms x 5
folds = 75 rows, with the full arm x fold grid. A file that exists is not a file that is
finished, because each task appends as it goes.

usage: python combine_hca_10seed.py [results_dir]
"""
import os, sys
import pandas as pd

D = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser('~/spacelink/aHCP/derivatives/regress')
MIRROR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'regress')
EXPECTED = {25: 10, 50: 10, 100: 10, 200: 10, 1000: 5}
N_MODELS, N_FOLDS = 15, 5
ROWS_PER_SEED = N_MODELS * N_FOLDS


def seed_frame(d, n, s):
    """Rows for one seed, from its per-seed file or else the five-seed combined file."""
    f = os.path.join(d, f'hca_reg_n{n}_ep500_s{s}.csv')
    if os.path.exists(f):
        return pd.read_csv(f), f'_s{s}.csv'
    pooled = os.path.join(d, f'hca_reg_n{n}_ep500.csv')
    if s < 5 and os.path.exists(pooled):
        df = pd.read_csv(pooled)
        return df[df.Seed == s].copy(), 'combined'
    return None, None


def complete(df):
    """A seed is done when it carries the whole arm x fold grid, not just enough rows."""
    return (len(df) == ROWS_PER_SEED and df.Model.nunique() == N_MODELS
            and df.Fold.nunique() == N_FOLDS
            and len(df.drop_duplicates(['Model', 'Fold'])) == ROWS_PER_SEED)


for n, want in EXPECTED.items():
    frames, short, absent, origin = [], [], [], {}
    for s in range(want):
        df, src = seed_frame(D, n, s)
        if df is None or df.empty:
            absent.append(s)
        elif not complete(df):
            short.append((s, len(df)))
        else:
            frames.append(df.assign(Seed=s))
            origin[s] = src
    if absent or short:
        have = len(frames)
        msg = f'n={n:5d}: {have}/{want} seeds ready'
        if short:
            msg += f', in flight {[f"s{s}@{r}/{ROWS_PER_SEED}" for s, r in short]}'
        if absent:
            msg += f', not started {["s%d" % s for s in absent]}'
        print(msg + ' -- skipped')
        continue
    out = pd.concat(frames, ignore_index=True)
    # Match the row order of the five-seed files rather than the order tasks finished.
    out = out.sort_values(['Model', 'Seed', 'Fold'], kind='stable').reset_index(drop=True)
    name = f'hca_reg_n{n}_ep500_10s.csv'
    for base in [D] + ([MIRROR] if os.path.isdir(MIRROR) and MIRROR != D else []):
        p = os.path.join(base, name)
        out.to_csv(p + '.tmp', index=False); os.replace(p + '.tmp', p)
    carried = sum(1 for v in origin.values() if v == 'combined')
    print(f'n={n:5d}: {want} seeds -> {len(out)} rows  ({carried} carried from the '
          f'five-seed combined file)  -> {name}')
