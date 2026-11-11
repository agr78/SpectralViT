"""Combine the per-seed fixed-epoch HCA regression CSVs into one file per sample size.

slurm/hca_regress_ep500.sbatch arrays over (size, seed) so that the five draws of a size
run concurrently rather than in series, which is what makes the conventional fixed-epoch
budget affordable at n=1000. Each task therefore writes hca_reg_n{N}_ep500_s{seed}.csv,
and this reassembles them into hca_reg_n{N}_ep500.csv, the layout the already-complete
n=25 and n=50 files use and the one make_fig_data_efficiency.py and the table builder read.

A size is written only when all five seeds are present. A partial size is reported and
skipped rather than written, since a table assembled from three draws of one size and
five of another would be silently inconsistent.

usage: python combine_hca_ep500.py [results_dir]
"""
import os, sys, glob
import pandas as pd

D = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser('~/spacelink/aHCP/derivatives/regress')
MIRROR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'regress')
SIZES = [25, 50, 100, 200, 1000]
SEEDS = range(5)
# 15 arms x 5 folds per seed; a short file means the task has not finished.
ROWS_PER_SEED = 75

for n in SIZES:
    parts, missing = [], []
    for s in SEEDS:
        f = os.path.join(D, f'hca_reg_n{n}_ep500_s{s}.csv')
        (parts if os.path.exists(f) else missing).append(f if os.path.exists(f) else s)
    done = os.path.join(D, f'hca_reg_n{n}_ep500.csv')
    if not parts:
        if os.path.exists(done):
            print(f'n={n:5d}: already combined ({len(pd.read_csv(done))} rows)')
        else:
            print(f'n={n:5d}: nothing yet')
        continue
    if missing:
        print(f'n={n:5d}: {len(parts)}/5 seeds present, missing {missing} -- skipped')
        continue
    frames = [pd.read_csv(f) for f in parts]
    # A task writes its CSV incrementally, so a file that exists is not a file that is
    # finished. Every seed must carry the full arm x fold grid or the size is still in
    # flight; combining early produces a short file that looks complete.
    rows = [len(d) for d in frames]
    if len(set(rows)) != 1 or max(rows) != max(ROWS_PER_SEED, max(rows)):
        pass
    if min(rows) < ROWS_PER_SEED:
        print(f'n={n:5d}: seeds still running (rows {rows}, expect {ROWS_PER_SEED} each)'
              f' -- skipped')
        continue
    df = pd.concat(frames, ignore_index=True)
    # Seed is carried in the per-seed files; sort so the row order matches the
    # single-process runs at n=25 and n=50 rather than the order tasks happened to finish.
    if {'Model', 'Seed'} <= set(df.columns):
        df = df.sort_values(['Model', 'Seed'], kind='stable').reset_index(drop=True)
    tmp = done + '.tmp'
    df.to_csv(tmp, index=False); os.replace(tmp, done)
    print(f'n={n:5d}: combined {len(parts)} seeds -> {len(df)} rows')
    if os.path.isdir(MIRROR):
        m = os.path.join(MIRROR, os.path.basename(done))
        df.to_csv(m + '.tmp', index=False); os.replace(m + '.tmp', m)
        print(f'        mirrored to results/regress/')
