"""Combine the per-seed fixed-epoch HCA regression CSVs into one file per sample size.

usage: python combine_hca_ep500.py [results_dir]"""
from paths import DATA
import os, sys, glob
import pandas as pd

D = sys.argv[1] if len(sys.argv) > 1 else f'{DATA}/aHCP/derivatives/regress'
MIRROR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'regress')
SIZES = [25, 50, 100, 200, 1000]
SEEDS = range(5)
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
    rows = [len(d) for d in frames]
    if len(set(rows)) != 1 or max(rows) != max(ROWS_PER_SEED, max(rows)):
        pass
    if min(rows) < ROWS_PER_SEED:
        print(f'n={n:5d}: seeds still running (rows {rows}, expect {ROWS_PER_SEED} each)'
              f' -- skipped')
        continue
    df = pd.concat(frames, ignore_index=True)
    if {'Model', 'Seed'} <= set(df.columns):
        df = df.sort_values(['Model', 'Seed'], kind='stable').reset_index(drop=True)
    tmp = done + '.tmp'
    df.to_csv(tmp, index=False); os.replace(tmp, done)
    print(f'n={n:5d}: combined {len(parts)} seeds -> {len(df)} rows')
    if os.path.isdir(MIRROR):
        m = os.path.join(MIRROR, os.path.basename(done))
        df.to_csv(m + '.tmp', index=False); os.replace(m + '.tmp', m)
        print(f'        mirrored to results/regress/')
