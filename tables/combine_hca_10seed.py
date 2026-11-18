"""Combine the fixed-epoch HCA regression CSVs across seeds.

usage: python tables/combine_hca_10seed.py [results_dir]"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repository root: shared modules and paths.py
from paths import DATA
import os, sys
import pandas as pd

D = sys.argv[1] if len(sys.argv) > 1 else f'{DATA}/aHCP/derivatives/regress'
MIRROR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'regress')
EXPECTED = {25: 10, 50: 10, 100: 10, 200: 10, 1000: 5}
N_MODELS, N_FOLDS = 15, 5
ROWS_PER_SEED = N_MODELS * N_FOLDS


def seed_frame(d, n, s):
    f = os.path.join(d, f'hca_reg_n{n}_ep500_s{s}.csv')
    if os.path.exists(f):
        return pd.read_csv(f), f'_s{s}.csv'
    pooled = os.path.join(d, f'hca_reg_n{n}_ep500.csv')
    if s < 5 and os.path.exists(pooled):
        df = pd.read_csv(pooled)
        return df[df.Seed == s].copy(), 'combined'
    return None, None


def complete(df):
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
    out = out.sort_values(['Model', 'Seed', 'Fold'], kind='stable').reset_index(drop=True)
    name = f'hca_reg_n{n}_ep500_10s.csv'
    for base in [D] + ([MIRROR] if os.path.isdir(MIRROR) and MIRROR != D else []):
        p = os.path.join(base, name)
        out.to_csv(p + '.tmp', index=False); os.replace(p + '.tmp', p)
    carried = sum(1 for v in origin.values() if v == 'combined')
    print(f'n={n:5d}: {want} seeds -> {len(out)} rows  ({carried} carried from the '
          f'five-seed combined file)  -> {name}')
