"""Diagnose HCP subject matching: how many files, how many subjects, how many matched.

The loader appends one volume per matching file with no deduplication, so a subject with
several T1w files enters the dataset more than once and StratifiedKFold can place copies
on both sides of a split. This reports the unmatched count and the duplicate count
without reading any image data.

usage: python hcp_match_check.py
"""
import os, re, collections
import pandas as pd

HCP_DIR = '/autofs/vast/lemon/data_instantsurfer_uncurated/HCP'
TSV = os.path.join(HCP_DIR, 'participants.tsv')

try:
    df = pd.read_csv(TSV, sep='\t')
except Exception:
    df = pd.read_csv(TSV, sep=r'\s+')
print(f'participants.tsv: {len(df)} rows, columns: {list(df.columns)[:8]}')
age_col = [c for c in df.columns if 'AGE' in c.upper()][0]
print(f'age column: {age_col!r}; values: {sorted(df[age_col].astype(str).unique())[:8]}')

# the id column the notebook would pick, and what the ids look like
for c in df.columns:
    if re.search(r'particip|subject|sub_?id|^id$', str(c), re.I):
        print(f'id column {c!r}: e.g. {list(df[c].astype(str))[:5]}')

files = []
for root, _, fs in os.walk(HCP_DIR):
    for f in fs:
        if f.endswith(('.nii', '.nii.gz')) and 't1w' in f.lower():
            files.append(os.path.join(root, f))
print(f'\nT1w files on disk: {len(files)}')

subs = collections.Counter()
nosub = 0
for p in files:
    m = re.search(r'sub-([a-zA-Z0-9]+)', p, re.IGNORECASE)
    if m:
        subs[m.group(1)] += 1
    else:
        nosub += 1
print(f'  files with a sub-XXXX pattern : {len(files)-nosub}')
print(f'  files WITHOUT one            : {nosub}')
print(f'  distinct subject ids         : {len(subs)}')
dup = {s: n for s, n in subs.items() if n > 1}
print(f'  subjects with >1 T1w file    : {len(dup)}  '
      f'(these would be loaded {sum(dup.values())} times -> {sum(dup.values())-len(dup)} duplicate volumes)')
if dup:
    print(f'  example: {list(dup.items())[:4]}')
print(f'\n  file basenames seen: {collections.Counter(os.path.basename(p) for p in files).most_common(5)}')
