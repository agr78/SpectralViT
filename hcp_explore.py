"""Characterize a candidate HCP download: structure, metadata, subject count, labels.

usage: python hcp_explore.py [root]
"""
import os, sys, re, collections
import pandas as pd

root = sys.argv[1] if len(sys.argv) > 1 else '/autofs/vast/lemon/data_original_downloads/ahcp_data'
print(f'root: {root}')
print(f'exists: {os.path.isdir(root)}')
if not os.path.isdir(root):
    sys.exit('not a directory')

print('\n=== top level (first 25) ===')
for e in sorted(os.listdir(root))[:25]:
    p = os.path.join(root, e)
    print(f'  {"d" if os.path.isdir(p) else "f"}  {e}')
print(f'  ({len(os.listdir(root))} entries total)')

print('\n=== metadata files (csv/tsv/json/txt, <50MB) ===')
meta = []
for dp, _, fs in os.walk(root):
    if dp.count(os.sep) - root.count(os.sep) > 3: continue
    for f in fs:
        if f.lower().endswith(('.csv', '.tsv', '.json', '.txt')):
            fp = os.path.join(dp, f)
            try:
                if os.path.getsize(fp) < 50_000_000: meta.append(fp)
            except OSError: pass
for m in meta[:15]:
    print(f'  {os.path.relpath(m, root)}  ({os.path.getsize(m)/1024:.0f} KB)')
print(f'  ({len(meta)} found)')

print('\n=== T1w-like images ===')
imgs = collections.Counter()
subs = set()
for dp, _, fs in os.walk(root):
    for f in fs:
        if f.endswith(('.nii', '.nii.gz', '.mgz')):
            low = f.lower()
            tag = 'T1w' if ('t1w' in low or 't1' in low) else 'other'
            imgs[tag] += 1
            m = re.search(r'(sub-[A-Za-z0-9]+|HCA\d+|HCD\d+|\b\d{6}\b)', dp + '/' + f)
            if m and tag == 'T1w': subs.add(m.group(1))
print(f'  T1w-like: {imgs["T1w"]}   other NIfTI/MGZ: {imgs["other"]}')
print(f'  distinct subject tokens in T1w paths: {len(subs)}')
print(f'  examples: {sorted(subs)[:6]}')

for m in meta[:6]:
    if not m.lower().endswith(('.csv', '.tsv')): continue
    print(f'\n=== {os.path.relpath(m, root)} ===')
    try:
        sep = '\t' if m.lower().endswith('.tsv') else ','
        d = pd.read_csv(m, sep=sep, nrows=5000, low_memory=False)
        print(f'  {len(d)} rows (first 5000), {len(d.columns)} cols')
        hits = [c for c in d.columns if re.search(r'age|sex|gender|subject|id', str(c), re.I)]
        for c in hits[:8]:
            print(f'    {c!r}: non-null {d[c].notna().sum()}, distinct {d[c].nunique()}, '
                  f'e.g. {list(d[c].dropna().unique()[:5])}')
    except Exception as e:
        print(f'  (unreadable: {type(e).__name__})')
