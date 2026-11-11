"""Find age/sex labels for HCP-Aging (HCA) subjects anywhere under a search root.

usage: python hca_labels.py [search_root ...]
"""
import os, sys, re
import pandas as pd

roots = sys.argv[1:] or ['/autofs/vast/lemon']
pat = re.compile(r'(ndar|subject0|demograph|behav|pheno|unrestricted|hca|aging|participant)', re.I)
cands = []
for root in roots:
    for dp, dns, fs in os.walk(root):
        if dp.count(os.sep) - root.count(os.sep) > 4:
            dns[:] = []; continue
        for f in fs:
            if f.lower().endswith(('.csv', '.tsv', '.txt', '.xlsx')) and pat.search(f):
                fp = os.path.join(dp, f)
                try:
                    if os.path.getsize(fp) < 80_000_000: cands.append(fp)
                except OSError: pass
print(f'{len(cands)} candidate label files\n')
for fp in cands[:30]:
    print(f'--- {fp}  ({os.path.getsize(fp)/1024:.0f} KB)')
    try:
        sep = '\t' if fp.lower().endswith(('.tsv', '.txt')) else ','
        d = pd.read_csv(fp, sep=sep, nrows=3000, low_memory=False)
        cols = [c for c in d.columns if re.search(r'age|sex|gender|subject|src_subject|interview', str(c), re.I)]
        has_hca = any(d[c].astype(str).str.contains('HCA', na=False).any() for c in d.columns[:40])
        print(f'    {len(d)} rows, {len(d.columns)} cols, contains HCA ids: {has_hca}')
        for c in cols[:8]:
            print(f'      {c!r}: non-null {d[c].notna().sum()}, distinct {d[c].nunique()}, '
                  f'e.g. {list(d[c].dropna().astype(str).unique()[:4])}')
    except Exception as e:
        print(f'    (unreadable: {type(e).__name__})')
