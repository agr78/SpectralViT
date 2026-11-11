"""Which participants.tsv columns are complete enough to label all 897 subjects?"""
import pandas as pd
p='/autofs/vast/lemon/data_instantsurfer_uncurated/HCP/participants.tsv'
try: d=pd.read_csv(p, sep='\t')
except Exception: d=pd.read_csv(p, sep=r'\s+')
print(f'{len(d)} rows\n')
print(f"{'column':24s} {'non-null':>9} {'distinct':>9}  example values")
for c in d.columns:
    nn=d[c].notna().sum(); u=d[c].nunique(dropna=True)
    ex=', '.join(map(str, d[c].dropna().unique()[:4]))
    print(f'{str(c):24s} {nn:9d} {u:9d}  {ex[:52]}')
if 'Gender' in d.columns:
    print('\nGender breakdown:'); print(d['Gender'].value_counts(dropna=False).to_string())
