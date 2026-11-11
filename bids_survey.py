"""Survey participants.tsv across the BIDS datasets: rows, age and sex completeness.

Reads only the small metadata files by explicit path -- no directory walking, so the
automounter is touched once per dataset.

usage: python bids_survey.py
"""
import os, re
import pandas as pd

ROOT = '/autofs/vast/lemon/data_instantsurfer_uncurated'
EXTRA = ['/autofs/vast/lemon/datasets/ABIDE_full/participants.tsv']
sets = ['ABIDE','ADHD200','ADNI3','AIBL','ATLAS','BAGhDAD','Buckner40','COBRE',
        'Chinese-HCP','HCP','MCIC','MGH-PACS-SynthSR','OASIS3','PPMI']

def load(p):
    for sep in ['\t', ',']:
        try:
            d = pd.read_csv(p, sep=sep, low_memory=False)
            if d.shape[1] > 1: return d
        except Exception: pass
    return None

print(f"{'dataset':20s} {'rows':>6} {'age col':>16} {'age ok':>7} {'sex col':>10} {'sex ok':>7}  age range")
paths = [(s, os.path.join(ROOT, s, f)) for s in sets for f in ('participants.tsv','participants.csv')]
paths += [(os.path.basename(os.path.dirname(p)), p) for p in EXTRA]
seen = set()
for name, p in paths:
    if name in seen or not os.path.exists(p): continue
    d = load(p)
    if d is None: continue
    seen.add(name)
    ac = next((c for c in d.columns if re.fullmatch(r'age|Age|AGE|age_at_scan.*|AgeGroup', str(c))), None) \
         or next((c for c in d.columns if 'age' in str(c).lower()), None)
    sc = next((c for c in d.columns if re.search(r'sex|gender', str(c), re.I)), None)
    an = d[ac].notna().sum() if ac else 0
    sn = d[sc].notna().sum() if sc else 0
    rng = ''
    if ac and an:
        v = pd.to_numeric(d[ac], errors='coerce').dropna()
        rng = f'{v.min():.0f}-{v.max():.0f} (numeric)' if len(v) > an*0.5 else \
              ', '.join(map(str, d[ac].dropna().unique()[:4]))[:34]
    print(f'{name:20s} {len(d):6d} {str(ac)[:16]:>16} {an:7d} {str(sc)[:10]:>10} {sn:7d}  {rng}')
