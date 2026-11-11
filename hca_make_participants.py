"""Convert the HCA age table into a participants.tsv the image loader can use.

The source has one row per subject-event. Only V1/V2/V3 are in-person MR visits; F*, CR
and AF* are surveys with no imaging, so they are dropped. Ages of "90 or older" are
censored at 90 and flagged, and the binary label is split at the cohort median rather than
at a threshold inherited from a young-adult cohort.

usage: python hca_make_participants.py <raw.tsv> [out.tsv]
"""
import sys, os
import pandas as pd, numpy as np

src = sys.argv[1] if len(sys.argv) > 1 else 'hca_ages_raw.tsv'
out = sys.argv[2] if len(sys.argv) > 2 else 'participants_hca.tsv'

d = pd.read_csv(src, sep='\t', dtype=str).dropna(how='all')
d.columns = [c.strip() for c in d.columns]
need = {'id_event', 'id', 'event', 'study', 'age_open'}
missing = need - set(d.columns)
if missing:
    sys.exit(f'missing columns: {missing}; found {list(d.columns)}')
d = d[d['id'].notna() & (d['id'].str.strip() != '')]
print(f'{len(d)} rows, {d["id"].nunique()} subjects, events: '
      f'{sorted(d["event"].dropna().unique())}')

d['censored'] = d['age_open'].str.contains('90 or older', na=False)
d['age'] = pd.to_numeric(d['age_open'].where(~d['censored'], '90'), errors='coerce')
print(f'  censored at 90: {d["censored"].sum()} rows; unparseable ages: {d["age"].isna().sum()}')

mr = d[d['event'].str.fullmatch(r'V\d', na=False)].copy()
print(f'  MR visits (V*): {len(mr)} rows, {mr["id"].nunique()} subjects')
print(f'  surveys dropped: {len(d)-len(mr)} rows')

mr['visit'] = mr['event'].str.extract(r'V(\d)').astype(int)
first = mr.sort_values('visit').groupby('id', as_index=False).first()
print(f'  one scan per subject (earliest visit): {len(first)} subjects')

med = first['age'].median()
first['AgeGroup'] = np.where(first['age'] >= med, f'{med:.0f} and above', f'below {med:.0f}')
first['participant_id'] = first['id_event']
cols = ['participant_id', 'id', 'id_event', 'event', 'study', 'age', 'censored', 'AgeGroup']
first[cols].to_csv(out, sep='\t', index=False)

print(f'\nage range {first["age"].min():.0f}-{first["age"].max():.0f}, median {med:.0f}')
print(first['AgeGroup'].value_counts().to_string())
print(f'\nwrote {out} ({len(first)} rows)')

# coverage against the images themselves: the only check that answers
# "do we have an age for every subject we can actually load?"
IMG = os.environ.get('HCA_IMG_DIR',
                     '/autofs/vast/lemon/data_instantsurfer_uncurated/ahcp_data')
if os.path.isdir(IMG):
    import re
    dirs = [e for e in os.listdir(IMG) if os.path.isdir(os.path.join(IMG, e))]
    img_ev, img_sub = set(), set()
    for e in dirs:
        m = re.match(r'(HCA\d+)_(V\d)_', e)
        if m:
            img_ev.add(f'{m.group(1)}_{m.group(2)}'); img_sub.add(m.group(1))
    tbl_ev = set(mr['id_event']); tbl_sub = set(mr['id'])
    print(f'\n=== coverage against {IMG} ===')
    print(f'  image dirs with a V-visit : {len(img_ev)} visits, {len(img_sub)} subjects')
    print(f'  table MR rows             : {len(tbl_ev)} visits, {len(tbl_sub)} subjects')
    print(f'  imaged visits WITH an age : {len(img_ev & tbl_ev)}')
    print(f'  imaged visits MISSING age : {len(img_ev - tbl_ev)}')
    print(f'  imaged subjects with age  : {len(img_sub & tbl_sub)} of {len(img_sub)}')
    miss = sorted(img_sub - tbl_sub)
    if miss:
        print(f'  subjects with images but no age ({len(miss)}): {miss[:10]}'
              + (' ...' if len(miss) > 10 else ''))
    extra = len(tbl_sub - img_sub)
    print(f'  table subjects with no images : {extra}')
else:
    print(f'\n(image dir {IMG} not reachable; set HCA_IMG_DIR to cross-check coverage)')
print('all MR visits (for a longitudinal/test-retest analysis) would be '
      f'{len(mr)} rows across {mr["id"].nunique()} subjects')
