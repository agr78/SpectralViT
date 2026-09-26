import os
import pandas as pd, numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get('DBS_CLINICAL_DIR', os.path.join(REPO, 'docs'))
def clin_tables():
    d=pd.read_csv(os.path.join(DATA,'dbs_03292024.csv'),header=1)
    d.columns=[str(c).strip().replace('\n',' ') for c in d.columns]
    num=lambda c: pd.to_numeric(d[c],errors='coerce')
    A=pd.DataFrame(dict(sid=num('CORNELL ID'),age=num('Age'),
        sex=num('Sex'),
        dur=num('Disease Duration (year)'),ledd=num('pre op levadopa equivalent dose (mg)'),
        off=num('OFF (pre-dbs updrs)'),on=num('ON (pre-dbs updrs)'))).dropna(subset=['sid'])
    c=pd.read_csv(os.path.join(DATA,'chh_subjects_table1_20240729.csv'),header=None).iloc[2:]
    g=lambda i: pd.to_numeric(c[i],errors='coerce')
    B=pd.DataFrame(dict(sid=g(0),age=g(1),sex=g(2),dur=g(3),ledd=g(4),off=g(9),on=g(10))).dropna(subset=['sid'])
    for T in (A,B):
        T['sid']=T['sid'].astype(int)
        T['ldopa']=(T['off']-T['on'])/T['off']     # levodopa responsiveness: the classic DBS predictor
    return A.set_index('sid'),B.set_index('sid')
SAFE=['age','sex','dur','ledd']                     # fully decoupled from the target
FULL=['age','sex','dur','ledd','off','on','ldopa']  # includes OFF_pre, the target's denominator
