import sys, os
SPDIR = os.path.dirname(os.path.abspath(globals().get('__file__', '/workspace/dbs_3d/x')))
REPO  = os.path.dirname(SPDIR)
OUT   = os.path.join(REPO, 'docs', 'dbs_3d')          # docs/ is gitignored
DATA  = os.environ.get('DBS_CLINICAL_DIR', os.path.join(REPO, 'docs'))
CACHE = os.environ.get('DBS_CACHE_DIR', '/checkpoints')
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, REPO); sys.path.insert(0, SPDIR); os.chdir(REPO)
import numpy as np,pandas as pd,torch,pickle,warnings; warnings.filterwarnings('ignore')
torch.set_num_threads(12)
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
src=open(os.path.join(SPDIR,'concat_attn.py')).read(); exec(src.split("rows=[]")[0])
D=pickle.load(open(os.path.join(OUT, 'concat_attn_preds.pkl'),'rb')); PRED,bB=D['PRED'],D['bB']
# 1. family-wise null over the 30 cells
ks=list(PRED); R=np.array([rankdata(PRED[k]) for k in ks])
n=len(bB); n1=int(bB.sum()); n0=n-n1
auc=lambda y:(R@y-n1*(n1+1)/2)/(n1*n0)
obs=auc(bB); rng=np.random.RandomState(0)
mx=np.array([auc(bB[rng.permutation(n)]).max() for _ in range(5000)])
i=int(obs.argmax())
print(f"FAMILY-WISE NULL over {len(ks)} cells")
print(f"  best external {obs[i]:.3f} ({ks[i]})   chance {mx.mean():.3f} (95th {np.percentile(mx,95):.3f})   p = {(mx>=obs[i]).mean():.3f}")
pb=PRED[('3DDCT',16,'typ_resid')]
o_=roc_auc_score(bB,pb); nl=np.array([roc_auc_score(bB[rng.permutation(n)],pb) for _ in range(5000)])
print(f"  pre-specified cell 3DDCT/16/typ_resid external {o_:.3f}   single-test p = {(nl>=o_).mean():.3f}")
pg=PRED[('3DgraphLap',8,'typ_resid')]; og=roc_auc_score(bB,pg)
nlg=np.array([roc_auc_score(bB[rng.permutation(n)],pg) for _ in range(5000)])
print(f"  best-external cell 3DgraphLap/8/typ_resid external {og:.3f}   single-test p = {(nlg>=og).mean():.3f}")
# 2. internal permutation + seed stability for that cell
X0=basis3(VA,'3DDCT',16); X1=basis3(VB,'3DDCT',16)
def run(y,seed):
    pr=np.full(len(y),np.nan)
    for tr,va in StratifiedKFold(4,shuffle=True,random_state=0).split(X0,y):
        s2=StandardScaler().fit(X0[tr]); s3=StandardScaler().fit(Ca[tr])
        pr[va]=fit(s2.transform(X0[tr]),s3.transform(Ca[tr]),y[tr].astype(float),
                   s2.transform(X0[va]),s3.transform(Ca[va]),X0.shape[1],Ca.shape[1],True,True,seed=seed)
    s2=StandardScaler().fit(X0); s3=StandardScaler().fit(Ca)
    pb=fit(s2.transform(X0),s3.transform(Ca),y.astype(float),s2.transform(X1),s3.transform(Cb),X0.shape[1],Ca.shape[1],True,True,seed=seed)
    return roc_auc_score(y,pr),pb
print("\nSEED STABILITY (same cell, different init):")
ints,exts=[],[]
for sd in range(5):
    ai,pb=run(bA,sd); ae=roc_auc_score(bB,pb); ints.append(ai); exts.append(ae)
    print(f"  seed {sd}: internal {ai:.3f}  external {ae:.3f}")
print(f"  internal {np.mean(ints):.3f}+-{np.std(ints):.3f}   external {np.mean(exts):.3f}+-{np.std(exts):.3f}")
print("\nINTERNAL PERMUTATION (labels shuffled, full OOF refit, 200x):")
null=[run(bA[rng.permutation(len(bA))],0)[0] for _ in range(200)]
null=np.array(null)
print(f"  observed {ints[0]:.3f}   null mean {null.mean():.3f} SD {null.std():.3f}   p = {(null>=ints[0]).mean():.3f}")

print("\nIMAGING ABLATION (best-internal cell, test-time):")
import torch.nn as nn
s2=StandardScaler().fit(X0); s3=StandardScaler().fit(Ca)
torch.manual_seed(0); m=ConcatAttn(X0.shape[1],Ca.shape[1],typ=True,resid=True)
opt=torch.optim.AdamW(m.parameters(),3e-3,weight_decay=0.1); lf=nn.BCEWithLogitsLoss()
xt=torch.tensor(s2.transform(X0),dtype=torch.float32); ct=torch.tensor(s3.transform(Ca),dtype=torch.float32)
yt=torch.tensor(bA,dtype=torch.float32)
for e in range(1500):
    opt.zero_grad(); lf(m(xt,ct),yt).backward(); opt.step()
m.eval()
ZB,DB=s2.transform(X1),s3.transform(Cb)
def ex(Z,Dd):
    with torch.no_grad(): return roc_auc_score(bB,m(torch.tensor(Z,dtype=torch.float32),torch.tensor(Dd,dtype=torch.float32)).numpy())
rs=np.random.RandomState(0)
print(f"  intact              {ex(ZB,DB):.3f}")
print(f"  imaging ZEROED      {ex(np.zeros_like(ZB),DB):.3f}")
print(f"  imaging SHUFFLED    {ex(ZB[rs.permutation(len(ZB))],DB):.3f}")
print(f"  covariates ZEROED   {ex(ZB,np.zeros_like(DB)):.3f}")
print(f"  covariates SHUFFLED {ex(ZB,DB[rs.permutation(len(DB))]):.3f}")
