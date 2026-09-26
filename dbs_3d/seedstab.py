import sys, os
SPDIR = os.path.dirname(os.path.abspath(globals().get('__file__', '/workspace/dbs_3d/x')))
REPO  = os.path.dirname(SPDIR)
OUT   = os.path.join(REPO, 'docs', 'dbs_3d')          # docs/ is gitignored
DATA  = os.environ.get('DBS_CLINICAL_DIR', os.path.join(REPO, 'docs'))
CACHE = os.environ.get('DBS_CACHE_DIR', '/checkpoints')
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, REPO); sys.path.insert(0, SPDIR); os.chdir(REPO)
import numpy as np,torch,torch.nn as nn,warnings,time,pickle; warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
src=open(os.path.join(SPDIR,'concat_attn.py')).read(); exec(src.split("rows=[]")[0])
dev=torch.device('cuda'); T=lambda a: torch.tensor(a,dtype=torch.float32,device=dev)
t0=time.time(); log=lambda m: print(f"[{time.time()-t0:5.0f}s] {m}",flush=True)
def gfit(Xtr,Ctr,ytr,Xte,Cte,K,C,seed):
    torch.manual_seed(seed); m=ConcatAttn(K,C,typ=True,resid=True).to(dev)
    opt=torch.optim.AdamW(m.parameters(),3e-3,weight_decay=0.1); lf=nn.BCEWithLogitsLoss()
    xt,ct,yt=T(Xtr),T(Ctr),T(ytr)
    for e in range(1500):
        opt.zero_grad(); lf(m(xt,ct),yt).backward(); opt.step()
    m.eval()
    with torch.no_grad(): return m(T(Xte),T(Cte)).cpu().numpy()
X0,X1=basis3(VA,'3DDCT',16),basis3(VB,'3DDCT',16)
N=20
log(f"seed stability, 3DDCT K=16 typ_resid, {N} independent inits")
ii,ee,pbs=[],[],[]
for sd in range(N):
    pr=np.full(len(bA),np.nan)
    for tr,va in StratifiedKFold(4,shuffle=True,random_state=0).split(X0,bA):
        s2=StandardScaler().fit(X0[tr]); s3=StandardScaler().fit(Ca[tr])
        pr[va]=gfit(s2.transform(X0[tr]),s3.transform(Ca[tr]),bA[tr].astype(float),
                    s2.transform(X0[va]),s3.transform(Ca[va]),X0.shape[1],Ca.shape[1],sd)
    s2=StandardScaler().fit(X0); s3=StandardScaler().fit(Ca)
    pb=gfit(s2.transform(X0),s3.transform(Ca),bA.astype(float),s2.transform(X1),s3.transform(Cb),
            X0.shape[1],Ca.shape[1],sd)
    ai,ae=roc_auc_score(bA,pr),roc_auc_score(bB,pb); ii.append(ai); ee.append(ae); pbs.append(pb)
    print(f"  seed {sd:2d}: internal {ai:.3f}  external {ae:.3f}",flush=True)
ii,ee=np.array(ii),np.array(ee)
print(f"\n  internal  mean {ii.mean():.3f}  SD {ii.std():.3f}  min {ii.min():.3f}  max {ii.max():.3f}")
print(f"  external  mean {ee.mean():.3f}  SD {ee.std():.3f}  min {ee.min():.3f}  max {ee.max():.3f}")
print(f"  external below 0.5 in {int((ee<0.5).sum())}/{N} seeds")
pe=np.mean(pbs,0); ae=roc_auc_score(bB,pe)
rng=np.random.RandomState(0); n=len(bB)
nl=np.array([roc_auc_score(bB[rng.permutation(n)],pe) for _ in range(20000)])
print(f"\n  SEED-ENSEMBLED external AUC {ae:.3f}   permutation p = {(nl>=ae).mean():.4f}")
bs=[]
for _ in range(20000):
    ix=rng.randint(0,n,n)
    if len(np.unique(bB[ix]))>1: bs.append(roc_auc_score(bB[ix],pe[ix]))
print(f"  bootstrap 95% CI over patients [{np.percentile(bs,2.5):.3f}, {np.percentile(bs,97.5):.3f}]")
pickle.dump(dict(internal=ii,external=ee,ens=pe,bB=bB),open(os.path.join(OUT, 'seedstab.pkl'),'wb'))
log("DONE")
