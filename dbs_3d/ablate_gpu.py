import sys, os
SPDIR = os.path.dirname(os.path.abspath(globals().get('__file__', '/workspace/dbs_3d/x')))
REPO  = os.path.dirname(SPDIR)
OUT   = os.path.join(REPO, 'docs', 'dbs_3d')          # docs/ is gitignored
DATA  = os.environ.get('DBS_CLINICAL_DIR', os.path.join(REPO, 'docs'))
CACHE = os.environ.get('DBS_CACHE_DIR', '/checkpoints')
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, REPO); sys.path.insert(0, SPDIR); os.chdir(REPO)
import numpy as np,torch,torch.nn as nn,warnings,time; warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
src=open(os.path.join(SPDIR,'concat_attn.py')).read(); exec(src.split("rows=[]")[0])
dev=torch.device('cuda'); print(f"device: {torch.cuda.get_device_name(0)}",flush=True)
T=lambda a: torch.tensor(a,dtype=torch.float32,device=dev)
def gfit(Xtr,Ctr,ytr,K,C,typ,resid,ep=1500,seed=0):
    torch.manual_seed(seed); m=ConcatAttn(K,C,typ=typ,resid=resid).to(dev)
    opt=torch.optim.AdamW(m.parameters(),3e-3,weight_decay=0.1); lf=nn.BCEWithLogitsLoss()
    xt,ct,yt=T(Xtr),T(Ctr),T(ytr)
    for e in range(ep):
        opt.zero_grad(); lf(m(xt,ct),yt).backward(); opt.step()
    return m.eval()
print("\nIMAGING ABLATION on the pre-specified cell (3DDCT K=16, typ_resid)")
print("trained normally; imaging or covariates removed only at TEST time\n")
for bn,K in [('3DDCT',16),('3DgraphLap',8),('3D|FFT|',16)]:
    X0,X1=basis3(VA,bn,K),basis3(VB,bn,K)
    s2=StandardScaler().fit(X0); s3=StandardScaler().fit(Ca)
    ZA,ZB,DA,DB=s2.transform(X0),s2.transform(X1),s3.transform(Ca),s3.transform(Cb)
    res={k:[] for k in ['intact','img0','imgshuf','cov0','covshuf']}
    for sd in range(5):
        m=gfit(ZA,DA,bA.astype(float),X0.shape[1],Ca.shape[1],True,True,seed=sd)
        rs=np.random.RandomState(sd)
        def ex(Z,D):
            with torch.no_grad(): return roc_auc_score(bB,m(T(Z),T(D)).cpu().numpy())
        res['intact'].append(ex(ZB,DB))
        res['img0'].append(ex(np.zeros_like(ZB),DB))
        res['imgshuf'].append(ex(ZB[rs.permutation(len(ZB))],DB))
        res['cov0'].append(ex(ZB,np.zeros_like(DB)))
        res['covshuf'].append(ex(ZB,DB[rs.permutation(len(DB))]))
    print(f"  {bn} K={K}   (mean +- SD over 5 seeds)")
    for k,lab in [('intact','intact            '),('img0','imaging ZEROED    '),('imgshuf','imaging SHUFFLED  '),
                  ('cov0','covariates ZEROED '),('covshuf','covariates SHUFFLED')]:
        v=np.array(res[k]); print(f"    {lab} {v.mean():.3f} +- {v.std():.3f}")
    d=np.array(res['intact'])-np.array(res['img0'])
    print(f"    -> imaging contribution (intact - zeroed): {d.mean():+.3f} +- {d.std():.3f}\n",flush=True)
