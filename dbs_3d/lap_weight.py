import sys, os
SPDIR = os.path.dirname(os.path.abspath(globals().get('__file__', '/workspace/dbs_3d/x')))
REPO  = os.path.dirname(SPDIR); OUT = os.path.join(REPO,'docs','dbs_3d')
DATA  = os.environ.get('DBS_CLINICAL_DIR', os.path.join(REPO,'docs'))
CACHE = os.environ.get('DBS_CACHE_DIR', '/checkpoints')
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, REPO); sys.path.insert(0, SPDIR); os.chdir(REPO)
import numpy as np, pandas as pd, torch, torch.nn as nn, itertools, time, pickle, warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
exec(open(os.path.join(SPDIR,'concat_attn.py')).read().split("rows=[]")[0])
dev=torch.device('cuda'); T=lambda a: torch.tensor(a,dtype=torch.float32,device=dev)
t0=time.time(); log=lambda m: print(f"[{time.time()-t0:5.0f}s] {m}",flush=True)

def dct_eigs(K, shape):
    """Neumann-Laplacian eigenvalues of the DCT modes actually used, in the
       same order basis3 emits them: lambda = sum_d 2(1-cos(pi k_d / N_d))"""
    k=int(round(K**(1/3)))+1
    idx=[(a,b,c) for a in range(k) for b in range(k) for c in range(k)][:K]
    N=shape
    return np.array([sum(2*(1-np.cos(np.pi*i/N[d])) for d,i in enumerate(t)) for t in idx])

class LapAttn(nn.Module):
    def __init__(s,K,C,lam,d=16,mode='lap',tau=1.0):
        super().__init__(); s.K,s.C,s.mode=K,C,mode
        l=torch.tensor(lam,dtype=torch.float32)
        s.register_buffer('lam', l/ (l.max()+1e-12))
        if mode=='lap':       s.register_buffer('w', torch.exp(-tau*s.lam)); s.p=None
        elif mode=='lap_tau': s.p=nn.Parameter(torch.tensor(float(tau))); s.w=None
        else:                 s.register_buffer('w', torch.ones(K)); s.p=None
        s.tok=nn.Linear(1,d); s.pos=nn.Parameter(torch.zeros(1,K+C,d))
        s.enc=nn.TransformerEncoder(nn.TransformerEncoderLayer(d,2,d*2,0.1,batch_first=True),1)
        s.hd=nn.Linear(d,1); s.lin=nn.Linear(K+C,1)
    def forward(s,x,c):
        w = torch.exp(-torch.nn.functional.softplus(s.p)*s.lam) if s.p is not None else s.w
        v=torch.cat([x*w,c],1)
        z=s.tok(v.unsqueeze(-1))+s.pos
        return s.hd(s.enc(z).mean(1)).squeeze(-1)+s.lin(v).squeeze(-1)

def gfit(Xtr,Ctr,ytr,Xte,Cte,K,C,lam,mode,tau,seed,ep=1500):
    torch.manual_seed(seed); m=LapAttn(K,C,lam,mode=mode,tau=tau).to(dev)
    opt=torch.optim.AdamW(m.parameters(),3e-3,weight_decay=0.1); lf=nn.BCEWithLogitsLoss()
    xt,ct,yt=T(Xtr),T(Ctr),T(ytr)
    for e in range(ep):
        opt.zero_grad(); lf(m(xt,ct),yt).backward(); opt.step()
    m.eval()
    with torch.no_grad(): return m(T(Xte),T(Cte)).cpu().numpy()

NS=10
shape=(BB[0].stop-BB[0].start, BB[1].stop-BB[1].start, BB[2].stop-BB[2].start)
rows=[]
for K in [8,16]:
    X0,X1=basis3(VA,'3DDCT',K),basis3(VB,'3DDCT',K)
    lam=dct_eigs(K,shape)
    for mode,tau in [('none',0.0),('lap',0.5),('lap',1.0),('lap',2.0),('lap',4.0),('lap_tau',1.0)]:
        ii,ee=[],[]
        for sd in range(NS):
            pr=np.full(len(bA),np.nan)
            for tr,va in StratifiedKFold(4,shuffle=True,random_state=0).split(X0,bA):
                s2=StandardScaler().fit(X0[tr]); s3=StandardScaler().fit(Ca[tr])
                pr[va]=gfit(s2.transform(X0[tr]),s3.transform(Ca[tr]),bA[tr].astype(float),
                            s2.transform(X0[va]),s3.transform(Ca[va]),K,Ca.shape[1],lam,mode,tau,sd)
            s2=StandardScaler().fit(X0); s3=StandardScaler().fit(Ca)
            pb=gfit(s2.transform(X0),s3.transform(Ca),bA.astype(float),s2.transform(X1),s3.transform(Cb),
                    K,Ca.shape[1],lam,mode,tau,sd)
            ii.append(roc_auc_score(bA,pr)); ee.append(roc_auc_score(bB,pb))
        ii,ee=np.array(ii),np.array(ee)
        lab=f"{mode}" + (f"_tau{tau}" if mode=='lap' else "")
        rows.append(dict(K=K,weight=lab,int_mean=ii.mean(),int_sd=ii.std(),ext_mean=ee.mean(),ext_sd=ee.std()))
        log(f"  K={K} {lab:12s} int {ii.mean():.3f}+-{ii.std():.3f}  ext {ee.mean():.3f}+-{ee.std():.3f}")
df=pd.DataFrame(rows); df.to_csv(os.path.join(OUT,'lap_weight.csv'), index=False)
print("\n=== Laplacian rank weight  omega_i = exp(-tau * lambda_i)  on 3D DCT ===")
print(df.sort_values('int_mean',ascending=False).round(3).to_string(index=False))
log("DONE")
