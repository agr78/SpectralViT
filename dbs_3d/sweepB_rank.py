import sys, os
SPDIR = os.path.dirname(os.path.abspath(globals().get('__file__', '/workspace/dbs_3d/x')))
REPO  = os.path.dirname(SPDIR)
OUT   = os.path.join(REPO, 'docs', 'dbs_3d')
DATA  = os.environ.get('DBS_CLINICAL_DIR', os.path.join(REPO, 'docs'))
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

class RankAttn(nn.Module):
    """concat-token attention with an explicit RANK prior on the spectral tokens.

    rank_mode scales token i of the imaging block before embedding:
      none      1
      inv       1/i          (the omega_i = 1/i weighting from the spectral ViT)
      invsqrt   1/sqrt(i)
      loginv    1/log(1+i)
      learned   a free per-rank scalar, initialised at 1/i
    """
    def __init__(s,K,C,d=16,rank_mode='none'):
        super().__init__(); s.K,s.C=K,C
        i=torch.arange(1,K+1,dtype=torch.float32)
        w={'none':torch.ones(K),'inv':1.0/i,'invsqrt':1.0/torch.sqrt(i),'loginv':1.0/torch.log1p(i)}
        if rank_mode=='learned':
            s.w=nn.Parameter(1.0/i); s.fixed=None
        else:
            s.w=None; s.register_buffer('fixed',w[rank_mode])
        s.tok=nn.Linear(1,d); s.pos=nn.Parameter(torch.zeros(1,K+C,d))
        s.ty=nn.Parameter(torch.zeros(2,d))
        s.enc=nn.TransformerEncoder(nn.TransformerEncoderLayer(d,2,d*2,0.1,batch_first=True),1)
        s.hd=nn.Linear(d,1); s.lin=nn.Linear(K+C,1)
    def forward(s,x,c):
        w=s.w if s.w is not None else s.fixed
        xw=x*w                                   # rank-weighted spectral coefficients
        v=torch.cat([xw,c],1)
        z=s.tok(v.unsqueeze(-1))+s.pos
        z=z+torch.cat([s.ty[0].expand(s.K,-1),s.ty[1].expand(s.C,-1)],0).unsqueeze(0)
        return s.hd(s.enc(z).mean(1)).squeeze(-1)+s.lin(v).squeeze(-1)

def gfit(Xtr,Ctr,ytr,Xte,Cte,K,C,rank_mode,seed,ep=1500):
    torch.manual_seed(seed); m=RankAttn(K,C,rank_mode=rank_mode).to(dev)
    opt=torch.optim.AdamW(m.parameters(),3e-3,weight_decay=0.1); lf=nn.BCEWithLogitsLoss()
    xt,ct,yt=T(Xtr),T(Ctr),T(ytr)
    for e in range(ep):
        opt.zero_grad(); lf(m(xt,ct),yt).backward(); opt.step()
    m.eval()
    with torch.no_grad(): return m(T(Xte),T(Cte)).cpu().numpy()

NS=5
rows=[];PRED={}
for bn,K in itertools.product(['3DDCT','3DgraphLap','3D|FFT|'],[8,16]):
    X0,X1=basis3(VA,bn,K),basis3(VB,bn,K)
    for rm in ['none','inv','invsqrt','loginv','learned']:
        ii,ee,pbs=[],[],[]
        for sd in range(NS):
            pr=np.full(len(bA),np.nan)
            for tr,va in StratifiedKFold(4,shuffle=True,random_state=0).split(X0,bA):
                s2=StandardScaler().fit(X0[tr]); s3=StandardScaler().fit(Ca[tr])
                pr[va]=gfit(s2.transform(X0[tr]),s3.transform(Ca[tr]),bA[tr].astype(float),
                            s2.transform(X0[va]),s3.transform(Ca[va]),K,Ca.shape[1],rm,sd)
            s2=StandardScaler().fit(X0); s3=StandardScaler().fit(Ca)
            pb=gfit(s2.transform(X0),s3.transform(Ca),bA.astype(float),s2.transform(X1),s3.transform(Cb),
                    K,Ca.shape[1],rm,sd)
            ii.append(roc_auc_score(bA,pr)); ee.append(roc_auc_score(bB,pb)); pbs.append(pb)
        ii,ee=np.array(ii),np.array(ee)
        rows.append(dict(basis=bn,K=K,rank_mode=rm,int_mean=ii.mean(),int_sd=ii.std(),
                         ext_mean=ee.mean(),ext_sd=ee.std(),ens=roc_auc_score(bB,np.mean(pbs,0))))
        PRED[(bn,K,rm)]=np.mean(pbs,0)
        log(f"  {bn} K={K} rank={rm:8s} int {ii.mean():.3f}+-{ii.std():.3f}  ext {ee.mean():.3f}+-{ee.std():.3f}")
    pd.DataFrame(rows).to_csv(os.path.join(OUT,'sweepB_rank.csv'), index=False)
pickle.dump(dict(PRED=PRED,bB=bB),open(os.path.join(OUT,'sweepB_rank_preds.pkl'),'wb'))
df=pd.DataFrame(rows).sort_values('int_mean',ascending=False)
print("\n=== rank-embedding variants, ranked by seed-averaged internal ===")
print(df.round(3).to_string(index=False))
print("\n=== marginal effect of the rank prior ===")
print(df.groupby('rank_mode').agg(int_best=('int_mean','max'),int_mean=('int_mean','mean'),
                                   ext_best=('ext_mean','max'),ext_mean=('ext_mean','mean')).round(3).to_string())
log("DONE")
