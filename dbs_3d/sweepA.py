import sys, os
SPDIR = os.path.dirname(os.path.abspath(globals().get('__file__', '/workspace/dbs_3d/x')))
REPO  = os.path.dirname(SPDIR)
OUT   = os.path.join(REPO, 'docs', 'dbs_3d')          # docs/ is gitignored
DATA  = os.environ.get('DBS_CLINICAL_DIR', os.path.join(REPO, 'docs'))
CACHE = os.environ.get('DBS_CACHE_DIR', '/checkpoints')
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, REPO); sys.path.insert(0, SPDIR); os.chdir(REPO)
import numpy as np,pandas as pd,torch,torch.nn as nn,itertools,time,pickle,warnings; warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as _P
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
exec(open(os.path.join(SPDIR,'concat_attn.py')).read().split("rows=[]")[0])
dev=torch.device('cuda'); T=lambda a: torch.tensor(a,dtype=torch.float32,device=dev)
t0=time.time(); log=lambda m: print(f"[{time.time()-t0:5.0f}s] {m}",flush=True)
NS=3
def combat(Xa,Xb):
    g=np.vstack([Xa,Xb]); gr=g.mean(0); po=g.std(0)+1e-8; out=[]
    for Z in ((Xa-gr)/po,(Xb-gr)/po):
        gh,dh=Z.mean(0),Z.var(0)+1e-8; gb,t2=gh.mean(),gh.var()+1e-8
        gs=(t2*gh*len(Z)+dh*gb)/(len(Z)*t2+dh); ds=(dh+dh.mean())/2
        out.append((Z-gs)/np.sqrt(ds)*po+gr)
    return out
def qnorm(Xa,Xb):
    """quantile-map each CHH feature onto the MSW marginal"""
    out=np.empty_like(Xb)
    for j in range(Xa.shape[1]):
        r=np.argsort(np.argsort(Xb[:,j]))
        q=(r+0.5)/len(Xb)
        out[:,j]=np.quantile(Xa[:,j],q)
    return Xa,out
def harm(Xa,Xb,mode):
    if mode=='none':   return Xa,Xb
    if mode=='zscore': return (Xa-Xa.mean(0))/(Xa.std(0)+1e-8),(Xb-Xb.mean(0))/(Xb.std(0)+1e-8)
    if mode=='combat': return combat(Xa,Xb)
    if mode=='qnorm':  return qnorm(Xa,Xb)
def oversample(X,C,y,seed):
    rs=np.random.RandomState(seed); mi=np.where(y==0)[0]; ma=np.where(y==1)[0]
    if len(mi)==0 or len(mi)>=len(ma): return X,C,y
    add=rs.choice(mi,len(ma)-len(mi),replace=True)
    return np.vstack([X,X[add]]),np.vstack([C,C[add]]),np.concatenate([y,y[add]])
def gfit(Xtr,Ctr,ytr,Xte,Cte,K,C,seed,bal,wpos):
    torch.manual_seed(seed); m=ConcatAttn(K,C,typ=True,resid=True).to(dev)
    opt=torch.optim.AdamW(m.parameters(),3e-3,weight_decay=0.1)
    if bal=='ros': Xtr,Ctr,ytr=oversample(Xtr,Ctr,ytr,seed)
    lf=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(wpos,device=dev)) if bal=='cw' else nn.BCEWithLogitsLoss()
    xt,ct,yt=T(Xtr),T(Ctr),T(ytr)
    for e in range(1500):
        opt.zero_grad(); lf(m(xt,ct),yt).backward(); opt.step()
    m.eval()
    with torch.no_grad(): return m(T(Xte),T(Cte)).cpu().numpy()
BASES=['3DDCT','3DgraphLap','3D|FFT|','3DlogFFT','3DPCA','3DradPSD','3DroiPCA']
rows=[];PRED={}
for bn,K in itertools.product(BASES,[8,16]):
    try:
        if bn=='3DPCA':
            f=lambda V: V[:,BB[0],BB[1],BB[2]].reshape(len(V),-1); s1=StandardScaler().fit(f(VA))
            t=_P(K,random_state=0,whiten=True).fit(s1.transform(f(VA)))
            A0,B0=t.transform(s1.transform(f(VA))),t.transform(s1.transform(f(VB)))
        elif bn=='3DroiPCA':
            fa,fb=VA.reshape(len(VA),-1)[:,NF3],VB.reshape(len(VB),-1)[:,NF3]
            s1=StandardScaler().fit(fa); t=_P(K,random_state=0,whiten=True).fit(s1.transform(fa))
            A0,B0=t.transform(s1.transform(fa)),t.transform(s1.transform(fb))
        else: A0,B0=basis3(VA,bn,K),basis3(VB,bn,K)
        if A0 is None: continue
    except Exception: continue
    for hm,bal in itertools.product(['none','zscore','combat','qnorm'],['none','ros','cw']):
        A_,B_=harm(A0,B0,hm)
        ii,ee,pbs=[],[],[]
        for sd in range(NS):
            pr=np.full(len(bA),np.nan)
            for tr,va in StratifiedKFold(4,shuffle=True,random_state=0).split(A_,bA):
                s2=StandardScaler().fit(A_[tr]); s3=StandardScaler().fit(Ca[tr])
                wp=float((bA[tr]==1).sum()/max(1,(bA[tr]==0).sum()))
                pr[va]=gfit(s2.transform(A_[tr]),s3.transform(Ca[tr]),bA[tr].astype(float),
                            s2.transform(A_[va]),s3.transform(Ca[va]),A_.shape[1],Ca.shape[1],sd,bal,1.0/wp)
            s2=StandardScaler().fit(A_); s3=StandardScaler().fit(Ca)
            wp=float((bA==1).sum()/max(1,(bA==0).sum()))
            pb=gfit(s2.transform(A_),s3.transform(Ca),bA.astype(float),s2.transform(B_),s3.transform(Cb),
                    A_.shape[1],Ca.shape[1],sd,bal,1.0/wp)
            ii.append(roc_auc_score(bA,pr)); ee.append(roc_auc_score(bB,pb)); pbs.append(pb)
        ii,ee=np.array(ii),np.array(ee)
        rows.append(dict(basis=bn,K=K,harm=hm,balance=bal,int_mean=ii.mean(),int_sd=ii.std(),
                         ext_mean=ee.mean(),ext_sd=ee.std(),ens=roc_auc_score(bB,np.mean(pbs,0))))
        PRED[(bn,K,hm,bal)]=np.mean(pbs,0)
    log(f"{bn} K={K} done ({len(rows)} cells)")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'sweepA.csv'),index=False)
pickle.dump(dict(PRED=PRED,bB=bB),open(os.path.join(OUT, 'sweepA_preds.pkl'),'wb'))
df=pd.DataFrame(rows).sort_values('int_mean',ascending=False)
print("\n=== top 20 by seed-averaged INTERNAL ==="); print(df.head(20).round(3).to_string(index=False))
log(f"DONE {len(rows)} cells")
