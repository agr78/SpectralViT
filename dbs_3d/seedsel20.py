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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
src=open(os.path.join(SPDIR,'concat_attn.py')).read(); exec(src.split("rows=[]")[0])
dev=torch.device('cuda'); T=lambda a: torch.tensor(a,dtype=torch.float32,device=dev)
t0=time.time(); log=lambda m: print(f"[{time.time()-t0:5.0f}s] {m}",flush=True)
NS=20
def gfit(Xtr,Ctr,ytr,Xte,Cte,K,C,typ,seed):
    torch.manual_seed(seed); m=ConcatAttn(K,C,typ=typ,resid=True).to(dev)
    opt=torch.optim.AdamW(m.parameters(),3e-3,weight_decay=0.1); lf=nn.BCEWithLogitsLoss()
    xt,ct,yt=T(Xtr),T(Ctr),T(ytr)
    for e in range(1500):
        opt.zero_grad(); lf(m(xt,ct),yt).backward(); opt.step()
    m.eval()
    with torch.no_grad(): return m(T(Xte),T(Cte)).cpu().numpy()
rows=[];PRED={}
for bn,K in itertools.product(['3D|FFT|','3DlogFFT','3DDCT','3DPCA','3DgraphLap','3DradPSD'],[8,16]):
    try:
        if bn=='3DPCA':
            from sklearn.decomposition import PCA as _P
            f=lambda V: V[:,BB[0],BB[1],BB[2]].reshape(len(V),-1); s1=StandardScaler().fit(f(VA))
            t=_P(K,random_state=0,whiten=True).fit(s1.transform(f(VA)))
            X0,X1=t.transform(s1.transform(f(VA))),t.transform(s1.transform(f(VB)))
        else: X0,X1=basis3(VA,bn,K),basis3(VB,bn,K)
        if X0 is None: continue
    except Exception: continue
    for typ in [True]:
        ii,ee,pbs=[],[],[]
        for sd in range(NS):
            pr=np.full(len(bA),np.nan)
            for tr,va in StratifiedKFold(4,shuffle=True,random_state=0).split(X0,bA):
                s2=StandardScaler().fit(X0[tr]); s3=StandardScaler().fit(Ca[tr])
                pr[va]=gfit(s2.transform(X0[tr]),s3.transform(Ca[tr]),bA[tr].astype(float),
                            s2.transform(X0[va]),s3.transform(Ca[va]),X0.shape[1],Ca.shape[1],typ,sd)
            s2=StandardScaler().fit(X0); s3=StandardScaler().fit(Ca)
            pb=gfit(s2.transform(X0),s3.transform(Ca),bA.astype(float),s2.transform(X1),s3.transform(Cb),
                    X0.shape[1],Ca.shape[1],typ,sd)
            ii.append(roc_auc_score(bA,pr)); ee.append(roc_auc_score(bB,pb)); pbs.append(pb)
        ii,ee=np.array(ii),np.array(ee)
        rows.append(dict(basis=bn,K=K,int_mean=ii.mean(),int_se=ii.std()/np.sqrt(NS),int_sd=ii.std(),
                         ext_mean=ee.mean(),ext_sd=ee.std(),ens_auc=roc_auc_score(bB,np.mean(pbs,0))))
        PRED[(bn,K)]=np.mean(pbs,0)
    log(f"{bn} K={K}: int {rows[-1]['int_mean']:.3f}+-{rows[-1]['int_se']:.3f}  ext {rows[-1]['ext_mean']:.3f}")
df=pd.DataFrame(rows).sort_values('int_mean',ascending=False)
df.to_csv(os.path.join(OUT, 'seedsel20.csv'),index=False)
pickle.dump(dict(PRED=PRED,bB=bB),open(os.path.join(OUT, 'seedsel20_preds.pkl'),'wb'))
print("\n=== all bases, 20-seed averaged (typ_resid) ===")
print(df.round(3).to_string(index=False))
b=df.iloc[0]; pe=PRED[(b['basis'],int(b['K']))]; ae=roc_auc_score(bB,pe)
rng=np.random.RandomState(0); n=len(bB)
nl=np.array([roc_auc_score(bB[rng.permutation(n)],pe) for _ in range(20000)])
bs=[roc_auc_score(bB[i],pe[i]) for i in (rng.randint(0,n,(20000,n))) if len(np.unique(bB[i]))>1]
print(f"\nPRE-SPECIFIED on 20-seed internal: {b['basis']} K={int(b['K'])}")
print(f"  internal {b['int_mean']:.3f} +- {b['int_se']:.3f} (SE)")
print(f"  external {b['ext_mean']:.3f} +- {b['ext_sd']:.3f} (SD over seeds)")
print(f"  ensembled external {ae:.3f}  permutation p = {(nl>=ae).mean():.4f}  bootstrap CI [{np.percentile(bs,2.5):.3f},{np.percentile(bs,97.5):.3f}]")
d=df.iloc[0]['int_mean']-df.iloc[1]['int_mean']; se=np.hypot(df.iloc[0]['int_se'],df.iloc[1]['int_se'])
print(f"  margin over 2nd place ({df.iloc[1]['basis']} K={int(df.iloc[1]['K'])}): {d:+.3f} +- {se:.3f}  -> {'SEPARATED' if d>2*se else 'NOT separated'}")
log("DONE")
