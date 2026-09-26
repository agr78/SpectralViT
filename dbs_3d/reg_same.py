import sys, os
SPDIR = os.path.dirname(os.path.abspath(globals().get('__file__', '/workspace/dbs_3d/x')))
REPO  = os.path.dirname(SPDIR)
OUT   = os.path.join(REPO, 'docs', 'dbs_3d')          # docs/ is gitignored
DATA  = os.environ.get('DBS_CLINICAL_DIR', os.path.join(REPO, 'docs'))
CACHE = os.environ.get('DBS_CACHE_DIR', '/checkpoints')
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, REPO); sys.path.insert(0, SPDIR); os.chdir(REPO)
import numpy as np,pandas as pd,torch,torch.nn as nn,warnings,time,pickle,itertools; warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.stats import pearsonr,spearmanr
src=open(os.path.join(SPDIR,'concat_attn.py')).read(); exec(src.split("rows=[]")[0])
dev=torch.device('cuda'); T=lambda a: torch.tensor(a,dtype=torch.float32,device=dev)
t0=time.time(); log=lambda m: print(f"[{time.time()-t0:5.0f}s] {m}",flush=True)
def gfit(Xtr,Ctr,ytr,Xte,Cte,K,C,seed,typ=True):
    torch.manual_seed(seed); m=ConcatAttn(K,C,typ=typ,resid=True).to(dev)
    opt=torch.optim.AdamW(m.parameters(),3e-3,weight_decay=0.1); lf=nn.MSELoss()
    xt,ct,yt=T(Xtr),T(Ctr),T(ytr)
    for e in range(1500):
        opt.zero_grad(); lf(m(xt,ct),yt).backward(); opt.step()
    m.eval()
    with torch.no_grad(): return m(T(Xte),T(Cte)).cpu().numpy()
def run(X0,X1,seed):
    pr=np.full(len(yA),np.nan)
    for tr,va in KFold(5,shuffle=True,random_state=0).split(X0):
        s2=StandardScaler().fit(X0[tr]); s3=StandardScaler().fit(Ca[tr])
        pr[va]=gfit(s2.transform(X0[tr]),s3.transform(Ca[tr]),yA[tr],
                    s2.transform(X0[va]),s3.transform(Ca[va]),X0.shape[1],Ca.shape[1],seed)
    s2=StandardScaler().fit(X0); s3=StandardScaler().fit(Ca)
    pb=gfit(s2.transform(X0),s3.transform(Ca),yA,s2.transform(X1),s3.transform(Cb),X0.shape[1],Ca.shape[1],seed)
    return pearsonr(yA,pr)[0],pb
print("SAME ARCHITECTURE, CONTINUOUS OUTCOME (all 37 CHH patients contribute -- no 3-negative wall)\n")
rows=[]
_sel=pd.read_csv(os.path.join(OUT, 'seedsel.csv')).sort_values('int_mean',ascending=False).iloc[0]
print(f"basis chosen by the CLASSIFICATION seed-averaged selection: {_sel['basis']} K={int(_sel['K'])} {_sel['variant']}\n")
for bn,K in [(_sel['basis'],int(_sel['K']))]:
    if bn=='3DPCA':
        from sklearn.decomposition import PCA as _P
        f=lambda V: V[:,BB[0],BB[1],BB[2]].reshape(len(V),-1); s1=StandardScaler().fit(f(VA))
        t=_P(K,random_state=0,whiten=True).fit(s1.transform(f(VA)))
        X0,X1=t.transform(s1.transform(f(VA))),t.transform(s1.transform(f(VB)))
    elif bn=='3DroiPCA':
        from sklearn.decomposition import PCA as _P
        fa,fb=VA.reshape(len(VA),-1)[:,NF3],VB.reshape(len(VB),-1)[:,NF3]
        s1=StandardScaler().fit(fa); t=_P(K,random_state=0,whiten=True).fit(s1.transform(fa))
        X0,X1=t.transform(s1.transform(fa)),t.transform(s1.transform(fb))
    else: X0,X1=basis3(VA,bn,K),basis3(VB,bn,K)
    ii,ee,pbs=[],[],[]
    for sd in range(20):
        ri,pb=run(X0,X1,sd); ii.append(ri); ee.append(pearsonr(yB,pb)[0]); pbs.append(pb)
    pe=np.mean(pbs,0)
    r,p=pearsonr(yB,pe); rho=spearmanr(yB,pe).statistic
    r2=1-np.sum((yB-pe)**2)/np.sum((yB-yB.mean())**2)
    rows.append(dict(basis=bn,K=K,int_mean=np.mean(ii),int_sd=np.std(ii),
                     ext_mean=np.mean(ee),ext_sd=np.std(ee),ens_r=r,ens_p=p,ens_rho=rho,ens_r2=r2))
    print(f"  {bn:11s} K={K:2d}  internal r {np.mean(ii):+.3f}+-{np.std(ii):.3f}   external r {np.mean(ee):+.3f}+-{np.std(ee):.3f}   ensembled r {r:+.3f} (p={p:.3f}, rho={rho:+.3f}, R2={r2:+.3f})",flush=True)
df=pd.DataFrame(rows); df.to_csv(os.path.join(OUT, 'reg_same.csv'),index=False)
b=df.loc[df.int_mean.idxmax()]
print(f"\nPRE-SPECIFIED (best seed-averaged internal): {b['basis']} K={int(b['K'])}")
print(f"  internal r {b['int_mean']:+.3f}   external r {b['ext_mean']:+.3f}   ensembled r {b['ens_r']:+.3f}  p={b['ens_p']:.4f}")
print(f"  reference: clinical-only external r = +0.236 ; imaging-only linear probe = +0.111")
pickle.dump(dict(df=df),open(os.path.join(OUT, 'reg_same.pkl'),'wb'))
log("DONE")
