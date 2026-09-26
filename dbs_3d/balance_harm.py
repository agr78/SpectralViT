import sys, os
SPDIR = os.path.dirname(os.path.abspath(globals().get('__file__', '/workspace/dbs_3d/x')))
REPO  = os.path.dirname(SPDIR)
OUT   = os.path.join(REPO, 'docs', 'dbs_3d')          # docs/ is gitignored
DATA  = os.environ.get('DBS_CLINICAL_DIR', os.path.join(REPO, 'docs'))
CACHE = os.environ.get('DBS_CACHE_DIR', '/checkpoints')
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, REPO); sys.path.insert(0, SPDIR); os.chdir(REPO)
import numpy as np,pandas as pd,itertools,time,pickle,warnings; warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
src=open(os.path.join(SPDIR,'vol3d_attn.py')).read()
exec(src.split("\nREG=")[0])
t0=time.time(); log=lambda m: print(f"[{time.time()-t0:5.0f}s] {m}",flush=True)

def ros(X,y,seed=0):
    """simple resampling with replacement: draw the minority up to majority size"""
    rs=np.random.RandomState(seed); mi=np.where(y==0)[0]; ma=np.where(y==1)[0]
    if len(mi)>=len(ma) or len(mi)==0: return X,y
    add=rs.choice(mi,len(ma)-len(mi),replace=True)
    return np.vstack([X,X[add]]),np.concatenate([y,y[add]])
def boot_fit(make,X,y,Xte,B=50,seed=0):
    """balanced bootstrap ensemble: each member trained on an equal-size
       with-replacement draw from each class; predictions averaged"""
    rs=np.random.RandomState(seed); mi=np.where(y==0)[0]; ma=np.where(y==1)[0]
    n=max(len(ma),2); out=[]
    for b in range(B):
        ix=np.concatenate([rs.choice(mi,n,replace=True),rs.choice(ma,n,replace=True)])
        yb=y[ix]
        if len(np.unique(yb))<2: continue
        m=make().fit(X[ix],yb)
        out.append(m.decision_function(Xte))
    return np.mean(out,0)
def combat(Xa,Xb):
    g=np.vstack([Xa,Xb]); grand=g.mean(0); pooled=g.std(0)+1e-8
    out=[]
    for Z in ((Xa-grand)/pooled,(Xb-grand)/pooled):
        gh,dh=Z.mean(0),Z.var(0)+1e-8
        gbar,tau2=gh.mean(),gh.var()+1e-8
        gs=(tau2*gh*len(Z)+dh*gbar)/(len(Z)*tau2+dh); ds=(dh+dh.mean())/2
        out.append((Z-gs)/np.sqrt(ds)*pooled+grand)
    return out
def harmonize(Xa,Xb,mode):
    if mode=='none':   return Xa,Xb
    if mode=='zscore': return ((Xa-Xa.mean(0))/(Xa.std(0)+1e-8),(Xb-Xb.mean(0))/(Xb.std(0)+1e-8))
    return combat(Xa,Xb)
MODELS={'LogReg':lambda w: LogisticRegression(C=1.0,max_iter=2000,class_weight=w),
        'SVC':   lambda w: SVC(kernel='rbf',class_weight=w)}
rows=[];PRED={}
for bn,K in itertools.product(['3D|FFT|','3DlogFFT','3DDCT','3DPCA','3DgraphLap'],[8,16]):
    if bn=='3DPCA':
        from sklearn.decomposition import PCA as _P
        f=lambda V: V[:,BB[0],BB[1],BB[2]].reshape(len(V),-1); s1=StandardScaler().fit(f(VA))
        t=_P(K,random_state=0,whiten=True).fit(s1.transform(f(VA)))
        X0,X1=t.transform(s1.transform(f(VA))),t.transform(s1.transform(f(VB)))
    else: X0,X1=basis3(VA,bn,K),basis3(VB,bn,K)
    for harm in ['none','zscore','combat']:
        XA_,XB_=harmonize(X0,X1,harm)
        for bal,mo in itertools.product(['none','cw','ros','boot'],MODELS):
            try:
                w='balanced' if bal=='cw' else None
                pr=np.full(len(bA),np.nan)
                for tr,va in StratifiedKFold(4,shuffle=True,random_state=0).split(XA_,bA):
                    s2=StandardScaler().fit(XA_[tr]); Xt,yt=s2.transform(XA_[tr]),bA[tr]
                    if bal=='boot':
                        pr[va]=boot_fit(lambda: MODELS[mo](w),Xt,yt,s2.transform(XA_[va]))
                    else:
                        if bal=='ros': Xt,yt=ros(Xt,yt)
                        pr[va]=MODELS[mo](w).fit(Xt,yt).decision_function(s2.transform(XA_[va]))
                s3=StandardScaler().fit(XA_); Xt,yt=s3.transform(XA_),bA
                if bal=='boot':
                    pb=boot_fit(lambda: MODELS[mo](w),Xt,yt,s3.transform(XB_))
                else:
                    if bal=='ros': Xt,yt=ros(Xt,yt)
                    pb=MODELS[mo](w).fit(Xt,yt).decision_function(s3.transform(XB_))
                rows.append(dict(basis=bn,K=K,harm=harm,balance=bal,model=mo,
                                 m_int=roc_auc_score(bA,pr),m_ext=roc_auc_score(bB,pb)))
                PRED[(bn,K,harm,bal,mo)]=pb.copy()
            except Exception: continue
    log(f"{bn} K={K}: {len(rows)} cells")
df=pd.DataFrame(rows); df.to_csv(os.path.join(OUT, 'balance_harm.csv'),index=False)
pickle.dump(dict(PRED=PRED,bB=bB),open(os.path.join(OUT, 'balance_harm_preds.pkl'),'wb'))
log(f"DONE {len(rows)} cells")
