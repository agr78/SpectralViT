import sys, os
SPDIR = os.path.dirname(os.path.abspath(globals().get('__file__', '/workspace/dbs_3d/x')))
REPO  = os.path.dirname(SPDIR)
OUT   = os.path.join(REPO, 'docs', 'dbs_3d')          # docs/ is gitignored
DATA  = os.environ.get('DBS_CLINICAL_DIR', os.path.join(REPO, 'docs'))
CACHE = os.environ.get('DBS_CACHE_DIR', '/checkpoints')
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, REPO); sys.path.insert(0, SPDIR); os.chdir(REPO)
import numpy as np,pandas as pd,itertools,time,pickle,warnings; warnings.filterwarnings('ignore')
from scipy.fft import fftn,dctn
from scipy.stats import pearsonr,spearmanr,rankdata,norm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV,LogisticRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR,SVC
from sklearn.neural_network import MLPRegressor,MLPClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
exec(open(os.path.join(SPDIR,'_shared_data.py')).read())
t0=time.time(); log=lambda m: print(f"[{time.time()-t0:6.0f}s] {m}",flush=True)
THR=0.30; bA=(yA>=THR).astype(int); bB=(yB>=THR).astype(int); bxA=(yxA>=THR).astype(int)

def basis(F,name,K):
    sq=F.reshape(-1,2*H,2*H); k=int(np.ceil(np.sqrt(K)))
    if name=='|FFT|':  return np.abs(fftn(sq,axes=(1,2)))[:,:k,:k].reshape(len(F),-1)[:,:K]
    if name=='logFFT': return np.log1p(np.abs(fftn(sq,axes=(1,2)))[:,:k,:k]).reshape(len(F),-1)[:,:K]
    if name=='DCT':    return dctn(sq,axes=(1,2),norm='ortho')[:,:k,:k].reshape(len(F),-1)[:,:K]
    return None

def combat(Xa,Xb):
    """Empirical-Bayes location/scale harmonization, 2 batches, NO outcome covariates."""
    g=np.vstack([Xa,Xb]); grand=g.mean(0); pooled=g.std(0)+1e-8
    Za,Zb=(Xa-grand)/pooled,(Xb-grand)/pooled
    out=[]
    for Z in (Za,Zb):
        gh,dh=Z.mean(0),Z.var(0)+1e-8                      # per-feature batch effects
        gbar,tau2=gh.mean(),gh.var()+1e-8                  # EB priors across features
        gstar=(tau2*gh*len(Z)+ (dh)*gbar)/(len(Z)*tau2+dh) # shrunk location
        dbar=dh.mean(); dstar=(dh+dbar)/2                  # shrunk scale
        out.append((Z-gstar)/np.sqrt(dstar)*pooled+grand)
    return out

def harmonize(Xa,Xb,mode):
    if mode=='none':   return Xa,Xb
    if mode=='zscore': return ((Xa-Xa.mean(0))/(Xa.std(0)+1e-8), (Xb-Xb.mean(0))/(Xb.std(0)+1e-8))
    if mode=='combat': return combat(Xa,Xb)
    if mode=='rank':
        f=lambda X: norm.ppf((np.apply_along_axis(rankdata,0,X))/(len(X)+1))
        return f(Xa),f(Xb)

REG={'Ridge':lambda:RidgeCV(alphas=np.logspace(-2,4,20)),'KRR':lambda:KernelRidge(kernel='rbf',alpha=1.0),
     'SVR':lambda:SVR(kernel='rbf',C=1.0),'MLP':lambda:MLPRegressor(hidden_layer_sizes=(64,32),max_iter=600,random_state=0)}
CLF={'LogReg':lambda:LogisticRegression(C=1.0,max_iter=2000),'SVC':lambda:SVC(kernel='rbf',C=1.0),
     'MLP':lambda:MLPClassifier(hidden_layer_sizes=(64,32),max_iter=800,random_state=0)}
sc_=lambda m,X: m.decision_function(X) if hasattr(m,'decision_function') else m.predict_proba(X)[:,1]
rows=[];PRED={}; S=np.array(subsA); IDX={s:i for i,s in enumerate(subsA)}
for bname,K,harm in itertools.product(['|FFT|','logFFT','DCT','PCA'],[8,32],['none','zscore','combat','rank']):
    if bname=='PCA':
        s1=StandardScaler().fit(FA); t=PCA(K,random_state=0,whiten=True).fit(s1.transform(FA))
        XA,XB=t.transform(s1.transform(FA)),t.transform(s1.transform(FB))
    else: XA,XB=basis(FA,bname,K),basis(FB,bname,K)
    XA,XB=harmonize(XA,XB,harm)
    for task in ['reg','clf']:
        zoo,yv,ytr_all = (REG,yA,yxA) if task=='reg' else (CLF,bA,bxA)
        for mname,agg in itertools.product(zoo,['mean','median']):
            try:
                pr=np.full(len(subsA),np.nan)
                for tr,va in KFold(5,shuffle=True,random_state=0).split(subsA):
                    mtr,mva=np.isin(PA,S[tr]),np.isin(PA,S[va])
                    s2=StandardScaler().fit(XA[mtr]); m=zoo[mname]().fit(s2.transform(XA[mtr]),ytr_all[mtr])
                    pv=m.predict(s2.transform(XA[mva])) if task=='reg' else sc_(m,s2.transform(XA[mva]))
                    for kk in np.unique(PA[mva]):
                        v=pv[PA[mva]==kk]; pr[IDX[kk]]=np.median(v) if agg=='median' else v.mean()
                ok=~np.isnan(pr)
                m_in = pearsonr(yA[ok],pr[ok])[0] if task=='reg' else roc_auc_score(bA[ok],pr[ok])
                s3=StandardScaler().fit(XA); m=zoo[mname]().fit(s3.transform(XA),ytr_all)
                pv=m.predict(s3.transform(XB)) if task=='reg' else sc_(m,s3.transform(XB))
                pb=np.array([(np.median(pv[PB==kk]) if agg=='median' else pv[PB==kk].mean()) for kk in subsB])
                m_ex = pearsonr(yB,pb)[0] if task=='reg' else roc_auc_score(bB,pb)
                rows.append(dict(task=task,basis=bname,K=K,harm=harm,model=mname,agg=agg,m_int=m_in,m_ext=m_ex))
                PRED[(task,bname,K,harm,mname,agg)]=pb.copy()
            except Exception: continue
    log(f"{bname} K={K} harm={harm}: {len(rows)} cells")
pd.DataFrame(rows).to_csv(os.path.join(OUT, 'harmonize.csv'),index=False)
pickle.dump(dict(PRED=PRED,yB=yB,bB=bB),open(os.path.join(OUT, 'harm_preds.pkl'),'wb'))
log(f"DONE {len(rows)} cells")
