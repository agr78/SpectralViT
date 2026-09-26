import sys, os
SPDIR = os.path.dirname(os.path.abspath(globals().get('__file__', '/workspace/dbs_3d/x')))
REPO  = os.path.dirname(SPDIR)
OUT   = os.path.join(REPO, 'docs', 'dbs_3d')          # docs/ is gitignored
DATA  = os.environ.get('DBS_CLINICAL_DIR', os.path.join(REPO, 'docs'))
CACHE = os.environ.get('DBS_CACHE_DIR', '/checkpoints')
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, REPO); sys.path.insert(0, SPDIR); os.chdir(REPO)
import numpy as np,pandas as pd,itertools,time,pickle,warnings; warnings.filterwarnings('ignore')
from scipy import ndimage
from scipy.fft import fftn,dctn
from scipy.stats import pearsonr,rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV,LogisticRegression
from sklearn.svm import SVR,SVC
from sklearn.neural_network import MLPRegressor,MLPClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
exec(open(os.path.join(SPDIR,'_shared_data.py')).read())
t0=time.time(); log=lambda m: print(f"[{time.time()-t0:5.0f}s] {m}",flush=True)
THR=0.30; bA=(yA>=THR).astype(int); bB=(yB>=THR).astype(int); bxA=(yxA>=THR).astype(int)
SQA,SQB=FA.reshape(-1,20,20),FB.reshape(-1,20,20)
def bbox(dil):
    P=ndimage.binary_dilation(POP,iterations=dil) if dil else POP
    r,c=np.where(P); return P,(r.min(),r.max()+1,c.min(),c.max()+1)
def haar(sq,lv=1):
    out=[];a=sq.copy()
    for _ in range(lv):
        e,od=a[:,0::2,:],a[:,1::2,:]; L,Hh=(e+od)/2,(e-od)/2
        e2,o2=L[:,:,0::2],L[:,:,1::2]; LL,LH=(e2+o2)/2,(e2-o2)/2
        e3,o3=Hh[:,:,0::2],Hh[:,:,1::2]; HL,HH=(e3+o3)/2,(e3-o3)/2
        out+=[LH.reshape(len(sq),-1),HL.reshape(len(sq),-1),HH.reshape(len(sq),-1)]; a=LL
    return np.hstack([a.reshape(len(sq),-1)]+out)
def lap_eigs(P):
    N=np.array(np.where(P)).T; I={(a,b):i for i,(a,b) in enumerate(map(tuple,N))}
    W=np.zeros((len(N),)*2)
    for (a,b),i in I.items():
        for da,db in ((0,1),(1,0)):
            j=I.get((a+da,b+db))
            if j is not None: W[i,j]=W[j,i]=1
    ev,U=np.linalg.eigh(np.diag(W.sum(1))-W)
    return np.array([a*20+b for a,b in N]),U[:,1:]
REG={'Ridge':lambda:RidgeCV(alphas=np.logspace(-2,4,20)),'SVR':lambda:SVR(kernel='rbf'),
     'MLP':lambda:MLPRegressor(hidden_layer_sizes=(64,32),max_iter=600,random_state=0)}
CLF={'LogReg':lambda:LogisticRegression(C=1.0,max_iter=2000),'SVC':lambda:SVC(kernel='rbf'),
     'MLP':lambda:MLPClassifier(hidden_layer_sizes=(64,32),max_iter=800,random_state=0)}
sc_=lambda m,X: m.decision_function(X) if hasattr(m,'decision_function') else m.predict_proba(X)[:,1]
S=np.array(subsA); IDX={s:i for i,s in enumerate(subsA)}
rows=[];PRED={}
for dil in [0,2]:
    P,(r0,r1,c0,c1)=bbox(dil); NF,U=lap_eigs(P)
    ca,cb=SQA[:,r0:r1,c0:c1],SQB[:,r0:r1,c0:c1]
    npx=(r1-r0)*(c1-c0)
    for bn,K in itertools.product(['|FFT|','logFFT','DCT','Haar','PCA','graphLap','roiPCA','roiRaw'],[8,16]):
        k=int(np.ceil(np.sqrt(K)))
        try:
            if   bn=='|FFT|':   XA_,XB_=[np.abs(fftn(x,axes=(1,2)))[:,:k,:k].reshape(len(x),-1)[:,:K] for x in (ca,cb)]
            elif bn=='logFFT':  XA_,XB_=[np.log1p(np.abs(fftn(x,axes=(1,2)))[:,:k,:k]).reshape(len(x),-1)[:,:K] for x in (ca,cb)]
            elif bn=='DCT':     XA_,XB_=[dctn(x,axes=(1,2),norm='ortho')[:,:k,:k].reshape(len(x),-1)[:,:K] for x in (ca,cb)]
            elif bn=='Haar':    XA_,XB_=[haar(x)[:,:K] for x in (ca,cb)]
            elif bn=='PCA':
                f=lambda x: x.reshape(len(x),-1); s1=StandardScaler().fit(f(ca))
                t=PCA(min(K,npx-1),random_state=0,whiten=True).fit(s1.transform(f(ca)))
                XA_,XB_=t.transform(s1.transform(f(ca))),t.transform(s1.transform(f(cb)))
            elif bn=='graphLap':XA_,XB_=FA[:,NF]@U[:,:K],FB[:,NF]@U[:,:K]
            elif bn=='roiPCA':
                s1=StandardScaler().fit(FA[:,NF])
                t=PCA(min(K,len(NF)-1),random_state=0,whiten=True).fit(s1.transform(FA[:,NF]))
                XA_,XB_=t.transform(s1.transform(FA[:,NF])),t.transform(s1.transform(FB[:,NF]))
            elif bn=='roiRaw':
                if K!=8: continue
                XA_,XB_=FA[:,NF],FB[:,NF]
        except Exception: continue
        for task in ['reg','clf']:
            zoo=REG if task=='reg' else CLF; ys=yA if task=='reg' else bA
            ytr=yxA if task=='reg' else bxA; ye=yB if task=='reg' else bB
            for mo in zoo:
                try:
                    pr=np.full(len(subsA),np.nan)
                    for tr,va in KFold(5,shuffle=True,random_state=0).split(subsA):
                        mtr,mva=np.isin(PA,S[tr]),np.isin(PA,S[va])
                        s2=StandardScaler().fit(XA_[mtr]); m=zoo[mo]().fit(s2.transform(XA_[mtr]),ytr[mtr])
                        pv=m.predict(s2.transform(XA_[mva])) if task=='reg' else sc_(m,s2.transform(XA_[mva]))
                        for kk in np.unique(PA[mva]): pr[IDX[kk]]=pv[PA[mva]==kk].mean()
                    ok=~np.isnan(pr)
                    m_in=pearsonr(ys[ok],pr[ok])[0] if task=='reg' else roc_auc_score(ys[ok],pr[ok])
                    s3=StandardScaler().fit(XA_); m=zoo[mo]().fit(s3.transform(XA_),ytr)
                    pv=m.predict(s3.transform(XB_)) if task=='reg' else sc_(m,s3.transform(XB_))
                    pb=np.array([pv[PB==kk].mean() for kk in subsB])
                    m_ex=pearsonr(ye,pb)[0] if task=='reg' else roc_auc_score(ye,pb)
                    rows.append(dict(dil=dil,bbox=f"{r1-r0}x{c1-c0}",basis=bn,K=K,task=task,model=mo,m_int=m_in,m_ext=m_ex))
                    PRED[(task,dil,bn,K,mo)]=pb.copy()
                except Exception: continue
    log(f"dilate={dil} bbox {r1-r0}x{c1-c0}: {len(rows)} cells")
pd.DataFrame(rows).to_csv(os.path.join(OUT, 'masked.csv'),index=False)
pickle.dump(dict(PRED=PRED,yB=yB,bB=bB),open(os.path.join(OUT, 'masked_preds.pkl'),'wb'))
log(f"DONE {len(rows)} cells")
