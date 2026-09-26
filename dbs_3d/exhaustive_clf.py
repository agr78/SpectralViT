import sys, os
SPDIR = os.path.dirname(os.path.abspath(globals().get('__file__', '/workspace/dbs_3d/x')))
REPO  = os.path.dirname(SPDIR)
OUT   = os.path.join(REPO, 'docs', 'dbs_3d')          # docs/ is gitignored
DATA  = os.environ.get('DBS_CLINICAL_DIR', os.path.join(REPO, 'docs'))
CACHE = os.environ.get('DBS_CACHE_DIR', '/checkpoints')
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, REPO); sys.path.insert(0, SPDIR); os.chdir(REPO)
import numpy as np, pandas as pd, warnings, itertools, time; warnings.filterwarnings('ignore')
from scipy import ndimage
from scipy.fft import fftn, dctn
from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from reader import prepare_qsm_dataset
from util import mask_crop
mcf=lambda d,m,p: mask_crop(d,m,p)
t0=time.time(); log=lambda m: print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
ds_tr=prepare_qsm_dataset('MSW','/x','/y',os.path.join(DATA,'dbs_03292024.csv'),os.path.join(CACHE,'msw_cache_6d_cv.pt'),load_cache=True,mask_crop_fn=mcf,cv_pad=False)
ds_te=prepare_qsm_dataset('CHH','/x','/y',os.path.join(DATA,'chh_subjects_table1_20240729.csv'),os.path.join(CACHE,'chh_cache_6d_cv.pt'),load_cache=True,mask_crop_fn=mcf,cv_pad=False)
d1=pd.read_csv(os.path.join(DATA,'dbs_03292024.csv'),header=1); d1.columns=[str(c).strip().replace('\n',' ') for c in d1.columns]
o=pd.to_numeric(d1['OFF (pre-dbs updrs)'],errors='coerce'); q=pd.to_numeric(d1['OFF meds ON stim 6mo'],errors='coerce'); s_=pd.to_numeric(d1['CORNELL ID'],errors='coerce')
impA={int(a):(b-c)/b for a,b,c in zip(s_,o,q) if pd.notna(a) and pd.notna(b) and pd.notna(c) and b>0}
d2=pd.read_csv(os.path.join(DATA,'chh_subjects_table1_20240729.csv'),header=None).iloc[2:]
impB={}
for _,r in d2.iterrows():
    try:
        sid=int(float(r[0])); a=pd.to_numeric(r[9],errors='coerce'); b=pd.to_numeric(r[11],errors='coerce')
        if pd.notna(a) and pd.notna(b) and a>0: impB[sid]=(a-b)/a
    except Exception: pass

def raw_crops(ds, imp, H, grad):
    subs=[int(s) for s in sorted(ds.volumes) if int(s) in imp]; F,P=[],[]
    for s in subs:
        v=np.asarray(ds.volumes[s],np.float32); m=np.asarray(ds.seg_masks[s])>0
        lb,n=ndimage.label(m); sz=ndimage.sum(m,lb,range(1,n+1))
        if n<2: continue
        big=np.argsort(sz)[::-1][:2]+1
        info=sorted([(float(v[lb==b].mean()), ndimage.center_of_mass(m,lb,[b])[0]) for b in big],key=lambda t:t[0])
        c=info[1][1]; cx,cy=int(round(c[0])),int(round(c[1]))
        vv=v
        if grad:
            gx,gy=np.gradient(v,axis=0),np.gradient(v,axis=1); vv=np.sqrt(gx**2+gy**2).astype(np.float32)
        for i in range(v.shape[2]):
            if m[:,:,i].sum()==0: continue
            pa=vv[max(0,cx-H):max(0,cx-H)+2*H, max(0,cy-H):max(0,cy-H)+2*H, i]
            if pa.shape==(2*H,2*H): F.append(pa.ravel()); P.append(s)
    return np.array(F,np.float32), np.array(P)

def haar(sq,lv=2):
    out=[]; a=sq.copy()
    for _ in range(lv):
        e,od=a[:,0::2,:],a[:,1::2,:]; L,Hh=(e+od)/2,(e-od)/2
        e2,o2=L[:,:,0::2],L[:,:,1::2]; LL,LH=(e2+o2)/2,(e2-o2)/2
        e3,o3=Hh[:,:,0::2],Hh[:,:,1::2]; HL,HH=(e3+o3)/2,(e3-o3)/2
        out+=[LH.reshape(len(sq),-1),HL.reshape(len(sq),-1),HH.reshape(len(sq),-1)]; a=LL
    return np.hstack([a.reshape(len(sq),-1)]+out)
def radial(sq,K):
    n=sq.shape[1]; yy,xx=np.mgrid[:n,:n]; rr=np.hypot(xx-n/2,yy-n/2).astype(int)
    Pw=np.abs(np.fft.fftshift(fftn(sq,axes=(1,2)),axes=(1,2)))**2
    return np.stack([[Pw[i][rr==k].mean() for k in range(K)] for i in range(len(sq))])

def basis(F,H,name,K):
    sq=F.reshape(-1,2*H,2*H); k=int(np.ceil(np.sqrt(K)))
    if name=='|FFT|':    return np.abs(fftn(sq,axes=(1,2)))[:,:k,:k].reshape(len(F),-1)[:,:K]
    if name=='logFFT':   return np.log1p(np.abs(fftn(sq,axes=(1,2)))[:,:k,:k]).reshape(len(F),-1)[:,:K]
    if name=='DCT':      return dctn(sq,axes=(1,2),norm='ortho')[:,:k,:k].reshape(len(F),-1)[:,:K]
    if name=='Haar':     return haar(sq)[:,:K]
    if name=='radialPSD':return radial(sq,K)
    if name=='raw':      return F
    return None

MODELS={
 'LogReg':     lambda: LogisticRegression(C=1.0,max_iter=2000),
 'SVC-rbf':    lambda: SVC(kernel='rbf',C=1.0,probability=False),
 'RF':         lambda: RandomForestClassifier(n_estimators=300,random_state=0,n_jobs=4),
 'GBM':        lambda: GradientBoostingClassifier(random_state=0),
 'MLP':        lambda: MLPClassifier(hidden_layer_sizes=(64,32),max_iter=800,random_state=0),
}
def score(m,X):
    return m.decision_function(X) if hasattr(m,'decision_function') else m.predict_proba(X)[:,1]
BASES=['|FFT|','logFFT','DCT','Haar','radialPSD','PCA','kPCA-rbf','raw']
Ks=[8,16,32]; AGGS=['mean','median']; GRADS=[False,True]; H=10

rows=[]; PRED={}; YSTORE={}
for grad in GRADS:
    FA,PA=raw_crops(ds_tr,impA,H,grad); FB,PB=raw_crops(ds_te,impB,H,grad)
    subsA=sorted(set(PA)); subsB=sorted(set(PB))
    yA=np.array([impA[s] for s in subsA]); yB=np.array([impB[s] for s in subsB])
    yxA=np.array([impA[p] for p in PA])
    THR=0.30
    bA=(yA>=THR).astype(int); bB=(yB>=THR).astype(int); bxA=(yxA>=THR).astype(int)
    print(f'  responders: MSW {bA.sum()}/{len(bA)}  CHH {bB.sum()}/{len(bB)}',flush=True)
    YSTORE[grad]=(bA.copy(),bB.copy())
    for bname,K in itertools.product(BASES,Ks):
        if bname=='raw' and K!=Ks[0]: continue
        try:
            if bname=='PCA':
                sc=StandardScaler().fit(FA); t=PCA(K,random_state=0,whiten=True).fit(sc.transform(FA))
                XA,XB=t.transform(sc.transform(FA)),t.transform(sc.transform(FB))
            elif bname=='kPCA-rbf':
                sc=StandardScaler().fit(FA); idx=np.random.RandomState(0).choice(len(FA),min(1200,len(FA)),replace=False)
                t=KernelPCA(n_components=K,kernel='rbf',random_state=0).fit(sc.transform(FA)[idx])
                XA,XB=t.transform(sc.transform(FA)),t.transform(sc.transform(FB))
            else:
                XA,XB=basis(FA,H,bname,K),basis(FB,H,bname,K)
        except Exception as e:
            continue
        for mname,agg in itertools.product(MODELS,AGGS):
            try:
                # internal OOF
                pr=np.full(len(subsA),np.nan); IDX={s:i for i,s in enumerate(subsA)}
                for tr,va in KFold(5,shuffle=True,random_state=0).split(subsA):
                    S=np.array(subsA); mtr,mva=np.isin(PA,S[tr]),np.isin(PA,S[va])
                    sc2=StandardScaler().fit(XA[mtr]); mdl=MODELS[mname]().fit(sc2.transform(XA[mtr]),bxA[mtr])
                    pv=score(mdl,sc2.transform(XA[mva]))
                    for k in np.unique(PA[mva]):
                        v=pv[PA[mva]==k]; pr[IDX[k]]=np.median(v) if agg=='median' else v.mean()
                ok=~np.isnan(pr)
                r_in=roc_auc_score(bA[ok],pr[ok])
                # external
                sc3=StandardScaler().fit(XA); mdl=MODELS[mname]().fit(sc3.transform(XA),bxA)
                pv=score(mdl,sc3.transform(XB))
                pb=np.array([ (np.median(pv[PB==k]) if agg=='median' else pv[PB==k].mean()) for k in subsB])
                r_ex=roc_auc_score(bB,pb); s_ex=spearmanr(bB,pb).statistic
                r2_ex=np.nan
                PRED[(grad,bname,K,mname,agg)]=(pr.copy(),pb.copy())
                rows.append(dict(grad=grad,basis=bname,K=K,model=mname,agg=agg,
                                 r_int=r_in,r_ext=r_ex,sp_ext=s_ex,r2_ext=r2_ex))
            except Exception: continue
        log(f"grad={grad} {bname} K={K}: {len(rows)} combos so far")
df=pd.DataFrame(rows); df.to_csv(os.path.join(OUT, 'exhaustive_clf.csv'), index=False)
import pickle
pickle.dump(dict(PRED=PRED,Y=YSTORE),open(os.path.join(OUT, 'preds_clf.pkl'),'wb'))
log(f"DONE: {len(df)} combinations -> {os.path.join(OUT,"exhaustive.csv")}")
