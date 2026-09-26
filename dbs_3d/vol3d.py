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
from reader import prepare_qsm_dataset
from util import mask_crop
mcf=lambda d,m,p: mask_crop(d,m,p)
t0=time.time(); log=lambda m: print(f"[{time.time()-t0:5.0f}s] {m}",flush=True)
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
H,DZ=10,8                                   # 20 x 20 x 16 volume
def vols(ds,imp):
    V,M,S=[],[],[]
    for s in sorted(ds.volumes):
        s=int(s)
        if s not in imp: continue
        v=np.asarray(ds.volumes[s],np.float32); m=np.asarray(ds.seg_masks[s])>0
        lb,n=ndimage.label(m); sz=ndimage.sum(m,lb,range(1,n+1))
        if n<2: continue
        big=np.argsort(sz)[::-1][:2]+1
        info=sorted([(float(v[lb==b].mean()),b) for b in big],key=lambda t:t[0])
        b=info[1][1]                        # higher-susceptibility side
        cx,cy,cz=[int(round(c)) for c in ndimage.center_of_mass(m,lb,[b])[0]]
        _x=min(max(0,cx-H), v.shape[0]-2*H); _y=min(max(0,cy-H), v.shape[1]-2*H); _z=min(max(0,cz-DZ), v.shape[2]-2*DZ)
        sl=(slice(_x,_x+2*H),slice(_y,_y+2*H),slice(_z,_z+2*DZ))
        pv,pm=v[sl],m[sl]
        if pv.shape!=(2*H,2*H,2*DZ): continue
        V.append(pv); M.append(pm); S.append(s)
    return np.array(V,np.float32),np.array(M),np.array(S)
VA,MA3,SA=vols(ds_tr,impA); VB,MB3,SB=vols(ds_te,impB)
yA=np.array([impA[s] for s in SA]); yB=np.array([impB[s] for s in SB])
THR=0.30; bA=(yA>=THR).astype(int); bB=(yB>=THR).astype(int)
print(f"3D volumes: MSW {VA.shape} ({bA.sum()}/{(bA==0).sum()})   CHH {VB.shape} ({bB.sum()}/{(bB==0).sum()})",flush=True)
POP3=(MA3.mean(0)>0.5)
if POP3.sum()<20: POP3=(MA3.mean(0)>MA3.mean())
r,c,z=np.where(POP3); BB=(slice(r.min(),r.max()+1),slice(c.min(),c.max()+1),slice(z.min(),z.max()+1))
print(f"3D ROI footprint {POP3.sum()} vox; bbox {[BB[i].stop-BB[i].start for i in range(3)]}",flush=True)
NF3=np.where(ndimage.binary_dilation(POP3,iterations=1).ravel())[0]
def lap3():
    P=ndimage.binary_dilation(POP3,iterations=1); N=np.array(np.where(P)).T
    I={tuple(x):i for i,x in enumerate(N)}; W=np.zeros((len(N),)*2)
    for t_,i in I.items():
        for d in [(1,0,0),(0,1,0),(0,0,1)]:
            j=I.get((t_[0]+d[0],t_[1]+d[1],t_[2]+d[2]))
            if j is not None: W[i,j]=W[j,i]=1
    ev,U=np.linalg.eigh(np.diag(W.sum(1))-W); return U[:,1:]
U3=lap3()
def basis3(V,name,K):
    k=max(1,int(round(K**(1/3))))
    fl=V.reshape(len(V),-1)
    cb=V[:,BB[0],BB[1],BB[2]]
    if name=='3D|FFT|':  return np.abs(fftn(cb,axes=(1,2,3)))[:,:k+1,:k+1,:k+1].reshape(len(V),-1)[:,:K]
    if name=='3DlogFFT': return np.log1p(np.abs(fftn(cb,axes=(1,2,3)))[:,:k+1,:k+1,:k+1]).reshape(len(V),-1)[:,:K]
    if name=='3DDCT':    return dctn(cb,axes=(1,2,3),norm='ortho')[:,:k+1,:k+1,:k+1].reshape(len(V),-1)[:,:K]
    if name=='3DradPSD':
        n=cb.shape[1:]; g=np.mgrid[:n[0],:n[1],:n[2]]
        rr=np.sqrt(sum((g[i]-n[i]/2)**2 for i in range(3))).astype(int)
        Pw=np.abs(np.fft.fftshift(fftn(cb,axes=(1,2,3)),axes=(1,2,3)))**2
        return np.stack([[Pw[i][rr==j].mean() for j in range(K)] for i in range(len(V))])
    if name=='3DgraphLap': return fl[:,NF3]@U3[:,:K]
    return None
REG={'Ridge':lambda:RidgeCV(alphas=np.logspace(-2,4,20)),'SVR':lambda:SVR(kernel='rbf'),
     'MLP':lambda:MLPRegressor(hidden_layer_sizes=(32,),max_iter=2000,random_state=0)}
CLF={'LogReg':lambda:LogisticRegression(C=1.0,max_iter=2000),'SVC':lambda:SVC(kernel='rbf'),
     'MLP':lambda:MLPClassifier(hidden_layer_sizes=(32,),max_iter=2000,random_state=0)}
sc_=lambda m,X: m.decision_function(X) if hasattr(m,'decision_function') else m.predict_proba(X)[:,1]
rows=[];PRED={}
BASES=['3D|FFT|','3DlogFFT','3DDCT','3DradPSD','3DgraphLap','3DPCA','3DroiPCA','3Draw']
for bn,K in itertools.product(BASES,[8,16]):
    try:
        if bn=='3DPCA':
            f=lambda V: V[:,BB[0],BB[1],BB[2]].reshape(len(V),-1)
            s1=StandardScaler().fit(f(VA)); t=PCA(min(K,len(VA)-1),random_state=0,whiten=True).fit(s1.transform(f(VA)))
            XA_,XB_=t.transform(s1.transform(f(VA))),t.transform(s1.transform(f(VB)))
        elif bn=='3DroiPCA':
            fa,fb=VA.reshape(len(VA),-1)[:,NF3],VB.reshape(len(VB),-1)[:,NF3]
            s1=StandardScaler().fit(fa); t=PCA(min(K,len(VA)-1),random_state=0,whiten=True).fit(s1.transform(fa))
            XA_,XB_=t.transform(s1.transform(fa)),t.transform(s1.transform(fb))
        elif bn=='3Draw':
            if K!=8: continue
            XA_,XB_=VA.reshape(len(VA),-1)[:,NF3],VB.reshape(len(VB),-1)[:,NF3]
        else: XA_,XB_=basis3(VA,bn,K),basis3(VB,bn,K)
        if XA_ is None: continue
    except Exception as e: log(f"  {bn} K={K} skipped: {e}"); continue
    for task in ['reg','clf']:
        zoo=REG if task=='reg' else CLF; ys=yA if task=='reg' else bA; ye=yB if task=='reg' else bB
        for mo in zoo:
            try:
                pr=np.full(len(ys),np.nan)
                for tr,va in KFold(5,shuffle=True,random_state=0).split(XA_):
                    s2=StandardScaler().fit(XA_[tr]); m=zoo[mo]().fit(s2.transform(XA_[tr]),ys[tr])
                    pr[va]=m.predict(s2.transform(XA_[va])) if task=='reg' else sc_(m,s2.transform(XA_[va]))
                m_in=pearsonr(ys,pr)[0] if task=='reg' else roc_auc_score(ys,pr)
                s3=StandardScaler().fit(XA_); m=zoo[mo]().fit(s3.transform(XA_),ys)
                pb=m.predict(s3.transform(XB_)) if task=='reg' else sc_(m,s3.transform(XB_))
                m_ex=pearsonr(ye,pb)[0] if task=='reg' else roc_auc_score(ye,pb)
                rows.append(dict(basis=bn,K=K,task=task,model=mo,m_int=m_in,m_ext=m_ex))
                PRED[(task,bn,K,mo)]=pb.copy()
            except Exception: continue
    log(f"{bn} K={K}: {len(rows)} cells")
pd.DataFrame(rows).to_csv(os.path.join(OUT, 'vol3d.csv'), index=False)
pickle.dump(dict(PRED=PRED,yB=yB,bB=bB),open(os.path.join(OUT, 'vol3d_preds.pkl'),'wb'))
log(f"DONE {len(rows)} cells")
