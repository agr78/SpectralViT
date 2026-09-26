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
from sklearn.linear_model import RidgeCV
from sklearn.kernel_ridge import KernelRidge
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


FA,PA=raw_crops(ds_tr,impA,10,False); FB,PB=raw_crops(ds_te,impB,10,False)
subsA=sorted(set(PA)); subsB=sorted(set(PB))
yA=np.array([impA[s_] for s_ in subsA]); yB=np.array([impB[s_] for s_ in subsB])
yxA=np.array([impA[p] for p in PA]); H=10
print(f"MSW {len(subsA)} patients / {len(FA)} slices ; CHH {len(subsB)} patients / {len(FB)} slices")

def mask_crops(ds, imp, H):
    subs=[int(s) for s in sorted(ds.volumes) if int(s) in imp]; M=[]
    for s in subs:
        v=np.asarray(ds.volumes[s],np.float32); m=np.asarray(ds.seg_masks[s])>0
        lb,n=ndimage.label(m); sz=ndimage.sum(m,lb,range(1,n+1))
        if n<2: continue
        big=np.argsort(sz)[::-1][:2]+1
        info=sorted([(float(v[lb==b].mean()), ndimage.center_of_mass(m,lb,[b])[0]) for b in big],key=lambda t:t[0])
        c=info[1][1]; cx,cy=int(round(c[0])),int(round(c[1]))
        for i in range(v.shape[2]):
            if m[:,:,i].sum()==0: continue
            pa=m[max(0,cx-H):max(0,cx-H)+2*H, max(0,cy-H):max(0,cy-H)+2*H, i]
            if pa.shape==(2*H,2*H): M.append(pa.ravel())
    return np.array(M,bool)
MA=mask_crops(ds_tr,impA,10)
POP=(MA.mean(0)>0.5).reshape(2*10,2*10)          # population ROI footprint in crop coords
if POP.sum()<8: POP=(MA.mean(0)>MA.mean()).reshape(2*10,2*10)
print(f"population ROI footprint: {POP.sum()} of {POP.size} pixels")
