import sys, os
SPDIR = os.path.dirname(os.path.abspath(globals().get('__file__', '/workspace/dbs_3d/x')))
REPO  = os.path.dirname(SPDIR)
OUT   = os.path.join(REPO, 'docs', 'dbs_3d')          # docs/ is gitignored
DATA  = os.environ.get('DBS_CLINICAL_DIR', os.path.join(REPO, 'docs'))
CACHE = os.environ.get('DBS_CACHE_DIR', '/checkpoints')
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, REPO); sys.path.insert(0, SPDIR); os.chdir(REPO)
import numpy as np,pandas as pd,torch,torch.nn as nn,time,pickle,warnings; warnings.filterwarnings('ignore')
torch.set_num_threads(12)
from scipy import ndimage
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
exec(open(os.path.join(SPDIR,'vol3d.py')).read().split('REG={')[0])
t0=time.time(); log=lambda m: print(f"[{time.time()-t0:5.0f}s] {m}",flush=True)
# ---- build the UNLABELED pool: every volume with a bilateral ROI, labeled or not ----
def all_vols(ds):
    V,S=[],[]
    for s in sorted(ds.volumes):
        s=int(s); v=np.asarray(ds.volumes[s],np.float32); m=np.asarray(ds.seg_masks[s])>0
        lb,n=ndimage.label(m); sz=ndimage.sum(m,lb,range(1,n+1))
        if n<2: continue
        big=np.argsort(sz)[::-1][:2]+1
        info=sorted([(float(v[lb==b].mean()),b) for b in big],key=lambda t:t[0]); b=info[1][1]
        cx,cy,cz=[int(round(c)) for c in ndimage.center_of_mass(m,lb,[b])[0]]
        _x=min(max(0,cx-H), v.shape[0]-2*H); _y=min(max(0,cy-H), v.shape[1]-2*H); _z=min(max(0,cz-DZ), v.shape[2]-2*DZ)
        sl=(slice(_x,_x+2*H),slice(_y,_y+2*H),slice(_z,_z+2*DZ))
        p=v[sl]
        if p.shape==(2*H,2*H,2*DZ): V.append(p); S.append(s)
    return np.array(V,np.float32),np.array(S)
UA,UAS=all_vols(ds_tr); UB,UBS=all_vols(ds_te)
lab=set(SA.tolist()); unlab_msw=UA[[i for i,s in enumerate(UAS) if s not in lab]]
log(f"SSL pools: MSW unlabeled {len(unlab_msw)}, CHH {len(UB)}, labeled MSW {len(VA)}")
PS=(5,5,4); NP=(2*H//PS[0])*(2*H//PS[1])*(2*DZ//PS[2]); PD=PS[0]*PS[1]*PS[2]
def patchify(V):
    n=len(V); v=V.reshape(n,4,5,4,5,4,4).transpose(0,1,3,5,2,4,6).reshape(n,NP,PD)
    return v
class MAE(nn.Module):
    def __init__(s,d=64,L=2):
        super().__init__()
        s.emb=nn.Linear(PD,d); s.pos=nn.Parameter(torch.zeros(1,NP,d)); s.mtok=nn.Parameter(torch.zeros(1,1,d))
        s.enc=nn.TransformerEncoder(nn.TransformerEncoderLayer(d,4,d*2,0.1,batch_first=True),L)
        s.dec=nn.Linear(d,PD); s.cls=nn.Parameter(torch.zeros(1,1,d)); s.d=d
    def encode(s,x,mask=None):
        z=s.emb(x)+s.pos
        if mask is not None: z=torch.where(mask.unsqueeze(-1),s.mtok.expand_as(z),z)
        z=torch.cat([s.cls.expand(len(x),-1,-1),z],1)
        return s.enc(z)
    def forward(s,x,mask): return s.dec(s.encode(x,mask)[:,1:])
def pretrain(pool,ep=300,ratio=0.5,seed=0):
    torch.manual_seed(seed); m=MAE()
    opt=torch.optim.AdamW(m.parameters(),1e-3,weight_decay=0.05)
    X=torch.tensor(patchify(pool),dtype=torch.float32)
    mu,sd=X.mean(),X.std(); X=(X-mu)/sd
    for e in range(ep):
        idx=torch.randperm(len(X))
        for i in range(0,len(X),32):
            b=idx[i:i+32]; xb=X[b]
            msk=torch.rand(len(b),NP)<ratio
            opt.zero_grad(); out=m(xb,msk)
            loss=((out-xb)**2)[msk].mean(); loss.backward(); opt.step()
        if e%100==99: log(f"   pretrain ep{e+1} masked-MSE {loss.item():.4f}")
    return m,(mu,sd)
class Head(nn.Module):
    def __init__(s,mae,d=64):
        super().__init__(); s.mae=mae; s.hd=nn.Linear(d,1)
    def forward(s,x): return s.hd(s.mae.encode(x)[:,0]).squeeze(-1)
def finetune(mae,norm,Xtr,ytr,Xte,ep=200,lr=3e-4,seed=0):
    torch.manual_seed(seed)
    import copy; m=Head(copy.deepcopy(mae))
    opt=torch.optim.AdamW(m.parameters(),lr,weight_decay=0.05); lf=nn.BCEWithLogitsLoss()
    mu,sd=norm
    xt=(torch.tensor(patchify(Xtr),dtype=torch.float32)-mu)/sd
    yt=torch.tensor(ytr,dtype=torch.float32)
    for e in range(ep):
        opt.zero_grad(); lf(m(xt),yt).backward(); opt.step()
    m.eval()
    with torch.no_grad(): return m((torch.tensor(patchify(Xte),dtype=torch.float32)-mu)/sd).numpy()
def evaluate(mae,norm,tag):
    pr=np.full(len(bA),np.nan)
    for tr,va in StratifiedKFold(4,shuffle=True,random_state=0).split(VA,bA):
        pr[va]=finetune(mae,norm,VA[tr],bA[tr].astype(float),VA[va])
    ain=roc_auc_score(bA,pr)
    pb=finetune(mae,norm,VA,bA.astype(float),VB)
    aex=roc_auc_score(bB,pb)
    print(f"  {tag:34s} internal AUC {ain:.3f}   external AUC {aex:.3f}",flush=True)
    return ain,aex,pr,pb
res={}
log("--- baseline: NO pretraining (random init) ---")
torch.manual_seed(0); res['none']=evaluate(MAE(),(torch.tensor(0.),torch.tensor(1.)),'no pretraining')
log("--- SSL-A: pretrain on 50 UNLABELED MSW only (no test data touched) ---")
mA,nA=pretrain(unlab_msw); res['ssl_msw']=evaluate(mA,nA,'SSL on unlabeled MSW')
log("--- SSL-B: + CHH inputs (domain-adaptive, no CHH labels used) ---")
mB,nB=pretrain(np.concatenate([unlab_msw,UB])); res['ssl_msw_chh']=evaluate(mB,nB,'SSL on unlabeled MSW + CHH')
pickle.dump({k:(v[0],v[1],v[2],v[3]) for k,v in res.items()},open(os.path.join(OUT, 'ssl3d.pkl'),'wb'))
log("DONE")
