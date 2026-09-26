import sys, os
SPDIR = os.path.dirname(os.path.abspath(globals().get('__file__', '/workspace/dbs_3d/x')))
REPO  = os.path.dirname(SPDIR)
OUT   = os.path.join(REPO, 'docs', 'dbs_3d')          # docs/ is gitignored
DATA  = os.environ.get('DBS_CLINICAL_DIR', os.path.join(REPO, 'docs'))
CACHE = os.environ.get('DBS_CACHE_DIR', '/checkpoints')
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, REPO); sys.path.insert(0, SPDIR); os.chdir(REPO)
import numpy as np,pandas as pd,torch,torch.nn as nn,itertools,time,pickle,warnings; warnings.filterwarnings('ignore')
torch.set_num_threads(12)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
src=open(os.path.join(SPDIR,'vol3d_attn.py')).read()
exec(src.split("\nREG=")[0])
t0=time.time(); log=lambda m: print(f"[{time.time()-t0:5.0f}s] {m}",flush=True)

class ConcatAttn(nn.Module):
    """self-attention over the CONCATENATED [imaging ; covariate] token set.
       mean-pool readout, plus a linear residual on the raw concat vector so the
       model can always fall back to plain concat."""
    def __init__(s,K,C,d=16,typ=False,resid=True):
        super().__init__(); s.K,s.C,s.typ,s.resid=K,C,typ,resid
        s.tok=nn.Linear(1,d); s.pos=nn.Parameter(torch.zeros(1,K+C,d))
        s.ty=nn.Parameter(torch.zeros(2,d))
        s.enc=nn.TransformerEncoder(nn.TransformerEncoderLayer(d,2,d*2,0.1,batch_first=True),1)
        s.hd=nn.Linear(d,1); s.lin=nn.Linear(K+C,1)
    def forward(s,x,c):
        v=torch.cat([x,c],1)
        z=s.tok(v.unsqueeze(-1))+s.pos
        if s.typ:
            z=z+torch.cat([s.ty[0].expand(s.K,-1),s.ty[1].expand(s.C,-1)],0).unsqueeze(0)
        o=s.hd(s.enc(z).mean(1)).squeeze(-1)
        return o+s.lin(v).squeeze(-1) if s.resid else o
def fit(Xtr,Ctr,ytr,Xte,Cte,K,C,typ,resid,ep=1500,seed=0):
    torch.manual_seed(seed); m=ConcatAttn(K,C,typ=typ,resid=resid)
    opt=torch.optim.AdamW(m.parameters(),3e-3,weight_decay=0.1); lf=nn.BCEWithLogitsLoss()
    xt,ct=torch.tensor(Xtr,dtype=torch.float32),torch.tensor(Ctr,dtype=torch.float32)
    yt=torch.tensor(ytr,dtype=torch.float32)
    for e in range(ep):
        opt.zero_grad(); lf(m(xt,ct),yt).backward(); opt.step()
    m.eval()
    with torch.no_grad(): return m(torch.tensor(Xte,dtype=torch.float32),torch.tensor(Cte,dtype=torch.float32)).numpy()
rows=[];PRED={}
for bn,K in itertools.product(['3D|FFT|','3DlogFFT','3DDCT','3DPCA','3DgraphLap'],[8,16]):
    if bn=='3DPCA':
        from sklearn.decomposition import PCA as _P
        f=lambda V: V[:,BB[0],BB[1],BB[2]].reshape(len(V),-1); s1=StandardScaler().fit(f(VA))
        t=_P(K,random_state=0,whiten=True).fit(s1.transform(f(VA)))
        XA_,XB_=t.transform(s1.transform(f(VA))),t.transform(s1.transform(f(VB)))
    else: XA_,XB_=basis3(VA,bn,K),basis3(VB,bn,K)
    for typ,resid in [(False,True),(True,True),(False,False)]:
        tag=f"{'typ' if typ else 'notyp'}_{'resid' if resid else 'nores'}"
        pr=np.full(len(bA),np.nan)
        for tr,va in StratifiedKFold(4,shuffle=True,random_state=0).split(XA_,bA):
            s2=StandardScaler().fit(XA_[tr]); s3=StandardScaler().fit(Ca[tr])
            pr[va]=fit(s2.transform(XA_[tr]),s3.transform(Ca[tr]),bA[tr].astype(float),
                       s2.transform(XA_[va]),s3.transform(Ca[va]),XA_.shape[1],Ca.shape[1],typ,resid)
        s2=StandardScaler().fit(XA_); s3=StandardScaler().fit(Ca)
        pb=fit(s2.transform(XA_),s3.transform(Ca),bA.astype(float),s2.transform(XB_),s3.transform(Cb),XA_.shape[1],Ca.shape[1],typ,resid)
        rows.append(dict(basis=bn,K=K,variant=tag,m_int=roc_auc_score(bA,pr),m_ext=roc_auc_score(bB,pb)))
        PRED[(bn,K,tag)]=pb.copy()
    log(f"{bn} K={K}: {len(rows)} cells")
pd.DataFrame(rows).to_csv(os.path.join(OUT, 'concat_attn.csv'),index=False)
pickle.dump(dict(PRED=PRED,bB=bB),open(os.path.join(OUT, 'concat_attn_preds.pkl'),'wb'))
log(f"DONE {len(rows)} cells")
