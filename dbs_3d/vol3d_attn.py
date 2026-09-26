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
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV,LogisticRegression
from sklearn.svm import SVR,SVC
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.metrics import roc_auc_score
exec(open(os.path.join(SPDIR,'vol3d.py')).read().split('REG={')[0])
from _clin import clin_tables,FULL
t0=time.time(); log=lambda m: print(f"[{time.time()-t0:5.0f}s] {m}",flush=True)
CA,CB=clin_tables()
def cmat(tab,subs,cols):
    M=np.array([[tab.loc[s,c] if (s in tab.index and pd.notna(tab.loc[s,c])) else np.nan for c in cols] for s in subs],float)
    med=np.nanmedian(M,0); ix=np.where(np.isnan(M)); M[ix]=np.take(med,ix[1]); return M
Ca,Cb=cmat(CA,list(SA),FULL),cmat(CB,list(SB),FULL)
class Joint(nn.Module):
    def __init__(s,K,C,d=16,mode='jointattn'):
        super().__init__(); s.mode=mode
        s.itok=nn.Linear(1,d); s.ctok=nn.Linear(1,d)
        s.ip=nn.Parameter(torch.zeros(1,K,d)); s.cp=nn.Parameter(torch.zeros(1,C,d))
        s.ty=nn.Parameter(torch.zeros(2,d)); s.cls=nn.Parameter(torch.zeros(1,1,d))
        s.enc=nn.TransformerEncoder(nn.TransformerEncoderLayer(d,2,d*2,0.1,batch_first=True),1)
        s.cl=nn.Sequential(nn.Linear(C,d),nn.GELU(),nn.Linear(d,2*d if mode=='film' else d))
        s.at=nn.MultiheadAttention(d,2,batch_first=True); s.hd=nn.Linear(d,1)
    def forward(s,x,c):
        if s.mode=='film':
            t=s.itok(x.unsqueeze(-1))+s.ip; g,b=s.cl(c).chunk(2,-1)
            return s.hd((t*(1+g.unsqueeze(1))+b.unsqueeze(1)).mean(1)).squeeze(-1)
        if s.mode=='xattn':
            t=s.itok(x.unsqueeze(-1))+s.ip; q=s.cl(c).unsqueeze(1)
            o,_=s.at(q,t,t); return s.hd(o.squeeze(1)+q.squeeze(1)).squeeze(-1)
        ti=s.itok(x.unsqueeze(-1))+s.ip+s.ty[0]; tc=s.ctok(c.unsqueeze(-1))+s.cp+s.ty[1]
        z=torch.cat([s.cls.expand(len(x),-1,-1),ti,tc],1)
        return s.hd(s.enc(z)[:,0]).squeeze(-1)
def tfit(Xtr,Ctr,ytr,Xte,Cte,K,C,mode,task,ep=400):
    torch.manual_seed(0); m=Joint(K,C,mode=mode)
    opt=torch.optim.AdamW(m.parameters(),3e-3,weight_decay=0.1)
    lf=nn.MSELoss() if task=='reg' else nn.BCEWithLogitsLoss()
    xt,ct=torch.tensor(Xtr,dtype=torch.float32),torch.tensor(Ctr,dtype=torch.float32)
    yt=torch.tensor(ytr,dtype=torch.float32)
    for e in range(ep):
        opt.zero_grad(); lf(m(xt,ct),yt).backward(); opt.step()
    m.eval()
    with torch.no_grad(): return m(torch.tensor(Xte,dtype=torch.float32),torch.tensor(Cte,dtype=torch.float32)).numpy()
REG={'Ridge':lambda:RidgeCV(alphas=np.logspace(-2,4,20)),'SVR':lambda:SVR(kernel='rbf')}
CLF={'LogReg':lambda:LogisticRegression(C=1.0,max_iter=2000),'SVC':lambda:SVC(kernel='rbf')}
sc_=lambda m,X: m.decision_function(X) if hasattr(m,'decision_function') else m.predict_proba(X)[:,1]
rows=[];PRED={}
def run(bn,K,task,arm,model=None):
    ys=yA if task=='reg' else bA; ye=yB if task=='reg' else bB
    if bn=='-': XA_,XB_=Ca.copy(),Cb.copy()
    elif bn=='3DPCA':
        from sklearn.decomposition import PCA as _P
        f=lambda V: V[:,BB[0],BB[1],BB[2]].reshape(len(V),-1); s1=StandardScaler().fit(f(VA))
        t=_P(K,random_state=0,whiten=True).fit(s1.transform(f(VA)))
        XA_,XB_=t.transform(s1.transform(f(VA))),t.transform(s1.transform(f(VB)))
    else: XA_,XB_=basis3(VA,bn,K),basis3(VB,bn,K)
    if arm=='concat': XA_,XB_=np.hstack([XA_,Ca]),np.hstack([XB_,Cb])
    spl=StratifiedKFold(4,shuffle=True,random_state=0).split(XA_,bA) if task=='clf' else KFold(5,shuffle=True,random_state=0).split(XA_)
    pr=np.full(len(ys),np.nan)
    for tr,va in spl:
        s2=StandardScaler().fit(XA_[tr]); s3=StandardScaler().fit(Ca[tr])
        if arm in ('film','xattn','jointattn'):
            pr[va]=tfit(s2.transform(XA_[tr]),s3.transform(Ca[tr]),ys[tr].astype(float),
                        s2.transform(XA_[va]),s3.transform(Ca[va]),XA_.shape[1],Ca.shape[1],arm,task)
        else:
            m=(REG if task=='reg' else CLF)[model]().fit(s2.transform(XA_[tr]),ys[tr])
            pr[va]=m.predict(s2.transform(XA_[va])) if task=='reg' else sc_(m,s2.transform(XA_[va]))
    m_in=pearsonr(ys,pr)[0] if task=='reg' else roc_auc_score(ys,pr)
    s2=StandardScaler().fit(XA_); s3=StandardScaler().fit(Ca)
    if arm in ('film','xattn','jointattn'):
        pb=tfit(s2.transform(XA_),s3.transform(Ca),ys.astype(float),s2.transform(XB_),s3.transform(Cb),XA_.shape[1],Ca.shape[1],arm,task)
    else:
        m=(REG if task=='reg' else CLF)[model]().fit(s2.transform(XA_),ys)
        pb=m.predict(s2.transform(XB_)) if task=='reg' else sc_(m,s2.transform(XB_))
    m_ex=pearsonr(ye,pb)[0] if task=='reg' else roc_auc_score(ye,pb)
    rows.append(dict(basis=bn,K=K,task=task,arm=arm,model=model or arm,m_int=m_in,m_ext=m_ex))
    PRED[(task,bn,K,arm,model or arm)]=pb.copy()
for task in ['reg','clf']:
    for mo in (REG if task=='reg' else CLF): run('-',0,task,'clin_only',mo)
log(f"clin_only done ({len(rows)})")
for bn,K in itertools.product(['3D|FFT|','3DlogFFT','3DDCT','3DPCA','3DgraphLap'],[8,16]):
    for task in ['reg','clf']:
        for mo in (REG if task=='reg' else CLF):
            run(bn,K,task,'img_only',mo); run(bn,K,task,'concat',mo)
        for arm in ['film','xattn','jointattn']: run(bn,K,task,arm)
    log(f"{bn} K={K}: {len(rows)} cells")
pd.DataFrame(rows).to_csv(os.path.join(OUT, 'vol3d_attn.csv'),index=False)
pickle.dump(dict(PRED=PRED,yB=yB,bB=bB),open(os.path.join(OUT, 'vol3d_attn_preds.pkl'),'wb'))
log(f"DONE {len(rows)} cells")
