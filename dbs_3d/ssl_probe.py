import sys, os
SPDIR = os.path.dirname(os.path.abspath(globals().get('__file__', '/workspace/dbs_3d/x')))
REPO  = os.path.dirname(SPDIR)
OUT   = os.path.join(REPO, 'docs', 'dbs_3d')          # docs/ is gitignored
DATA  = os.environ.get('DBS_CLINICAL_DIR', os.path.join(REPO, 'docs'))
CACHE = os.environ.get('DBS_CACHE_DIR', '/checkpoints')
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, REPO); sys.path.insert(0, SPDIR); os.chdir(REPO)
import numpy as np,torch,torch.nn as nn,warnings,pickle,time; warnings.filterwarnings('ignore')
torch.set_num_threads(12)
src=open(os.path.join(SPDIR,'ssl3d.py')).read(); exec(src.split("res={}")[0])
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
t0=time.time(); log=lambda m: print(f"[{time.time()-t0:5.0f}s] {m}",flush=True)
def feats(mae,norm,V):
    mu,sd=norm
    x=(torch.tensor(patchify(V),dtype=torch.float32)-mu)/sd
    mae.eval()
    with torch.no_grad(): z=mae.encode(x)
    return torch.cat([z[:,0],z[:,1:].mean(1)],1).numpy()      # CLS + mean-pooled
def probe(FA_,FB_,npc,tag):
    p=PCA(npc,random_state=0).fit(StandardScaler().fit_transform(FA_))
    sA=StandardScaler().fit(FA_); XA_=p.transform(sA.transform(FA_)); XB_=p.transform(sA.transform(FB_))
    pr=np.full(len(bA),np.nan)
    for tr,va in StratifiedKFold(4,shuffle=True,random_state=0).split(XA_,bA):
        s=StandardScaler().fit(XA_[tr])
        m=LogisticRegression(C=1.0,max_iter=2000).fit(s.transform(XA_[tr]),bA[tr])
        pr[va]=m.decision_function(s.transform(XA_[va]))
    s=StandardScaler().fit(XA_)
    m=LogisticRegression(C=1.0,max_iter=2000).fit(s.transform(XA_),bA)
    pb=m.decision_function(s.transform(XB_))
    ai,ae=roc_auc_score(bA,pr),roc_auc_score(bB,pb)
    print(f"  {tag:46s} npc={npc:2d}  internal {ai:.3f}   external {ae:.3f}",flush=True)
    return ai,ae,pr,pb
out={}
log("--- FROZEN-ENCODER LINEAR PROBE ---")
torch.manual_seed(0); m0=MAE()
n0=(torch.tensor(0.),torch.tensor(1.))
FA0,FB0=feats(m0,n0,VA),feats(m0,n0,VB)
for npc in [8,16]: out[f'rand_{npc}']=probe(FA0,FB0,npc,'random init (no pretraining)')
log("pretraining on 50 unlabeled MSW ...")
mA,nA=pretrain(unlab_msw,ep=300)
FA1,FB1=feats(mA,nA,VA),feats(mA,nA,VB)
for npc in [8,16]: out[f'sslA_{npc}']=probe(FA1,FB1,npc,'SSL: unlabeled MSW only')
log("pretraining on unlabeled MSW + CHH inputs ...")
mB,nB=pretrain(np.concatenate([unlab_msw,UB]),ep=300)
FA2,FB2=feats(mB,nB,VA),feats(mB,nB,VB)
for npc in [8,16]: out[f'sslB_{npc}']=probe(FA2,FB2,npc,'SSL: unlabeled MSW + CHH (domain-adaptive)')
log("reference: 3D|FFT| K=16 logistic regression")
X=basis3(VA,'3D|FFT|',16); Xb=basis3(VB,'3D|FFT|',16)
out['fft']=probe(X,Xb,16,'3D|FFT| K=16 (no pretraining, handcrafted)')
pickle.dump(out,open(os.path.join(OUT, 'ssl_probe.pkl'),'wb'))
log("DONE")
