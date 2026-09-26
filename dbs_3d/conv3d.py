import sys, os
SPDIR = os.path.dirname(os.path.abspath(globals().get('__file__', '/workspace/dbs_3d/x')))
REPO  = os.path.dirname(SPDIR)
OUT   = os.path.join(REPO, 'docs', 'dbs_3d')          # docs/ is gitignored
DATA  = os.environ.get('DBS_CLINICAL_DIR', os.path.join(REPO, 'docs'))
CACHE = os.environ.get('DBS_CACHE_DIR', '/checkpoints')
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, REPO); sys.path.insert(0, SPDIR); os.chdir(REPO)
import numpy as np,torch,torch.nn as nn,warnings,time; warnings.filterwarnings('ignore')
torch.set_num_threads(12)
src=open(os.path.join(SPDIR,'vol3d_attn.py')).read()
exec(src.split("\nREG=")[0])
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
X=basis3(VA,'3D|FFT|',16); Xs=StandardScaler().fit_transform(X); Cs=StandardScaler().fit_transform(Ca)
xt,ct=torch.tensor(Xs,dtype=torch.float32),torch.tensor(Cs,dtype=torch.float32)
yt=torch.tensor(bA,dtype=torch.float32)
print("3D attention arms: convergence on the full labeled set (n=59)\n")
for mode in ['xattn','jointattn','film']:
    for ep in [400,1500,4000]:
        torch.manual_seed(0); m=Joint(16,Ca.shape[1],mode=mode)
        opt=torch.optim.AdamW(m.parameters(),3e-3,weight_decay=0.1); lf=nn.BCEWithLogitsLoss()
        t0=time.time()
        for e in range(ep):
            opt.zero_grad(); lf(m(xt,ct),yt).backward(); opt.step()
        m.eval()
        with torch.no_grad(): o=m(xt,ct)
        print(f"  {mode:10s} ep={ep:5d}  train loss {lf(o,yt).item():.4f}  train AUC {roc_auc_score(bA,o.numpy()):.3f}  ({time.time()-t0:.0f}s)")
    print()
