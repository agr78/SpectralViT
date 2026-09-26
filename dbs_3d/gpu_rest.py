import sys, os
SPDIR = os.path.dirname(os.path.abspath(globals().get('__file__', '/workspace/dbs_3d/x')))
REPO  = os.path.dirname(SPDIR)
OUT   = os.path.join(REPO, 'docs', 'dbs_3d')          # docs/ is gitignored
DATA  = os.environ.get('DBS_CLINICAL_DIR', os.path.join(REPO, 'docs'))
CACHE = os.environ.get('DBS_CACHE_DIR', '/checkpoints')
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, REPO); sys.path.insert(0, SPDIR); os.chdir(REPO)
import numpy as np,pandas as pd,torch,torch.nn as nn,itertools,time,pickle,warnings; warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
src=open(os.path.join(SPDIR,'concat_attn.py')).read(); exec(src.split("rows=[]")[0])
dev=torch.device('cuda'); T=lambda a: torch.tensor(a,dtype=torch.float32,device=dev)
t0=time.time(); log=lambda m: print(f"[{time.time()-t0:5.0f}s] {m}",flush=True)
log(f"device {torch.cuda.get_device_name(0)}")
def gfit(Xtr,Ctr,ytr,Xte,Cte,K,C,typ,resid,ep=1500,seed=0):
    torch.manual_seed(seed); m=ConcatAttn(K,C,typ=typ,resid=resid).to(dev)
    opt=torch.optim.AdamW(m.parameters(),3e-3,weight_decay=0.1); lf=nn.BCEWithLogitsLoss()
    xt,ct,yt=T(Xtr),T(Ctr),T(ytr)
    for e in range(ep):
        opt.zero_grad(); lf(m(xt,ct),yt).backward(); opt.step()
    m.eval()
    with torch.no_grad(): return m(T(Xte),T(Cte)).cpu().numpy()
def oof_ext(X0,X1,y,typ,resid,seed):
    pr=np.full(len(y),np.nan)
    for tr,va in StratifiedKFold(4,shuffle=True,random_state=0).split(X0,y):
        s2=StandardScaler().fit(X0[tr]); s3=StandardScaler().fit(Ca[tr])
        pr[va]=gfit(s2.transform(X0[tr]),s3.transform(Ca[tr]),y[tr].astype(float),
                    s2.transform(X0[va]),s3.transform(Ca[va]),X0.shape[1],Ca.shape[1],typ,resid,seed=seed)
    s2=StandardScaler().fit(X0); s3=StandardScaler().fit(Ca)
    pb=gfit(s2.transform(X0),s3.transform(Ca),y.astype(float),s2.transform(X1),s3.transform(Cb),
            X0.shape[1],Ca.shape[1],typ,resid,seed=seed)
    return roc_auc_score(y,pr),pb
# ---------- TEST 4: internal permutation on the pre-specified cell ----------
X0,X1=basis3(VA,'3DDCT',16),basis3(VB,'3DDCT',16)
obs,_=oof_ext(X0,X1,bA,True,True,0)
log(f"TEST 4: observed internal OOF AUC {obs:.3f}; running 200 label permutations")
rng=np.random.RandomState(0); null=[]
for b in range(200):
    yp=bA[rng.permutation(len(bA))]
    null.append(oof_ext(X0,X1,yp,True,True,0)[0])
    if (b+1)%50==0: log(f"   {b+1}/200  running null mean {np.mean(null):.3f}")
null=np.array(null)
print(f"\nTEST 4 INTERNAL PERMUTATION: observed {obs:.3f}   null {null.mean():.3f}+-{null.std():.3f}   p = {(null>=obs).mean():.4f}\n",flush=True)
# ---------- seed-averaged cell selection ----------
log("SEED-AVERAGED SELECTION across all cells x 5 seeds")
rows=[];PRED={}
for bn,K in itertools.product(['3D|FFT|','3DlogFFT','3DDCT','3DPCA','3DgraphLap'],[8,16]):
    if bn=='3DPCA':
        from sklearn.decomposition import PCA as _P
        f=lambda V: V[:,BB[0],BB[1],BB[2]].reshape(len(V),-1); s1=StandardScaler().fit(f(VA))
        t=_P(K,random_state=0,whiten=True).fit(s1.transform(f(VA)))
        A_,B_=t.transform(s1.transform(f(VA))),t.transform(s1.transform(f(VB)))
    else: A_,B_=basis3(VA,bn,K),basis3(VB,bn,K)
    for typ in [False,True]:
        ii,ee,pbs=[],[],[]
        for sd in range(5):
            ai,pb=oof_ext(A_,B_,bA,typ,True,sd); ii.append(ai); ee.append(roc_auc_score(bB,pb)); pbs.append(pb)
        tag=('typ' if typ else 'notyp')+'_resid'
        rows.append(dict(basis=bn,K=K,variant=tag,int_mean=np.mean(ii),int_sd=np.std(ii),
                         ext_mean=np.mean(ee),ext_sd=np.std(ee)))
        PRED[(bn,K,tag)]=np.mean(pbs,0)
    log(f"  {bn} K={K} done ({len(rows)} cells)")
df=pd.DataFrame(rows).sort_values('int_mean',ascending=False)
df.to_csv(os.path.join(OUT, 'seedsel.csv'),index=False)
pickle.dump(dict(PRED=PRED,bB=bB),open(os.path.join(OUT, 'seedsel_preds.pkl'),'wb'))
print("\n=== ranked by SEED-AVERAGED internal AUC ===")
print(df.round(3).to_string(index=False))
b=df.iloc[0]; pe=PRED[(b['basis'],int(b['K']),b['variant'])]
ae=roc_auc_score(bB,pe); n=len(bB)
nl=np.array([roc_auc_score(bB[rng.permutation(n)],pe) for _ in range(5000)])
print(f"\nPRE-SPECIFIED WINNER (seed-averaged internal): {b['basis']} K={int(b['K'])} {b['variant']}")
print(f"  internal {b['int_mean']:.3f} +- {b['int_sd']:.3f}")
print(f"  external {b['ext_mean']:.3f} +- {b['ext_sd']:.3f}")
print(f"  seed-ENSEMBLED external AUC {ae:.3f}   single-test p = {(nl>=ae).mean():.4f}")
log("DONE")
