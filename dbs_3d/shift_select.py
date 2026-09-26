import sys, os
SPDIR = os.path.dirname(os.path.abspath(globals().get('__file__', '/workspace/dbs_3d/x')))
REPO  = os.path.dirname(SPDIR)
OUT   = os.path.join(REPO, 'docs', 'dbs_3d')
DATA  = os.environ.get('DBS_CLINICAL_DIR', os.path.join(REPO, 'docs'))
CACHE = os.environ.get('DBS_CACHE_DIR', '/checkpoints')
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, REPO); sys.path.insert(0, SPDIR); os.chdir(REPO)
import numpy as np, pandas as pd, itertools, warnings; warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as _P
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.stats import pearsonr, spearmanr, wasserstein_distance
exec(open(os.path.join(SPDIR,'concat_attn.py')).read().split("rows=[]")[0])

def mmd(A,B,gamma=None):
    """RBF maximum mean discrepancy between the two feature clouds"""
    Z=np.vstack([A,B]); d2=((Z[:,None,:]-Z[None,:,:])**2).sum(-1)
    g=1.0/np.median(d2[d2>0]) if gamma is None else gamma
    k=lambda X,Y: np.exp(-g*((X[:,None,:]-Y[None,:,:])**2).sum(-1))
    return k(A,A).mean()+k(B,B).mean()-2*k(A,B).mean()
def domain_auc(A,B):
    """how separable are the two sites? 0.5 = indistinguishable = no shift"""
    X=np.vstack([A,B]); y=np.r_[np.zeros(len(A)),np.ones(len(B))]
    s=StandardScaler().fit_transform(X)
    return cross_val_score(LogisticRegression(max_iter=2000),s,y,cv=4,scoring='roc_auc').mean()

sel=pd.read_csv(os.path.join(OUT.replace('/docs/dbs_3d',''),'docs','dbs_3d','seedsel.csv')) \
    if os.path.exists(os.path.join(OUT,'seedsel.csv')) else pd.read_csv('/agent-home/runs/seedsel.csv')
rows=[]
for bn,K in itertools.product(['3D|FFT|','3DlogFFT','3DDCT','3DPCA','3DgraphLap'],[8,16]):
    if bn=='3DPCA':
        f=lambda V: V[:,BB[0],BB[1],BB[2]].reshape(len(V),-1); s1=StandardScaler().fit(f(VA))
        t=_P(K,random_state=0,whiten=True).fit(s1.transform(f(VA)))
        A,B=t.transform(s1.transform(f(VA))),t.transform(s1.transform(f(VB)))
    else: A,B=basis3(VA,bn,K),basis3(VB,bn,K)
    s=StandardScaler().fit(A); A,B=s.transform(A),s.transform(B)
    wd=np.mean([wasserstein_distance(A[:,j],B[:,j]) for j in range(A.shape[1])])
    rows.append(dict(basis=bn,K=K,mmd=mmd(A,B),wasserstein=wd,domain_auc=domain_auc(A,B)))
sh=pd.DataFrame(rows)
m=sh.merge(sel[['basis','K','variant','int_mean','ext_mean']].query("variant=='notyp_resid'"),on=['basis','K'])
print("=== label-free domain-shift metrics vs. actual external performance ===")
print(m[['basis','K','mmd','wasserstein','domain_auc','int_mean','ext_mean']].round(3).sort_values('domain_auc').to_string(index=False))
print()
for col in ['mmd','wasserstein','domain_auc']:
    r=pearsonr(m[col],m.ext_mean)[0]; rho=spearmanr(m[col],m.ext_mean).statistic
    print(f"  corr({col:12s}, external AUC) = {r:+.3f}   Spearman {rho:+.3f}")
print(f"  corr(internal AUC , external AUC) = {pearsonr(m.int_mean,m.ext_mean)[0]:+.3f}   Spearman {spearmanr(m.int_mean,m.ext_mean).statistic:+.3f}")
print()
b=m.loc[m.domain_auc.idxmin()]
print(f"cell chosen by MINIMUM domain shift (no labels used): {b['basis']} K={int(b['K'])}")
print(f"  domain AUC {b['domain_auc']:.3f} (0.5 = sites indistinguishable)")
print(f"  internal {b['int_mean']:.3f}   EXTERNAL {b['ext_mean']:.3f}")
m.to_csv(os.path.join(OUT,'shift_select.csv'), index=False)
