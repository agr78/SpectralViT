"""Can a linear probe on the 25 training volumes pick the right tokenizer?

Honest by construction: for each (dataset, seed) the probe sees only the same 25 volumes
the ViT trains on, scored by repeated inner CV inside them. Everything data-dependent --
the PCA basis, the scalers -- is fit on the inner-train rows alone. The MedMNIST test
split is never opened here; it enters only afterwards, when the arm the probe chose is
looked up in the ViT results that were already computed.

Compared against fixing pca_energy, the best single arm, on the same cells.
"""
import numpy as np, pandas as pd, os, sys
from scipy import fft
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import mutual_info_classif

NCLS={'organmnist3d':11,'nodulemnist3d':2,'adrenalmnist3d':2,
      'fracturemnist3d':3,'vesselmnist3d':2,'synapsemnist3d':2}
PATH=lambda d: ('/agent-home/medmnist/organmnist3d.npz' if d=='organmnist3d'
                else f'/workspace/data/medmnist/{d}.npz')
SIDE, NB, REP = 14, 12, 4
ARMS=['pca16','dct','energy','pca_energy','concat','energy_cat','dct_energy','square']


def dct_blocks(V, freq):
    offs=list(range(0,V.shape[1]-SIDE+1,SIDE))
    out=np.zeros((len(V),len(offs)**3,freq**3),dtype=np.float32)
    for i in range(len(V)):
        bi=0
        for a in offs:
            for b in offs:
                for c in offs:
                    t=fft.dctn(V[i,a:a+SIDE,b:b+SIDE,c:c+SIDE],type=2,norm='ortho')
                    out[i,bi]=t[:freq,:freq,:freq].ravel(); bi+=1
    out=out.reshape(len(V),-1)
    return np.sign(out)*np.log1p(np.abs(out))


def bandE(V):
    offs=list(range(0,V.shape[1]-SIDE+1,SIDE))
    kz,ky,kx=np.meshgrid(*[np.arange(SIDE)]*3,indexing='ij')
    kr=np.sqrt(kz**2+ky**2+kx**2).ravel()
    edges=np.linspace(0,kr.max()+1e-6,NB+1); masks=[np.digitize(kr,edges)-1==b for b in range(NB)]
    out=np.zeros((len(V),len(offs)**3,NB),dtype=np.float32)
    for i in range(len(V)):
        bi=0
        for a in offs:
            for b in offs:
                for c in offs:
                    c2=fft.dctn(V[i,a:a+SIDE,b:b+SIDE,c:c+SIDE],type=2,norm='ortho').ravel()**2
                    for q,m in enumerate(masks): out[i,bi,q]=c2[m].mean()
                    bi+=1
    return np.log1p(out.reshape(len(V),-1))


def build(arm, Vfit, Vapp, D_fit, D_app, E_fit, E_app):
    """Fit anything data-dependent on Vfit/D_fit/E_fit only."""
    f=Vfit.reshape(len(Vfit),-1)
    parts_fit, parts_app = [], []
    if arm.startswith('pca') or arm in ('concat','energy_cat','square'):
        sp=StandardScaler().fit(f)
        pca=PCA(n_components=min(16,len(f)-1)).fit(sp.transform(f))
        parts_fit.append(pca.transform(sp.transform(f)))
        parts_app.append(pca.transform(sp.transform(Vapp.reshape(len(Vapp),-1))))
    if arm in ('dct','concat','energy_cat','dct_energy','square'):
        sd=StandardScaler().fit(D_fit)
        parts_fit.append(sd.transform(D_fit)); parts_app.append(sd.transform(D_app))
    if arm in ('energy','pca_energy','energy_cat','dct_energy'):
        se=StandardScaler().fit(E_fit)
        parts_fit.append(se.transform(E_fit)); parts_app.append(se.transform(E_app))
    A=np.hstack(parts_fit); B=np.hstack(parts_app)
    if arm=='square': A=np.hstack([A,A**2]); B=np.hstack([B,B**2])
    return A,B


rows=[]
for ds,ncls in NCLS.items():
    z=np.load(PATH(ds)); X=z['train_images'].astype(np.float32); y=z['train_labels'].ravel()
    X=(X-X.mean())/X.std()
    pool={c:np.where(y==c)[0] for c in range(ncls)}
    avail=np.array([len(pool[c]) for c in range(ncls)])
    for seed in range(5):
        np.random.seed(seed)
        take=np.minimum(avail,25//ncls)
        while take.sum()<25:
            room=np.where(avail-take>0)[0]
            take[np.random.choice(room,min(25-take.sum(),len(room)),replace=False)]+=1
        idx=np.concatenate([np.random.choice(pool[c],take[c],replace=False)
                            for c in range(ncls) if take[c]])
        Xs,ys=X[idx],y[idx]
        Dall,Eall=dct_blocks(Xs,3),bandE(Xs)
        cnt=np.bincount(ys,minlength=ncls); k=max(2,min(5,int(cnt[cnt>0].min())))
        for arm in ARMS:
            # Model-free alternative to the probe: mutual information between each
            # tokenized feature and the label. Reported four ways because the obvious
            # one is biased -- summing per-feature MI rewards more features regardless
            # of how redundant they are, so an arm with 464 coefficients beats one with
            # 16 by construction. Everything is fit on the same 25 volumes the ViT sees.
            A, _ = build(arm, Xs, Xs, Dall, Dall, Eall, Eall)
            mi = mutual_info_classif(A, ys, random_state=0, n_neighbors=3)
            srt = np.sort(mi)[::-1]
            rows.append(dict(dataset=ds, seed=seed, arm=arm, nfeat=A.shape[1],
                             mi_sum=float(mi.sum()), mi_mean=float(mi.mean()),
                             mi_max=float(srt[0]), mi_top16=float(srt[:16].mean())))
            print('%-17s s%d %-12s feats %4d  sum %6.2f mean %.4f max %.3f top16 %.3f'
                  % (ds, seed, arm, A.shape[1], mi.sum(), mi.mean(), srt[0],
                     srt[:16].mean()), flush=True)
pd.DataFrame(rows).to_csv('/workspace/results/mi_select.csv',index=False)
print('\nwrote results/mi_select.csv')
