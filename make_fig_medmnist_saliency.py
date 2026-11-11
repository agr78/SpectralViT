"""Gradient and occlusion attribution for both arms on OrganMNIST3D at n=25.

Why both. The spectral arm sees 16 PCA components, so a voxel-space gradient is
    d(logit)/dv = sum_i (d logit/d s_i) * e_i / scale
a linear combination of 16 eigenimages: it is confined to a 16-dimensional subspace of
the 21,952 voxels and cannot show structure outside that span whatever the model learned.
This is the same band-limit that constrains a magnitude-Fourier tokenizer. Occlusion is a
finite perturbation rather than a derivative, so it escapes the span -- provided the
occluded VOLUME is re-projected through the scaler and the PCA, which is what makes the
perturbation travel through the tokenizer instead of around it.

The spectral occlusion is computed in closed form. Scaler and PCA are affine, so zeroing
a voxel set S (zero is the cohort mean, the volumes being globally standardized) gives
    z' = z - (x[S] / scale[S]) @ C[:, S].T
exactly, which is a 16x27 dot per window rather than a 21,952-dimensional reprojection.

usage: python make_fig_medmnist_saliency.py
"""
import os
import numpy as np, torch, torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from networks import MultiClassSpectralViT, MultiClassSpatialViT

NPZ = '/agent-home/medmnist/organmnist3d.npz'
OUT = 'paper/cvpr/fig/medmnist_saliency.png'
N, SEED, NCOMP, EPOCHS, BATCH, LR, WD, PATCH = 25, 0, 16, 100, 128, 1e-3, 0.01, 4
W, NVOL = 3, 6                 # occlusion window; test volumes averaged per class
CLASSES = [0, 1, 6]            # liver, kidney-right, pancreas in the MedMNIST ordering
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

z = np.load(NPZ)
Xtr, ytr = z['train_images'].astype(np.float32), z['train_labels'].ravel().astype(np.int64)
Xte, yte = z['test_images'].astype(np.float32), z['test_labels'].ravel().astype(np.int64)
NCLS, VOL = int(ytr.max()) + 1, Xtr.shape[1]
mu, sd = Xtr.mean(), Xtr.std()
Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd

# Reproduce the sweep's draw for this (n, seed) exactly: same calls in the same order.
torch.manual_seed(SEED); np.random.seed(SEED)
pool = {c: np.where(ytr == c)[0] for c in range(NCLS)}
avail = np.array([len(pool[c]) for c in range(NCLS)])
take = np.minimum(avail, N // NCLS)
while take.sum() < N:
    room = np.where(avail - take > 0)[0]
    pick = np.random.choice(room, min(N - take.sum(), len(room)), replace=False)
    take[pick] += 1
idx = np.concatenate([np.random.choice(pool[c], take[c], replace=False)
                      for c in range(NCLS) if take[c]])
tr_imgs = torch.from_numpy(Xtr[idx]).unsqueeze(1)
tr_flat = tr_imgs.view(len(idx), -1).numpy()
sc = StandardScaler().fit(tr_flat)
pca = PCA(n_components=min(NCOMP, len(tr_flat))).fit(sc.transform(tr_flat))
Ztr = torch.tensor(pca.transform(sc.transform(tr_flat)), dtype=torch.float32)
tr_y = torch.from_numpy(ytr[idx])

spec = MultiClassSpectralViT(n_inputs=Ztr.shape[1], num_classes=NCLS,
                             use_mode_weights=True).to(dev)
spat = MultiClassSpatialViT(num_classes=NCLS, size=VOL, patch_size=PATCH,
                            in_channels=1, volumetric=True).to(dev)
opt = optim.AdamW(list(spec.parameters()) + list(spat.parameters()), lr=LR,
                  weight_decay=WD)
loader = DataLoader(TensorDataset(tr_imgs, Ztr, tr_y), batch_size=BATCH, shuffle=True)
for _ in range(EPOCHS):
    spec.train(); spat.train()
    for bi, bz, by in loader:
        bi, bz, by = bi.to(dev), bz.to(dev), by.to(dev)
        opt.zero_grad()
        (F.cross_entropy(spec(bz), by) + F.cross_entropy(spat(bi), by)).backward()
        opt.step()
spec.eval(); spat.eval()

C = pca.components_.astype(np.float32)                  # (16, 21952)
scale = sc.scale_.astype(np.float32)
Cs = torch.tensor(C / scale, device=dev)                # eigenimages in volume units


def project(flat):
    return torch.tensor(pca.transform(sc.transform(flat)), dtype=torch.float32)


def grad_maps(vol, cls):
    """d p_cls / d voxel for each arm. The spectral path goes through the eigenimages."""
    v = torch.tensor(vol, device=dev).view(1, 1, VOL, VOL, VOL).requires_grad_(True)
    F.softmax(spat(v), 1)[0, cls].backward()
    gs = v.grad.detach().cpu().numpy().reshape(VOL, VOL, VOL)
    zz = project(vol.reshape(1, -1)).to(dev).requires_grad_(True)
    F.softmax(spec(zz), 1)[0, cls].backward()
    gz = zz.grad.detach()[0]                            # (16,)
    gv = (gz @ Cs).cpu().numpy().reshape(VOL, VOL, VOL)
    return gv, gs


def occl_maps(vol, cls):
    """Drop in p_cls when a WxWxW window is set to the cohort mean, per window center."""
    x = vol.reshape(-1)
    centres, zrows, vols = [], [], []
    r = range(W // 2, VOL - W // 2)
    for a in r:
        for b in r:
            for c in r:
                sl = (slice(a - W // 2, a + W // 2 + 1), slice(b - W // 2, b + W // 2 + 1),
                      slice(c - W // 2, c + W // 2 + 1))
                m = np.zeros((VOL, VOL, VOL), dtype=bool); m[sl] = True
                S = m.reshape(-1)
                centres.append((a, b, c))
                zrows.append(np.asarray((x[S] / scale[S]) @ C[:, S].T, dtype=np.float32))
                vv = vol.copy(); vv[sl] = 0.0
                vols.append(vv)
    z0 = project(vol.reshape(1, -1)).to(dev)
    with torch.no_grad():
        p0s = F.softmax(spec(z0), 1)[0, cls].item()
        zz = z0.repeat(len(zrows), 1) - torch.tensor(np.stack(zrows), device=dev)
        ps = torch.cat([F.softmax(spec(c), 1)[:, cls]
                        for c in torch.split(zz, 4096)]).cpu().numpy()
        v = torch.tensor(np.stack(vols), device=dev).unsqueeze(1)
        p0p = F.softmax(spat(torch.tensor(vol, device=dev).view(1, 1, VOL, VOL, VOL)),
                        1)[0, cls].item()
        pp = torch.cat([F.softmax(spat(c), 1)[:, cls]
                        for c in torch.split(v, 512)]).cpu().numpy()
    mz = np.zeros((VOL, VOL, VOL)); mp = np.zeros((VOL, VOL, VOL))
    for (a, b, c), q, w in zip(centres, ps, pp):
        mz[a, b, c] = p0s - q; mp[a, b, c] = p0p - w
    return mz, mp


# Average over correctly classified test volumes so a panel is not one draw of noise.
with torch.no_grad():
    zte = project(Xte.reshape(len(Xte), -1)).to(dev)
    pred_s = spec(zte).argmax(1).cpu().numpy()
    pred_p = spat(torch.tensor(Xte, device=dev).unsqueeze(1)).argmax(1).cpu().numpy()

acc = {}
for cls in CLASSES:
    ok = np.where((yte == cls) & (pred_s == cls) & (pred_p == cls))[0][:NVOL]
    if len(ok) < 2:
        ok = np.where(yte == cls)[0][:NVOL]
    stack = {k: [] for k in ('gz', 'gp', 'oz', 'op')}
    for i in ok:
        gv, gs = grad_maps(Xte[i], cls)
        oz, op = occl_maps(Xte[i], cls)
        stack['gz'].append(gv); stack['gp'].append(gs)
        stack['oz'].append(oz); stack['op'].append(op)
    acc[cls] = ({k: np.mean(v, 0) for k, v in stack.items()},
                Xte[ok].mean(0), len(ok))
    print(f'class {cls}: averaged {len(ok)} volumes', flush=True)

ROWS = [('gz', 'Gradient, spectral'), ('gp', 'Gradient, spatial'),
        ('oz', 'Occlusion, spectral'), ('op', 'Occlusion, spatial')]
fig, ax = plt.subplots(len(ROWS) + 1, len(CLASSES),
                       figsize=(2.1 * len(CLASSES), 2.1 * (len(ROWS) + 1)))
mid = VOL // 2
for j, cls in enumerate(CLASSES):
    maps, anat, k = acc[cls]
    ax[0, j].imshow(anat[mid], cmap='gray')
    ax[0, j].set_title(f'class {cls}  ($k={k}$)', fontsize=9)
    if j == 0:
        ax[0, j].set_ylabel('Anatomy', fontsize=9)
    for i, (key, lab) in enumerate(ROWS, start=1):
        m = maps[key][mid]
        # A handful of extreme voxels otherwise set the scale and wash the panel out,
        # so the limit is a high percentile of the volume rather than its maximum.
        lim = np.percentile(np.abs(maps[key]), 99.5) + 1e-12
        ax[i, j].imshow(anat[mid], cmap='gray')
        ax[i, j].imshow(m, cmap='bwr', vmin=-lim, vmax=lim, alpha=0.65)
        if j == 0:
            ax[i, j].set_ylabel(lab, fontsize=9)
for a in ax.ravel():
    a.set_xticks([]); a.set_yticks([])
fig.suptitle(f'OrganMNIST3D attribution at $n={N}$, central axial slice', fontsize=10)
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=170, bbox_inches='tight')
print('wrote', OUT)

# The band-limit claim is checkable: project each map onto the span of the 16
# eigenimages and report the share of its energy the subspace explains. The rows of
# C/scale are not orthogonal -- dividing by the per-voxel scale destroys the PCA's
# orthogonality -- so an orthonormal basis for their span is taken first; projecting on
# the raw rows double-counts the overlap and can exceed 100%.
B = np.linalg.qr((C / scale).T)[0].T          # (16, 21952), orthonormal rows
np.savez_compressed('results/medmnist/saliency_maps.npz', basis=B,
                    **{f'{k}_{cls}': acc[cls][0][k]
                       for cls in CLASSES for k, _ in ROWS})
for cls in CLASSES:
    maps = acc[cls][0]
    for key, lab in ROWS:
        f = maps[key].reshape(-1)
        share = np.sum((B @ f) ** 2) / (f @ f + 1e-30)
        print(f'class {cls}  {lab:22s} energy in the 16-component span: {share:6.1%}')
