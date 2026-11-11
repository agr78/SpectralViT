"""Build the Figure 1 eigenimage cache from a volume cohort.

Figure 1 illustrates Equation 1 by showing the leading basis vectors w_i as images. The
archived cache was built from IXI, which is acquired in native space, so its leading modes
track head position and size rather than anatomy -- the opening figure then displays the
registration artefact that the Precondition paragraph and sec:corr exist to address. This
rebuilds the cache from any cohort, so a consistently positioned one can be used instead.

The cache is written next to the figure, NOT under results/, because .gitignore excludes
results/ and the Figure 1 inputs must survive a clean clone -- the DBS figure's inputs were
left in /tmp and are now unrecoverable.

usage:
  HCA_VOLS=/agent-home/hca_smoke/hca_vols_96.npy python make_fig1_cache.py   # smoke only: synthetic
  HCA_VOLS=~/spacelink/aHCP/derivatives/hca_vols_96.npy python make_fig1_cache.py   # cluster
  FIG1_SET=mnist python make_fig1_cache.py        # the published Figure 1: fig1_cache_mnist.npz
      (FIG1_MNIST_N=1000 FIG1_SEED=0: the reported run's own draw; FIG1_SPLIT=test FIG1_DIGIT=0:
       the subject shown, the first test digit)
"""
import os
import numpy as np
from sklearn.decomposition import PCA

SRC = os.path.expanduser(os.environ.get('HCA_VOLS', '/agent-home/hca_smoke/hca_vols_96.npy'))
SET = os.environ.get('FIG1_SET', 'hca')
OUT = os.environ.get('FIG1_CACHE', os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                'paper', 'cvpr', 'fig', 'data',
                                                f'fig1_cache_{SET}.npz'))
NMODE = int(os.environ.get('FIG1_MODES', 16 if SET == 'mnist' else 8))   # 16: the digit experiment's count

if SET == 'mnist':
    # MNIST: the experiment in sec:dc fits PCA on the training digits (mnist_sweep.py,
    # 16 components), so the w_i shown are eigen-digits from the same 60k images. The
    # sweep's Normalize is a global affine map and leaves the component directions unchanged.
    import torch
    from torchvision import datasets, transforms
    from sklearn.preprocessing import StandardScaler
    from util import get_stratified_subset
    ROOT = os.environ.get('MNIST_ROOT', '/checkpoints/data')
    tr = datasets.MNIST(root=ROOT, train=True, download=False)
    sag = (tr.data.numpy().astype(np.float32) / 255.0)
    # The subject shown is the first test image, the 7 -- with training index 0 (the 5) the
    # only single MNIST image that appears everywhere, so neither is a cherry-pick. It is
    # an input to illustrate the projection with, not part of the basis fit.
    SPLIT = os.environ.get('FIG1_SPLIT', 'test')
    DIGIT = int(os.environ.get('FIG1_DIGIT', 0))
    ex_set = datasets.MNIST(root=ROOT, train=(SPLIT == 'train'), download=False)
    exemplar = ex_set.data.numpy().astype(np.float32)[DIGIT] / 255.0
    # The basis is the one the reported run fitted: mnist_sweep.py draws a stratified subset
    # of the training set under a fixed seed, standardizes every pixel, and fits the PCA on
    # that. Replicate the draw exactly (same transform, same seeding order, same helper), so
    # the w_i shown are the n = FIG1_MNIST_N, seed FIG1_SEED run's own components -- not the
    # full training set's, which the sweep never used. The sweep's PCA takes sklearn's
    # randomized solver with no seed; the full SVD here gives the same leading components up
    # to numerical tolerance and sign, and is identical on every run.
    N_DRAW = int(os.environ.get('FIG1_MNIST_N', 1000))
    SEED = int(os.environ.get('FIG1_SEED', 0))
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))])
    train_full = datasets.MNIST(root=ROOT, train=True, download=False, transform=tf)
    torch.manual_seed(SEED); np.random.seed(SEED)
    tr_imgs, tr_y = get_stratified_subset(train_full, N_DRAW // 10)
    sc = StandardScaler().fit(tr_imgs.view(len(tr_imgs), -1).numpy())
    flat = sc.transform(tr_imgs.view(len(tr_imgs), -1).numpy())
    # The schematic projects the exemplar onto the basis and sums the pieces back, so it
    # needs the same per-pixel standardization the coefficients were defined under. The
    # sweep's Normalize is folded into mu/sd here, so the stored pair maps raw [0,1] pixels.
    extra = {'mu': (sc.mean_ * 0.3081 + 0.1307).astype(np.float32),
             'sd': (sc.scale_ * 0.3081).astype(np.float32),
             # one digit per class from the draw itself (it is stratified, 100 per class in
             # order), un-normalized, for the schematic's 'training set -> PCA' icon
             'train_examples': (tr_imgs[::len(tr_imgs) // 10, 0] * 0.3081 + 0.1307)
                               .clamp(0, 1).numpy().astype(np.float32)}
    SRC = f'MNIST train, n={len(flat)} stratified draw seed {SEED}; subject {SPLIT}[{DIGIT}]'
    print(f'MNIST train: {sag.shape}; basis fitted on the n={len(flat)} draw, seed {SEED}')
    disp = lambda a: a                                   # already in display orientation
else:
    V = np.load(SRC, mmap_mode='r')
    print(f'{os.path.basename(SRC)}: {V.shape}')
    mid = V.shape[1] // 2
    sag = np.asarray(V[:, mid, :, :], dtype=np.float32)  # one mid-sagittal slice per subject
    flat = sag.reshape(len(sag), -1)
    exemplar = sag[0]
    extra = {}
    # make_fig_schematic.py shows the arrays with a plain imshow, and the IXI cache was
    # stored already in display orientation. The HCA volumes index (x, y, z) with z
    # superior, so the raw slice comes out rotated; transpose and flip so anterior is left
    # and superior is up.
    disp = lambda a: a.T[::-1]
# These volumes are standardized to a common per-scan SD but not to a common mean: the
# per-subject means span -0.42 to 0.66, so the leading component is otherwise just that DC
# offset and absorbs 92% of the variance. Remove each subject's own mean first, then the
# cohort mean image, so the modes describe anatomy rather than intensity offset.
if SET != 'mnist':
    flat = flat - flat.mean(1, keepdims=True)
    flat = flat - flat.mean(0)

k = min(NMODE, len(flat) - 1)
# Full SVD rather than sklearn's default randomized solver, so the cache is identical on
# every run; the randomized path takes no seed here and can flip component signs.
p = PCA(n_components=k, svd_solver='full').fit(flat)
eig = p.components_.reshape(k, sag.shape[1], sag.shape[2])
eig = np.stack([disp(e) for e in eig])

os.makedirs(os.path.dirname(OUT), exist_ok=True)
if SET == 'mnist':
    extra['pca_mean'] = p.mean_.astype(np.float32)
np.savez_compressed(OUT, **extra, eig_sag=eig.astype(np.float32),
                    brain_sag=disp(exemplar).astype(np.float32),
                    var=p.explained_variance_ratio_[:k].astype(np.float32),
                    source=os.path.basename(SRC) if os.path.exists(SRC) else SRC,
                    n_subjects=len(flat))
print(f'wrote {OUT}: {k} modes from {len(flat)} subjects, '
      f'leading variance {p.explained_variance_ratio_[:4].round(3).tolist()}')
