"""MedMNIST3D classification by training-set size: a 3D medical benchmark at our own sizes.

The volumetric cohorts are ours; this is a public standardized 3D medical benchmark at the
same sample sizes, which makes the data-efficiency claim checkable by anyone. organmnist3d
ships 971 training and 610 test volumes at 28^3 over 11 organ classes, so n=25 to 500
matches the volumetric experiments and the test set is fixed rather than resampled.

Both arms carry the same 128 dimensional, two layer, four head encoder. The spatial arm
takes cubic patches through the volumetric path of MultiClassSpatialViT; its positional
embedding covers 343 patch positions against the spectral arm's 16 coefficients, so it
carries 1,238,530 weights at patch 4 and 1,538,690 at patch 14, against 1,190,402-1,310,347
for the bases the cross-validation selects. The baseline is therefore between 5% SMALLER
and 29% larger than the spectral arm depending on the cohort, so the comparison is close to
matched but NOT uniformly conservative -- on organ the baseline is the smaller model. An
earlier version of this docstring quoted a single '4% more weights, therefore
conservative', which held only while the patch was hardcoded to 4 on one cohort. Both
counts are written to the CSV; read them rather than restating a ratio.

Class counts are uneven (three organ classes hold ~39 training volumes against ~115 for the
rest), so a size is allocated as evenly as the available counts allow and the remainder is
spread at random among the classes that still have room, rather than failing or silently
dropping classes.

env: MEDMNIST_NPZ MEDMNIST_COMP MEDMNIST_SEEDS MEDMNIST_N MEDMNIST_OUT
usage: python medmnist_sweep.py [--smoke]
"""
import os, sys, time
import numpy as np, pandas as pd, torch, torch.optim as optim
import torch.nn.functional as F
from scipy import fft
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, top_k_accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from networks import MultiClassSpectralViT, MultiClassSpatialViT

# Token-identity switches; defaults reproduce every result already on disk.
_POS = os.environ.get('MEDMNIST_POS', '1') != '0'


SMOKE = '--smoke' in sys.argv
NPZ = os.environ.get('MEDMNIST_NPZ', '/agent-home/medmnist/organmnist3d.npz')
N_SEEDS = 2 if SMOKE else int(os.environ.get('MEDMNIST_SEEDS', 5))
SEED0 = int(os.environ.get('MEDMNIST_SEED0', 0))
N_COMP = int(os.environ.get('MEDMNIST_COMP', 16))
# MEDMNIST_COMP=0 means "as many components as the sample supports" rather than a fixed
# count: PCA on n centred volumes has rank n-1, so taking all of them discards nothing and
# estimates nothing beyond what the data carries. 16 was inherited from the MNIST setting
# and never re-justified for 3D volumes, where the leading 16 components hold as little as
# 13.5% of the class-discriminative signal (synapsemnist3d) against 76.2% (organmnist3d).
FULL_RANK = N_COMP == 0
# 'pca' refits a fold-wise PCA at every draw; 'dct' uses the paper's fixed block cosine
# transform, which costs no data to estimate. 28 = 2^2 * 7, so the block grid must divide
# into {1,2,4,7,14,28}. GRID=2, FREQ=3 keeps the 3^3 lowest frequencies of each of eight
# 14^3 blocks: 216 coefficients, which matches the spatial arm's weight count to 0.2%.
BASIS = os.environ.get('MEDMNIST_BASIS', 'pca')
GRID = int(os.environ.get('MEDMNIST_GRID', 2))
FREQ = int(os.environ.get('MEDMNIST_FREQ', 3))
# Block side and stride. Stride == side reproduces the disjoint partition used on the
# volumetric cohorts; a smaller stride overlaps the windows, which is what the CIFAR
# experiments report. Disjoint blocks impose a hard partition at fixed coordinates, so a
# structure sitting near a boundary splits between two blocks and a few voxels of shift
# moves it into a different one. These crops are centred on anatomy rather than
# registered to a template, so that discontinuity is a live concern here in a way it is
# not for the template-aligned brain volumes.
BLOCK = int(os.environ.get('MEDMNIST_BLOCK', 0)) or None
STRIDE = int(os.environ.get('MEDMNIST_STRIDE', 0)) or None
SIZES = [25, 50] if SMOKE else [int(v) for v in
                                os.environ.get('MEDMNIST_N', '25,50,100,200,500').split(',')]
EPOCHS, BATCH, LR, WD = (2 if SMOKE else 100), 128, 1e-3, 0.01
# The spatial baseline's only capacity knob. It was fixed at 4 while the spectral arm was
# swept over 89 basis x capacity configurations and allowed to pick one per dataset by
# cross-validation, which is not a fair comparison: the baseline has to be granted the
# same per-dataset choice or the margin is partly just tuning. 28 = 2^2 * 7, so the patch
# side must divide it: {1,2,4,7,14,28}, giving {21952,2744,343,64,8,1} tokens.
PATCH = int(os.environ.get('MEDMNIST_PATCH', 4))
if 28 % PATCH:
    sys.exit(f'MEDMNIST_PATCH must divide 28; got {PATCH}')
OUT = os.environ.get('MEDMNIST_OUT',
                     os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'results', 'medmnist'))
if N_COMP and N_COMP % 4:
    sys.exit(f'MEDMNIST_COMP must be a multiple of 4 (chunk_size); got {N_COMP}')
os.makedirs(OUT, exist_ok=True)
# The spatial arm's Conv3d patch embedding uses non-deterministic cuDNN kernels: an
# identical rerun moved its top-1 by up to 0.003, while the spectral arm reproduced
# exactly. That noise entered every spectral-minus-spatial contrast, so the comparison was
# paired in name only. Deterministic kernels cost a little speed and make the arm
# reproducible at fixed (dataset, n, seed), which is what lets close tokenizers separate.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
SPAT_CACHE = os.environ.get('MEDMNIST_SPAT_CACHE',
                           '/agent-home/medmnist_spatial_cache')
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def block_offsets(side):
    """Start indices of each block along one axis, given the side and stride."""
    b = BLOCK or side // GRID
    st = STRIDE or b
    return [o for o in range(0, side - b + 1, st)], b


def n_blocks(side=28):
    return len(block_offsets(side)[0]) ** 3


def block_spectra(vols):
    """Per-block low-frequency cosine coefficients, sign preserved and log compressed.

    The same transform as the volumetric experiments. Sign is kept because the DCT is
    real: unlike a Fourier magnitude it discards no phase, which is what carries
    position. With STRIDE < BLOCK the windows overlap, so every voxel is covered by
    several of them and a small translation moves the coefficients smoothly rather than
    across a partition boundary.
    """
    B = len(vols)
    offs, b = block_offsets(vols.shape[1])
    nb = len(offs) ** 3
    out = np.zeros((B, nb, FREQ ** 3), dtype=np.float32)
    for i in range(B):
        bi = 0
        for z in offs:
            for y in offs:
                for x in offs:
                    blk = vols[i, z:z+b, y:y+b, x:x+b]
                    c = fft.dctn(blk, type=2, norm='ortho')[:FREQ, :FREQ, :FREQ].ravel()
                    out[i, bi] = np.sign(c) * np.log1p(np.abs(c))
                    bi += 1
    return out.reshape(B, nb * FREQ ** 3)


def band_energy(vols, nbands=None):
    """Mean squared coefficient within each radial frequency shell, per block.

    Quadratic in the coefficients, so no linear basis can express it however it is chosen.
    That matters for texture: in a dense texture the phase of a high-frequency coefficient
    is near-random across samples and the signed value carries little, while the power in
    a band is stable. On SynapseMNIST3D 96 of these beat all 21,952 signed coefficients
    (0.676 against 0.613 probe AUC); on NoduleMNIST3D the same construction loses 0.108,
    which is why it is offered concatenated with the linear coefficients rather than
    instead of them. log1p because the log version probed better on every dataset tried.
    """
    nbands = nbands or int(os.environ.get('MEDMNIST_BANDS', 12))
    offs, side = block_offsets(vols.shape[1])
    kz, ky, kx = np.meshgrid(*[np.arange(side)] * 3, indexing='ij')
    kr = np.sqrt(kz ** 2 + ky ** 2 + kx ** 2).ravel()
    edges = np.linspace(0, kr.max() + 1e-6, nbands + 1)
    masks = [np.digitize(kr, edges) - 1 == b for b in range(nbands)]
    out = np.zeros((len(vols), len(offs) ** 3, nbands), dtype=np.float32)
    for i in range(len(vols)):
        bi = 0
        for z in offs:
            for y in offs:
                for x in offs:
                    c2 = fft.dctn(vols[i, z:z+side, y:y+side, x:x+side],
                                  type=2, norm='ortho').ravel() ** 2
                    for b, m in enumerate(masks):
                        out[i, bi, b] = c2[m].mean()
                    bi += 1
    return np.log1p(out.reshape(len(vols), -1))


def scatter_energy(vols, J=None):
    """Radial wavelet scattering energies, first and second order, per block.

    Band energy is a marginal: it reports the power in each shell and nothing about how
    the shells co-vary, yet band-to-band coupling is most of what distinguishes one dense
    texture from another. Scattering repairs exactly that gap and strictly contains the
    band-energy construction -- the first-order terms mean|x * psi_j| ARE a band energy,
    smoothed rather than squared, and the second-order terms
    mean||x * psi_j1| * psi_j2| carry the interaction between shells j1 < j2, which no
    quadratic statistic of a single shell can express.

    The filters are fixed Gaussian band-passes on dyadic centre frequencies, so nothing
    is estimated from the sample, as with the cosine transform. J=4 gives 4 first-order
    and 6 second-order terms per block, 80 features over the eight blocks, which puts it
    in the same budget as the 96 band energies it is meant to replace.
    """
    J = J or int(os.environ.get('MEDMNIST_SCAT_J', 4))
    offs, side = block_offsets(vols.shape[1])
    f1 = np.fft.fftfreq(side)
    kz, ky, kx = np.meshgrid(f1, f1, f1, indexing='ij')
    kr = np.sqrt(kz ** 2 + ky ** 2 + kx ** 2)
    cents = 0.5 / 2.0 ** np.arange(J)[::-1]
    filts = [np.exp(-0.5 * ((kr - c) / (0.5 * c)) ** 2) for c in cents]
    pairs = [(a, b) for a in range(J) for b in range(a + 1, J)]
    out = np.zeros((len(vols), len(offs) ** 3, J + len(pairs)), dtype=np.float32)
    for i in range(len(vols)):
        bi = 0
        for z in offs:
            for y in offs:
                for x in offs:
                    Fb = np.fft.fftn(vols[i, z:z+side, y:y+side, x:x+side])
                    u = [np.abs(np.fft.ifftn(Fb * f)) for f in filts]
                    out[i, bi, :J] = [v.mean() for v in u]
                    for j, (a, b) in enumerate(pairs):
                        Fa = np.fft.fftn(u[a])
                        out[i, bi, J + j] = np.abs(np.fft.ifftn(Fa * filts[b])).mean()
                    bi += 1
    return np.log1p(out.reshape(len(vols), -1))


def oriented_energy(vols, nrad=None, nori=None):
    """Power in radial x angular bins of the block cosine spectrum.

    The radial pooling in band_energy discards orientation entirely, so an anisotropic
    structure and isotropic noise with the same radial spectrum give identical features.
    That is the wrong invariance for vessel and fracture, where the orientation of the
    structure IS the class. Binning the same power by direction as well as by |k| keeps
    it. Cosine frequency indices are non-negative, so the directions span one octant and
    two polar by three azimuthal bins partition it.
    """
    nrad = nrad or int(os.environ.get('MEDMNIST_ORAD', 6))
    nori = nori or int(os.environ.get('MEDMNIST_ORI', 6))
    offs, side = block_offsets(vols.shape[1])
    kz, ky, kx = [a.ravel() for a in
                  np.meshgrid(*[np.arange(side)] * 3, indexing='ij')]
    kr = np.sqrt(kz ** 2 + ky ** 2 + kx ** 2)
    th = np.arccos(np.clip(kz / (kr + 1e-9), -1, 1))
    ph = np.arctan2(ky, kx)
    rb = np.clip((kr / (kr.max() + 1e-9) * nrad).astype(int), 0, nrad - 1)
    nth = 2
    nph = max(1, nori // nth)
    tb = np.clip((th / (np.pi / 2 + 1e-9) * nth).astype(int), 0, nth - 1)
    pb = np.clip((ph / (np.pi / 2 + 1e-9) * nph).astype(int), 0, nph - 1)
    cell = rb * (nth * nph) + tb * nph + pb
    nc = nrad * nth * nph
    masks = [cell == c for c in range(nc)]
    keep = [m for m in masks if m.any()]
    out = np.zeros((len(vols), len(offs) ** 3, len(keep)), dtype=np.float32)
    for i in range(len(vols)):
        bi = 0
        for z in offs:
            for y in offs:
                for x in offs:
                    c2 = fft.dctn(vols[i, z:z+side, y:y+side, x:x+side],
                                  type=2, norm='ortho').ravel() ** 2
                    for b, m in enumerate(keep):
                        out[i, bi, b] = c2[m].mean()
                    bi += 1
    return np.log1p(out.reshape(len(vols), -1))


def topo_features(vols, nth=None):
    """Connected-component and Euler-characteristic curves over an intensity sweep.

    Every other basis here is a bandlimited functional of the volume -- linear (cosine,
    PCA) or quadratic (band energy, squares) -- and none of them can express connectivity.
    A vertebral fracture is a DISCONNECTION of cortical bone, a change in topology rather
    than in frequency content, which is a property no projection onto a fixed basis
    represents at any rank. Sweeping a threshold and counting foreground components,
    background components and the occupied fraction gives a filtration summary: the curves
    are what persistent homology would integrate, at a fraction of the cost and with no
    new dependency.

    Nothing is estimated from the sample beyond the per-feature standardisation applied by
    the caller; the thresholds are percentiles of each volume, so the descriptor is
    invariant to the intensity windowing that differs between these cohorts.
    """
    from scipy import ndimage
    nth = nth or int(os.environ.get('MEDMNIST_THRESH', 12))
    out = np.zeros((len(vols), nth, 3), dtype=np.float32)
    for i, v in enumerate(vols):
        lo, hi = np.percentile(v, [5, 95])
        for j, t in enumerate(np.linspace(lo, hi, nth)):
            B = v > t
            out[i, j, 0] = ndimage.label(B)[1]
            out[i, j, 1] = ndimage.label(~B)[1]
            out[i, j, 2] = B.mean()
    return np.log1p(out.reshape(len(vols), -1))


def block_mag(vols):
    """Per-block Fourier MAGNITUDE, log compressed -- power without phase.

    block_spectra keeps sign(c)*log1p|c|, which retains phase; on these cohorts that is
    the wrong choice almost everywhere. A full-training-split probe on the leading PCs of
    the cosine coefficients scores 0.560 signed against 0.672 on |c| for fracture, 0.472
    against 0.664 for synapse (signed is BELOW chance there), and 0.909 against 1.000 for
    organ. Only nodule prefers the signed form (0.728 against 0.682).

    band_energy already discards phase but pools power into radial shells, which destroys
    the per-coefficient structure the probe exploits. This keeps magnitude per coefficient,
    which is the representation the probe says the information actually lives in. Mirrors
    the Block-|FFT| arm the volumetric experiments already carry.
    """
    B = len(vols)
    offs, b = block_offsets(vols.shape[1])
    nb = len(offs) ** 3
    out = np.zeros((B, nb, FREQ ** 3), dtype=np.float32)
    for i in range(B):
        bi = 0
        for z in offs:
            for y in offs:
                for x in offs:
                    m = np.abs(fft.fftn(vols[i, z:z+b, y:y+b, x:x+b]))
                    out[i, bi] = np.log1p(m[:FREQ, :FREQ, :FREQ].ravel())
                    bi += 1
    return out.reshape(B, nb * FREQ ** 3)


def moment_features(vols):
    """Marginal intensity profiles and low-order spatial moments.

    Absent from every other basis here. FractureMNIST3D is vertebral compression fracture,
    whose clinical sign is loss of vertebral HEIGHT -- a morphometric property of where the
    mass sits along an axis, not a frequency, magnitude, variance or connectivity property.
    A marginal profile along each axis states it directly: a compressed vertebra has its
    intensity concentrated over fewer slices, which shows up as a narrower profile and a
    smaller second central moment, and no projection onto a fixed basis above encodes that.

    Three marginal profiles (28 values each) plus the first three normalised central
    moments of each profile: 93 features, nothing fitted from the sample.
    """
    out = np.zeros((len(vols), 3 * vols.shape[1] + 9), dtype=np.float32)
    for i, v in enumerate(vols):
        # The caller hands over GLOBALLY STANDARDISED volumes, so intensities are signed
        # and a marginal sum can be negative -- log1p and the variance term then return
        # NaN. Taking the positive part first restores a mass distribution, and after a
        # global standardisation v > 0 means above-average attenuation, which on these CT
        # crops is the dense tissue whose extent the fracture sign is about.
        w = np.maximum(v, 0.0)
        parts = []
        mom = []
        for ax in (0, 1, 2):
            p = w.sum(axis=tuple(a for a in (0, 1, 2) if a != ax)).astype(np.float64)
            parts.append(np.log1p(p))
            q = p / (p.sum() + 1e-9)
            c = np.arange(len(q), dtype=np.float64)
            mu = (c * q).sum()
            sd = np.sqrt(((c - mu) ** 2 * q).sum()) + 1e-9
            mom += [mu, sd, ((c - mu) ** 3 * q).sum() / sd ** 3]
        out[i] = np.concatenate(parts + [np.asarray(mom)])
    return out


def haar_features(vols, signed=False):
    """Two-level 3D Haar coefficients, pooled per subband per octant.

    The scattering arms here use isotropic radial band-passes and report energies averaged
    over space, so they keep scale and discard position. The block cosine arms keep
    position but work at ONE scale -- the block side was swept over {14, 7} as separate
    configs, never combined. A Haar decomposition is the missing middle: multiscale, with
    localisation retained at every scale, and orthogonal so nothing is estimated.

    28 = 4 x 7, so two levels are available (28 -> 14 -> 7). Each level produces eight
    subbands by low/high filtering along each axis; the LLL band is decomposed again, so
    two levels give fifteen subbands. Each is summarised by its mean coefficient over the
    eight spatial octants, which keeps coarse position without reinstating the full
    21952-dimensional transform: 15 x 8 = 120 features.

    Magnitudes by default, because signed coefficients lose to |c| on three of four
    cohorts here (fracture 0.560 against 0.672 on the full split); signed=True gives the
    phase-retaining variant for the contrast.
    """
    r2 = np.sqrt(2.0)

    def split(a, ax):
        s = np.take(a, np.arange(0, a.shape[ax] - 1, 2), axis=ax)
        d = np.take(a, np.arange(1, a.shape[ax], 2), axis=ax)
        return (s + d) / r2, (s - d) / r2

    def level(a):
        out = []
        for x0 in split(a, 0):
            for x1 in split(x0, 1):
                out += list(split(x1, 2))
        return out                                    # LLL first, then 7 detail bands

    def octants(b):
        h = [max(1, s // 2) for s in b.shape]
        return np.array([b[i*h[0]:(i+1)*h[0] or None,
                           j*h[1]:(j+1)*h[1] or None,
                           k*h[2]:(k+1)*h[2] or None].mean()
                         for i in (0, 1) for j in (0, 1) for k in (0, 1)])

    out = np.zeros((len(vols), 15 * 8), dtype=np.float32)
    for i, v in enumerate(vols):
        l1 = level(v.astype(np.float64))
        bands = l1[1:] + level(l1[0])                 # 7 detail at level 1 + 8 at level 2
        rows = []
        for b in bands:
            c = b if signed else np.abs(b)
            o = octants(c)
            rows.append(o if signed else np.log1p(o))
        out[i] = np.concatenate(rows)
    return out


def _freq_grid(side):
    k = np.fft.fftfreq(side)
    kz, ky, kx = np.meshgrid(k, k, k, indexing='ij')
    kr = np.sqrt(kz ** 2 + ky ** 2 + kx ** 2)
    return kz, ky, kx, kr


def steerable_features(vols, nsc=None):
    """Oriented band-pass energies: a 3D steerable decomposition.

    The scattering arms use ISOTROPIC radial band-passes and the Haar arms are
    axis-aligned, so no basis here resolves orientation jointly with scale. A fracture
    line is an oriented discontinuity, which is exactly the property neither captures.
    Second-order steerable filters are the six products k_a k_b / |k|^2 modulating a
    radial band-pass, which span the l=2 angular subspace, so any orientation is a linear
    combination of the six responses -- steerable in the usual sense.
    """
    nsc = nsc or int(os.environ.get('MEDMNIST_SCALES', 4))
    side = vols.shape[1]
    kz, ky, kx, kr = _freq_grid(side)
    cents = 0.5 / 2.0 ** np.arange(nsc)[::-1]
    rad = [np.exp(-0.5 * ((kr - c) / (0.5 * c)) ** 2) for c in cents]
    k2 = kr ** 2 + 1e-12
    ang = [kz * kz / k2, ky * ky / k2, kx * kx / k2,
           kz * ky / k2, kz * kx / k2, ky * kx / k2]
    out = np.zeros((len(vols), nsc * 6 * 8), dtype=np.float32)
    h = side // 2
    for i, v in enumerate(vols):
        Fv = np.fft.fftn(v)
        row = []
        for r in rad:
            for a in ang:
                resp = np.abs(np.fft.ifftn(Fv * r * a))
                row += [resp[p::h if h else 1].mean() if False else
                        resp[zi*h:(zi+1)*h, yi*h:(yi+1)*h, xi*h:(xi+1)*h].mean()
                        for zi in (0, 1) for yi in (0, 1) for xi in (0, 1)
                        for p in (0,)]
        out[i] = np.log1p(np.asarray(row))
    return out


def bispectrum_features(vols, nsh=None):
    """Radially reduced bispectrum: third-order, translation invariant.

    The power spectrum discards phase. The bispectrum B(k1,k2)=F(k1)F(k2)F*(k1+k2) keeps
    phase COUPLING while staying invariant to translation, so it reaches structure that no
    second-order statistic represents -- and the magnitude probe showed phase is not simply
    useless here (nodule prefers signed coefficients). Reduced over radial shells, which is
    what makes it affordable: nsh shells give nsh*(nsh+1)/2 pairs rather than |k|^6 terms.
    """
    nsh = nsh or int(os.environ.get('MEDMNIST_SHELLS', 6))
    side = vols.shape[1]
    _, _, _, kr = _freq_grid(side)
    e = np.linspace(0, kr.max() + 1e-9, nsh + 1)
    masks = [(kr >= e[b]) & (kr < e[b + 1]) for b in range(nsh)]
    pairs = [(a, b) for a in range(nsh) for b in range(a, nsh)]
    out = np.zeros((len(vols), 2 * len(pairs)), dtype=np.float32)
    for i, v in enumerate(vols):
        Fv = np.fft.fftn(v)
        comp = [np.fft.ifftn(Fv * m) for m in masks]
        row = []
        for a, b in pairs:
            # <x_a x_b conj(x_{a+b})> in the spatial domain: the shell-reduced bispectrum
            z = comp[a] * comp[b] * np.conj(comp[min(a + b, nsh - 1)])
            row += [np.sign(z.real.mean()) * np.log1p(abs(z.real.mean())),
                    np.log1p(abs(z.imag.mean()))]
        out[i] = np.asarray(row)
    return out


def shell_harmonic_features(vols, lmax=None, nsh=None):
    """Spherical-harmonic power per radial shell: rotation invariant angular structure.

    band_energy pools each shell to a single number, discarding all angular information;
    oriented_energy bins angle crudely into octants of direction. Decomposing each shell's
    angular pattern onto spherical harmonics and taking power per degree l is the proper
    version: invariant to rotation within the shell, but sensitive to how anisotropic it
    is, which octant binning confounds with orientation.
    """
    from scipy.special import sph_harm_y
    lmax = lmax or int(os.environ.get('MEDMNIST_LMAX', 3))
    nsh = nsh or int(os.environ.get('MEDMNIST_SHELLS', 8))
    side = vols.shape[1]
    kz, ky, kx, kr = _freq_grid(side)
    th = np.arccos(np.clip(kz / (kr + 1e-12), -1, 1)).ravel()
    ph = np.arctan2(ky, kx).ravel()
    e = np.linspace(0, kr.max() + 1e-9, nsh + 1)
    sh_idx = np.digitize(kr.ravel(), e) - 1
    Y = {}
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            Y[(l, m)] = sph_harm_y(l, m, th, ph)
    out = np.zeros((len(vols), nsh * (lmax + 1)), dtype=np.float32)
    for i, v in enumerate(vols):
        P = np.abs(np.fft.fftn(v)).ravel()
        row = np.zeros((nsh, lmax + 1))
        for b in range(nsh):
            sel = sh_idx == b
            if not sel.any():
                continue
            p = P[sel]
            for l in range(lmax + 1):
                acc = 0.0
                for m in range(-l, l + 1):
                    acc += abs((p * np.conj(Y[(l, m)][sel])).sum()) ** 2
                row[b, l] = acc / (sel.sum() ** 2 + 1e-12)
        out[i] = np.log1p(row.ravel())
    return out


def glcm_features(vols, nlev=None):
    """Grey-level co-occurrence statistics: second-order SPATIAL, not spectral.

    Every other basis here is a functional of the Fourier or PCA coefficients. A
    co-occurrence matrix is a joint histogram of intensity pairs at a fixed offset, which
    is a second-order statistic in SPACE rather than in frequency, and it is the classical
    representation for trabecular bone texture -- the tissue fracture grading is read off.
    Levels are percentile bins, so the signed globally standardised input is handled
    without assuming a range.
    """
    nlev = nlev or int(os.environ.get('MEDMNIST_GLEV', 8))
    offs = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)]
    out = np.zeros((len(vols), len(offs) * 5), dtype=np.float32)
    lv = np.arange(nlev)
    ii, jj = np.meshgrid(lv, lv, indexing='ij')
    for i, v in enumerate(vols):
        q = np.clip(np.digitize(v, np.percentile(v, np.linspace(0, 100, nlev + 1)[1:-1])),
                    0, nlev - 1)
        row = []
        for dz, dy, dx in offs:
            a = q[dz:q.shape[0] or None, dy:q.shape[1] or None, dx:q.shape[2] or None]
            b = q[:q.shape[0] - dz or None, :q.shape[1] - dy or None, :q.shape[2] - dx or None]
            G = np.bincount((a.ravel() * nlev + b.ravel()), minlength=nlev ** 2)
            G = G.reshape(nlev, nlev).astype(np.float64)
            G = (G + G.T) / (G.sum() * 2 + 1e-12)
            mu = (G * ii).sum()
            sd = np.sqrt((G * (ii - mu) ** 2).sum()) + 1e-12
            row += [(G * (ii - jj) ** 2).sum(),                       # contrast
                    (G / (1 + (ii - jj) ** 2)).sum(),                 # homogeneity
                    (G ** 2).sum(),                                   # energy
                    (G * (ii - mu) * (jj - mu)).sum() / sd ** 2,      # correlation
                    -(G * np.log(G + 1e-12)).sum()]                   # entropy
        out[i] = np.asarray(row)
    return out


def patch_dct(vols, k=None, patch=None):
    """Per-patch DCT keeping the k lowest-frequency coefficients of each cubic patch.

    The diagnostic that isolates LOSSINESS from LEARNEDNESS. The spatial baseline applies
    nn.Linear(patch^3, embed) to every patch, so it sees all 21952 voxels and fits the
    projection against the label. At k = patch^3 this function is an orthogonal rotation
    of each patch, and a learned Linear can absorb any rotation, so the two are the same
    function class and must score the same up to seed noise. Lowering k truncates the
    rotation, and the drop measures exactly what a lossy tokenizer gives up -- per dataset,
    with everything else held fixed.

    No per-feature standardisation here, deliberately: the spatial arm receives raw
    (globally standardised) voxels, and rescaling each coefficient separately would break
    the orthogonal equivalence that makes the comparison exact.
    """
    patch = patch or PATCH
    k = k or int(os.environ.get('MEDMNIST_PK', patch ** 3))
    side = vols.shape[1]
    npx = side // patch
    f = np.arange(patch)
    fz, fy, fx = np.meshgrid(f, f, f, indexing='ij')
    order = np.argsort((fz ** 2 + fy ** 2 + fx ** 2).ravel(), kind='stable')[:k]
    out = np.zeros((len(vols), npx ** 3, k), dtype=np.float32)
    for i, v in enumerate(vols):
        t = 0
        for z in range(npx):
            for y in range(npx):
                for x in range(npx):
                    blk = v[z*patch:(z+1)*patch, y*patch:(y+1)*patch, x*patch:(x+1)*patch]
                    out[i, t] = fft.dctn(blk, type=2, norm='ortho').ravel()[order]
                    t += 1
    return out.reshape(len(vols), npx ** 3 * k)


def rank_perm():
    """Block-major to frequency-major, frequencies sorted by |k|.

    The rank weights are 1/i over the flat vector, so index order has to be variance
    order. Left block-major, the profile would downweight whole spatial blocks rather
    than high frequencies, which carries no variance interpretation.
    """
    nb, nf = n_blocks(), FREQ ** 3
    kz, ky, kx = np.unravel_index(np.arange(nf), (FREQ, FREQ, FREQ))
    order = np.argsort(kz ** 2 + ky ** 2 + kx ** 2, kind='stable')
    return np.array([b * nf + order[r] for r in range(nf) for b in range(nb)])


def metrics(probs, y_true, ncls):
    """Top-1, top-5 and one-vs-rest AUC, guarded for the small label sets.

    Only OrganMNIST3D has more than five classes. Top-5 is undefined at or below five
    and is reported as NaN rather than the trivial 1.0, and roc_auc_score takes a single
    column of positive-class scores in the binary case rather than one column per class,
    so four of the six MedMNIST3D sets need that branch.
    """
    lab = np.arange(ncls)
    return {'T1': accuracy_score(y_true, np.argmax(probs, axis=1)),
            'T5': (top_k_accuracy_score(y_true, probs, k=5, labels=lab)
                   if ncls > 5 else float('nan')),
            'AUC': (roc_auc_score(y_true, probs[:, 1]) if ncls == 2 else
                    roc_auc_score(y_true, probs, multi_class='ovr', labels=lab))}


def allocate(avail, n):
    """Spread n draws over classes as evenly as the available counts permit."""
    k = len(avail)
    take = np.minimum(avail, n // k)
    while take.sum() < n:
        room = np.where(avail - take > 0)[0]
        if not len(room):
            break
        pick = np.random.choice(room, min(n - take.sum(), len(room)), replace=False)
        take[pick] += 1
    return take


z = np.load(NPZ)
Xtr_all = z['train_images'].astype(np.float32)
ytr_all = z['train_labels'].ravel().astype(np.int64)
Xte = z['test_images'].astype(np.float32)
yte = z['test_labels'].ravel().astype(np.int64)
NCLS, VOL = int(ytr_all.max()) + 1, Xtr_all.shape[1]
# One global standardization, applied identically to train and test; the per-subsample
# StandardScaler below is what is fit on training data only.
mu, sd = Xtr_all.mean(), Xtr_all.std()
Xtr_all, Xte = (Xtr_all - mu) / sd, (Xte - mu) / sd

# MEDMNIST_OCCLUDE=<npz> replaces the test split with an OCCLUSION BATCH: element 0 is an
# unmodified volume and every other element is a copy of it with one window zeroed. Zero is
# the cohort mean, because the volumes are globally standardised on the line above, so a
# zeroed window reads as "no evidence" rather than as an out-of-distribution intensity.
# The batch travels the SAME tokenizer branch and the SAME trained models as the reported
# run. That is the point: the perturbation goes THROUGH the basis rather than around it, and
# unlike an input-space gradient it is not confined to the span of the basis -- a 16-component
# tokenizer has a 16-dimensional gradient subspace and can never show structure outside it,
# whatever the model learned. Per-sample probabilities for both arms are written out and the
# run then exits; nothing about training or featurisation changes.
OCCLUDE = os.environ.get('MEDMNIST_OCCLUDE')
if OCCLUDE:
    _occ = np.load(OCCLUDE)
    Xte = (_occ['volumes'].astype(np.float32) - mu) / sd
    yte = _occ['labels'].ravel().astype(np.int64)
te_imgs = torch.from_numpy(Xte).unsqueeze(1)
te_flat = te_imgs.view(len(te_imgs), -1).numpy()
pool = {c: np.where(ytr_all == c)[0] for c in range(NCLS)}
avail = np.array([len(pool[c]) for c in range(NCLS)])
print(f'{os.path.basename(NPZ)}: {len(Xtr_all)} train / {len(Xte)} test, {VOL}^3, '
      f'{NCLS} classes on {dev}; per-class train {avail.tolist()}', flush=True)

rows, t0 = [], time.time()
for N in SIZES:
    if N > avail.sum():
        print(f'  skipping n={N}: only {avail.sum()} training volumes'); continue
    GROUPS = None
    # MEDMNIST_SEED0 offsets the draw index, so a configuration selected on seeds 0-4 can
    # be evaluated on a disjoint block of fresh draws. Selection and evaluation then use
    # different data, which is what lets the reported spread be read as the spread of the
    # whole procedure rather than of one fixed basis.
    for seed in range(SEED0, SEED0 + N_SEEDS):
        torch.manual_seed(seed); np.random.seed(seed)
        take = allocate(avail, N)
        tr_idx = np.concatenate([np.random.choice(pool[c], take[c], replace=False)
                                 for c in range(NCLS) if take[c]])
        tr_imgs = torch.from_numpy(Xtr_all[tr_idx]).unsqueeze(1)
        tr_y = torch.from_numpy(ytr_all[tr_idx])
        tr_flat = tr_imgs.view(len(tr_imgs), -1).numpy()

        if BASIS in ('energy', 'energy_cat', 'dct_energy', 'pca_energy', 'pca_sq_energy'):
            Etr, Ete = band_energy(Xtr_all[tr_idx]), band_energy(Xte)
            se = StandardScaler().fit(Etr)
            if BASIS == 'energy':
                Ztr = torch.tensor(se.transform(Etr), dtype=torch.float32)
                Zte = torch.tensor(se.transform(Ete), dtype=torch.float32)
            elif BASIS == 'dct_energy':
                # Which linear base does the work beside the quadratic statistic?
                perm = rank_perm()
                Ftr = block_spectra(Xtr_all[tr_idx])[:, perm]
                Fte = block_spectra(Xte)[:, perm]
                sd = StandardScaler().fit(Ftr)
                Ztr = torch.tensor(np.hstack([sd.transform(Ftr), se.transform(Etr)]),
                                   dtype=torch.float32)
                Zte = torch.tensor(np.hstack([sd.transform(Fte), se.transform(Ete)]),
                                   dtype=torch.float32)
            elif BASIS == 'pca_sq_energy':
                # Both quadratic statistics at once, and they are not the same quantity:
                # squaring a retained coefficient is power in a direction the tokenizer
                # kept, while band energy pools power over bands it discarded. On synapse
                # the squares arm is the only one that beats the spatial baseline, and
                # band energy alone is not enough, so neither subsumes the other.
                sp = StandardScaler().fit(tr_flat)
                nc = len(tr_flat) - 1 if FULL_RANK else min(N_COMP, len(tr_flat))
                pca = PCA(n_components=max(1, nc)).fit(sp.transform(tr_flat))
                P0 = pca.transform(sp.transform(tr_flat))
                P1 = pca.transform(sp.transform(te_flat))
                Ztr = torch.tensor(np.hstack([P0, P0 ** 2, se.transform(Etr)]),
                                   dtype=torch.float32)
                Zte = torch.tensor(np.hstack([P1, P1 ** 2, se.transform(Ete)]),
                                   dtype=torch.float32)
            elif BASIS == 'pca_energy':
                sp = StandardScaler().fit(tr_flat)
                nc = len(tr_flat) - 1 if FULL_RANK else min(N_COMP, len(tr_flat))
                pca = PCA(n_components=max(1, nc)).fit(sp.transform(tr_flat))
                Ztr = torch.tensor(np.hstack([pca.transform(sp.transform(tr_flat)),
                                              se.transform(Etr)]), dtype=torch.float32)
                Zte = torch.tensor(np.hstack([pca.transform(sp.transform(te_flat)),
                                              se.transform(Ete)]), dtype=torch.float32)
            else:
                perm = rank_perm()
                Ftr = block_spectra(Xtr_all[tr_idx])[:, perm]
                Fte = block_spectra(Xte)[:, perm]
                sd = StandardScaler().fit(Ftr); sp = StandardScaler().fit(tr_flat)
                nc = len(tr_flat) - 1 if FULL_RANK else min(N_COMP, len(tr_flat))
                pca = PCA(n_components=max(1, nc)).fit(sp.transform(tr_flat))
                Ztr = torch.tensor(np.hstack([pca.transform(sp.transform(tr_flat)),
                                              sd.transform(Ftr), se.transform(Etr)]),
                                   dtype=torch.float32)
                Zte = torch.tensor(np.hstack([pca.transform(sp.transform(te_flat)),
                                              sd.transform(Fte), se.transform(Ete)]),
                                   dtype=torch.float32)
            GROUPS = None
            # The 1/i rank profile is only meaningful per basis: left to run across a
            # concatenation it decays through the first block and then keeps decaying into the
            # second, penalising the band energies for nothing but their position in the array.
            # networks.py takes rank_groups to restart the profile at each block, so the group
            # sizes are recorded here. Without this, MEDMNIST_RANK=1 on dct_energy would be
            # testing a mis-specified profile rather than the hierarchy.
            if BASIS == 'dct_energy':
                GROUPS = [Ftr.shape[1], Etr.shape[1]]
            elif BASIS == 'pca_energy':
                GROUPS = [Ztr.shape[1] - Etr.shape[1], Etr.shape[1]]
            elif BASIS == 'energy_cat':
                GROUPS = [Ztr.shape[1] - Ftr.shape[1] - Etr.shape[1], Ftr.shape[1], Etr.shape[1]]
            elif BASIS == 'pca_sq_energy':
                _np = (Ztr.shape[1] - Etr.shape[1]) // 2
                GROUPS = [_np, _np, Etr.shape[1]]
        elif BASIS in ('square', 'rbf', 'triple'):
            # Three further ways to combine linear bases nonlinearly, all keeping the
            # projections themselves linear and fixed.
            #   square  z and z^2: per-coefficient energy, the simplest second-order
            #           statistic, and the one attention cannot form at all since it mixes
            #           across tokens rather than within one.
            #   rbf     random Fourier features cos(Wz + b) on the concatenated vector:
            #           approximates an RBF kernel, so it is a nonlinear combination of
            #           every coefficient at once, yet W is drawn once from a fixed seed
            #           and never fitted, so nothing is estimated from the sample.
            #   triple  pca + disjoint DCT + overlapping DCT, to separate "more bases"
            #           from "nonlinearity" as the source of any gain.
            perm = rank_perm()
            Ftr = block_spectra(Xtr_all[tr_idx])[:, perm]
            Fte = block_spectra(Xte)[:, perm]
            sd = StandardScaler().fit(Ftr)
            sp = StandardScaler().fit(tr_flat)
            nc = len(tr_flat) - 1 if FULL_RANK else min(N_COMP, len(tr_flat))
            pca = PCA(n_components=max(1, nc)).fit(sp.transform(tr_flat))
            if BASIS == 'triple':
                g0, s0 = BLOCK, STRIDE
                globals()['BLOCK'], globals()['STRIDE'] = 14, 7
                g2 = rank_perm()
                Otr, Ote = block_spectra(Xtr_all[tr_idx])[:, g2], block_spectra(Xte)[:, g2]
                globals()['BLOCK'], globals()['STRIDE'] = g0, s0
                so = StandardScaler().fit(Otr)
            W = b = None
            if BASIS == 'rbf':
                D = int(os.environ.get('MEDMNIST_RBF', 128))
                rs = np.random.RandomState(0)

            def build(flat, F, O=None):
                P = pca.transform(sp.transform(flat)); Dd = sd.transform(F)
                Z = np.hstack([P, Dd])
                if BASIS == 'square':
                    return np.hstack([Z, Z ** 2])
                if BASIS == 'triple':
                    return np.hstack([Z, so.transform(O)])
                return np.hstack([Z, np.cos(Z @ W + b)])

            if BASIS == 'rbf':
                Z0 = np.hstack([pca.transform(sp.transform(tr_flat)), sd.transform(Ftr)])
                sig = np.median(np.abs(Z0)) + 1e-9
                W = rs.normal(0, 1.0 / (sig * np.sqrt(Z0.shape[1])), (Z0.shape[1], D))
                b = rs.uniform(0, 2 * np.pi, D)
            Ztr = torch.tensor(build(tr_flat, Ftr, Otr if BASIS == 'triple' else None),
                               dtype=torch.float32)
            Zte = torch.tensor(build(te_flat, Fte, Ote if BASIS == 'triple' else None),
                               dtype=torch.float32)
            GROUPS = None
        elif BASIS == 'cross':
            # Nonlinear COMBINATION of two linear bases, as distinct from a nonlinear
            # transform of the image. Both projections stay linear and fixed; what is added
            # is the set of pairwise products between the leading coefficients of each,
            # z_cross[i,j] = pca_i * dct_j. The encoder can only reach such interactions
            # through attention, which has to learn them from 25 volumes; writing them down
            # costs nothing to estimate. Restricted to the leading CROSS of each basis
            # because the full product is 16 x 216 and would swamp the sample.
            K = int(os.environ.get('MEDMNIST_CROSS', 8))
            perm = rank_perm()
            Ftr = block_spectra(Xtr_all[tr_idx])[:, perm]
            Fte = block_spectra(Xte)[:, perm]
            sd = StandardScaler().fit(Ftr)
            sp = StandardScaler().fit(tr_flat)
            nc = len(tr_flat) - 1 if FULL_RANK else min(N_COMP, len(tr_flat))
            pca = PCA(n_components=max(1, nc)).fit(sp.transform(tr_flat))

            def build(flat, F):
                P = pca.transform(sp.transform(flat))
                D = sd.transform(F)
                X = np.einsum('ni,nj->nij', P[:, :K], D[:, :K]).reshape(len(P), -1)
                return np.hstack([P, D, X])

            Ztr = torch.tensor(build(tr_flat, Ftr), dtype=torch.float32)
            Zte = torch.tensor(build(te_flat, Fte), dtype=torch.float32)
            GROUPS = None
        elif BASIS == 'concat':
            # Union rather than choice: standardise each basis separately, then hand the
            # model both. Selection cannot be got wrong if nothing is selected, and the
            # oracle ceiling over {pca, dct} is only +0.029 mean delta -- so if attention
            # can ignore the uninformative half, concatenation should capture most of it
            # for free. The risk runs the other way: at n=25 the best single arm measured
            # is pca24, the one with the FEWEST features, so 16+216 may simply overfit.
            # That tension is the experiment.
            perm = rank_perm()
            Ftr = block_spectra(Xtr_all[tr_idx])[:, perm]
            Fte = block_spectra(Xte)[:, perm]
            sd = StandardScaler().fit(Ftr)
            sp = StandardScaler().fit(tr_flat)
            nc = len(tr_flat) - 1 if FULL_RANK else min(N_COMP, len(tr_flat))
            pca = PCA(n_components=max(1, nc)).fit(sp.transform(tr_flat))
            Ptr = pca.transform(sp.transform(tr_flat))
            Ztr = torch.tensor(np.hstack([Ptr, sd.transform(Ftr)]), dtype=torch.float32)
            Zte = torch.tensor(np.hstack([pca.transform(sp.transform(te_flat)),
                                          sd.transform(Fte)]), dtype=torch.float32)
            GROUPS = [Ptr.shape[1], Ftr.shape[1]]
        elif BASIS == 'blockpca':
            # The data-driven twin of the block cosine transform: same blocks, same
            # coefficient count, but the within-block basis is fit rather than fixed.
            # Pooling blocks across volumes lifts the rank cap that binds whole-volume
            # PCA -- 25 volumes give 8 x 25 = 200 block-samples at side 14 -- so the
            # count is reachable at n=25 where a global PCA would be clamped to 25.
            offs, side = block_offsets(Xtr_all.shape[1])
            def tiles(V):
                return np.stack([V[:, z:z+side, y:y+side, x:x+side].reshape(len(V), -1)
                                 for z in offs for y in offs for x in offs], 1)
            Ttr, Tte = tiles(Xtr_all[tr_idx]), tiles(Xte)
            nb, dim = Ttr.shape[1], Ttr.shape[2]
            # Not 'pool': that name holds the class-to-indices dict the subsample draw
            # uses, and shadowing it here made every seed after the first fail.
            tiled = Ttr.reshape(-1, dim)
            k = min(*tiled.shape) if FULL_RANK else min(N_COMP, *tiled.shape)
            sc = StandardScaler().fit(tiled)
            pca = PCA(n_components=k, random_state=0).fit(sc.transform(tiled))
            def enc(T):
                return pca.transform(sc.transform(T.reshape(-1, dim))).reshape(len(T), nb * k)
            Ztr = torch.tensor(enc(Ttr), dtype=torch.float32)
            Zte = torch.tensor(enc(Tte), dtype=torch.float32)
        elif BASIS in ('scatter', 'dct_scatter', 'oriented', 'dct_oriented',
                       'dct_energy_scatter', 'energy_gram'):
            # Four ways of carrying more than a marginal band power, all fixed bases with
            # nothing estimated from the 25 volumes beyond the per-feature standardisation.
            Str = Gtr = None
            if 'scatter' in BASIS:
                Str, Ste = scatter_energy(Xtr_all[tr_idx]), scatter_energy(Xte)
            if 'oriented' in BASIS:
                Str, Ste = oriented_energy(Xtr_all[tr_idx]), oriented_energy(Xte)
            if 'energy' in BASIS or BASIS == 'energy_gram':
                Gtr, Gte = band_energy(Xtr_all[tr_idx]), band_energy(Xte)
            parts_tr, parts_te, names = [], [], []
            if BASIS.startswith('dct'):
                perm = rank_perm()
                Ftr = block_spectra(Xtr_all[tr_idx])[:, perm]
                Fte = block_spectra(Xte)[:, perm]
                sdd = StandardScaler().fit(Ftr)
                parts_tr.append(sdd.transform(Ftr)); parts_te.append(sdd.transform(Fte))
                names.append('dct')
            if Gtr is not None:
                sg = StandardScaler().fit(Gtr)
                parts_tr.append(sg.transform(Gtr)); parts_te.append(sg.transform(Gte))
                names.append('energy')
            if BASIS == 'energy_gram':
                # Cheap stand-in for the band interactions: the outer product of the
                # per-block band energies, pooled over blocks. Quadratic in a statistic
                # that is already quadratic, so it reaches fourth order in the signal.
                nb = n_blocks()
                def gram(E):
                    R = E.reshape(len(E), nb, -1)
                    G = np.einsum('nbi,nbj->nij', R, R) / nb
                    iu = np.triu_indices(G.shape[1])
                    return G[:, iu[0], iu[1]]
                Qtr, Qte = gram(Gtr), gram(Gte)
                sq = StandardScaler().fit(Qtr)
                parts_tr.append(sq.transform(Qtr)); parts_te.append(sq.transform(Qte))
                names.append('gram')
            if Str is not None:
                ss2 = StandardScaler().fit(Str)
                parts_tr.append(ss2.transform(Str)); parts_te.append(ss2.transform(Ste))
                names.append('scatter' if 'scatter' in BASIS else 'oriented')
            Ztr = torch.tensor(np.hstack(parts_tr), dtype=torch.float32)
            Zte = torch.tensor(np.hstack(parts_te), dtype=torch.float32)
            GROUPS = [p.shape[1] for p in parts_tr]
            print('      %s: ' % BASIS + ' '.join(
                '%s %d' % (n, p.shape[1]) for n, p in zip(names, parts_tr)), flush=True)
        elif BASIS.startswith('pool'):
            # Pool the tokens of every basis considered and select WITHIN the pool, rather
            # than choosing one basis. The motivation is that arm selection kept failing
            # for a reason that pooling sidesteps: ranking eight arms from 25 labels is a
            # choice between whole feature sets, and the measured criteria (linear probe
            # r=+0.34, mutual information r<=+0.08) cannot resolve it. Pooling makes no
            # choice at all, and selection inside the pool is then a question about
            # features rather than about bases.
            #
            # The pool is also a falsifiable test of whether selection can work at this n.
            # The nested-pair analysis says band energy helps wherever it is added
            # (dct -> dct_energy +0.039, p<0.001) and the squares hurt wherever they are
            # added (pca_energy -> pca_sq_energy -0.064, p<0.001). So the pool contains one
            # block known to help and one known to hurt. Any selector worth using has to
            # rediscover that from the 25 training volumes alone; one that keeps the
            # squares and drops the energies has told us selection is out of reach here.
            SELECT = os.environ.get('MEDMNIST_SELECT', 'none')
            SELK = int(os.environ.get('MEDMNIST_SELK', 312))

            sp = StandardScaler().fit(tr_flat)
            nc = len(tr_flat) - 1 if FULL_RANK else min(N_COMP, len(tr_flat))
            pca = PCA(n_components=max(1, nc), random_state=0).fit(sp.transform(tr_flat))
            Ptr = pca.transform(sp.transform(tr_flat))
            Pte = pca.transform(sp.transform(te_flat))
            perm = rank_perm()
            Ftr = block_spectra(Xtr_all[tr_idx])[:, perm]
            Fte = block_spectra(Xte)[:, perm]
            sd = StandardScaler().fit(Ftr)
            Dtr, Dte = sd.transform(Ftr), sd.transform(Fte)
            Etr, Ete = band_energy(Xtr_all[tr_idx]), band_energy(Xte)
            se = StandardScaler().fit(Etr)
            Gtr, Gte = se.transform(Etr), se.transform(Ete)

            # Overlapping cosine windows and block PCA both live on the stride-7 grid, so
            # the globals are swapped once for both and restored immediately after.
            g0, s0 = BLOCK, STRIDE
            globals()['BLOCK'], globals()['STRIDE'] = 14, 7
            pov = rank_perm()
            Otr = block_spectra(Xtr_all[tr_idx])[:, pov]
            Ote = block_spectra(Xte)[:, pov]
            offs, side = block_offsets(Xtr_all.shape[1])
            globals()['BLOCK'], globals()['STRIDE'] = g0, s0
            so = StandardScaler().fit(Otr)
            Vtr, Vte = so.transform(Otr), so.transform(Ote)

            def tiles(V):
                return np.stack([V[:, z:z+side, y:y+side, x:x+side].reshape(len(V), -1)
                                 for z in offs for y in offs for x in offs], 1)
            Ttr, Tte = tiles(Xtr_all[tr_idx]), tiles(Xte)
            nb, dim = Ttr.shape[1], Ttr.shape[2]
            tiled = Ttr.reshape(-1, dim)
            kb = min(N_COMP or min(*tiled.shape), *tiled.shape)
            sb = StandardScaler().fit(tiled)
            bp = PCA(n_components=kb, random_state=0).fit(sb.transform(tiled))
            def enc(T):
                return bp.transform(sb.transform(T.reshape(-1, dim))).reshape(len(T), nb * kb)
            Btr, Bte = enc(Ttr), enc(Tte)

            # Cross terms and squares are the quadratic blocks; both are built from the
            # already-standardised pca and dct coefficients so their scale is comparable.
            K = int(os.environ.get('MEDMNIST_CROSS', 8))
            def xprod(P, D):
                return np.einsum('ni,nj->nij', P[:, :K], D[:, :K]).reshape(len(P), -1)
            Xtr_c, Xte_c = xprod(Ptr, Dtr), xprod(Pte, Dte)
            Qtr = np.hstack([Ptr, Dtr]) ** 2
            Qte = np.hstack([Pte, Dte]) ** 2
            RD = int(os.environ.get('MEDMNIST_RBF', 128))
            rs = np.random.RandomState(0)
            Z0 = np.hstack([Ptr, Dtr])
            sig = np.median(np.abs(Z0)) + 1e-9
            Wr = rs.normal(0, 1.0 / (sig * np.sqrt(Z0.shape[1])), (Z0.shape[1], RD))
            br = rs.uniform(0, 2 * np.pi, RD)
            Rtr = np.cos(Z0 @ Wr + br)
            Rte = np.cos(np.hstack([Pte, Dte]) @ Wr + br)

            BLK = [('pca', Ptr, Pte), ('dct', Dtr, Dte), ('dct_ov', Vtr, Vte),
                   ('blockpca', Btr, Bte), ('energy', Gtr, Gte), ('cross', Xtr_c, Xte_c),
                   ('square', Qtr, Qte), ('rbf', Rtr, Rte)]
            Atr = np.hstack([b[1] for b in BLK]).astype(np.float64)
            Ate = np.hstack([b[2] for b in BLK]).astype(np.float64)
            owner = np.concatenate([[i] * b[1].shape[1] for i, b in enumerate(BLK)])
            ytr = tr_y.numpy()

            # Every selector below sees the 25 training rows and their labels only. The
            # MedMNIST test split is transformed by the objects fitted above and then
            # subset by the indices the selector returns; it is never scored or ranked.
            idx, proj = np.arange(Atr.shape[1]), None
            k = min(SELK, Atr.shape[1])
            if SELECT == 'fscore':
                from sklearn.feature_selection import f_classif
                sc = np.nan_to_num(f_classif(Atr, ytr)[0])
                idx = np.sort(np.argsort(-sc)[:k])
            elif SELECT == 'mi':
                from sklearn.feature_selection import mutual_info_classif
                sc = mutual_info_classif(Atr, ytr, random_state=0, n_neighbors=3)
                idx = np.sort(np.argsort(-sc)[:k])
            elif SELECT == 'mrmr':
                # Max relevance, min redundancy. Summed per-feature MI failed as an arm
                # criterion precisely because it ignores redundancy and so rewards whichever
                # arm has the most features; inside a pool that failure mode is addressable,
                # because redundancy against the already-selected set can be subtracted.
                from sklearn.feature_selection import mutual_info_classif
                rel = mutual_info_classif(Atr, ytr, random_state=0, n_neighbors=3)
                C = np.abs(np.nan_to_num(np.corrcoef(Atr, rowvar=False)))
                pick = [int(np.argmax(rel))]
                while len(pick) < k:
                    red = C[:, pick].mean(1)
                    cand = rel - red
                    cand[pick] = -np.inf
                    pick.append(int(np.argmax(cand)))
                idx = np.sort(np.array(pick))
            elif SELECT == 'stab':
                # Stability selection: the standard answer to selection at tiny n. An L1
                # logistic fit on 25 rows is pure variance, so it is refit on bootstrap
                # resamples and features are ranked by how often they survive.
                from sklearn.linear_model import LogisticRegression
                B = int(os.environ.get('MEDMNIST_BOOT', 100))
                cnt = np.zeros(Atr.shape[1])
                rb = np.random.RandomState(seed)
                for _ in range(B):
                    bi = rb.choice(len(Atr), len(Atr), replace=True)
                    if len(np.unique(ytr[bi])) < 2:
                        continue
                    lr = LogisticRegression(penalty='l1', C=0.1, solver='liblinear',
                                            max_iter=2000)
                    try:
                        lr.fit(Atr[bi], ytr[bi])
                    except Exception:
                        continue
                    cnt += (np.abs(lr.coef_).max(0) > 1e-8)
                idx = np.sort(np.argsort(-cnt)[:k]) if cnt.any() else idx
            elif SELECT == 'group':
                # Coarse selection: keep or drop whole blocks, so the decision is eight
                # binary choices instead of 1600. Each block is scored by cross-validated
                # logistic AUC inside the 25 volumes, with the PCA and scalers already
                # fitted above; blocks at or above the median score are kept. If the
                # nested-pair structure is learnable at all, it should be learnable here.
                from sklearn.linear_model import LogisticRegression
                from sklearn.model_selection import RepeatedStratifiedKFold
                from sklearn.metrics import roc_auc_score as _auc
                mn = np.bincount(ytr).min()
                cv = RepeatedStratifiedKFold(n_splits=min(5, mn), n_repeats=4,
                                             random_state=0) if mn >= 2 else None
                sc = np.zeros(len(BLK))
                for i, (nm, tb, _) in enumerate(BLK):
                    if cv is None:
                        sc[i] = 0.5; continue
                    aucs = []
                    for a, b in cv.split(tb, ytr):
                        if len(np.unique(ytr[a])) < 2:
                            continue
                        lr = LogisticRegression(max_iter=2000, C=1.0).fit(tb[a], ytr[a])
                        pr = lr.predict_proba(tb[b])
                        try:
                            aucs.append(_auc(ytr[b], pr[:, 1] if pr.shape[1] == 2 else pr,
                                             multi_class='ovr',
                                             labels=np.arange(NCLS)) if pr.shape[1] > 2
                                        else _auc(ytr[b], pr[:, 1]))
                        except Exception:
                            continue
                    sc[i] = np.mean(aucs) if aucs else 0.5
                keep = np.where(sc >= np.median(sc))[0]
                idx = np.sort(np.where(np.isin(owner, keep))[0])
                print('      group scores ' + ' '.join(
                    '%s=%.3f%s' % (BLK[i][0], sc[i], '*' if i in keep else '')
                    for i in range(len(BLK))), flush=True)
            elif SELECT == 'cons':
                # Selection moved from inference time to DESIGN time. The group selector's
                # per-cell decisions are too noisy to pay off (it keeps the squares in 23%
                # of cells and drops the band energies in 27%), but its ORDERING over
                # blocks, pooled across all 30 cells, reproduces the nested-pair ground
                # truth exactly: energy and the cosine blocks on top, squares at the
                # bottom. So fix the blocks it prefers in aggregate and select nothing per
                # draw. This consumes no labels at run time; it is a design choice informed
                # by the whole experiment, and has to be reported as such rather than as a
                # selector -- the leave-one-dataset-out column is the honest version.
                keep = [i for i, b in enumerate(BLK)
                        if b[0] in ('dct', 'dct_ov', 'blockpca', 'energy')]
                idx = np.sort(np.where(np.isin(owner, keep))[0])
            elif SELECT == 'cons_lite':
                # The same idea restricted to the two blocks that the nested pairs show to
                # help on their own, so the comparison against dct_energy differs only in
                # the overlapping windows.
                keep = [i for i, b in enumerate(BLK) if b[0] in ('dct', 'dct_ov', 'energy')]
                idx = np.sort(np.where(np.isin(owner, keep))[0])
            elif SELECT == 'pcared':
                # Label-free reduction of the pool. If the 25 labels are the bottleneck
                # rather than the estimator, a selector that never looks at them is the
                # fair control.
                ss = StandardScaler().fit(Atr)
                proj = PCA(n_components=min(k, len(Atr) - 1), random_state=0).fit(
                    ss.transform(Atr))
                Ztr = torch.tensor(proj.transform(ss.transform(Atr)), dtype=torch.float32)
                Zte = torch.tensor(proj.transform(ss.transform(Ate)), dtype=torch.float32)
            if proj is None:
                Ztr = torch.tensor(Atr[:, idx], dtype=torch.float32)
                Zte = torch.tensor(Ate[:, idx], dtype=torch.float32)
                kept = [int((owner[idx] == i).sum()) for i in range(len(BLK))]
                GROUPS = [c for c in kept if c]
                print('      pool %d -> %d via %s | ' % (Atr.shape[1], len(idx), SELECT)
                      + ' '.join('%s %d/%d' % (BLK[i][0], kept[i], BLK[i][1].shape[1])
                                 for i in range(len(BLK))), flush=True)
            else:
                GROUPS = None
        elif BASIS == 'blockpca_energy':
            # The untested concatenation. Every pairing with band energy so far used a
            # fixed basis (dct_energy) or whole-volume PCA (pca_energy); the per-block PCA
            # was never combined with it, even though block PCA is the strongest single arm
            # on exactly the two datasets the method is weakest on (fracture 0.573 best of
            # 89, vessel 0.721) and band energy is the component that helps wherever it is
            # added (+0.039 onto dct p<0.001, +0.024 onto pca p=0.002).
            Etr, Ete = band_energy(Xtr_all[tr_idx]), band_energy(Xte)
            se = StandardScaler().fit(Etr)
            offs, side = block_offsets(Xtr_all.shape[1])

            def tiles(V):
                return np.stack([V[:, z:z+side, y:y+side, x:x+side].reshape(len(V), -1)
                                 for z in offs for y in offs for x in offs], 1)
            Ttr, Tte = tiles(Xtr_all[tr_idx]), tiles(Xte)
            nb, dim = Ttr.shape[1], Ttr.shape[2]
            tiled = Ttr.reshape(-1, dim)
            kb = min(*tiled.shape) if FULL_RANK else min(N_COMP, *tiled.shape)
            sb = StandardScaler().fit(tiled)
            bp = PCA(n_components=kb, random_state=0).fit(sb.transform(tiled))

            def enc(T):
                return bp.transform(sb.transform(T.reshape(-1, dim))).reshape(len(T), nb * kb)
            Ztr = torch.tensor(np.hstack([enc(Ttr), se.transform(Etr)]), dtype=torch.float32)
            Zte = torch.tensor(np.hstack([enc(Tte), se.transform(Ete)]), dtype=torch.float32)
            GROUPS = [nb * kb, Etr.shape[1]]
        elif BASIS in ('pls', 'pls_energy'):
            # The one selection criterion absent from every other basis here. PCA ranks
            # directions by VARIANCE, the cosine transform by FIXED FREQUENCY, band energy
            # by POWER -- none by class discriminability, and the two cohorts the method
            # fails on are exactly those whose class signal is diffuse and low-amplitude
            # (fracture d_max 0.31 conc 0.339, synapse 0.32 / 0.319), which is the kind a
            # variance ordering discards. Partial least squares ranks directions by
            # covariance with the label instead, so it is the missing family.
            #
            # It is also the first SUPERVISED tokenizer tried, so it is the one place a
            # leak could enter: the decomposition is fitted on the inner-training rows and
            # their labels only, and the held-out rows are transformed by it, never used to
            # fit it. At n=25 with 20 inner rows this will overfit; whether the extra
            # relevance outweighs that is the experiment.
            from sklearn.cross_decomposition import PLSRegression
            nc = max(1, min(N_COMP or 16, len(tr_flat) - 2))
            sp = StandardScaler().fit(tr_flat)
            Y = np.eye(NCLS)[tr_y.numpy()] if NCLS > 2 else tr_y.numpy().astype(float)
            pls = PLSRegression(n_components=nc, scale=False).fit(sp.transform(tr_flat), Y)
            Ptr = pls.transform(sp.transform(tr_flat))
            Pte = pls.transform(sp.transform(te_flat))
            ss = StandardScaler().fit(Ptr)
            if BASIS == 'pls_energy':
                Etr, Ete = band_energy(Xtr_all[tr_idx]), band_energy(Xte)
                se = StandardScaler().fit(Etr)
                Ztr = torch.tensor(np.hstack([ss.transform(Ptr), se.transform(Etr)]),
                                   dtype=torch.float32)
                Zte = torch.tensor(np.hstack([ss.transform(Pte), se.transform(Ete)]),
                                   dtype=torch.float32)
                GROUPS = [Ptr.shape[1], Etr.shape[1]]
            else:
                Ztr = torch.tensor(ss.transform(Ptr), dtype=torch.float32)
                Zte = torch.tensor(ss.transform(Pte), dtype=torch.float32)
        elif BASIS in ('topo', 'topo_energy', 'topo_dct'):
            Ttr, Tte = topo_features(Xtr_all[tr_idx]), topo_features(Xte)
            st = StandardScaler().fit(Ttr)
            parts_tr, parts_te = [st.transform(Ttr)], [st.transform(Tte)]
            if BASIS == 'topo_energy':
                Etr, Ete = band_energy(Xtr_all[tr_idx]), band_energy(Xte)
                se2 = StandardScaler().fit(Etr)
                parts_tr.append(se2.transform(Etr)); parts_te.append(se2.transform(Ete))
            elif BASIS == 'topo_dct':
                perm = rank_perm()
                Ftr = block_spectra(Xtr_all[tr_idx])[:, perm]
                Fte = block_spectra(Xte)[:, perm]
                sd2 = StandardScaler().fit(Ftr)
                parts_tr.append(sd2.transform(Ftr)); parts_te.append(sd2.transform(Fte))
            Ztr = torch.tensor(np.hstack(parts_tr), dtype=torch.float32)
            Zte = torch.tensor(np.hstack(parts_te), dtype=torch.float32)
            GROUPS = [p.shape[1] for p in parts_tr] if len(parts_tr) > 1 else None
        elif BASIS in ('absfft', 'absfft_energy', 'absfft_pca'):
            Mtr, Mte = block_mag(Xtr_all[tr_idx]), block_mag(Xte)
            sm = StandardScaler().fit(Mtr)
            parts_tr, parts_te = [sm.transform(Mtr)], [sm.transform(Mte)]
            if BASIS == 'absfft_pca':
                # whole-volume magnitude spectrum, reduced -- the probe used 64 PCs of the
                # full |c| vector, so this is the closest tokenizer to what it measured
                At = np.abs(np.stack([fft.dctn(v, type=2, norm='ortho').ravel()
                                      for v in Xtr_all[tr_idx]]))
                Ae = np.abs(np.stack([fft.dctn(v, type=2, norm='ortho').ravel()
                                      for v in Xte]))
                sa = StandardScaler().fit(At)
                nc = max(1, min(N_COMP or 16, len(At) - 1))
                pa = PCA(n_components=nc, random_state=0).fit(sa.transform(At))
                parts_tr = [pa.transform(sa.transform(At))]
                parts_te = [pa.transform(sa.transform(Ae))]
            elif BASIS == 'absfft_energy':
                Etr, Ete = band_energy(Xtr_all[tr_idx]), band_energy(Xte)
                se3 = StandardScaler().fit(Etr)
                parts_tr.append(se3.transform(Etr)); parts_te.append(se3.transform(Ete))
            Ztr = torch.tensor(np.hstack(parts_tr), dtype=torch.float32)
            Zte = torch.tensor(np.hstack(parts_te), dtype=torch.float32)
            GROUPS = [p.shape[1] for p in parts_tr] if len(parts_tr) > 1 else None
        elif BASIS in ('magblockpca', 'magblockpca_energy'):
            # Magnitude spectra with the rank cap lifted. The diagnosis for fracture is
            # that its class signal is in the MAGNITUDE spectrum (full-split probe: 0.672
            # on |c| against 0.560 signed) but that estimating a magnitude basis from 25
            # volumes caps PCA at n-1 = 24 components where the probe used 64, so
            # absfft_pca reaches only 0.575 of the 0.672 available.
            #
            # Fitting the decomposition on per-block magnitude spectra pooled ACROSS blocks
            # turns 25 volumes into 25 x nb block-samples, which is the same trick block
            # PCA uses on signed voxels -- and no magnitude basis here has used it. If the
            # diagnosis is right this should recover part of the gap; if it does not, the
            # limit is not the rank cap.
            g0, s0 = BLOCK, STRIDE
            offs, side = block_offsets(Xtr_all.shape[1])
            nbk = len(offs) ** 3

            def magtiles(V):
                out = np.zeros((len(V), nbk, FREQ ** 3), dtype=np.float32)
                for i in range(len(V)):
                    bi = 0
                    for z in offs:
                        for y in offs:
                            for x in offs:
                                m = np.abs(fft.fftn(V[i, z:z+side, y:y+side, x:x+side]))
                                out[i, bi] = np.log1p(m[:FREQ, :FREQ, :FREQ].ravel())
                                bi += 1
                return out
            Ttr, Tte = magtiles(Xtr_all[tr_idx]), magtiles(Xte)
            dim = Ttr.shape[2]
            pooled = Ttr.reshape(-1, dim)
            kk = min(N_COMP or min(*pooled.shape), *pooled.shape)
            sbm = StandardScaler().fit(pooled)
            pbm = PCA(n_components=kk, random_state=0).fit(sbm.transform(pooled))

            def encm(T):
                return pbm.transform(sbm.transform(T.reshape(-1, dim))).reshape(len(T), nbk * kk)
            parts_tr, parts_te = [encm(Ttr)], [encm(Tte)]
            if BASIS == 'magblockpca_energy':
                Etr, Ete = band_energy(Xtr_all[tr_idx]), band_energy(Xte)
                se4 = StandardScaler().fit(Etr)
                parts_tr.append(se4.transform(Etr)); parts_te.append(se4.transform(Ete))
            Ztr = torch.tensor(np.hstack(parts_tr), dtype=torch.float32)
            Zte = torch.tensor(np.hstack(parts_te), dtype=torch.float32)
            GROUPS = [p.shape[1] for p in parts_tr] if len(parts_tr) > 1 else None
        elif BASIS in ('moment', 'moment_energy', 'moment_dct'):
            Mo_tr, Mo_te = moment_features(Xtr_all[tr_idx]), moment_features(Xte)
            smo = StandardScaler().fit(Mo_tr)
            # A marginal profile is near constant at the volume edges, so after standardising
            # on 25 volumes a test volume can land 20 SD out on one coordinate where the
            # training draw spans +-5, and the encoder then returns non-finite logits.
            # Clipping bounds the input without touching the ordering; every other basis
            # here is already log compressed, which does the same job implicitly.
            parts_tr = [np.clip(smo.transform(Mo_tr), -5, 5)]
            parts_te = [np.clip(smo.transform(Mo_te), -5, 5)]
            if BASIS == 'moment_energy':
                Etr, Ete = band_energy(Xtr_all[tr_idx]), band_energy(Xte)
                se5 = StandardScaler().fit(Etr)
                parts_tr.append(se5.transform(Etr)); parts_te.append(se5.transform(Ete))
            elif BASIS == 'moment_dct':
                perm = rank_perm()
                Ftr = block_spectra(Xtr_all[tr_idx])[:, perm]
                Fte = block_spectra(Xte)[:, perm]
                sd5 = StandardScaler().fit(Ftr)
                parts_tr.append(sd5.transform(Ftr)); parts_te.append(sd5.transform(Fte))
            Ztr = torch.tensor(np.hstack(parts_tr), dtype=torch.float32)
            Zte = torch.tensor(np.hstack(parts_te), dtype=torch.float32)
            GROUPS = [p.shape[1] for p in parts_tr] if len(parts_tr) > 1 else None
        elif BASIS in ('haar', 'haar_signed', 'haar_energy', 'haar_dct'):
            Ha_tr = haar_features(Xtr_all[tr_idx], signed=(BASIS == 'haar_signed'))
            Ha_te = haar_features(Xte, signed=(BASIS == 'haar_signed'))
            sha = StandardScaler().fit(Ha_tr)
            parts_tr = [np.clip(sha.transform(Ha_tr), -5, 5)]
            parts_te = [np.clip(sha.transform(Ha_te), -5, 5)]
            if BASIS == 'haar_energy':
                Etr, Ete = band_energy(Xtr_all[tr_idx]), band_energy(Xte)
                se6 = StandardScaler().fit(Etr)
                parts_tr.append(se6.transform(Etr)); parts_te.append(se6.transform(Ete))
            elif BASIS == 'haar_dct':
                perm = rank_perm()
                Ftr = block_spectra(Xtr_all[tr_idx])[:, perm]
                Fte = block_spectra(Xte)[:, perm]
                sd6 = StandardScaler().fit(Ftr)
                parts_tr.append(sd6.transform(Ftr)); parts_te.append(sd6.transform(Fte))
            Ztr = torch.tensor(np.hstack(parts_tr), dtype=torch.float32)
            Zte = torch.tensor(np.hstack(parts_te), dtype=torch.float32)
            GROUPS = [p.shape[1] for p in parts_tr] if len(parts_tr) > 1 else None
        elif BASIS == 'patchdct':
            Ztr = torch.tensor(patch_dct(Xtr_all[tr_idx]), dtype=torch.float32)
            Zte = torch.tensor(patch_dct(Xte), dtype=torch.float32)
            GROUPS = None
        elif BASIS == 'dct':
            # The basis is fixed, so nothing is fit on the training split except the
            # per-feature standardisation, which matches the volumetric experiments.
            perm = rank_perm()
            Ftr = block_spectra(Xtr_all[tr_idx])[:, perm]
            Fte = block_spectra(Xte)[:, perm]
            sc = StandardScaler().fit(Ftr)
            Ztr = torch.tensor(sc.transform(Ftr), dtype=torch.float32)
            Zte = torch.tensor(sc.transform(Fte), dtype=torch.float32)
        else:
            sc = StandardScaler().fit(tr_flat)
            nc = len(tr_flat) - 1 if FULL_RANK else min(N_COMP, len(tr_flat))
            pca = PCA(n_components=max(1, nc)).fit(sc.transform(tr_flat))
            Ztr = torch.tensor(pca.transform(sc.transform(tr_flat)), dtype=torch.float32)
            Zte = torch.tensor(pca.transform(sc.transform(te_flat)), dtype=torch.float32)

        # The 1/i hierarchy is off by default here, as it was for every MedMNIST result
        # already computed. MEDMNIST_RANK=1 turns it on; with a concatenated basis it
        # restarts per basis, since the two orderings are not commensurable.
        rk = os.environ.get('MEDMNIST_RANK') == '1'
        groups = GROUPS if (rk and (BASIS == 'concat' or BASIS.startswith('pool') or GROUPS)) else None
        if BASIS == 'patchdct':
            # chunk_size = coefficients per patch, mode weights off, so this is
            # 343 tokens through one shared Linear(PK, embed) -- the spatial arm's
            # own projection, applied to a rotation of the same voxels.
            spec = MultiClassSpectralViT(
                n_inputs=Ztr.shape[1], chunk_size=int(os.environ.get('MEDMNIST_PK', PATCH ** 3)),
                num_classes=NCLS, use_mode_weights=False).to(dev)
        else:
            spec = MultiClassSpectralViT(n_inputs=Ztr.shape[1], num_classes=NCLS,
                                         use_mode_weights=True, use_rank_weights=rk,
                                         rank_groups=groups,
                                         use_pos_embed=_POS).to(dev)
        # The spatial arm's input does not depend on the tokenizer, so its score should
        # not either -- but building `spec` first consumes a different amount of RNG for
        # every coefficient count, which changed the spatial initialisation and the batch
        # order with it. Its test score then moved by SD 0.032 between runs whose spatial
        # input was identical, and that noise entered every spectral-minus-spatial
        # contrast. Reseeding here, and giving the loader its own generator, makes the
        # spatial arm bit-identical across tokenizers at fixed (dataset, n, seed).
        torch.manual_seed(seed)
        spat = MultiClassSpatialViT(num_classes=NCLS, size=VOL, patch_size=PATCH,
                                    in_channels=1, volumetric=True).to(dev)
        # Trained in separate loops, each from its own seeded stream. Interleaved in one
        # loop they share the global RNG: both encoders use dropout(0.1), so the spectral
        # arm's draws at step t shift the state the spatial arm's dropout samples at step
        # t+1, and the spatial score then moved with the tokenizer even though its input
        # never changed. AdamW is per-parameter, so splitting one optimizer over disjoint
        # parameter sets into two optimizers is the same computation -- only the RNG
        # stream and the ordering change.
        def train(model, tensors, sd):
            torch.manual_seed(sd)
            o = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
            ld = DataLoader(TensorDataset(*tensors, tr_y), batch_size=BATCH, shuffle=True,
                            generator=torch.Generator().manual_seed(seed))
            model.train()
            for _ in range(EPOCHS):
                for bx, by in ld:
                    o.zero_grad()
                    F.cross_entropy(model(bx.to(dev)), by.to(dev)).backward()
                    o.step()

        # The spatial arm's input does not depend on the tokenizer, so its predictions are
        # computed once per (dataset, n, seed) and cached. Chasing bitwise determinism
        # through cudnn flags did not hold -- an identical rerun still moved its top-1 by
        # up to 0.003 -- and that noise sat inside every spectral-minus-spatial contrast,
        # making the comparison paired in name only. Reusing one fit makes the spatial
        # column literally the same number across arms, which is what lets close
        # tokenizers separate. Delete the cache directory to recompute.
        # The key must carry the patch size, or varying MEDMNIST_PATCH silently reuses the
        # patch-4 predictions. The suffix is empty at PATCH=4 so the 60 entries already
        # cached -- and every paired result computed against them -- remain valid.
        _ps = '' if PATCH == 4 else f'_p{PATCH}'
        ck = os.path.join(SPAT_CACHE,
                          f'{os.path.basename(NPZ)[:-4]}_n{int(take.sum())}_s{seed}{_ps}.npy')
        if os.path.exists(ck) and not OCCLUDE:
            pp_arr = np.load(ck)
        else:
            train(spat, (tr_imgs,), seed)
            spat.eval()
            with torch.no_grad():
                pp_arr = torch.cat([F.softmax(spat(c.to(dev)), 1).cpu()
                                    for c in torch.split(te_imgs, 256)]).numpy()
            os.makedirs(SPAT_CACHE, exist_ok=True)
            np.save(ck, pp_arr)
        train(spec, (Ztr,), seed + 1_000_000)

        spec.eval()
        ps = []
        with torch.no_grad():
            for bz in torch.split(Zte, 256):
                ps.append(F.softmax(spec(bz.to(dev)), 1).cpu())
        pp = [torch.from_numpy(pp_arr)]
        if OCCLUDE:
            # Saved before metrics(): an occlusion batch is one volume repeated, so every
            # label is identical and roc_auc_score would raise.
            np.savez_compressed(os.path.join(OUT, 'occlusion_probs.npz'),
                                spec=torch.cat(ps).numpy(), spat=torch.cat(pp).numpy(),
                                labels=yte, comp=int(Ztr.shape[1]), patch=PATCH)
            print(f'  wrote {OUT}/occlusion_probs.npz '
                  f'({len(yte)} volumes, {Ztr.shape[1]} features)', flush=True)
            sys.exit(0)
        ms = metrics(torch.cat(ps).numpy(), yte, NCLS)
        mp = metrics(torch.cat(pp).numpy(), yte, NCLS)
        rows.append(dict(n=int(take.sum()), requested=N, seed=seed, ncls=NCLS,
                         comp=int(Ztr.shape[1]),
                         np_spec=sum(q.numel() for q in spec.parameters()),
                         np_spat=sum(q.numel() for q in spat.parameters()),
                         **{f'spec_{k}': v for k, v in ms.items()},
                         **{f'spat_{k}': v for k, v in mp.items()}))
        print(f'  n={int(take.sum()):4d} seed {seed}: spec T1 {ms["T1"]:.3f} '
              f'spat T1 {mp["T1"]:.3f} | AUC {ms["AUC"]:.3f}/{mp["AUC"]:.3f} '
              f'({time.time()-t0:.0f}s)', flush=True)
        del spec, spat
        if dev.type == 'cuda':
            torch.cuda.empty_cache()
        tmp = os.path.join(OUT, 'medmnist_perseed.csv.tmp')
        pd.DataFrame(rows).to_csv(tmp, index=False)
        os.replace(tmp, os.path.join(OUT, 'medmnist_perseed.csv'))

print(f'\nwrote {OUT}/medmnist_perseed.csv ({len(rows)} cells)', flush=True)
