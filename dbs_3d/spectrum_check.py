"""Compare the MSW and CHH spectral content of the SN/STN crops.

MSW is native 0.5 mm; CHH is acquired at 0.9 mm and interpolated onto the MSW
grid. If that is what limits cross-site transport, CHH power should fall away
from MSW above the frequency CHH actually resolves -- a knee near

    f_cut / f_nyquist = 0.5 / 0.9 = 0.556

This uses no outcome labels, so any cutoff it suggests is available a priori.
"""
import sys, os
SPDIR = os.path.dirname(os.path.abspath(__file__))
REPO  = os.path.dirname(SPDIR)
OUT   = os.path.join(REPO, 'docs', 'dbs_3d')
DATA  = os.environ.get('DBS_CLINICAL_DIR', os.path.join(REPO, 'docs'))
CACHE = os.environ.get('DBS_CACHE_DIR', '/checkpoints')
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, REPO); sys.path.insert(0, SPDIR); os.chdir(REPO)
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from scipy.fft import fftn, fftshift
exec(open(os.path.join(SPDIR, 'concat_attn.py')).read().split("rows=[]")[0])

MSW_RES, CHH_RES = 0.5, 0.9
PRED_CUT = MSW_RES / CHH_RES

cbA = VA[:, BB[0], BB[1], BB[2]]
cbB = VB[:, BB[0], BB[1], BB[2]]
shape = cbA.shape[1:]
print(f"crops: MSW {cbA.shape}  CHH {cbB.shape}   (bbox {shape})")


def radial_power(X, nb=12):
    """mean radial power spectrum, frequency normalised to Nyquist"""
    P = np.abs(fftshift(fftn(X, axes=(1, 2, 3)), axes=(1, 2, 3))) ** 2
    g = np.mgrid[:shape[0], :shape[1], :shape[2]].astype(float)
    c = [(s - 1) / 2 for s in shape]
    # normalise each axis by its own Nyquist so the radius is in cycles/Nyquist
    r = np.sqrt(sum(((g[d] - c[d]) / (shape[d] / 2.0)) ** 2 for d in range(3)) / 3.0)
    edges = np.linspace(0, 1, nb + 1)
    out = np.zeros((len(X), nb))
    for b in range(nb):
        m = (r >= edges[b]) & (r < edges[b + 1])
        if m.sum() == 0: continue
        out[:, b] = P[:, m].mean(1)
    return out, 0.5 * (edges[:-1] + edges[1:])


pa, f = radial_power(cbA)
pb, _ = radial_power(cbB)
# normalise each subject by its own total power so overall scaling cancels
pa = pa / pa.sum(1, keepdims=True)
pb = pb / pb.sum(1, keepdims=True)
ma, mb = pa.mean(0), pb.mean(0)
ratio = mb / (ma + 1e-30)

print(f"\npredicted CHH cutoff at f/f_nyq = {PRED_CUT:.3f}\n")
print(f"{'f/f_nyq':>8}  {'MSW power':>11}  {'CHH power':>11}  {'CHH/MSW':>9}")
for i in range(len(f)):
    flag = '   <-- predicted cutoff' if i > 0 and f[i-1] < PRED_CUT <= f[i] else ''
    print(f"{f[i]:8.3f}  {ma[i]:11.3e}  {mb[i]:11.3e}  {ratio[i]:9.3f}{flag}")

lo = f < PRED_CUT
print(f"\nmean CHH/MSW power ratio below the predicted cutoff: {ratio[lo].mean():.3f}")
print(f"mean CHH/MSW power ratio above it:                   {ratio[~lo].mean():.3f}")
drop = np.argmax(ratio < 0.5) if (ratio < 0.5).any() else -1
if drop > 0:
    print(f"first band where CHH holds <50% of MSW power: f/f_nyq = {f[drop]:.3f}")
else:
    print("CHH never falls below 50% of MSW power in these bands")

# per-axis, since the resample is isotropic but the grid is not
print("\nper-axis 1-D spectra (normalised power beyond half-Nyquist):")
for d, nm in enumerate('xyz'):
    ax = tuple(i for i in (1, 2, 3) if i != d + 1)
    sa = np.abs(fftn(cbA.mean(axis=ax), axes=(1,))) ** 2
    sb = np.abs(fftn(cbB.mean(axis=ax), axes=(1,))) ** 2
    sa, sb = sa / sa.sum(1, keepdims=True), sb / sb.sum(1, keepdims=True)
    n = sa.shape[1]; half = slice(n // 4, n // 2)
    print(f"  {nm}: MSW {sa[:, half].sum(1).mean():.4f}   CHH {sb[:, half].sum(1).mean():.4f}"
          f"   ratio {sb[:, half].sum(1).mean() / (sa[:, half].sum(1).mean() + 1e-30):.3f}")

pd.DataFrame(dict(f=f, msw=ma, chh=mb, ratio=ratio)).to_csv(
    os.path.join(OUT, 'spectrum_check.csv'), index=False)
print(f"\nwrote {os.path.join(OUT, 'spectrum_check.csv')}")
