"""The same brain at the test-time SNRs of the distribution-shift experiment.

usage: python figures/make_fig_snr_shift.py [out.png]"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repository root: shared modules and paths.py
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

args = [a for a in sys.argv[1:] if not a.startswith('--')]
OUT = args[0] if args else 'paper/cvpr/fig/snr_shift.png'
CACHE = os.environ.get('SNRFIG_CACHE', '/checkpoints/data/IXI_cached_npy')
SUBJ = os.environ.get('SNRFIG_SUBJ', '')
AXIS = int(os.environ.get('SNRFIG_AXIS', 2))
SEED = int(os.environ.get('SNRFIG_SEED', 0))
ALPHA = float(os.environ.get('SNRFIG_ALPHA', 0.75))
SNRS = [None, 9, 4, 1, 0.25, 0.111]
LABELS = ['Clean', 'SNR 9', 'SNR 4', 'SNR 1', 'SNR 1/4', 'SNR 1/9']


def rician(x, snr, rng):
    if snr is None: return x
    xm = x.min(); xp = x - xm
    sigma = np.mean(xp[xp > 0]) / snr
    n1 = rng.normal(0, sigma, x.shape); n2 = rng.normal(0, sigma, x.shape)
    xn = np.sqrt((xp + n1) ** 2 + n2 ** 2)
    return ((xn - np.mean(xn)) / (np.std(xn) + 1e-8)).astype(np.float32)


files = sorted(f for f in os.listdir(CACHE) if f.endswith('.npy'))
pick = next((f for f in files if SUBJ in f), files[0]) if SUBJ else files[0]
vol = np.load(os.path.join(CACHE, pick)).astype(np.float32)
sl = int(os.environ.get('SNRFIG_SLICE', vol.shape[AXIS] // 2))
print(f'{pick} {vol.shape}, axis {AXIS} slice {sl}', flush=True)

panels = [np.take(rician(vol, s, np.random.RandomState(SEED + i)), sl, axis=AXIS)
          for i, s in enumerate(SNRS)]
clean = panels[0]
shown = [clean] + [ALPHA * p + (1 - ALPHA) * clean for p in panels[1:]]
vmin, vmax = np.percentile(clean, [1, 99.5])

fig, axes = plt.subplots(1, len(panels), figsize=(6.9, 1.62))
for ax, img, lab in zip(axes, shown, LABELS):
    ax.imshow(np.rot90(img), cmap='gray', vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.set_title(lab, fontsize=8, pad=3)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

plt.tight_layout(pad=0.35)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
tmp = OUT + '.tmp.png'
plt.savefig(tmp, dpi=300)
os.replace(tmp, OUT)
print(f'wrote {OUT}: window [{vmin:.2f}, {vmax:.2f}], noise opacity {ALPHA}')
