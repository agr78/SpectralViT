"""The same brain at the test-time SNRs of the distribution-shift experiment.

sec:ds reports a rank reversal between SNR 4 and SNR 1, and a reader has no way to judge
whether that range is a reasonable corruption or a destroyed image. These are the actual
cached volumes the experiment scores, under the same Rician noise at the same levels, so
the figure answers that directly: the crossover happens while the anatomy is still legible.

The noise function is duplicated from sfcn_shift.py rather than imported, matching how the
sweep scripts each carry their own feature machinery -- importing it would execute that
script's module-level data loading. The seed is figure-only; the experiment draws one noise
realization per fold and SNR, and this is one more draw from the same generator.

Every panel shares one display window, taken from the clean slice, so the loss of contrast
is visible rather than normalized away per panel. The cached volumes are already z-scored,
and rician() restandardizes after corrupting, so the panels are on a common scale.

At SNR 1/4 and below the corrupted slice is visually pure noise, which shows the severity
but not what is being destroyed. Each panel is therefore DISPLAYED as a convex blend of the
corrupted slice with the clean one, at SNRFIG_ALPHA, so the anatomy stays faintly visible
underneath. This is a display choice and nothing else: the arrays the models receive are
the unblended ones, and the caption states the blend so the figure is not read as the
model's input.

env: SNRFIG_CACHE SNRFIG_SUBJ SNRFIG_AXIS SNRFIG_SLICE SNRFIG_SEED
usage: python make_fig_snr_shift.py [out.png]
"""
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
ALPHA = float(os.environ.get('SNRFIG_ALPHA', 0.75))   # display opacity of the noise
SNRS = [None, 9, 4, 1, 0.25, 0.111]          # exactly sfcn_shift.py
LABELS = ['Clean', 'SNR 9', 'SNR 4', 'SNR 1', 'SNR 1/4', 'SNR 1/9']


def rician(x, snr, rng):
    """Rician noise at a given SNR, then renormalize. Duplicated from sfcn_shift.py."""
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

# Corrupt the whole volume, not the slice: sigma is set from the volume's mean positive
# intensity, so noising a single slice would scale it off a different statistic than the
# experiment uses.
panels = [np.take(rician(vol, s, np.random.RandomState(SEED + i)), sl, axis=AXIS)
          for i, s in enumerate(SNRS)]
clean = panels[0]
# Display only; see the module docstring. The clean panel is left untouched, so alpha
# changes nothing about what SNR means in the panels it applies to.
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
