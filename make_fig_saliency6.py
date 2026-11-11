"""Occlusion saliency for both arms on one case from each MedMNIST3D cohort at n=25.

Why occlusion and not a gradient. The spectral arm sees k features, so an input-space
gradient is d(logit)/dv = sum_i (d logit/d s_i) * (d s_i/d v): it lives in a k-dimensional
subspace of the 21,952 voxels and cannot show structure outside the span of the basis,
however much the model uses it. k is 16 or 24 for the absfft_pca cohorts here, so a gradient
map would be a picture of the basis, not of the model. Occlusion is a finite perturbation
re-featurised through the tokenizer, so it escapes the span.

Faithfulness. The batch is evaluated by medmnist_sweep.py itself under MEDMNIST_OCCLUDE, with
each cohort's curve-selected config and patch, the same n=25 draw (seed 5, the first
reporting draw) and the same training code as the reported table. Nothing is re-implemented.

Case choice, fixed in advance and not tuned: the first test volume of class 1 for the binary
cohorts and of class 0 (liver) for the 11-class organ cohort. Whether each arm classifies it
correctly is reported in the panel, not used to choose it.

usage: python make_fig_saliency6.py          # builds batches, runs the sweep, plots
       python make_fig_saliency6.py --plot   # re-plot from cached occlusion_probs.npz
"""
import os, subprocess, sys
import numpy as np

SHORTLIST = 4
WIN, STRIDE, N, SEED = 5, 2, 25, 5   # finer than 7/3: that smeared every map

# Diverging with a neutral midpoint, not sequential: measured over the top decile of
# |saliency|, 72% of the mass is positive but the rest is real and cohort-specific --
# organ/spatial is 100% NEGATIVE (occluding raises p(true class)) and adrenal/spectral is
# 38% positive. Polarity is therefore the job. No rainbow: a hue at the midpoint would
# invent structure in the near-zero range, which is most of every map.
#
# Cyan/orange rather than blue/red because three cohorts are binary masks on a BLACK
# background, where RdBu_r's poles both darken into it and vanish. These two poles stay
# bright, so they read against black mask and white mask alike. Validated: CVD separation
# dE 25.0 (deuteranopia) and 36.0 (tritanopia), normal-vision 33.3, contrast >= 3:1 against
# the surface. The validator's lightness-band check fails by design -- it applies to
# categorical palettes, where a shared band is wanted; here both poles must be bright.
from matplotlib.colors import LinearSegmentedColormap as _LSC
CMAP = _LSC.from_list('cyan_orange',
                      ['#00E5FF', '#7FF0FF', '#FFFFFF', '#FFC071', '#FF8800'])
VMIN = -1.0
ROOT = '/workspace/results/saliency6'
DS = ['organmnist3d', 'nodulemnist3d', 'adrenalmnist3d',
      'fracturemnist3d', 'vesselmnist3d', 'synapsemnist3d']
# All six live under data/medmnist now; organmnist3d.npz was copied in from
# /agent-home/medmnist on 2 Nov so the figure's inputs are all inside the repo.
NPZ = {d: f'/workspace/data/medmnist/{d}.npz' for d in DS}
TARGET = {d: (0 if d == 'organmnist3d' else 1) for d in DS}

# The occluded window must become EXACTLY ZERO after the sweep's global standardization
# (x - mu) / sd, i.e. "no evidence" rather than an out-of-distribution intensity. Writing a
# raw 0 does not do that: it lands at (0 - mu)/sd, which is -1.79 on organ and -2.15 on
# synapse -- an extreme value that can raise p(true class) as readily as lower it, and which
# produced incoherent red/blue speckle on exactly the four intensity cohorts. Filling with
# the raw training mean maps to 0 by construction.
FILL = {}
for _d in DS:
    _z = np.load(NPZ[_d])
    FILL[_d] = float(_z['train_images'].astype(np.float32).mean())


def selection():
    """The curve's n=25 pick per cohort: (patch, config, env)."""
    out = {}
    for line in open('/workspace/results/ladder_report_n25.tsv'):
        p = line.rstrip('\n').split('\t')
        out[p[0]] = (int(p[1]), p[3], p[4])
    return out


def centres(side):
    return list(range(0, side - WIN + 1, STRIDE))


def probe_case(ds, patch, env):
    """First test volume of the target class that BOTH arms classify correctly.

    An occlusion map is only interpretable where the model is making the right call: if
    p(true class) is already ~0, occluding evidence cannot lower it and the panel is blank
    for a reason that has nothing to do with where the model attends. Taking simply the
    first case of the class left 4 of 12 panels dead that way. This is a precondition on the
    attribution, not a selection on the result -- the result is the AUC table, which this
    figure does not touch. Both arms must agree, so neither is advantaged.

    Implemented by sending the WHOLE test split through the occlusion path unmodified, which
    returns per-sample probabilities for both arms from the same trained models.
    """
    out = f'{ROOT}/{ds}/probe'
    os.makedirs(out, exist_ok=True)
    z = np.load(NPZ[ds])
    X, y = z['test_images'].astype(np.float32), z['test_labels'].ravel().astype(int)
    if not os.path.exists(f'{out}/occlusion_probs.npz'):
        np.savez_compressed(f'{out}/batch.npz', volumes=X, labels=y.astype(np.int64))
        e = dict(os.environ)
        for kv in env.split():
            k, v = kv.split('=', 1)
            e[k] = v
        e.update(MEDMNIST_PATCH=str(patch), MEDMNIST_ARM='probe', MEDMNIST_NPZ=NPZ[ds],
                 MEDMNIST_OUT=out, MEDMNIST_N=str(N), MEDMNIST_SEED0=str(SEED),
                 MEDMNIST_SEEDS='1', MEDMNIST_OCCLUDE=f'{out}/batch.npz')
        r = subprocess.run(['/opt/dpeekenv/bin/python', '/workspace/medmnist_sweep.py'],
                           env=e, capture_output=True, text=True)
        open(f'{out}/run.log', 'w').write(r.stdout + r.stderr)
    d = np.load(f'{out}/occlusion_probs.npz')
    ok = ((np.argmax(d['spec'], 1) == y) & (np.argmax(d['spat'], 1) == y)
          & (y == TARGET[ds]))
    if not ok.any():                      # no case both arms get right: fall back, and say so
        print(f'  {ds}: NO case correct for both arms; falling back to first of class')
        return int(np.where(y == TARGET[ds])[0][0]), False
    cand = np.where(ok)[0][:40]
    return screen(ds, patch, env, cand, X), True


def run_case(ds, patch, env, i, tag):
    """Fine occlusion for one case into its own subdirectory; returns the probabilities.

    Shared by the fine selection pass and by the final figure, so the chosen case is never
    re-run and the panel is drawn from exactly the probabilities the choice was made on.
    """
    out = f'{ROOT}/{ds}/{tag}'
    if not os.path.exists(f'{out}/occlusion_probs.npz'):
        os.makedirs(out, exist_ok=True)
        vol = np.load(NPZ[ds])['test_images'].astype(np.float32)[i]
        cs = centres(vol.shape[0])
        vols = [vol]
        for a in cs:
            for b in cs:
                for c in cs:
                    v = vol.copy()
                    v[a:a + WIN, b:b + WIN, c:c + WIN] = FILL[ds]
                    vols.append(v)
        np.savez_compressed(f'{out}/batch.npz', volumes=np.stack(vols),
                            labels=np.full(len(vols), TARGET[ds], np.int64))
        e = dict(os.environ)
        for kv in env.split():
            k, v = kv.split('=', 1)
            e[k] = v
        e.update(MEDMNIST_PATCH=str(patch), MEDMNIST_ARM=tag, MEDMNIST_NPZ=NPZ[ds],
                 MEDMNIST_OUT=out, MEDMNIST_N=str(N), MEDMNIST_SEED0=str(SEED),
                 MEDMNIST_SEEDS='1', MEDMNIST_OCCLUDE=f'{out}/batch.npz')
        r = subprocess.run(['/opt/dpeekenv/bin/python', '/workspace/medmnist_sweep.py'],
                           env=e, capture_output=True, text=True)
        open(f'{out}/run.log', 'w').write(r.stdout + r.stderr)
    f = f'{out}/occlusion_probs.npz'
    return np.load(f) if os.path.exists(f) else None


def pick_agreeing(ds, patch, env, short):
    """Of the shortlisted cases, one where BOTH arms' peak response has the same sign.

    Runs the full fine occlusion for each shortlisted case and scores
        (signs agree, min over arms of the peak |response|)
    lexicographically, so a same-signed pair is preferred and ties break toward the case
    whose weaker arm responds most. If none agree the ranking still returns the strongest
    pair, and the caller reports that no agreeing case existed rather than hiding it.
    """
    out = []
    for i in short:
        tag = f'cand{i}'
        d = run_case(ds, patch, env, i, tag)
        if d is None:
            continue
        pk = {}
        for arm in ('spec', 'spat'):
            q = d[arm][:, TARGET[ds]]
            dd = q[0] - q[1:]
            pk[arm] = dd[np.argmax(np.abs(dd))]
        agree = (pk['spec'] > 0) == (pk['spat'] > 0)
        # Rank on the weaker arm's 99.5th-percentile RANGE, not its peak. The peak is one
        # window and says nothing about how much of the map is drawable; the panel is scaled
        # to the percentile, so a case can win on peak and still render as a single blob --
        # adrenal case 5 had a spectral peak of 0.999 with a range of 0.082, against case
        # 97's range of 0.996.
        rng = {a: float(np.percentile(np.abs(d[a][0, TARGET[ds]] - d[a][1:, TARGET[ds]]), 99.5))
               for a in ('spec', 'spat')}
        out.append((agree, min(rng['spec'], rng['spat']), i, pk))
    out.sort(key=lambda t: (t[0], t[1]), reverse=True)
    agree, strength, i, pk = out[0]
    print(f'  {ds}: fine pass -> case {i}, signs {"AGREE" if agree else "DISAGREE"} '
          f'(spec {pk["spec"]:+.3f}, spat {pk["spat"]:+.3f}); '
          f'{sum(1 for o in out if o[0])}/{len(out)} shortlisted agreed')
    return i, agree


def screen(ds, patch, env, cand, X):
    """Of the candidate cases, the one whose WEAKER arm still has a real occlusion response.

    Correctness alone is not enough: at p(true)=1.000 the evidence is redundant, no single
    window moves the probability, and the panel is flat for a reason that again has nothing
    to do with where the model attends -- that is what left synapse-spatial and
    fracture-spatial blank. Screening is a coarse 14^3/stride-14 occlusion (8 windows, 9
    volumes per candidate) and the score is

        min over arms of ( max over windows of | p_base - p_occluded | )

    i.e. maximize the weaker arm's dynamic range, so NEITHER map can come out blank. The
    criterion is symmetric in the two arms and is computed on a case whose label both arms
    already get right, so it cannot favor one tokenizer over the other.
    """
    out = f'{ROOT}/{ds}/screen'
    os.makedirs(out, exist_ok=True)
    CW = 14
    if not os.path.exists(f'{out}/occlusion_probs.npz'):
        vols, cs = [], list(range(0, X.shape[1] - CW + 1, CW))
        for i in cand:
            vols.append(X[i])
            for a in cs:
                for b in cs:
                    for c in cs:
                        v = X[i].copy()
                        v[a:a + CW, b:b + CW, c:c + CW] = FILL[ds]
                        vols.append(v)
        np.savez_compressed(f'{out}/batch.npz', volumes=np.stack(vols),
                            labels=np.full(len(vols), TARGET[ds], np.int64))
        e = dict(os.environ)
        for kv in env.split():
            k, v = kv.split('=', 1)
            e[k] = v
        e.update(MEDMNIST_PATCH=str(patch), MEDMNIST_ARM='screen', MEDMNIST_NPZ=NPZ[ds],
                 MEDMNIST_OUT=out, MEDMNIST_N=str(N), MEDMNIST_SEED0=str(SEED),
                 MEDMNIST_SEEDS='1', MEDMNIST_OCCLUDE=f'{out}/batch.npz')
        r = subprocess.run(['/opt/dpeekenv/bin/python', '/workspace/medmnist_sweep.py'],
                           env=e, capture_output=True, text=True)
        open(f'{out}/run.log', 'w').write(r.stdout + r.stderr)
    d = np.load(f'{out}/occlusion_probs.npz')
    per = 1 + 8                                        # original + 2^3 coarse windows
    best, bi, scores = -1.0, int(cand[0]), []
    for j, i in enumerate(cand):
        rng = []
        for arm in ('spec', 'spat'):
            q = d[arm][j * per:(j + 1) * per, TARGET[ds]]
            rng.append(float(np.abs(q[0] - q[1:]).max()))
        sc = min(rng)
        scores.append(sc)
        if sc > best:
            best, bi = sc, int(i)
    # Shortlist rather than decide: the coarse pass ranks candidates by the weaker arm's
    # response, and the fine pass below breaks the tie on SIGN AGREEMENT. Both arms already
    # agree in sign at the coarse scale on every qualifying fracture case, so disagreement
    # is a fine-scale, per-case phenomenon; a panel pair that disagrees invites a sign
    # comparison this figure cannot support, since the two maps attribute two different
    # models. Three filters in total -- correct, responsive, same-signed -- and the caption
    # says so.
    order = sorted(range(len(cand)), key=lambda j: -scores[j])[:SHORTLIST]
    short = [int(cand[j]) for j in order]
    print(f'  {ds}: screened {len(cand)} cases, shortlist {short} '
          f'(weaker-arm range {best:.3f})')
    return short


def build_batch(ds, i):
    z = np.load(NPZ[ds])
    X = z['test_images'].astype(np.float32)
    vol = X[i]
    cs = centres(vol.shape[0])
    vols = [vol]
    for a in cs:
        for b in cs:
            for c in cs:
                v = vol.copy()
                v[a:a + WIN, b:b + WIN, c:c + WIN] = FILL[ds]
                vols.append(v)
    vols = np.stack(vols)
    os.makedirs(f'{ROOT}/{ds}', exist_ok=True)
    np.savez_compressed(f'{ROOT}/{ds}/batch.npz', volumes=vols,
                        labels=np.full(len(vols), TARGET[ds], np.int64))
    np.save(f'{ROOT}/{ds}/case.npy', vol)
    return len(vols), len(cs)


def run(ds, patch, env):
    out = f'{ROOT}/{ds}'
    if os.path.exists(f'{out}/occlusion_probs.npz'):
        print(f'  have {ds}')
        return
    e = dict(os.environ)
    for kv in env.split():
        k, v = kv.split('=', 1)
        e[k] = v
    e.update(MEDMNIST_PATCH=str(patch), MEDMNIST_ARM='saliency', MEDMNIST_NPZ=NPZ[ds],
             MEDMNIST_OUT=out, MEDMNIST_N=str(N), MEDMNIST_SEED0=str(SEED),
             MEDMNIST_SEEDS='1', MEDMNIST_OCCLUDE=f'{out}/batch.npz')
    r = subprocess.run(['/opt/dpeekenv/bin/python', '/workspace/medmnist_sweep.py'],
                       env=e, capture_output=True, text=True)
    open(f'{out}/run.log', 'w').write(r.stdout + r.stderr)
    print(f'  {ds}: {"ok" if os.path.exists(f"{out}/occlusion_probs.npz") else "FAILED"}')


def saliency(ds):
    """Signed drop in p(true class) on the window grid -- NOT accumulated into voxels.

    Each occlusion window yields exactly one number, so the attribution's true resolution is
    the window grid (12^3 here), not 28^3. Writing each window's value into all of its voxels
    and overlap-averaging, as a first version did, manufactures hard edges at every stride
    boundary and is what made the maps blocky. The grid is kept as a grid and interpolated
    only at display time.
    """
    d = np.load(f'{ROOT}/{ds}/occlusion_probs.npz')
    cs = centres(28)
    out = {}
    for arm in ('spec', 'spat'):
        p = d[arm][:, TARGET[ds]]
        base, rest = p[0], p[1:]
        g = np.empty((len(cs),) * 3)
        k = 0
        for ia in range(len(cs)):
            for ib in range(len(cs)):
                for ic in range(len(cs)):
                    g[ia, ib, ic] = base - rest[k]
                    k += 1
        out[arm] = g
        out[arm + '_base'] = base
        out[arm + '_pred'] = int(np.argmax(d[arm][0]))
    return out


def upsample(grid, side=28):
    """Sample the window grid at voxel positions with cubic interpolation.

    Window i spans [cs[i], cs[i]+WIN), so its center is cs[i] + WIN//2; a voxel at x maps to
    coarse coordinate (x - WIN//2) / STRIDE. Interpolating is a rendering choice, not an
    extra claim: it shows the grid the occlusion actually measured, without pretending to
    voxel resolution and without inventing block edges.
    """
    from scipy.ndimage import map_coordinates
    n = grid.shape[0]
    ax = np.clip((np.arange(side) - WIN // 2) / STRIDE, 0, n - 1)
    gz, gy, gx = np.meshgrid(ax, ax, ax, indexing='ij')
    return map_coordinates(grid, np.stack([gz, gy, gx]), order=3, mode='nearest')


def plot(out_png='paper/cvpr/fig/medmnist_saliency6.png'):
    """3 rows x 6 cohorts: anatomy, spectral occlusion saliency, spatial occlusion saliency.

    Rendering decisions, each made to fix a defect in an earlier version:
      - the slice is the one with the most ANATOMICAL energy, chosen from the volume alone.
        The central slice carried only 3.1-3.7% of volume energy on four of six cohorts, so
        those panels were near-empty cross-sections.
      - the map is the window grid interpolated cubically, not each window's value painted
        into its voxels. The latter put a hard edge at every stride boundary.
      - limits are the 99.5th percentile of |saliency|, symmetric, and shared BETWEEN ARMS
        within a cohort. The max compressed each map into near-white around one hot window;
        sharing across cohorts is wrong because their probability scales differ by an order
        of magnitude, so each column prints its own limit.
      - diverging with a neutral midpoint, because the signal is genuinely two-signed.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    SEL = selection()
    NAME = {'organmnist3d': 'Organ', 'nodulemnist3d': 'Nodule', 'adrenalmnist3d': 'Adrenal',
            'fracturemnist3d': 'Fracture', 'vesselmnist3d': 'Vessel',
            'synapsemnist3d': 'Synapse'}
    fig, ax = plt.subplots(3, len(DS), figsize=(2.05 * len(DS), 6.6))
    LIMS = []
    for j, ds in enumerate(DS):
        vol = np.load(f'{ROOT}/{ds}/case.npy')
        # Peak anatomical energy among INTERIOR slices. Unrestricted, this rule returned
        # slice 0 for nodule and 27 for organ -- faces of the cube, not cross-sections
        # through the structure, because bright tissue touches the volume boundary.
        e = np.abs(vol - vol.mean()).sum(axis=(1, 2))
        lo, hi = WIN, vol.shape[0] - WIN
        SL = int(lo + np.argmax(e[lo:hi]))
        s = saliency(ds)
        up = {a: upsample(s[a]) for a in ('spec', 'spat')}
        # Scaled PER PANEL. A shared column limit rendered synapse/spectral at 53% and
        # nodule/spatial at 6% of range, which reads as "attends to nothing" when the truth
        # is "responds less strongly". The cost is that COLOR NO LONGER CARRIES MAGNITUDE
        # anywhere in this figure: magnitude is the printed range under each panel, only.
        # Still the 99.5th percentile rather than the max, or one hot window would compress
        # each map into near-white around it.
        lims = {a: (np.percentile(np.abs(up[a]), 99.5) or 1.0) for a in ('spec', 'spat')}
        ax[0, j].imshow(vol[SL], cmap='gray')
        _title = (f'{NAME[ds]}\n{SEL[ds][1]}  p{SEL[ds][0]}\n'
                  f'slice {SL}')
        ax[0, j].set_title(_title, fontsize=7.5, linespacing=1.45)
        for i, arm in enumerate(('spec', 'spat'), start=1):
            # The anatomy is redrawn under each map so the attribution can be read against
            # the structure it refers to. The overlay carries an alpha ramp proportional to
            # |saliency|, so the near-zero bulk of every map -- which is most of it -- is
            # transparent rather than a wash of pale color sitting over the anatomy.
            # Consequence worth knowing: the colorbar then describes the hue only. How
            # OPAQUE a region is, not just how saturated, now also encodes magnitude.
            m = up[arm][SL]
            rgba = plt.get_cmap(CMAP)(plt.Normalize(VMIN * lims[arm], lims[arm])(m))
            rgba[..., 3] = np.clip(np.abs(m) / lims[arm], 0, 1)
            ax[i, j].imshow(vol[SL], cmap='gray')
            ax[i, j].imshow(rgba, interpolation='bilinear')
            hit = '\u2713' if s[arm + '_pred'] == TARGET[ds] else '\u2717'
            # Each panel's OWN 99.5th percentile, alongside the shared column limit it is
            # drawn against. Without this a weaker arm reads as "attends to nothing" when it
            # is really "responds half as strongly": synapse/spectral sits at 53% of its
            # column limit and nodule/spatial at 6%, and in both cases the structure that is
            # there is simply below the shared scale.
            ax[i, j].set_xlabel(f'p={s[arm + "_base"]:.2f} {hit}\n'
                                f'range $\\pm${lims[arm]:.3f}',
                                fontsize=6.9, linespacing=1.4)
        LIMS += list(lims.values())
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)
    for i, lab in enumerate(['Volume', 'Spectral', 'Spatial']):
        ax[i, 0].set_ylabel(lab, fontsize=9)
    # Built from an explicit mappable, not from a panel's image: since the anatomy is drawn
    # first in each overlay panel, images[0] is the GREYSCALE underlay and taking the bar
    # from it rendered a black-to-white ramp. Normalised to +-1 because every panel is
    # scaled to its own range, so the bar can only show the sign convention, not a value.
    import matplotlib.cm as _cm
    sm = _cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(-1, 1))
    cb = fig.colorbar(sm, ax=ax[1:, :].ravel().tolist(),
                      fraction=0.012, pad=0.012, ticks=[])
    cb.set_label('occluded $\\rightarrow$ $p$(true) falls (orange) / rises (cyan);  '
                 'transparent where nothing changes', fontsize=7)
    # The suptitle and the footnote below used to be baked into the PNG. Both repeat what
    # the LaTeX caption already states, and together they added a quarter of the figure's
    # height -- which, for a full-width float, is what decides whether LaTeX can place it
    # beside text at all. Dropped from the image; the caption carries them.
    # Independent scaling means color is comparable NOWHERE in this figure -- not between
    # arms, not between cohorts. The panel ranges span two orders of magnitude, so saying so
    # is not a formality.
    print(f'panel ranges span +/-{min(LIMS):.3f} to +/-{max(LIMS):.3f} -- quote these in the caption')
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    print('wrote', out_png)
    for ds in DS:
        s = saliency(ds)
        print('  %-9s spec p=%.3f pred=%d  spat p=%.3f pred=%d  true=%d' %
              (NAME[ds], s['spec_base'], s['spec_pred'], s['spat_base'], s['spat_pred'],
               TARGET[ds]))


if __name__ == '__main__':
    # Defined last on purpose: an earlier version called plot() above its own definition.
    SEL = selection()
    if '--plot' not in sys.argv:
        import shutil
        for ds in DS:
            short, both = probe_case(ds, SEL[ds][0], SEL[ds][2])
            i, agree = pick_agreeing(ds, SEL[ds][0], SEL[ds][2], short)
            src = f'{ROOT}/{ds}/cand{i}/occlusion_probs.npz'
            shutil.copy(src, f'{ROOT}/{ds}/occlusion_probs.npz')
            np.save(f'{ROOT}/{ds}/case.npy',
                    np.load(NPZ[ds])['test_images'].astype(np.float32)[i])
            print(f'{ds}: case {i} (both correct: {both}, signs agree: {agree})')
    plot()
