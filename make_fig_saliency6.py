"""Occlusion saliency for both arms on one case from each MedMNIST3D cohort.

usage: python make_fig_saliency6.py
  python make_fig_saliency6.py --plot"""
from paths import DATA, PYTHON, REPO, RESULTS
import os, subprocess, sys
import numpy as np

SHORTLIST = 4
WIN, STRIDE, N, SEED = 5, 2, 25, 5

from matplotlib.colors import LinearSegmentedColormap as _LSC
CMAP = _LSC.from_list('cyan_orange',
                      ['#00E5FF', '#7FF0FF', '#FFFFFF', '#FFC071', '#FF8800'])
VMIN = -1.0
ROOT = f'{RESULTS}/saliency6'
DS = ['organmnist3d', 'nodulemnist3d', 'adrenalmnist3d',
      'fracturemnist3d', 'vesselmnist3d', 'synapsemnist3d']
NPZ = {d: f'{DATA}/medmnist/{d}.npz' for d in DS}
TARGET = {d: (0 if d == 'organmnist3d' else 1) for d in DS}

FILL = {}
for _d in DS:
    _z = np.load(NPZ[_d])
    FILL[_d] = float(_z['train_images'].astype(np.float32).mean())


def selection():
    out = {}
    for line in open(f'{RESULTS}/ladder_report_n25.tsv'):
        p = line.rstrip('\n').split('\t')
        out[p[0]] = (int(p[1]), p[3], p[4])
    return out


def centres(side):
    return list(range(0, side - WIN + 1, STRIDE))


def probe_case(ds, patch, env):
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
        r = subprocess.run([PYTHON, f'{REPO}/medmnist_sweep.py'],
                           env=e, capture_output=True, text=True)
        open(f'{out}/run.log', 'w').write(r.stdout + r.stderr)
    d = np.load(f'{out}/occlusion_probs.npz')
    ok = ((np.argmax(d['spec'], 1) == y) & (np.argmax(d['spat'], 1) == y)
          & (y == TARGET[ds]))
    if not ok.any():
        print(f'  {ds}: NO case correct for both arms; falling back to first of class')
        return int(np.where(y == TARGET[ds])[0][0]), False
    cand = np.where(ok)[0][:40]
    return screen(ds, patch, env, cand, X), True


def run_case(ds, patch, env, i, tag):
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
        r = subprocess.run([PYTHON, f'{REPO}/medmnist_sweep.py'],
                           env=e, capture_output=True, text=True)
        open(f'{out}/run.log', 'w').write(r.stdout + r.stderr)
    f = f'{out}/occlusion_probs.npz'
    return np.load(f) if os.path.exists(f) else None


def pick_agreeing(ds, patch, env, short):
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
        r = subprocess.run([PYTHON, f'{REPO}/medmnist_sweep.py'],
                           env=e, capture_output=True, text=True)
        open(f'{out}/run.log', 'w').write(r.stdout + r.stderr)
    d = np.load(f'{out}/occlusion_probs.npz')
    per = 1 + 8
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
    r = subprocess.run([PYTHON, f'{REPO}/medmnist_sweep.py'],
                       env=e, capture_output=True, text=True)
    open(f'{out}/run.log', 'w').write(r.stdout + r.stderr)
    print(f'  {ds}: {"ok" if os.path.exists(f"{out}/occlusion_probs.npz") else "FAILED"}')


def saliency(ds):
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
    from scipy.ndimage import map_coordinates
    n = grid.shape[0]
    ax = np.clip((np.arange(side) - WIN // 2) / STRIDE, 0, n - 1)
    gz, gy, gx = np.meshgrid(ax, ax, ax, indexing='ij')
    return map_coordinates(grid, np.stack([gz, gy, gx]), order=3, mode='nearest')


def plot(out_png='paper/cvpr/fig/medmnist_saliency6.png'):
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
        e = np.abs(vol - vol.mean()).sum(axis=(1, 2))
        lo, hi = WIN, vol.shape[0] - WIN
        SL = int(lo + np.argmax(e[lo:hi]))
        s = saliency(ds)
        up = {a: upsample(s[a]) for a in ('spec', 'spat')}
        lims = {a: (np.percentile(np.abs(up[a]), 99.5) or 1.0) for a in ('spec', 'spat')}
        ax[0, j].imshow(vol[SL], cmap='gray')
        _title = (f'{NAME[ds]}\n{SEL[ds][1]}  p{SEL[ds][0]}\n'
                  f'slice {SL}')
        ax[0, j].set_title(_title, fontsize=7.5, linespacing=1.45)
        for i, arm in enumerate(('spec', 'spat'), start=1):
            m = up[arm][SL]
            rgba = plt.get_cmap(CMAP)(plt.Normalize(VMIN * lims[arm], lims[arm])(m))
            rgba[..., 3] = np.clip(np.abs(m) / lims[arm], 0, 1)
            ax[i, j].imshow(vol[SL], cmap='gray')
            ax[i, j].imshow(rgba, interpolation='bilinear')
            hit = '\u2713' if s[arm + '_pred'] == TARGET[ds] else '\u2717'
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
    import matplotlib.cm as _cm
    sm = _cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(-1, 1))
    cb = fig.colorbar(sm, ax=ax[1:, :].ravel().tolist(),
                      fraction=0.012, pad=0.012, ticks=[])
    cb.set_label('occluded $\\rightarrow$ $p$(true) falls (orange) / rises (cyan);  '
                 'transparent where nothing changes', fontsize=7)
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
