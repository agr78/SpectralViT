"""How much volume energy one occlusion window holds, per MedMNIST3D cohort.

usage: python make_saliency_energy.py"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results', 'saliency_energy.txt')

sys.argv = ['x']
_src = open(os.path.join(HERE, 'make_fig_saliency6.py')).read()
exec(_src.split("if __name__ == '__main__':")[0])

SPARSE = ['adrenalmnist3d', 'vesselmnist3d']
DENSE = ['organmnist3d', 'nodulemnist3d', 'synapsemnist3d']
PUB_SPARSE = (14.0, 20.0)
PUB_DENSE = (0.9, 2.9)

share = {}
for ds in DS:
    v = np.load(f'{ROOT}/{ds}/case.npy')
    dev = (v - FILL[ds]) ** 2
    tot = dev.sum()
    cs = centres(28)
    best = 0.0
    for a in cs:
        for b in cs:
            for c in cs:
                best = max(best, dev[a:a + WIN, b:b + WIN, c:c + WIN].sum())
    share[ds] = 100 * best / tot

L = [f'Richest {WIN}^3 window as a share of volume energy, one case per cohort.', '']
for ds in DS:
    tag = ('sparse mask' if ds in SPARSE else 'dense texture' if ds in DENSE else '')
    L.append(f"  {ds.replace('mnist3d',''):<10} {share[ds]:5.1f}%   {tag}")
sp = [share[d] for d in SPARSE if d in share]
dn = [share[d] for d in DENSE if d in share]
L += ['', f'  sparse-mask range  {min(sp):.1f}-{max(sp):.1f}%  (published {PUB_SPARSE[0]:.0f}-{PUB_SPARSE[1]:.0f}%)',
      f'  dense-texture range {min(dn):.1f}-{max(dn):.1f}%  (published {PUB_DENSE[0]:.1f}-{PUB_DENSE[1]:.1f}%)']

report = '\n'.join(L)
print(report)
with open(OUT, 'w') as fh:
    fh.write(report + '\n')
print(f'\nwrote {OUT}')

bad = []
if not (PUB_SPARSE[0] <= min(sp) and max(sp) <= PUB_SPARSE[1] + 0.5):
    bad.append(f'sparse-mask range {min(sp):.1f}-{max(sp):.1f}% outside published 14-20%')
if not (PUB_DENSE[0] - 0.05 <= min(dn) and max(dn) <= PUB_DENSE[1] + 0.05):
    bad.append(f'dense-texture range {min(dn):.1f}-{max(dn):.1f}% outside published 0.9-2.9%')
if min(sp) <= max(dn):
    bad.append('the two regimes overlap, which is the claim the paragraph rests on')
if bad:
    raise SystemExit('DOES NOT MATCH the saliency paragraph:\n  ' + '\n  '.join(bad))
print('matches the saliency paragraph: both ranges and the separation between regimes')
