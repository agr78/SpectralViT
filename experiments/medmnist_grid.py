"""The basis x capacity grid for MedMNIST3D selection."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repository root: shared modules and paths.py
G = []


def add(name, **kw):
    G.append((name, ' '.join('MEDMNIST_%s=%s' % (k.upper(), v) for k, v in kw.items())))


for c in (4, 8, 16, 24):
    add('pca_c%d' % c, basis='pca', comp=c)
for f, s in ((2, 14), (3, 14), (2, 7), (3, 7)):
    add('dct_f%d_s%d' % (f, s), basis='dct', freq=f, block=14, stride=s)
for b in (6, 12, 24):
    add('energy_b%d' % b, basis='energy', bands=b)
for c in (8, 16, 28):
    for s in (14, 7):
        add('blockpca_c%d_s%d' % (c, s), basis='blockpca', comp=c, block=14, stride=s)
for r, o in ((4, 4), (6, 6), (8, 6)):
    add('oriented_r%d_o%d' % (r, o), basis='oriented', orad=r, ori=o)
for j in (3, 4, 5):
    add('scatter_j%d' % j, basis='scatter', scat_j=j)

for c in (8, 16, 24):
    for b in (6, 12):
        add('pca_energy_c%d_b%d' % (c, b), basis='pca_energy', comp=c, bands=b)
for f in (2, 3):
    for b in (6, 12, 24):
        add('dct_energy_f%d_b%d' % (f, b), basis='dct_energy', freq=f, bands=b)
for f in (2, 3):
    add('dct_oriented_f%d' % f, basis='dct_oriented', freq=f, orad=6, ori=6)
for f in (2, 3):
    add('dct_scatter_f%d' % f, basis='dct_scatter', freq=f, scat_j=4)
for c in (8, 16, 24):
    add('concat_c%d' % c, basis='concat', comp=c)
for c in (8, 16, 24):
    add('energy_cat_c%d' % c, basis='energy_cat', comp=c, bands=12)

for c in (8, 16, 24):
    add('square_c%d' % c, basis='square', comp=c)
for c in (8, 16, 24):
    add('pca_sq_energy_c%d' % c, basis='pca_sq_energy', comp=c, bands=12)
for b in (6, 12, 24):
    add('energy_gram_b%d' % b, basis='energy_gram', bands=b)

for k in (4, 8, 16):
    add('cross_k%d' % k, basis='cross', cross=k, comp=16)
for d in (64, 128, 256):
    add('rbf_d%d' % d, basis='rbf', rbf=d, comp=16)
add('triple_c16', basis='triple', comp=16)

for b, st in ((7, 7), (7, 4)):
    add('dct_b%d_s%d_f2' % (b, st), basis='dct', block=b, stride=st, freq=2)
    add('dct_energy_b%d_s%d' % (b, st), basis='dct_energy', block=b, stride=st, freq=2, bands=6)
for c in (4, 8):
    add('blockpca_b7_s7_c%d' % c, basis='blockpca', block=7, stride=7, comp=c)
add('blockpca_b7_s4_c4', basis='blockpca', block=7, stride=4, comp=4)
add('energy_b7_s7', basis='energy', block=7, stride=7, bands=12)
add('scatter_b7', basis='scatter', block=7, stride=7, scat_j=3)
for c in (8, 16, 28):
    for st in (14, 7):
        add('blockpca_energy_c%d_s%d' % (c, st), basis='blockpca_energy',
            comp=c, block=14, stride=st, bands=12)


for nm, kw in (('dct_energy_f2_b6', dict(basis='dct_energy', freq=2, bands=6)),
               ('dct_energy_f2_b12', dict(basis='dct_energy', freq=2, bands=12)),
               ('dct_oriented_f3', dict(basis='dct_oriented', freq=3, orad=6, ori=6)),
               ('energy_cat_c24', dict(basis='energy_cat', comp=24, bands=12)),
               ('concat_c24', dict(basis='concat', comp=24)),
               ('blockpca_energy_c16_s7', dict(basis='blockpca_energy', comp=16,
                                               block=14, stride=7, bands=12))):
    add(nm + '_rank', rank=1, **kw)

for c in (4, 8, 16):
    add('pls_c%d' % c, basis='pls', comp=c)
    add('pls_energy_c%d' % c, basis='pls_energy', comp=c, bands=12)

for t in (8, 12, 20):
    add('topo_t%d' % t, basis='topo', thresh=t)
add('topo_energy_t12', basis='topo_energy', thresh=12, bands=12)
add('topo_dct_t12', basis='topo_dct', thresh=12, freq=2)

for f in (2, 3):
    add('absfft_f%d' % f, basis='absfft', freq=f)
    add('absfft_energy_f%d' % f, basis='absfft_energy', freq=f, bands=12)
for c in (8, 16, 24):
    add('absfft_pca_c%d' % c, basis='absfft_pca', comp=c)

for c in (8, 16, 28):
    for st in (14, 7):
        add('magblockpca_c%d_s%d' % (c, st), basis='magblockpca', comp=c,
            block=14, stride=st, freq=3)
add('magblockpca_energy_c16_s7', basis='magblockpca_energy', comp=16,
    block=14, stride=7, freq=3, bands=12)

add('moment', basis='moment')
add('moment_energy', basis='moment_energy', bands=12)
add('moment_dct', basis='moment_dct', freq=2)

add('haar', basis='haar')
add('haar_signed', basis='haar_signed')
add('haar_energy', basis='haar_energy', bands=12)
add('haar_dct', basis='haar_dct', freq=2)

if __name__ == '__main__':
    import sys
    with open('results/medmnist_grid_configs.tsv', 'w') as fh:
        for n, e in G:
            fh.write('%s\t%s\n' % (n, e))
    print('%d configs' % len(G))
    for n, e in G:
        print('  %-22s %s' % (n, e))
