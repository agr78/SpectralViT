"""The basis x capacity grid for MedMNIST3D selection.

Every earlier sweep held each basis at one capacity -- COMP=16, FREQ=3, BANDS=12 -- which
confounds "which basis" with "how much of it". This grid sweeps each family over its own
capacity parameter so selection chooses a (basis, capacity) pair, and writes one config
table that both the test driver and the CV driver read, so the two can never drift.

pca_fullrank is omitted: PCA on n centred samples has rank n-1, so at n=25 COMP=0 clamps
to 24 and is bit-identical to COMP=24 (both gave mean spec_AUC 0.651541 over 30 cells).
COMP must be a multiple of 4 (the encoder's chunk size).
"""
G = []


def add(name, **kw):
    G.append((name, ' '.join('MEDMNIST_%s=%s' % (k.upper(), v) for k, v in kw.items())))


# ---- single bases, each over its own capacity axis
for c in (4, 8, 16, 24):                      # whole-volume PCA, up to the rank cap
    add('pca_c%d' % c, basis='pca', comp=c)
for f, s in ((2, 14), (3, 14), (2, 7), (3, 7)):   # cosine: retained frequencies x overlap
    add('dct_f%d_s%d' % (f, s), basis='dct', freq=f, block=14, stride=s)
for b in (6, 12, 24):                         # radial shells for the band energies
    add('energy_b%d' % b, basis='energy', bands=b)
for c in (8, 16, 28):                         # block PCA: components per block x overlap
    for s in (14, 7):
        add('blockpca_c%d_s%d' % (c, s), basis='blockpca', comp=c, block=14, stride=s)
for r, o in ((4, 4), (6, 6), (8, 6)):         # radial x angular energy bins
    add('oriented_r%d_o%d' % (r, o), basis='oriented', orad=r, ori=o)
for j in (3, 4, 5):                           # scattering orders
    add('scatter_j%d' % j, basis='scatter', scat_j=j)

# ---- concatenations, over the capacities of both parts. Band energy added to a linear
# basis, and orientation or scattering added to the cosine basis, are the pairings the
# nested-pair analysis supports (+0.039 and +0.024 for energy, p<0.001 and p=0.002).
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

# ---- quadratic and higher-order arms. These are the worst performers on average
# (square -0.033, pca_sq_energy -0.046, energy_gram -0.018 over 30 cells) and the only
# arms with a positive point estimate on synapse, where everything else loses. They are
# kept because dropping a candidate on the strength of a test-set p-value is itself a
# selection made on test data, which is the leakage this whole experiment exists to avoid.
# The candidate set has to be fixed in advance; CV is then free to reject them.
for c in (8, 16, 24):
    add('square_c%d' % c, basis='square', comp=c)
for c in (8, 16, 24):
    add('pca_sq_energy_c%d' % c, basis='pca_sq_energy', comp=c, bands=12)
for b in (6, 12, 24):
    add('energy_gram_b%d' % b, basis='energy_gram', bands=b)

# ---- nonlinear combinations of two linear bases, at their own capacities. Same
# a-priori argument: none of them won, but the set is fixed before selection sees it.
for k in (4, 8, 16):
    add('cross_k%d' % k, basis='cross', cross=k, comp=16)
for d in (64, 128, 256):
    add('rbf_d%d' % d, basis='rbf', rbf=d, comp=16)
add('triple_c16', basis='triple', comp=16)

# ---- finer block grids. BLOCK was fixed at 14 in every sweep above, so the block SIDE is
# the one capacity axis never explored, and it is the one that should matter most for a
# localized disruption: fracture's two strongest arms are the overlapping block PCA and
# scattering, both locality-sensitive, and no config in the 89 beats its patch-7 baseline
# (best 0.573 against 0.584, a gap of 0.011). Added for every dataset, not just fracture,
# so nothing here is tuned on a test outcome.
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


# ---- rank weights, the last knob never swept. Off in all 89 configs above; the single
# config that enabled them (concat_rank) was 0.014 WORSE than its non-rank twin, but that
# was the only test and the profile was mis-specified for every concatenation until the
# group sizes above were recorded. Tested here on the leading families only.
for nm, kw in (('dct_energy_f2_b6', dict(basis='dct_energy', freq=2, bands=6)),
               ('dct_energy_f2_b12', dict(basis='dct_energy', freq=2, bands=12)),
               ('dct_oriented_f3', dict(basis='dct_oriented', freq=3, orad=6, ori=6)),
               ('energy_cat_c24', dict(basis='energy_cat', comp=24, bands=12)),
               ('concat_c24', dict(basis='concat', comp=24)),
               ('blockpca_energy_c16_s7', dict(basis='blockpca_energy', comp=16,
                                               block=14, stride=7, bands=12))):
    add(nm + '_rank', rank=1, **kw)

# ---- supervised projections: the missing criterion. Fitted on inner-training rows and
# their labels only.
for c in (4, 8, 16):
    add('pls_c%d' % c, basis='pls', comp=c)
    add('pls_energy_c%d' % c, basis='pls_energy', comp=c, bands=12)

# ---- topology. Not a basis at all: a threshold filtration summarised by component
# counts. A fracture is a disconnection, which no bandlimited projection can express.
for t in (8, 12, 20):
    add('topo_t%d' % t, basis='topo', thresh=t)
add('topo_energy_t12', basis='topo_energy', thresh=12, bands=12)
add('topo_dct_t12', basis='topo_dct', thresh=12, freq=2)

# ---- magnitude spectra. A probe on the full training split scores |c| far above the
# signed coefficients on three of four cohorts (fracture 0.672 vs 0.560, synapse 0.664 vs
# 0.472, organ 1.000 vs 0.909; only nodule prefers signed). Every basis above is signed,
# and the only phase-free ones pool radially -- so per-coefficient magnitude is untested.
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

# ---- morphometry. Marginal profiles and spatial moments: where the mass sits along each
# axis. Vertebral compression fracture is height loss, which is exactly this and is not a
# frequency, magnitude, variance or connectivity property.
add('moment', basis='moment')
add('moment_energy', basis='moment_energy', bands=12)
add('moment_dct', basis='moment_dct', freq=2)

# ---- wavelets. Scattering above pools wavelet moduli over space; this keeps the
# coefficients, localised per subband per octant, at two scales.
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