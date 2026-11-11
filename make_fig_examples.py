"""Random examples from every MedMNIST3D set, and a class contrast for synapse."""
import numpy as np, glob, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

S = {'organmnist3d': '/agent-home/medmnist/organmnist3d.npz'}
for f in sorted(glob.glob('/workspace/data/medmnist/*.npz')):
    S[os.path.basename(f)[:-4]] = f
ORDER = ['organmnist3d', 'nodulemnist3d', 'adrenalmnist3d',
         'fracturemnist3d', 'vesselmnist3d', 'synapsemnist3d']
rs = np.random.RandomState(0)

# --- 10 random volumes per dataset, central slice -------------------------
fig, ax = plt.subplots(len(ORDER), 10, figsize=(14, 1.55 * len(ORDER)))
for r, name in enumerate(ORDER):
    z = np.load(S[name]); X = z['train_images']; y = z['train_labels'].ravel()
    idx = rs.choice(len(X), 10, replace=False)
    for c, i in enumerate(idx):
        ax[r, c].imshow(X[i, 14], cmap='gray')
        ax[r, c].set_title(f'y={y[i]}', fontsize=7, pad=2)
        ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
    ax[r, 0].set_ylabel(name.replace('mnist3d', ''), fontsize=9)
fig.suptitle('MedMNIST3D, 10 random training volumes each, central axial slice', fontsize=11)
fig.tight_layout()
fig.savefig('results/medmnist_examples.png', dpi=150, bbox_inches='tight')
print('wrote results/medmnist_examples.png')

# --- synapse by class, several slices -------------------------------------
z = np.load(S['synapsemnist3d']); X = z['train_images']; y = z['train_labels'].ravel()
SL = [7, 11, 14, 17, 21]
fig, ax = plt.subplots(10, len(SL), figsize=(2.0 * len(SL), 19))
rows = list(rs.choice(np.where(y == 0)[0], 5, replace=False)) + \
       list(rs.choice(np.where(y == 1)[0], 5, replace=False))
for r, i in enumerate(rows):
    for c, s in enumerate(SL):
        ax[r, c].imshow(X[i, s], cmap='gray')
        ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
        if r == 0: ax[r, c].set_title(f'z={s}', fontsize=9)
    ax[r, 0].set_ylabel(f'class {y[i]}', fontsize=9,
                        color='tab:blue' if y[i] == 0 else 'tab:red')
fig.suptitle('SynapseMNIST3D: five volumes of each class, five slices through each',
             fontsize=11)
fig.tight_layout()
fig.savefig('results/synapse_by_class.png', dpi=150, bbox_inches='tight')
print('wrote results/synapse_by_class.png')
print()
for name in ORDER:
    z = np.load(S[name]); X = z['train_images']
    print('%-17s dtype %s  range %d-%d  mean %.1f  sd %.1f' % (
        name, X.dtype, X.min(), X.max(), X.mean(), X.std()))
