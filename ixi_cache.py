"""Cache the IXI volumes into an array any host can read.

usage: python ixi_cache.py [out_prefix]
  writes <prefix>.npy (volumes, float32) and <prefix>_meta.npz (labels)"""
from paths import DATA
import os
import sys

import numpy as np

from loader import load_ixi_data

PRE = (sys.argv[1] if len(sys.argv) > 1
       else f'{DATA}/ixi_vols_96')
IXI_DIR = os.environ.get('IXI_DIR', '/checkpoints/data/IXI_extracted/')
VOL = int(os.environ.get('IXI_VOL_SIZE', '96'))

os.makedirs(os.path.dirname(PRE), exist_ok=True)
print(f'source {IXI_DIR}; crop {VOL}^3; dtype float32; out {PRE}.npy')

X, Y = load_ixi_data(IXI_DIR, os.path.join(IXI_DIR, 'IXI.csv'), VOL)
print(f'loaded {X.shape} {X.dtype}, labels {np.bincount(Y.astype(int))}'
      f' ({X.nbytes / 1e9:.2f} GB)')
assert X.ndim == 4 and X.shape[1:] == (VOL, VOL, VOL), X.shape
assert len(X) == len(Y), (len(X), len(Y))

np.save(PRE + '.npy', X)
np.savez(PRE + '_meta.npz', y=Y, vol=VOL, n=len(Y))
print(f'wrote {PRE}.npy and {PRE}_meta.npz')
