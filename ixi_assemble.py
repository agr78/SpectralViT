"""Assemble the IXI table from saved out-of-fold predictions.

Arms are given as name=path pairs so cells can come from whichever run produced them at
the selected capacity, while Swin and the U-net -- which depend on neither the component
count nor the patch size -- are reused rather than recomputed. All runs must share the
same KFold(random_state=0) assignment; the label vectors are checked for equality.
"""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, '/workspace')
from sklearn.metrics import roc_auc_score, accuracy_score
from util import compute_delong_paired, compute_mcnemar

LABELS = {'spec': 'Spectral ViT (proposed)', 'spat': 'Standard spatial ViT',
          'match': 'Compact spatial ViT', 'swin': 'Swin ViT',
          'unet': 'U-net with attention'}
ORDER = ['spec', 'spat', 'match', 'swin', 'unet']

args = dict(a.split('=', 1) for a in sys.argv[1:] if '=' in a)
out = args.pop('out', '/workspace/results/ixi_final/ixi_table.csv')
d = {k: np.load(v) for k, v in args.items()}
ref = d['spec']['y']
for k, v in d.items():
    assert np.array_equal(v['y'], ref), f'fold assignment differs for {k}'

rows = []
for k in ORDER:
    if k not in d: continue
    z = d[k]; y, p, sizes = z['y'], z['probs'], z['fold_sizes']
    cuts = np.cumsum(sizes)[:-1]
    yf, pf = np.split(y, cuts), np.split(p, cuts)
    accs = [accuracy_score(a, (b > .5).astype(int)) for a, b in zip(yf, pf)]
    aucs = [roc_auc_score(a, b) for a, b in zip(yf, pf)]
    n = int(z['nparam'])
    r = dict(Model=LABELS[k], Params=n, Accuracy=np.mean(accs), Acc_sd=np.std(accs),
             AUC=np.mean(aucs), AUC_sd=np.std(aucs), Efficiency=np.mean(aucs)/(n/1e6),
             AUC_pooled=roc_auc_score(y, p))
    if k == 'spec':
        r['DeLong_p'] = r['McNemar_p'] = float('nan')
    else:
        r['DeLong_z'], r['DeLong_p'] = compute_delong_paired(ref, p, d['spec']['probs'])
        r['McNemar_chi2'], r['McNemar_p'] = compute_mcnemar(
            ref, (p > .5).astype(int), (d['spec']['probs'] > .5).astype(int))
    rows.append(r)

t = pd.DataFrame(rows)
os.makedirs(os.path.dirname(out), exist_ok=True)
t.to_csv(out, index=False)
pd.set_option('display.width', 220)
print(t.round(4).to_string(index=False))
print(f'\nwrote {out}')
