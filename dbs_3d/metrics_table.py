"""Patient-level Table-3 metrics for the selected cell.

The original Table 3 reported AUC / balanced accuracy / specificity / F1 computed
slice-wise, which is why specificities like 0.939 appear despite there being only
3 external non-responders. At the patient level specificity is quantised to
{0, 1/3, 2/3, 1}, so it is reported here with that granularity made explicit.

Threshold follows the project convention: maximise balanced accuracy over
np.linspace(0.01, 0.9, 100) on the training cohort, then apply to the test set.
"""
import sys, os, glob, pickle
SPDIR = os.path.dirname(os.path.abspath(__file__))
REPO  = os.path.dirname(SPDIR)
OUT   = os.path.join(REPO, 'docs', 'dbs_3d')
sys.path.insert(0, REPO); sys.path.insert(0, SPDIR); os.chdir(REPO)
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score, confusion_matrix

sig = lambda z: 1.0 / (1.0 + np.exp(-z))


def calibrate_balanced(y, p):
    """threshold maximising balanced accuracy, project convention"""
    best, bt = -1, 0.5
    for t in np.linspace(0.01, 0.9, 100):
        b = balanced_accuracy_score(y, (p >= t).astype(int))
        if b > best: best, bt = b, t
    return bt


def metrics(y, p, thr):
    yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    return dict(AUC=roc_auc_score(y, p),
                BalAcc=balanced_accuracy_score(y, yh),
                Sens=tp / (tp + fn) if (tp + fn) else np.nan,
                Spec=tn / (tn + fp) if (tn + fp) else np.nan,
                F1=f1_score(y, yh, zero_division=0),
                TN=tn, FP=fp, FN=fn, TP=tp)


CELLS = ['lap_16_lap', 'lap_16_none', 'rowmajor_16_none', 'rowmajor_16_lap', 'l1_16_none']
rows = []
for tag in CELLS:
    f = os.path.join(OUT, tag + '.pkl')
    if not os.path.exists(f): continue
    r = pickle.load(open(f, 'rb'))
    bB, P = r['bB'], r['preds']
    bA = r['bA']

    # ensemble: average predictions, then score once
    pe = sig(P.mean(0))
    for lab, thr in [('ensemble @0.5', 0.5),
                     ('ensemble @test-cal', calibrate_balanced(bB, pe))]:
        m = metrics(bB, pe, thr); m.update(cell=tag, estimator=lab, thr=thr)
        rows.append(m)

    # per-seed: score each model, then average the metrics
    per = [metrics(bB, sig(p), 0.5) for p in P]
    avg = {k: np.mean([d[k] for d in per]) for k in ('AUC', 'BalAcc', 'Sens', 'Spec', 'F1')}
    sd = {k: np.std([d[k] for d in per]) for k in ('AUC', 'BalAcc', 'Spec', 'F1')}
    avg.update(cell=tag, estimator='per-seed mean @0.5', thr=0.5,
               TN=np.mean([d['TN'] for d in per]), FP=np.mean([d['FP'] for d in per]),
               FN=np.mean([d['FN'] for d in per]), TP=np.mean([d['TP'] for d in per]))
    avg.update({k + '_sd': v for k, v in sd.items()})
    rows.append(avg)

df = pd.DataFrame(rows)
cols = ['cell', 'estimator', 'AUC', 'BalAcc', 'Sens', 'Spec', 'F1', 'TN', 'FP', 'FN', 'TP']
print("=== patient-level metrics, CHH external (34 responders / 3 non-responders) ===")
print(df[cols].round(3).to_string(index=False))
print()
print("specificity is quantised: 3 non-responders -> only 0, 0.333, 0.667, 1.000 are attainable")
print()
sdc = [c for c in df.columns if c.endswith('_sd')]
if sdc:
    print("per-seed variability:")
    print(df[df.estimator == 'per-seed mean'][['cell'] + sdc].round(3).to_string(index=False))
df.to_csv(os.path.join(OUT, 'metrics_table.csv'), index=False)
print(f"\nwrote {os.path.join(OUT, 'metrics_table.csv')}")
print("\nNOTE: @0.5 is uncalibrated; @test-cal is calibrated ON CHH and is therefore")
print("optimistic. The honest value lies between them; a train-calibrated threshold")
print("needs the MSW out-of-fold predictions, which the cell pickles do not store.")
