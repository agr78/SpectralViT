"""Decide from the n=25/50 phi@4096 cells whether the n=1000 run is worth 25 h/task.

Submitted with --dependency=afterok on the phi4096 array, so it runs once those finish.
It submits the n=1000 array only if the criterion below passes, and otherwise exits
without submitting. There is nothing to cancel in the failing case: the expensive job is
never created.

Criterion, deliberately conservative and printed in full so the decision is auditable:
  * phi must not LOSE at n=25 by more than the run-to-run reproduction error (0.010 AUC),
    because n=25 is the sample size the paper's claim rests on; and
  * phi must WIN at n=50 by more than that same 0.010.
Both are paired on identical (seed, fold) against the reported arm at the same frequency.
Thresholds are the measured reproduction error, not a significance test: with 25 cells a
p-value here would mostly reflect cell count.
"""
import glob, subprocess, sys
import pandas as pd

MARGIN = 0.010          # measured run-to-run |AUC| difference, mean over duplicated cells
REPORTED = 'SpectralViT hcp (Block-DCT)'
R = '/workspace/results/sweep'


def delta(n):
    phi = pd.read_csv(f'{R}/hca_sweep_n{n}_phi4096.csv').drop_duplicates(['Seed', 'Fold', 'Model'])
    rep = pd.concat([pd.read_csv(f) for f in glob.glob(f'{R}/hca_sweep_n{n}_ep500_s*.csv')])
    rep = rep.drop_duplicates(['Seed', 'Fold', 'Model'])
    rep = rep[rep.Model == REPORTED]
    j = phi[['Seed', 'Fold', 'AUC']].merge(rep[['Seed', 'Fold', 'AUC']],
                                           on=['Seed', 'Fold'], suffixes=('_p', '_r'))
    if len(j) < 15:
        sys.exit(f'n={n}: only {len(j)} paired cells, refusing to decide on partial data')
    return j.AUC_p.mean() - j.AUC_r.mean(), len(j), phi.secs.median()


d25, k25, s25 = delta(25)
d50, k50, s50 = delta(50)
print(f'  n=25  phi - reported {d25:+.4f}  ({k25} paired cells, {s25:.0f} s/cell)')
print(f'  n=50  phi - reported {d50:+.4f}  ({k50} paired cells, {s50:.0f} s/cell)')
print(f'  rule: n=25 delta > {-MARGIN:+.3f} AND n=50 delta > {MARGIN:+.3f}')

if d25 > -MARGIN and d50 > MARGIN:
    cmd = ['sbatch', '-A', 'mlsclemon', '-p', 'rtx8000,rtx6000', '--gres=gpu:1',
           '-c3', '--mem=48G', '-t72:00:00', '-a', '0-4%3',
           '--wrap', 'HCA_FREQ=4 HCA_SUFFIX=_phi4096_s$SLURM_ARRAY_TASK_ID '
                     'HCA_SEED_START=$SLURM_ARRAY_TASK_ID '
                     'HCA_SEEDS=$((SLURM_ARRAY_TASK_ID+1)) '
                     '$HOME/miniforge3/envs/dpeekenv/bin/python '
                     '$HOME/SpectralViT/hca_sweep.py 1000']
    print('  PASS -> submitting n=1000')
    print(subprocess.run(cmd, capture_output=True, text=True).stdout.strip())
else:
    print('  FAIL -> not submitting n=1000; nothing to cancel, the job was never created')
