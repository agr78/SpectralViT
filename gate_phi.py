"""Decide from the completed cells whether the larger sample-size run is worth its cost."""
from paths import PYTHON, REPO, RESULTS
import glob, os, subprocess, sys
import pandas as pd

MARGIN = 0.010
REPORTED = 'SpectralViT hcp (Block-DCT)'
R = f'{RESULTS}/sweep'


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
    cmd = ['sbatch', '-A', os.environ['SLURM_ACCOUNT'], '-p', os.environ['SLURM_PARTITION'], '--gres=gpu:1',
           '-c3', '--mem=48G', '-t72:00:00', '-a', '0-4%3',
           '--wrap', 'HCA_FREQ=4 HCA_SUFFIX=_phi4096_s$SLURM_ARRAY_TASK_ID '
                     'HCA_SEED_START=$SLURM_ARRAY_TASK_ID '
                     'HCA_SEEDS=$((SLURM_ARRAY_TASK_ID+1)) '
                     f'{PYTHON} '
                     f'{REPO}/hca_sweep.py 1000']
    print('  PASS -> submitting n=1000')
    print(subprocess.run(cmd, capture_output=True, text=True).stdout.strip())
else:
    print('  FAIL -> not submitting n=1000; nothing to cancel, the job was never created')
