#!/bin/bash
#SBATCH --job-name=ixi-restore
#SBATCH --account=mlsclemon
#SBATCH --partition=rtx8000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/ixi-restore-%j.out
#SBATCH --error=slurm/logs/ixi-restore-%j.err

# Reproduce the committed Table 2 via the restore path (retrain = False):
# load stored out-of-fold predictions from the master state, retrain only the
# two live PCA+LR / PCA+MLP rows, then bootstrap + permutation over all seven.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
PY=/opt/dpeekenv/bin/python

export SPECTRALVIT_RETRAIN=0
export SPECTRALVIT_MASTER_STATE="${SPECTRALVIT_MASTER_STATE:-/checkpoints/spectral_vit_master_state_se.pth}"
export SPECTRALVIT_DATA="${SPECTRALVIT_DATA:-/homes/9/ar1326/spacelink/OneDrive/SpectralViT/data}"
export SPECTRALVIT_SEED=0
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "=== job $SLURM_JOB_ID on $(hostname) $(date -Is)"
echo "=== git $(git rev-parse --short HEAD) on $(git rev-parse --abbrev-ref HEAD)"
echo "=== master state: $SPECTRALVIT_MASTER_STATE"
md5sum "$SPECTRALVIT_MASTER_STATE"
nvidia-smi --query-gpu=name --format=csv,noheader
"$PY" -c "import sys,torch,sklearn,numpy;print('python',sys.version.split()[0],'torch',torch.__version__,'sklearn',sklearn.__version__,'numpy',numpy.__version__)"

OUT="slurm/logs/ixi_abl_cis.restore.${SLURM_JOB_ID}.ipynb"
"$PY" -m jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=-1 \
    --output "$(basename "$OUT")" --output-dir "$(dirname "$OUT")" \
    ixi_abl_cis.ipynb
echo "=== done: $OUT"
