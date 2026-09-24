#!/bin/bash
#SBATCH --job-name=hcp-%x
#SBATCH --account=mlsclemon
#SBATCH --partition=rtx8000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/hcp-%j.out
#SBATCH --error=slurm/logs/hcp-%j.err

# One script, three modes. Submit with --export (see slurm/README.md):
#   RUN_TAG=before  HCP_DISABLE_HEAD_FIX=1                  pre-fix baseline
#   RUN_TAG=after                                           classifier head trained
#   RUN_TAG=null    HCP_PERMUTE_LABELS=1                    training labels shuffled
#
# 'before' and 'after' differ only in whether the lazily-built classification
# head is materialised before AdamW captures model.parameters(). 'null' shuffles
# the TRAINING labels within each fold and leaves the TEST labels intact, giving
# the AUC floor this CV design produces for a model with nothing to learn.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
PY=/opt/dpeekenv/bin/python

export RUN_TAG="${RUN_TAG:-after}"
export HCP_DISABLE_HEAD_FIX="${HCP_DISABLE_HEAD_FIX:-0}"
export HCP_PERMUTE_LABELS="${HCP_PERMUTE_LABELS:-0}"
export SPECTRALVIT_RESULTS="${SPECTRALVIT_RESULTS:-$PWD/results/hcp_${RUN_TAG}}"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$SPECTRALVIT_RESULTS"

echo "=== job $SLURM_JOB_ID  tag=$RUN_TAG  host $(hostname)  $(date -Is)"
echo "=== git $(git rev-parse --short HEAD) on $(git rev-parse --abbrev-ref HEAD)"
git diff --quiet || echo "=== WARNING: dirty working tree"
echo "=== HCP_DISABLE_HEAD_FIX=$HCP_DISABLE_HEAD_FIX  HCP_PERMUTE_LABELS=$HCP_PERMUTE_LABELS"
echo "=== results -> $SPECTRALVIT_RESULTS"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"$PY" -c "import sys,torch,sklearn,numpy;print('python',sys.version.split()[0],'torch',torch.__version__,'sklearn',sklearn.__version__,'numpy',numpy.__version__)"

OUT="slurm/logs/hcp.${RUN_TAG}.${SLURM_JOB_ID}.ipynb"
"$PY" -m jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=-1 \
    --output "$(basename "$OUT")" --output-dir "$(dirname "$OUT")" \
    hcp.ipynb

echo "=== done: $OUT"
echo "=== per-fold metrics : $SPECTRALVIT_RESULTS/*/classification_results.csv"
echo "=== per-fold probs   : $SPECTRALVIT_RESULTS/*/per_fold_probs.npz"
