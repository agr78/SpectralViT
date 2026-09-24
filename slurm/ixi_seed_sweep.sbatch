#!/bin/bash
#SBATCH --job-name=ixi-seed
#SBATCH --account=mlsclemon
#SBATCH --partition=rtx8000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --output=slurm/logs/ixi-seed-%A_%a.out
#SBATCH --error=slurm/logs/ixi-seed-%A_%a.err

# Step 2 of 2: seed sweep. DO NOT SUBMIT until step 1 has run and you have
# decided whether the reproduction is close enough to build on -- and until
# you have filled in the two hyperparameters below from step 1's
# "SELECTED_HPARAMS" line.
#
# Only initialisation varies:
#   - SPECTRALVIT_SEED varies 0..4  (torch/numpy/random init, shuffle order)
#   - KFold stays pinned at random_state=0 in the notebook, so all five runs
#     see identical folds
#   - PATCH_SIZE and N_COMP are pinned, so the CV selection does not re-run
#     and cannot drift with the seed

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

PY=/opt/dpeekenv/bin/python

# >>> FILL THESE IN FROM STEP 1 <<<
PATCH_SIZE_PINNED="${PATCH_SIZE_PINNED:-12}"   # recovered from master-state config
N_COMP_PINNED="${N_COMP_PINNED:-128}"     # recovered from master-state config
if [[ -z "$PATCH_SIZE_PINNED" || -z "$N_COMP_PINNED" ]]; then
    echo "ERROR: pin the hyperparameters first." >&2
    echo "  grep SELECTED_HPARAMS slurm/logs/ixi-repro-*.out" >&2
    echo "  then: sbatch --export=ALL,PATCH_SIZE_PINNED=<p>,N_COMP_PINNED=<n> slurm/ixi_seed_sweep.sbatch" >&2
    exit 2
fi

export SPECTRALVIT_DATA="${SPECTRALVIT_DATA:-/homes/9/ar1326/spacelink/OneDrive/SpectralViT/data}"
export SPECTRALVIT_SEED="${SLURM_ARRAY_TASK_ID}"
export SPECTRALVIT_PATCH_SIZE="$PATCH_SIZE_PINNED"
export SPECTRALVIT_N_COMP="$N_COMP_PINNED"
export PYTHONHASHSEED="${SLURM_ARRAY_TASK_ID}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "=== array task $SLURM_ARRAY_TASK_ID | seed=$SPECTRALVIT_SEED | host $(hostname)"
echo "=== pinned PATCH_SIZE=$SPECTRALVIT_PATCH_SIZE N_COMP=$SPECTRALVIT_N_COMP"
echo "=== git: $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name --format=csv,noheader

OUT="slurm/logs/ixi_abl_cis.seed${SLURM_ARRAY_TASK_ID}.${SLURM_ARRAY_JOB_ID}.ipynb"
"$PY" -m jupyter nbconvert \
    --to notebook --execute \
    --ExecutePreprocessor.timeout=-1 \
    --output "$(basename "$OUT")" \
    --output-dir "$(dirname "$OUT")" \
    ixi_abl_cis.ipynb

echo "=== done seed $SPECTRALVIT_SEED -> $OUT"
