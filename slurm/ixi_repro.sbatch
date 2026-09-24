#!/bin/bash
#SBATCH --job-name=ixi-repro
#SBATCH --account=mlsclemon
#SBATCH --partition=rtx8000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/ixi-repro-%j.out
#SBATCH --error=slurm/logs/ixi-repro-%j.err

# Step 1 of 2: does ixi_abl_cis.ipynb still produce the committed Table 2
# (Spec-ViT AUC 0.848, U-Net 0.796) under Python 3.14.6 / torch 2.11.0?
#
# Read the CAVEAT in slurm/README.md before interpreting the result. The
# committed table was produced by LOADING spectral_vit_master_state_se.pth,
# which is gitignored and absent from the repo. This job RETRAINS. A mismatch
# therefore does not by itself implicate the library upgrade.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

PY=/opt/dpeekenv/bin/python

# --- environment ------------------------------------------------------------
# Data root: no $HOME assumption (cluster/cadiz/container homes differ).
export SPECTRALVIT_DATA="${SPECTRALVIT_DATA:-/homes/9/ar1326/spacelink/OneDrive/SpectralViT/data}"
export SPECTRALVIT_SEED=0

# Must be exported BEFORE the interpreter starts. util.seed_everything sets
# os.environ['PYTHONHASHSEED'] at runtime, which has no effect on the already
# running process -- hash randomisation is fixed at startup.
export PYTHONHASHSEED=0

# Deterministic cuBLAS reductions. Required for torch.use_deterministic_algorithms
# to be usable at all on CUDA >= 10.2; harmless otherwise.
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# --- provenance -------------------------------------------------------------
echo "=== job $SLURM_JOB_ID on $(hostname) at $(date -Is)"
echo "=== git: $(git rev-parse --short HEAD) $(git rev-parse --abbrev-ref HEAD)"
git diff --quiet || echo "=== WARNING: working tree is dirty"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"$PY" -c "import sys,torch,numpy,sklearn,scipy;\
print('python',sys.version.split()[0]);\
print('torch',torch.__version__,'cuda',torch.version.cuda,'cudnn',torch.backends.cudnn.version());\
print('numpy',numpy.__version__,'sklearn',sklearn.__version__,'scipy',scipy.__version__);\
print('tf32 matmul',torch.backends.cuda.matmul.allow_tf32,'| tf32 cudnn',torch.backends.cudnn.allow_tf32)"
echo "=== SPECTRALVIT_DATA=$SPECTRALVIT_DATA"
ls -d "$SPECTRALVIT_DATA/IXI_extracted" >/dev/null || { echo "DATA ROOT MISSING"; exit 2; }

# --- run --------------------------------------------------------------------
OUT="slurm/logs/ixi_abl_cis.repro.${SLURM_JOB_ID}.ipynb"
"$PY" -m jupyter nbconvert \
    --to notebook --execute \
    --ExecutePreprocessor.timeout=-1 \
    --output "$(basename "$OUT")" \
    --output-dir "$(dirname "$OUT")" \
    ixi_abl_cis.ipynb

echo "=== executed notebook: $OUT"
echo "=== grep the log above for 'SELECTED_HPARAMS' -- you need those two"
echo "=== numbers to pin the seed sweep (step 2)."
