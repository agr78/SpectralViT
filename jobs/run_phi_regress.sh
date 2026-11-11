#!/usr/bin/env bash
# Equation 2 (per-coefficient embedding) on the brain-age regression, every sample size,
# smallest first so the n=25 decision point lands before the expensive cells start.
#
# Only the 'SpectralViT phi (Block-DCT)' arm runs; every baseline is untouched and the
# existing CSVs are not overwritten, because HCA_SUFFIX puts the output in its own
# namespace: hca_reg_n{N}_phi4096.csv.
#
#   PY=~/miniforge3/envs/dpeekenv/bin/python VOLS=~/spacelink/aHCP/derivatives/hca_vols_96 \
#     bash run_phi_regress.sh
#
# Override SIZES to run a subset:  SIZES="25 50" bash run_phi_regress.sh
set -u
PY=${PY:-$HOME/miniforge3/envs/dpeekenv/bin/python}
VOLS=${VOLS:-$HOME/spacelink/aHCP/derivatives/hca_vols_96}
SIZES=${SIZES:-"25 50 100 200 1000"}
LOGS=${LOGS:-$HOME/SpectralViT/results/regress}

# Some hosts ship a libstdc++ older than the conda build needs, and numpy then fails
# with "GLIBCXX_3.4.29 not found" before torch is even reached. The env ships its own;
# put it first. Harmless where the system library is new enough.
export LD_LIBRARY_PATH="$(dirname "$PY")/../lib:${LD_LIBRARY_PATH:-}"

# Fail before burning a GPU-day on a missing path: the volumes are the binding
# requirement and a wrong prefix otherwise surfaces only after the cache load.
[ -f "$VOLS.npy" ]      || { echo "missing $VOLS.npy"; exit 1; }
[ -f "${VOLS}_meta.npz" ] || { echo "missing ${VOLS}_meta.npz"; exit 1; }
"$PY" -c 'import torch;print("torch",torch.__version__,"|",torch.cuda.get_device_name(0))' || exit 1
mkdir -p "$LOGS"

for N in $SIZES; do
  echo "=== n=$N  $(date +%H:%M:%S)"
  HCA_ARMS="SpectralViT phi" HCA_SEEDS=5 HCA_SUFFIX=_phi4096 \
    "$PY" hca_regress.py "$N" "$VOLS" 2>&1 | tee "$LOGS/phi_n${N}.log"
  echo "=== n=$N done $(date +%H:%M:%S)"
done
