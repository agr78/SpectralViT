#!/bin/bash
# Does selecting at n=20 (what 5-fold inner CV actually ranks) misrepresent n=25?
# Measures the 20->25 learning-curve gain for every spatial patch and the leading spectral
# configs. If the high-capacity patches gain much more than the spectral configs, inner CV
# systematically underrates the BASELINE's best option and the comparison is biased.
set -u
PY=/opt/dpeekenv/bin/python
PAR=${PAR:-3}
OUT=/workspace/results/medmnist_n20
NPZS="/workspace/data/medmnist/adrenalmnist3d.npz /workspace/data/medmnist/fracturemnist3d.npz
/workspace/data/medmnist/nodulemnist3d.npz /workspace/data/medmnist/synapsemnist3d.npz
/workspace/data/medmnist/vesselmnist3d.npz /agent-home/medmnist/organmnist3d.npz"
cell() {
  local tag=$1 envs=$2 npz=$3 ds=$4
  local o="$OUT/$tag/$ds"; mkdir -p "$o"
  env $envs MEDMNIST_NPZ="$npz" MEDMNIST_OUT="$o" MEDMNIST_N=20 MEDMNIST_SEEDS=5 \
      "$PY" medmnist_sweep.py > "$o/log.txt" 2>&1 && echo "ok $tag $ds" || echo "FAIL $tag $ds"
}
echo "=== start $(date +%H:%M:%S)"
while IFS=$'\t' read -r tag envs; do
  for npz in $NPZS; do
    ds=$(basename "$npz" .npz)
    [ -f "$OUT/$tag/$ds/medmnist_perseed.csv" ] && continue
    while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do wait -n; done
    cell "$tag" "$envs" "$npz" "$ds" &
  done
done <<'JOBS'
patch2	MEDMNIST_PATCH=2 MEDMNIST_BASIS=pca MEDMNIST_COMP=4
patch4	MEDMNIST_PATCH=4 MEDMNIST_BASIS=pca MEDMNIST_COMP=4
patch7	MEDMNIST_PATCH=7 MEDMNIST_BASIS=pca MEDMNIST_COMP=4
patch14	MEDMNIST_PATCH=14 MEDMNIST_BASIS=pca MEDMNIST_COMP=4
patch28	MEDMNIST_PATCH=28 MEDMNIST_BASIS=pca MEDMNIST_COMP=4
dct_energy_f2_b6	MEDMNIST_BASIS=dct_energy MEDMNIST_FREQ=2 MEDMNIST_BANDS=6
dct_energy_f2_b12	MEDMNIST_BASIS=dct_energy MEDMNIST_FREQ=2 MEDMNIST_BANDS=12
dct_oriented_f3	MEDMNIST_BASIS=dct_oriented MEDMNIST_FREQ=3 MEDMNIST_ORAD=6 MEDMNIST_ORI=6
blockpca_ov	MEDMNIST_BASIS=blockpca MEDMNIST_BLOCK=14 MEDMNIST_STRIDE=7 MEDMNIST_COMP=8
dct_f3_s7	MEDMNIST_BASIS=dct MEDMNIST_BLOCK=14 MEDMNIST_STRIDE=7 MEDMNIST_FREQ=3
JOBS
wait
echo "=== done $(date +%H:%M:%S)"
