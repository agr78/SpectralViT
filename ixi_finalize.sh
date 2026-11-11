#!/bin/bash
# Wait for capacity selection, rerun only the arms that depend on it, then assemble the
# table. Swin and the U-net depend on neither the component count nor the patch size, so
# their out-of-fold predictions are reused rather than recomputed (~4 h saved).
set -uo pipefail
SEL=/workspace/results/ixi_select/selection.json
PY=/opt/dpeekenv/bin/python
for i in $(seq 1 720); do [ -f "$SEL" ] && break; sleep 30; done
[ -f "$SEL" ] || { echo "selection.json never appeared"; exit 1; }
NC=$("$PY" -c "import json;print(json.load(open('$SEL'))['N_COMP'])")
PT=$("$PY" -c "import json;print(json.load(open('$SEL'))['patch'])")
echo "selected N_COMP=$NC patch=$PT; rerunning spec,spat,match"
IXI_ARMS=spec,spat,match IXI_NCOMP="$NC" IXI_PATCH="$PT" \
  IXI_OUT=/workspace/results/ixi_final PYTHONPATH=/workspace \
  "$PY" -u ixi_regen.py 2>&1 | tr '\r' '\n' | grep -vE "Loading Data:|UserWarning|self.transformer"
echo "assembling"
"$PY" ixi_assemble.py \
  spec=/workspace/results/ixi_final/oof_spec.npz \
  spat=/workspace/results/ixi_final/oof_spat.npz \
  match=/workspace/results/ixi_final/oof_match.npz \
  swin=/workspace/results/ixi_regen/oof_swin.npz \
  unet=/workspace/results/ixi_regen/oof_unet.npz \
  out=/workspace/results/ixi_final/ixi_table.csv
