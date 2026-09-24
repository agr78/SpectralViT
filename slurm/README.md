# Cluster runs — IXI Table 2 reproduction and seed sweep

Partition `rtx8000`, account `mlsclemon`. Submit from the repo root.

---

## Status: Table 2 reproduces exactly

Verified 2026-09-24 on Python 3.14.6 / torch 2.11.0+cu128 / numpy 2.5.1 /
scikit-learn 1.9.0, from `/checkpoints/spectral_vit_master_state_se.pth`
(md5 `fde45136c8aaefd4b6671e3e76b67768`). Point estimates **and** bootstrap CIs
are character-for-character identical to the committed table for all five
stored models:

| Model | recomputed AUC | committed | Δ |
|---|---|---|---|
| Spec-ViT | 0.848021 | 0.848 | +0.0000 |
| U-Net | 0.795869 | 0.796 | −0.0001 |
| Spat-H | 0.736555 | 0.737 | −0.0004 |
| Spat-M | 0.653878 | 0.654 | −0.0001 |
| Swin | 0.606476 | 0.606 | +0.0005 |

The evaluation pipeline is version-stable across the 3.8→3.14 / torch 2.11
migration. The earlier worry that library versions moved the numbers is not
supported: they did not move at all.

**Caveat on the `*` significance markers.** They are *not* reproduced above.
The permutation test does not reseed between models — it continues the numpy
stream left by the bootstrap — so the p-value for each model depends on how
many models precede it in `all_model_probs`. Reproducing the markers requires
all seven rows, and `PCA+LR` / `PCA+MLP` are **not** in the checkpoint; they are
retrained live from the IXI volumes on every run. Those two rows, and only
those, remain exposed to the sklearn/torch upgrade.

## What is in the checkpoint

154,905 bytes, and it holds **no model weights**. Contents:

| Key | Contents |
|---|---|
| `results` | out-of-fold predicted probabilities, 5 folds × (114,113,113,113,113) = 566, float32, for `spec`, `spat`, `spat_matched`, `swin`, `unet`, plus `oof_y_true` |
| `histories` | per-model training loss curves |
| `config` | `N_COMP=128`, `PATCH_SIZE=12`, `N_FOLDS=5`, `LR=1e-4`, `VOL_SIZE=96`, `EPOCHS=500` |
| `rng_state` | python / numpy / torch_cpu / torch_gpu (8 GPU states) |

The zip holds only 9 tensor storages (one 5,056 B CPU RNG state and eight 816 B
GPU RNG states); everything else lives inline in the 141 KB pickle as numpy
arrays. Six trained models would be megabytes — `spectral_vit_ix_master_checkpoints.pth`
(13 MB, md5 `26efd1ca`) is the one holding weights.

`spectral_vit_master_state.pth` and `..._bak.pth` are byte-identical to each
other (md5 `dd9bc608`) and **differ** from `_se.pth`. `_se` is the Table 2 one.

`config` also settles the open question about the rebuttal's hyperparameter
selection: **PATCH_SIZE = 12, N_COMP = 128**. These are now the defaults in
`ixi_seed_sweep.sbatch`, so the sweep pins to the actual rebuttal selection
rather than to a fresh CV run.

## Running it

```bash
export SPECTRALVIT_MASTER_STATE=/checkpoints/spectral_vit_master_state_se.pth
sbatch slurm/ixi_repro.sbatch                 # step 1
sbatch slurm/ixi_seed_sweep.sbatch            # step 2 — defaults already pinned to 12 / 128
```

Step 1 with `retrain = False` needs the checkpoint plus the IXI volumes (for the
two live baseline rows). Step 1 with `retrain = True` is a fresh retrain and is
a *new measurement*, not a reproduction — the committed table came from the
restore path.

## Fixes applied to make this work

Four defects blocked the restore path. All are patched in the working tree:

1. **`torch.load(filename, map_location='cpu')` fails outright on torch ≥ 2.6** —
   `weights_only` defaults to `True` and this checkpoint is a pickled dict of
   numpy arrays. Raises `UnpicklingError`. Now passes `weights_only=False`.
2. **`PCA(n_components=N_COMP, whiten=True)` had no `random_state`.** With
   n_features ≫ n_samples, `svd_solver='auto'` resolves to `randomized`, which
   draws from the global numpy RandomState. Since `seed_everything(SEED)` reseeds
   that stream, the PCA basis varied with the seed — so a "seed sweep" would have
   varied the basis as well as the initialisation, which is not the experiment.
   Measured: different seed → max |Δ| in the basis of 8.1e-02. Now `random_state=0`
   at both notebook call sites and at `validate.py:61` (`validate.py:224` already
   had it; `networks.py:416` still does not, but is not on this path).
3. **The checkpoint path was resolved against cwd.** Now `SPECTRALVIT_MASTER_STATE`
   with the old relative name as default.
4. **A missing checkpoint fell through silently** to a `NameError` at
   `y_final = np.concatenate(oof_y_true)`. Now raises `FileNotFoundError` naming
   the env var.

Still worth doing, not done: the `retrain = True` branch never writes a master
state, which is why this file was nearly lost. Adding a `torch.save` of
`{'results': …, 'config': {'N_COMP': …, 'PATCH_SIZE': …}}` at the end of that
branch would stop the gap recurring.

## What varies across the sweep

| | |
|---|---|
| varies | `SPECTRALVIT_SEED` 0–4 → torch/numpy/`random` init, shuffle order |
| pinned | `KFold(random_state=0)` — identical folds across all five runs |
| pinned | `PATCH_SIZE=12`, `N_COMP=128` via env → CV selection skipped entirely |
| pinned | PCA `random_state=0` → identical spectral basis across seeds (fix 2) |

`PYTHONHASHSEED` is exported from the job script, not from Python.
`util.seed_everything` assigns `os.environ['PYTHONHASHSEED']` at runtime, which
does nothing — CPython fixes hash randomisation at interpreter startup.

## Determinism expectations

`CUBLAS_WORKSPACE_CONFIG=:4096:8` is exported. The notebook does not call
`torch.use_deterministic_algorithms(True)`; `seed_everything` sets
`cudnn.deterministic=True` / `benchmark=False`, which covers most but not all
kernels. rtx8000 is Turing, so there is no TF32 path — one fewer cross-machine
numerical difference than Ampere-or-later would introduce. Step 1 logs GPU name,
driver, torch/cuda/cudnn versions and the TF32 flags for after-the-fact
attribution.

## Resources

`--mem=64G` covers the 566×96³ float32 array (~2.0 GB) plus the PCA over the
566×884,736 flattened matrix, which dominates. `--time=24:00:00` is generous for
6 models × 5 folds × 500 epochs. A `retrain=False` run needs far less of both —
tighten before submitting five of them.
