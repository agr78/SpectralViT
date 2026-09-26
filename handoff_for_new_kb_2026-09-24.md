# Handoff for a new knowledge base (2026-09-24): cluster map and rules, git preferences, results to date

## 1. Cluster: GPU/CPU map and rules
Accounts: `mlsclemon` (default), `lcnlemon`. Submit only from a login node (mlsc-login / cadiz); the analysis container has no SLURM client and no GPU (24 CPUs, 124 GB, /opt/synthseg python, cluster logs are EDT, container clock is UTC).

| partition | nodes (cores / GPUs) | GPU | CPU cap per job | limit | notes |
|---|---|---|---|---|---|
| dgx-a100 | A100-01..04, 128 / 8 | A100 40 GB | 16 (trainers use 4 with 2 GPUs) | 7 d | contended; all long runs live here |
| rtx8000 | rtx-04/05/07 32/10, rtx-06 32/9, rtx-08 32/8 | Quadro 8000 48 GB, rtx-08 A6000 | **3 (IT email 2026-09-22)** | 7 d | inference legs, head-only forks |
| rtx6000 | rtx-01/02, 32 / 8 | Quadro 6000 24 GB | 4 | 7 d | inference only; training this config OOMs |
| pubgpu-req | rtx-03 32/10 (tightest), luisa 52/10, sevilla 56/8, glaurung/gothmog/l40s-01..03/shelob 64/4, leo 64/1 | mixed | 3 | | multi-partition jobs take the smallest cap |
| lcna40 / lcna100 | luisa 52/10 / sevilla 56/8 | A40 / A100 | 5 / 7 | | |
| pubgpu | leo 64/1 (MaxCPUsPerNode 16) | | 16 | | |
| basic / pubcpu | CPU nodes (r440-xx) | none | | | scoring jobs; pubcpu skips the basic queue |
Cap rule: cap = floor(cores / GPUs) of the tightest node; a job listing several partitions takes the smallest. Every jobs/*.sbatch that can land on rtx8000/rtx6000 requests `--cpus-per-task=3`.

Rules that came from IT or from incidents:
- Never a job storm (a 60-job resubmission flood drew the IT email). Keep counts modest; inference legs are now chained one after another per arm (`LEG_SERIAL=1` in longrun.sbatch and fork.sbatch).
- No long-lived polling jobs (a squeue-dump job "consumed little while blocking a node"). Sequence with `--dependency=afterany:<ids>` or `--begin=<time>`; a `--begin` job that exits in seconds if its input is absent is fine.
- Run everything that does not need a GPU in the container: milestone WMH scoring (`tmp_mask_check/local_score.sh`), gates, gradient diagnostics, table building. ATLAS/BraTS scoring must stay on the cluster until the ground-truth path `/autofs/nas/churro_001/...` is bound into the container.
- Storage: the 1.9 TB map device (`/autofs/space/cadiz_001`, nvme) filled twice (old maps; unpruned milestone maps at ~50 GB per arm per milestone). Launchers refuse to submit legs below `50 x live long_* arms + 50` GB free and retry every 10 min; scored older milestones are pruned to MANIFEST.tsv + REGENERATE.sh; deletion lists live in `tmp_mask_check/delete_rejected.sh` (modes now / after-score / oldcode / superseded / variants).
- The home directory (`/homes/9/ar1326`, NFS vault) has a quota; a write under a full quota truncated train.py on 2026-09-24. Nothing of the pipeline should write there: snapshots can go to `claude-kb/jobs/snapshots` (`DEST=` in make_snapshot.sh), all launchers set `ulimit -c 0` and point torch/matplotlib/triton/bytecode caches to `claude-kb/jobs/.cache`. Known home consumers: miniforge3 14 GB, .local 5.5 GB, .vscode-server 1.8 GB, .cache 1.2 GB.
- Launch conventions: job names carry the run tag (`fork_<lever>_s<X>`, `long_<cfg>_s<seed>`, `inf_<tag>_<dataset>_<leg>_<frame>_<step>`); every job writes `claude-kb/jobs/<name>-<jobid>.out`; the user's terminal collapses multi-line pastes, so hand over single-line commands.

## 2. Git / GitHub preferences
- Repository `Pathology-Randomizer` (branch master, main for PRs). Author of record: Alexandra Grace Roberts <agr78@cornell.edu> (repo git config). She writes every commit message; stage and stop, never commit or push for her.
- No Claude attribution anywhere in the repo: no Co-Authored-By, no session links, no generated-by notes.
- One commit per run ("one HASH, one run"): incidental fixes wait and ride along with the next substantive change; the run-name hash is the main repo's HEAD at launch.
- Comments and docstrings: terse, describe the code as written, never a measurement or an experimental rationale (numbers only when structural, e.g. "56 channels"). A 30-line justification comment is rejected; 4 lines is right.
- American spelling everywhere (regularizer, randomizer, normalization, tumor, edema, center, artifact, gray, modeling).
- `.gitignore`: `*.sbatch`, `eval/`, `runs/`, `docs/` are ignored. New modules imported by train.py must be `git add`ed explicitly (target_utils.py was; it is staged). Launch scripts live in `claude-kb/jobs`, documentation in `claude-kb/jobs/docs`, never in the repo's ignored docs/.
- Manuscript (CVPR template under `claude-kb/jobs/manuscript`): bullet form, American spelling, oracle numbers only for benchmark comparison and labelled as such, whole datasets only.

## 3. Results to date (whole datasets, plain p1-p99 frame unless stated; per-case median AUC / AP)
Evaluation legs: WMH FLAIR (60), WMH T1 (60), ATLAS T1w (652-655), BraTS T1w necrotic core (1193), BraTS FLAIR whole tumor (1249). Rules: AUC primary, AP secondary, pooled AUPRC + global-oracle Dice only beside the UAD benchmark; nothing tuned on test data; one global readout; a lever must beat its paired same-seed control.

**Best row on record (2026-09-23): the frozen-trunk head branch `fork_headrel_s1` @150k** (control_s1 @100k trunk frozen; new MLP regression head with a per-scan tissue-relative channel; 50k head-only steps on composition + region-target data):
| leg | paired control | headrel plain | headrel batch-statistics readout |
|---|---|---|---|
| WMH FLAIR | 0.690 / 0.014 | 0.886 / 0.049 | 0.918 / 0.068 |
| WMH T1 | 0.583 / 0.010 | 0.625 / 0.017 | 0.614 / 0.013 |
| ATLAS | 0.700 / 0.009 | 0.793 / 0.020 | 0.775 / 0.021 |
| BraTS necrotic | 0.565 / 0.007 | 0.661 / 0.010 | 0.750 / 0.015 |
| BraTS FLAIR | 0.811 / 0.201 | 0.868 / 0.388 | 0.914 / 0.608 |
Passes the pre-registered rule on ancestor 1. Caveats: one ancestor (ancestor-2 copy scoring 2026-09-24 evening); its in-domain gate FAILED (0.70 on tissue-consistent composition cases vs 0.97 on ordinary synthetic lesions), so this is a readout result, not a generator result; the trunk ablation (same head, trunk trained: `headrel_full`) is worse on 4/5 legs by 0.04-0.07.

**Generator levers, two ancestors (fork design, 50k from control@100k):** region target = dark-leg lever both times (ATLAS +0.05 / +0.05, necrotic +0.01 / +0.04) with a bright-leg cost both times (WMH FLAIR -0.03 / -0.12) -> fails the rule; composition's wave-1 WMH FLAIR gain (+0.11) did not replicate (+0.01), dark legs down -> null; stacked both -> null. No generator lever passes.

**Long runs (5 x 500k, control x3 + coupled atrophy x2, plus stacked composition+region-target x2 started 2026-09-24):** at 300k WMH FLAIR plain 0.824 / 0.686 / 0.811 (controls), 0.718 / 0.825 (anat); ATLAS/BraTS at 300k pending scoring. The trajectory after 50k is noisy and roughly flat (0.66-0.90, swings ~0.1 per milestone); ATLAS and BraTS FLAIR crept up 50k -> 200k (ATLAS 0.58 -> 0.72 on seed 1); coupled atrophy is below control on both dark legs in both seeds at 200k.

**Readouts (same weights, different inference):** batch-statistics BN at 200k lifts BraTS necrotic (+0.10..+0.22) and WMH T1 (+0.07..+0.10) in 3/3 seeds and costs WMH FLAIR and BraTS FLAIR in 2/3 (the mirror of 50k) -> checkpoint-dependent, secondary row only. Skull-stripped input lifts WMH FLAIR +0.07..+0.12 and costs ATLAS -0.08..-0.14 and WMH T1 -0.09..-0.15 in 3/3 -> not a global readout. Median filter and seed ensembling both hurt.

**Diagnostics:** segmentation holds 70-72% of the shared-trunk gradient (2.3-2.6x the regression gradient, 12 synthetic cases at 100k/200k); in-domain regression AUC is flat at 0.94-0.96 from 60k on while anatomy Dice climbs 0.31 -> 0.53; linear probes on frozen features do not recover the BraTS necrotic cue (0.52-0.64 vs the per-class intensity rule's 0.75); uncertainty weighting fails structurally (weights ~ 1/loss value; collapsed anatomy Dice to 0.09 at commit 5a53f8b).

**Implemented, not yet run:** `--grad_balance imtl` (IMTL-G two-task gradient balancing on the trunk, scale-free, no hand-set weight; `norm` = equal-norm fallback) and `--trunk_lr_mult`; snapshot sprintN on cadiz; pre-registered branches J0 (head-only 50k more) / JA (joint plain sum, warm head) / JB (joint IMTL-G) from headrel_s1 @150k; ablation `headrel_nocomp`. Plan: `claude-kb/jobs/docs/trunk_dynamic_plan_2026-09-24.md`.

**Benchmark position (UAD benchmark, T1w only, arXiv 2512.01534):** our global-oracle Dice ceiling is 3-6x below the best benchmarked method on WMH-T1w, ATLAS and BraTS-T1w; FLAIR has no benchmark counterpart. Never quote the retracted "at par on WMH".

**BITS (Allen Brain format):** meeting held 2026-09-24; I/O only. Baseline: gzipped int16 NIfTI whole read 67 ms, raw 38 ms, uncompressed 24 ms, gz write 118 ms; reading is ~16% of a scoring pass. Read-only trial design and reader benchmark stub: `claude-kb/jobs/docs/bits_meeting_2026-09-24.md`, `tmp_mask_check/bench_reader.py`.
