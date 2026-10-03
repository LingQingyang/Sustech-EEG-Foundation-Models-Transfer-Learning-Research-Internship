# LaBraM × M3CV revised four-spectrum analysis v6.0

This package implements the 2026-09-07 revised research proposal as an **analysis-only** pipeline. It never re-runs full fine-tuning. It consumes the existing v5.0 final artifacts and performs SVD analysis, model reconstruction/evaluation, matched-random baselines, STI, tables, and plots.

## Files

- `analyze_labram_m3cv_four_spectra_v6_0.py` — main analysis pipeline.
- `run_labram_m3cv_four_spectra_v6_0.sh` — Volcengine launcher with the project defaults.

## Scientific contract implemented

1. The immutable analysis atom is the original rank-one operator `Z_i = u_i v_i^T`.
2. Individual Energy and Individual Functional use the same matrix-wise singular-value order.
3. The functional cutoff `q*` defines the candidate functional set used by both Shared spectra.
4. Single/Full contexts are constructed **within the identical matrix position only**. No cross-matrix projection is allowed.
5. Shared Energy uses
   `epsilon_i^sh = sigma_i^2 ||P_C Z_i||_F^2`
   directly. There is no separate “sharedness” variable.
6. Candidate functional components from all 72 matrices are globally ordered by `epsilon_i^sh` only after the per-matrix projection energy has been computed.
7. Shared Functional uses that exact order but always writes back the candidate's **original** `sigma_i Z_i`. Context-side or projected operators never enter the candidate model.
8. The functional reconstruction changes only the 72 analyzed weights; the candidate fitted head and all other fine-tuned parameters remain adapted.
9. Shared-Energy random baselines are shape/rank/rank-one matched and retain the candidate empirical singular values. The launcher now defaults to a lightweight `fast` profile; `PROFILE=full` restores the proposal-complete Monte Carlo and replicate grid.
10. STI observed values remain paper-aligned with `k = floor(r/T)` for five tasks and all `2^5 = 32` replicate combinations.

### Frobenius-space projection implementation

The code does **not** explicitly flatten every `Z_i` into a 40,000- or 160,000-dimensional vector. For context operators `Z_j = u_j v_j^T`, it uses the exact Gram identity

`<Z_j, Z_k>_F = (u_j^T u_k)(v_j^T v_k)`.

If `G` is the context-operator Gram matrix and `b_i` contains `<Z_j, Z_i>_F`, then

`||P_C Z_i||_F^2 = b_i^T G^+ b_i`.

This is algebraically identical to explicit vectorization + orthogonal projection, but it avoids building huge flattened matrices. The numerical rank tolerance and context dimensions are recorded in outputs/manifest. An internal synthetic self-test additionally checks this operator algebra and Full ≥ Single nesting.


## Fast profile for the current time budget

The launcher now defaults to `PROFILE=fast`. This does **not** change the four-spectrum definitions. It only reduces redundant replication and numerical resolution:

- Functional curves are still reported by **raw vertical-axis targets**. Fast mode uses 0.10 bACC targets rather than 0.05. Energy targets remain 0.05 because Energy is inference-free.
- Hidden bracketing probes are reduced from 21 q/fraction points to 9 anchors: `0, .05, .10, .20, .35, .50, .70, .85, 1`.
- The 99% functional cutoff uses a 2% local lattice rather than 1%.
- Shared analyses use aligned replicate tracks only: candidate rep01 with context rep01, candidate rep02 with context rep02. This gives 50 shared comparisons rather than the proposal-complete 240 while retaining two independent replicate observations.
- The same candidate model and evaluation loader are reused across all of its Shared Functional contexts.
- Fast mode trusts the already-certified stored full-checkpoint metric at evaluator initialization; q=1 and shared-endpoint equality checks still replay the relevant reconstructed models.
- Evaluation batch size defaults to 128.
- Fast profile defaults to 128 Shared-Energy random draws and 128 STI random draws. Shared Energy uses the matched-random 95th-percentile envelope in fast mode (full mode retains the 99th percentile). These are robust exploratory references, not publication-grade tail estimates.

To restore the denser proposal-complete analysis later:

```bash
PROFILE=full bash ~/run_labram_m3cv_four_spectra_v6_0.sh <mode>
```

Because the profile is included in the sampling policy/manifest, fast and full caches are not silently mixed.

## Server setup

Copy both executable files to the same directory on the server, for example `/home/linqy/`:

```bash
chmod +x ~/run_labram_m3cv_four_spectra_v6_0.sh
```

The launcher defaults to the existing final-run root:

```text
/omni-eeg-01/task calibration/results/tsv_labram_m3cv/tsv_labram_m3cv_v5_0
```

and writes the new analysis to:

```text
/omni-eeg-01/task calibration/results/tsv_labram_m3cv/tsv_labram_m3cv_v5_0/analysis_four_spectra_v6_0
```

The Python code tries to auto-discover the exact v5.0 training runner because it needs its LaBraM construction/data/evaluation functions for checkpoint-compatible inference. If auto-discovery fails, set it explicitly:

```bash
CORE_SCRIPT="$HOME/run_tsv_labram_m3cv_v0_3_4.py" \
  bash ~/run_labram_m3cv_four_spectra_v6_0.sh preflight
```

The certified V5 final reused `run_tsv_labram_m3cv_v0_3_4.py` as its frozen numerical core. The analyzer imports that core, patches the five-task registry exactly as the previous V5 analyzer did (including Rest), and uses it only for model/data/evaluation compatibility. No training entry point is called.

## Recommended staged run

For the current overnight + half-day window, `main` is the recommended one-shot run. It includes the matched-random Shared-Energy baseline, while keeping the fast replicate grid and sparse functional probing. Every expensive comparison writes resumable progress artifacts.

The quickest recommended primary run is a single process:

```bash
bash ~/run_labram_m3cv_four_spectra_v6_0.sh main
```

`main` runs heat map + Individual + Shared Energy + Shared Functional + STI + plots, but deliberately leaves the heavier Shared-Energy Monte Carlo null for later. This also avoids recomputing all 720 SVDs once per stage. If the primary run finishes early, add:

```bash
bash ~/run_labram_m3cv_four_spectra_v6_0.sh random
bash ~/run_labram_m3cv_four_spectra_v6_0.sh plot
```

```bash
bash ~/run_labram_m3cv_four_spectra_v6_0.sh preflight
bash ~/run_labram_m3cv_four_spectra_v6_0.sh heatmap
bash ~/run_labram_m3cv_four_spectra_v6_0.sh individual
bash ~/run_labram_m3cv_four_spectra_v6_0.sh shared-energy
bash ~/run_labram_m3cv_four_spectra_v6_0.sh shared-functional
bash ~/run_labram_m3cv_four_spectra_v6_0.sh sti
# lower priority if the clock is tight:
bash ~/run_labram_m3cv_four_spectra_v6_0.sh random
bash ~/run_labram_m3cv_four_spectra_v6_0.sh plot
```

`all` is available, but the random and functional stages are intentionally computationally substantial:

```bash
bash ~/run_labram_m3cv_four_spectra_v6_0.sh all
```

The launcher defaults to `RESUME=1`. To deliberately rebuild a stage:

```bash
RESUME=0 bash ~/run_labram_m3cv_four_spectra_v6_0.sh individual
```

Fast mode already uses a small Shared-Energy null. Override it explicitly if you want a different quick reference:

```bash
SHARED_RANDOM_N=100 bash ~/run_labram_m3cv_four_spectra_v6_0.sh random
```

STI's matched-random repeat count is independently tunable:

```bash
STI_RANDOM_N=100 bash ~/run_labram_m3cv_four_spectra_v6_0.sh sti
```

## Main outputs

The output directory contains, among others:

- `training_summary.csv`
- `weight_update_heatmap.csv`
- `individual_spectra.csv`
- `individual_summary.csv`
- `shared_energy_spectra.csv`
- `shared_energy_random_baseline.csv`
- `shared_functional_spectra.csv`
- `shared_summary.csv`
- `sti_paper.csv`
- `sti_random_baseline.csv`
- `analysis_manifest_v6_0.json`
- `figures/*.png` and matching `*.pdf`
- `progress/` with resumable per-run/per-comparison checkpoints

A partial stage writes `_STAGE_<MODE>_SUCCESS.json`. `_ANALYSIS_V6_0_SUCCESS.json` is emitted only when all proposal-required analysis tables exist, so a partial run cannot masquerade as a complete result.

## Mandatory sanity checks implemented

- SVD reconstruction error.
- Left/right singular-vector orthonormality.
- `q=1` Individual Functional reproduces the saved full checkpoint bACC.
- `epsilon_i^sh` is constrained to `[0, sigma_i^2]` via the projection fraction in `[0,1]`.
- Shared-Energy endpoint `Gamma` is constrained to `[0,1]` and is not renormalized to one.
- Full-context projection energy is never below the corresponding Single-context projection energy within numerical tolerance.
- Shared Functional endpoint equals Individual Functional at `q*` within the configured metric tolerance.
- Candidate reconstruction is separated from context selection: only original candidate SVD indices are passed to model reconstruction.
- Existing final artifacts, hashes, protocol fingerprint, W0 digest, checkpoint channel mapping, and saved metrics are audited before inference.

## Interpretation boundary

All functional curves use the same pooled task data used for fitting. Their bACC is therefore a **functional-retention audit**, not held-out generalization. STI is a structural interference-potential reference because this analysis does not merge task updates into an actual multi-task model.
