# LaBraM × M3CV Task Adaptation Geometry V5.0

V5.0 is the cleaned, archive-ready implementation of the five-task analysis protocol. The active V5 source set is deliberately reduced to four files:

```text
run_task_geometry_v5_0.py
analyze_task_geometry_v5_0.py
run_task_geometry_v5_0.sh
README_TASK_GEOMETRY_V5_0.md
```

The already-existing `run_tsv_labram_m3cv_v0_3_4.py` remains a **frozen numerical core dependency**. V5.0 does not add another adapter or helper around it. The new training Python file is the single orchestration/provenance entry point; the old core continues to provide the already-audited model loading, optimizer, epoch loop, checkpoint writing and 72-matrix Delta-W extraction.

## 1. Final V5.0 run grid

The final analysis grid is exactly 5 tasks × 2 replicates = 10 independent fine-tunes from the same pretrained LaBraM backbone $W_0$.

| Run | V5.0 source |
|---|---|
| Rest/rep01 | new V5.0 training |
| Rest/rep02 | new V5.0 training |
| Motor/rep01 | certified legacy v0.3.4 reuse |
| Motor/rep02 | **new V5.0 retraining** |
| P300/rep01 | certified legacy v0.3.4 reuse |
| P300/rep02 | certified legacy v0.3.4 reuse |
| SSS/rep01 | certified legacy v0.3.4 reuse |
| SSS/rep02 | certified legacy v0.3.4 reuse |
| TS/rep01 | certified legacy v0.3.4 reuse |
| TS/rep02 | certified legacy v0.3.4 reuse |

The old legacy `Motor/rep02` is **rejected by policy** and is never linked into the V5 grid. V5.0 leaves the historical v0.3.4 root untouched so the provenance trail remains auditable; the rejected artifact is simply excluded and replaced by a fresh V5 run.

The seven approved legacy runs are linked at the **replicate-directory level**, not at the whole-task-directory level. This is essential because Motor has mixed provenance: rep01 is legacy while rep02 is new.

## 2. Frozen common endpoint

V5.0 does not run a new Rest pilot. The common endpoint remains

$$
E^*=20.
$$

It is inherited from the certified original v0.3.4 eight-run pilot, where all 8 pilot trajectories failed the strict rolling plateau certification and the explicit `pilot_max_epoch_fallback` rule selected the full 20-epoch budget.

Therefore V5.0 makes the following precise claim:

> The endpoint was selected by the original four-task eight-run pilot and then frozen. The new Motor/rep02 and Rest/rep01–02 runs start from the same $W_0$, use the same optimization and seed contract, and train for exactly the same 20 epochs. V5.0 does not claim that a new ten-run pilot certified $E^*$.

The `pilot` launcher mode is intentionally disabled.

## 3. Rest contract

The actual Session-1 Rest H5 has been inspected. V5.0 uses the raw labels exactly as stored:

```text
0 = Beg_EC (Task 1)
2 = Beg_EO (Task 2)
```

The file is:

```text
/omni-eeg-01/task calibration/dataset/m3cv/m3cv_resting_session1_4s.h5
```

It contains 6300 samples, balanced 3150/3150, and 95 subjects. The preflight remains fail-closed on the H5 label metadata rather than silently remapping it.

## 4. Fixed training contract

All new V5 runs use the same contract as the accepted legacy runs:

- LaBraM base checkpoint: `labram-base.pth`
- exact positional channel mapping: EEG channels `1..64`, with CLS position `0` prepended internally
- pooled Session-1 subjects and samples, no held-out split
- full backbone + task head fine-tuning
- AdamW
- backbone LR `1e-4`
- head LR `1e-3`
- weight decay `0`
- betas `(0.9, 0.999)`
- epsilon `1e-8`
- class-weighted cross entropy
- batch size `32`, evaluation batch size `64`
- AMP off
- no gradient clipping, scheduler, warm-up or early stopping
- seed base `20260819`
- model initialization seed `314159`
- exactly `20` epochs for the final new runs

Each final run must produce a complete adapted checkpoint and exactly 72 Delta-W matrices: Q, K, V, O, fc1 and fc2 in each of 12 Transformer blocks.

## 5. What V5.0 fixes relative to the transitional V3/V4 scripts

V5.0 removes the adapter/helper pile-up and fixes several provenance and resume hazards:

1. **Per-run provenance.** Provenance is keyed by `task/repXX`, so mixed Motor provenance is represented correctly. A task-level `legacy/new` flag is no longer accepted.
2. **Rep-level legacy links.** Only the seven approved replicate directories are linked. The rejected legacy Motor/rep02 cannot leak into the V5 root.
3. **Truthful endpoint contract.** The V5 preflight explicitly records `frozen_E_star_from_certified_legacy_pilot`; it no longer carries the mature core's generic statement that a matching current-root pilot is required.
4. **Fresh V5 root.** Results go to `tsv_labram_m3cv_v5_0`, so partial V3/V4 analysis files cannot be mistaken for V5 products.
5. **Fail-closed input reuse.** Before inspection/training/analysis, H5 size/mtime, label hashes, subject hashes, checkpoint hash, modeling-file hash and the V5 orchestrator hash are checked against the certified preflight.
6. **Exact run-artifact audit.** Approved runs are checked for identity, endpoint, W0, deterministic seeds, optimizer contract, metadata/_SUCCESS hashes and the exact 72-module Delta-W grid/shapes.
7. **Checkpoint semantic audit during functional evaluation.** The analyzer checks task, replicate, protocol fingerprint, W0 digest and channel mapping inside each checkpoint before using it.
8. **Resume signatures.** Individual, principal-angle, overlap and STI stages bind resume markers to the final-manifest hash, analyzer code hash and analysis settings. Stale partial analysis is refused rather than silently reused.
9. **Matched-random cache fix.** The random seed and rank-1-Z geometry version are now part of the cache signature. Changing the null seed cannot accidentally reuse an older cache.
10. **Paper STI rank is exact.** The implementation now uses the literal `floor(r_full/T)` rule and fails if that were ever to produce an invalid zero rank instead of silently replacing it by one.

## 6. Recommended server layout

Put the four V5 files in `~/` next to the existing mature core:

```text
~/run_tsv_labram_m3cv_v0_3_4.py       # frozen dependency, already present
~/run_task_geometry_v5_0.py
~/analyze_task_geometry_v5_0.py
~/run_task_geometry_v5_0.sh
~/README_TASK_GEOMETRY_V5_0.md
```

The default roots are:

```text
V5 result root:
/omni-eeg-01/task calibration/results/tsv_labram_m3cv/tsv_labram_m3cv_v5_0

Legacy source root:
/omni-eeg-01/task calibration/results/tsv_labram_m3cv/tsv_labram_m3cv_v0_3_4
```

## 7. Execution sequence

### Step 1: fresh V5 preflight

```bash
bash run_task_geometry_v5_0.sh preflight
```

This performs the full five-task data/model audit and seals the V5-specific frozen-$E^*$ protocol.

### Step 2: inspect legacy provenance

```bash
bash run_task_geometry_v5_0.sh inspect
```

This verifies the historical pilot, the seven approved legacy final runs, compatibility between the legacy and V5 scientific contracts, and records diagnostics for the rejected old Motor/rep02. It performs no training.

### Step 3: train the three new final runs and assemble the grid

```bash
bash run_task_geometry_v5_0.sh final
```

This trains only:

```text
Motor/rep02
Rest/rep01
Rest/rep02
```

It then creates the seven approved replicate-level legacy links, certifies the exact 10-run grid, and writes the composite `run_manifest.json` and `_FINAL_SUCCESS.json`.

If a run was interrupted after producing a complete certified V5 run, `final` resumes by skipping it. A non-empty but uncertified V5 run directory is refused. Only after manual confirmation should it be replaced with:

```bash
OVERWRITE_NEW_RUN=1 bash run_task_geometry_v5_0.sh final
```

This overwrite switch can affect only the three designated new V5 runs and never modifies the legacy root.

### Step 4: V5 analysis

```bash
bash run_task_geometry_v5_0.sh analyze
```

The scientific defaults are 5% coarse functional grid, 1% refinement, 99% stable retention target, 1000 matched-random repetitions, and all Full-Context replicate combinations.

For a fast geometry-only code check before the expensive analysis:

```bash
bash run_task_geometry_v5_0.sh selftest
```

## 8. Analysis outputs

V5 analysis is isolated under:

```text
<ROOT>/analysis_v5_0/
<ROOT>/figures_v5_0/
```

The intended 22 key figures remain:

- 2 individual spectra
- 5 Single-Context principal-angle spectra
- 5 Full-Context principal-angle spectra
- 5 Single-Context overlapped-functional spectra
- 5 Full-Context overlapped-functional spectra

The mathematical definitions remain those of the mature protocol: rank-1 matrix directions $Z_j=u_jv_j^\top$ in Frobenius space, Single/Full Context principal angles, matched random reference, same-task Replicate Reference, overlapped functional reconstruction, and paper-aligned STI.

## 9. Scientific boundary

Functional retention is evaluated on the same pooled fitted task dataset. It measures how much of the fitted task function survives rank restriction or overlap projection; it is **not held-out generalization**. Cross-task geometry measures alignment/overlap/interference potential. V5.0 does not perform model merging.
