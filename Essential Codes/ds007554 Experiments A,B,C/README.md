# ds007554: experiments A, B, and C

[Read the pilot results](RESULTS.md) · [Original launcher notes](README_FINAL_LAUNCHERS.txt) · [Research overview](../../README.md)

The pipeline tests whether frozen EEG foundation-model embeddings contain static state geometry, reproducible temporal structure, and semantic information retained by a frozen slow subspace.

| Stage | Entry point | Role |
|---|---|---|
| Shared features | `ld_qd_feature_core_v5_0.py` | LD/QD/residual/tensor maps, ridge readout, metrics, and numerical checks. The retained filename contains the v5.1.1 covariance hotfix. |
| A | `run_experiment_A_five_state_ld_qd_tensor_v5_1_1.py` | Static discriminant geometry in training-fitted embedding coordinates. |
| B | `run_experiment_B_five_state_within_global_sfa_v5_1.py` | Within-state and Global slow-feature systems, temporal-order nulls, and frozen B-to-C handoffs. |
| C | `run_experiment_C_slowspace_ld_qd_tensor_v5_1.py` | Semantic readouts of frozen slow coordinates and control subspaces. |
| Integration | `integrate_experiments_ABC_five_state_v5_0.py` | Checks and aggregates the A/B/C artifacts. |

The five states are Baseline, MA, NB, NBMA, and Full. The outer split holds out one of three sessions within each subject. Global slow-space fitting is frozen before C; C cannot refit or select B using test labels.

## Launching

The supplied Bash launchers retain their original server paths. The integrated launcher invokes the standalone stages; inspect the launchers and configure their `SCRIPT`, `CORE`, input, and output settings for your environment before running.

```bash
python3 run_experiment_A_five_state_ld_qd_tensor_v5_1_1.py --help
python3 run_experiment_B_five_state_within_global_sfa_v5_1.py --help
python3 run_experiment_C_slowspace_ld_qd_tensor_v5_1.py --help

# After configuring all launcher paths and HDF5 inputs:
SMOKE=1 bash run_experiments_ABC_five_state_integrated_v5_1_1.sh
```

The original launch notes document formal runs and resume controls. The result adaptation explains the residual-definition difference between the slide report and the bundled implementation; an exact result reproduction requires the original run metadata.
