# ds007554 multiclass and semantic pairwise LDA

[Research overview](../../README.md)

This earlier five-state experiment performs three-fold leave-one-session-out evaluation within each subject. It fits up to four multiclass LD dimensions and ten named pairwise contrasts in the same training-fitted whitened geometry.

The ten pairwise axes form a redundant semantic dictionary whose span has at most four dimensions; they are not ten independent Euclidean dimensions. Near-degenerate axes should be interpreted as subspace blocks.

| File | Role |
|---|---|
| `run_experiment_A_five_state_semantic_lda_v3_1.py` | Effective-dimension diagnostics, pairwise alignment, and semantic decomposition. |
| `run_experiment_A_five_state_semantic_lda_v3_1.sh` | Original Bash launcher. |

```bash
python3 run_experiment_A_five_state_semantic_lda_v3_1.py --help
```

Required HDF5 metadata include embedding, subject, session, run, sample bounds, phase, task, and task-family fields. See the script's input contract. The neighboring [A/B/C study](../ds007554%20Experiments%20A%2CB%2CC/RESULTS.md) contains later pilot results, not a separate result table for this version.
