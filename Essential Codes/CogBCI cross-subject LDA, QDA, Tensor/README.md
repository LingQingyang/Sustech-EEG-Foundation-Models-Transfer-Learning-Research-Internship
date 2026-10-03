# CogBCI: cross-subject LDA, QDA, and tensor geometry

[Read the experimental results](RESULTS.md) · [Return to the research overview](../../README.md)

The experiment uses linear discriminant analysis (LDA), quadratic discriminant analysis (QDA), their score residual, and tensor interaction features. The feature arms use the abbreviations LD and QD for their respective discriminant scores.

## Entry points

| File | Purpose |
|---|---|
| `run_cross_subject_ld_qd_tensor48_v4_2_1.py` | Train-only geometry fitting, nested subject-heldout readout selection, and outer evaluation. |
| `run_cross_subject_ld_qd_tensor48_v4_2_1.sh` | Original Bash launcher and server settings. |

The seven states are Rest-EO, three N-Back levels, and three MATB levels. All sessions use five-second windows. The four arms share a class-balanced ridge readout and differ in their LD/QD-derived features. Global and dedicated family geometries are evaluated separately from strict audits of the original seven-class classifier.

Inspect available input and output arguments:

```bash
python3 run_cross_subject_ld_qd_tensor48_v4_2_1.py --help
```

The launcher retains the original server paths. Configure its inputs and output locations before execution. Code has been preserved unchanged; the accompanying result report documents an existing run.
