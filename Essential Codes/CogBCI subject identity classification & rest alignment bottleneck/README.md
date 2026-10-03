# CogBCI state geometry and identity controls

[Research overview](../../README.md)

The suite separates state discrimination from a closed-set subject-identity control. The latter splits windows within each subject and condition, so the same identities appear in training and testing; it is a diagnostic of identity information rather than unseen-subject identification.

| Entry point | Purpose |
|---|---|
| `run_experiment_A_discriminant_suite_v3.py` | Three-state subject-heldout probes, detailed geometry outputs, and identity controls. |
| `run_experiment_A_subject_heldout_task6_shared_vs_internal_5s.py` | Compare shared five-dimensional six-task LDA with family-specific two-dimensional N-Back/MATB LDA. |

[The original experiment guide](README_Experiment_A_discriminant_suite.md) documents HDF5 keys, dependencies, options, and outputs. The shared-vs-internal script excludes Rest and evaluates six cognitive conditions.

```bash
python3 run_experiment_A_discriminant_suite_v3.py --help
python3 run_experiment_A_subject_heldout_task6_shared_vs_internal_5s.py --help
```

No separate numeric result report for these scripts is supplied in the three source PDFs.
