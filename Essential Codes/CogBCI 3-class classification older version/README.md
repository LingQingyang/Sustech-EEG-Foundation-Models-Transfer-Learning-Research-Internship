# Earlier three-state subject-heldout LDA probes

[Research overview](../../README.md)

These original scripts classify Resting, N-Back, and MATB from frozen embeddings. Centering, SVD, discriminant axes, and class centroids are fitted using training subjects only. Default evaluation uses four subject-group folds; LOSO is also supported.

| Script | Role |
|---|---|
| `run_experiment_A_subject_heldout_discriminant_spectrum.py` | Original single-file three-state experiment. |
| `run_experiment_A_subject_heldout_discriminant_spectrum_5s.py` | Five-second windows, with batch and single-model modes. |

This is an earlier experiment family; it is distinct from the seven-state LD/QD/tensor report. No corresponding numeric results are provided in the three source reports.

```bash
python3 run_experiment_A_subject_heldout_discriminant_spectrum.py --help
python3 run_experiment_A_subject_heldout_discriminant_spectrum_5s.py --help
```

Provide an embedding HDF5 with the appropriate `embedding`, `task_id`, and `subject_id` keys. Original server defaults and script filenames are retained.
