# CogBCI rest-driven personalization

[Research overview](../../README.md)

The seven-class experiment merges four Rest conditions and retains three N-Back plus three MATB conditions. Training subjects fit the embedding reduction and a six-dimensional Fisher/LDA space. Held-out subjects are evaluated with:

| Method | Available information |
|---|---|
| Cross-subject baseline | No target-subject adaptation |
| All-state centering | Label-free transductive access to all target windows; diagnostic |
| Rest centering / CORAL | Target Rest mean or regularized mean/covariance |
| Rest residual MLP | Target Rest distribution alignment with anchor/displacement constraints |
| Oracle Procrustes / self-readout | Label-using diagnostic comparisons |

The primary outcome is **task6 bACC while seven-class predictions remain allowed**. This prevents improved Rest recall alone from being mistaken for transfer to cognitive tasks.

```bash
python3 run_experiment_A_rest_personalization_7class_ld_mlp_v3_2_metricfix_dedup22.py --help
```

The source lists NumPy, SciPy, h5py, and Matplotlib; PyTorch is optional because the small residual MLP has a NumPy backend. This experiment merges Rest conditions and therefore differs from the Rest-EO-only cross-subject LD/QD/tensor report. No dedicated numerical report for this personalization script is included.
