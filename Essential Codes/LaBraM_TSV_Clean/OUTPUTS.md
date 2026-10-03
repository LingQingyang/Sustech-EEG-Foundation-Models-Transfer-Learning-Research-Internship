# Output Tables and Assertions

## Run artifacts

Each `<root>/runs/<task>/repXX/` contains:

| File | Content |
|---|---|
| `adapted_checkpoint.pth` | complete adapted backbone, task head, identity and W0 contract |
| `delta_weights.npz` | exact 72 analysed matrices, stored as `Lxx__role` |
| `epoch_metrics.csv` | 20 online training-pass observations |
| `metadata.json` | seeds, optimizer, final fitted-data metrics, artifact hashes |
| `_SUCCESS.json` | run-level completion certificate |

`audit` requires the exact 5×2 grid. It checks identity, 20 epochs, declared hashes, one common protocol fingerprint, one common W0 digest, and all 72 shapes.

## Six spectra

| Spectrum | CSV | Essential endpoint/check |
|---|---|---|
| Individual Energy | `spectra/individual_energy_spectrum.csv` | `q=1 → energy=1` |
| Individual Functional | `spectra/individual_functional_spectrum.csv` | `q=1 → full checkpoint bACC` |
| Shared Energy | `spectra/shared_energy_spectrum.csv` | endpoint is absolute `Gamma`, not forced to 1 |
| Shared Functional | `spectra/shared_functional_spectrum.csv` | full prefix equals Individual at `q_star` |
| Principal Angle | `spectra/principal_angle_spectrum.csv` | exact rank-one operator cosines; matched random / replicate reference |
| Overlapped Functional | `spectra/overlapped_functional_spectrum.csv` | projected principal directions tested by actual inference |

Supporting, non-seventh-spectrum tables:

- `functional_cutoffs.csv`: candidate (q^*\), (K^*\), baselines and exact target metadata.
- `shared_energy_random_reference.csv`: matched Haar null envelope for Shared Energy.
- `shared_components/*.csv`: auditable component ordering for each comparison.
- `random_cache/*.npz`: rank-signature cache for expensive principal-angle nulls.

The six-spectrum manifest records SVD reconstruction/orthonormality errors and all output hashes.

## Heat map

- `heatmap/weight_update_heatmap.csv`: one row per run × matrix.
- `heatmap/weight_update_heatmap_summary.csv`: task × matrix replicate summaries.

Both linear energy and post-aggregation log10 values are explicit columns.

## STI

- `sti/sti_observed.csv`: 32 replicate combinations × 72 matrices.
- `sti/sti_random_reference.csv`: one matched-null distribution summary per matrix.
- `sti/sti_summary.csv`: observed median/range, null median/q99, and observed/null ratio.

## Figures

`plot` writes both PNG and vector PDF under `analysis/figures/`. It reads the CSV files above and never reruns SVD, inference, projection, or random sampling. The numerical results remain unchanged when figures are deleted and regenerated.
