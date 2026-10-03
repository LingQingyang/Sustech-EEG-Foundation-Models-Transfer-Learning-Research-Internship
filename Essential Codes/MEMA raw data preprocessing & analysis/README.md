# MEMA raw EEG preprocessing and geometry

[Research overview](../../README.md)

[Read the original pipeline documentation](MEMA_pipeline_README.md).

This early pipeline cleans continuous raw EEG, builds aligned HDF5 datasets, and studies low-dimensional geometry and temporal trajectories.

| Script | Purpose |
|---|---|
| `mema_preprocessing.py` | EEG/EOG filtering, resampling, and normalization. |
| `build_mema_all_subjects_continuous_labeled_h5_fast.py` | Continuous HDF5 construction with task/event alignment. |
| `analyze_raw_eeg_geometry.py` | PCA, participation ratio, local neighborhoods, velocity, and curvature. |
| `analyze_single_trial_geometry.py` | Trajectories across complete trial stages. |
| `analyze_subject_condition_geometry.py` | Task/non-task geometry with segment-aware temporal analysis. |

The original guide describes a conceptual staged directory layout; in this preserved code collection, the five scripts are together in the current directory. The original files and guide remain unchanged.

The documented defaults include 50 Hz notch, 0.1 Hz high-pass, 40 Hz low-pass, 500→200 Hz resampling, 30 EEG channels and two EOG channels. Dependencies include NumPy, SciPy, pandas, scikit-learn, Matplotlib, h5py, and MNE. Consult each script for its actual inputs and path settings.

No corresponding numeric MEMA result report is supplied in `Essential Results`.
