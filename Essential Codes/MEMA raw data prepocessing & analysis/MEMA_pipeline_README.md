# MEMA EEG Processing and Geometry Analysis Pipeline

## Overview

This repository contains the preprocessing, dataset construction, and
geometry analysis pipeline developed for the MEMA EEG dataset.

The pipeline transforms continuous raw EEG recordings into structured
representations for studying:

-   EEG state-space geometry
-   low-dimensional neural trajectories
-   temporal continuity of brain states
-   intrinsic dimensionality
-   local manifold structure

Workflow:

    Raw EEG TXT
        |
        v
    EEG preprocessing
        |
        v
    Continuous labeled HDF5 dataset
        |
        v
    Geometry exploration
        |
        v
    Trial dynamics analysis
        |
        v
    Condition-specific geometry analysis

------------------------------------------------------------------------

## Directory Structure

    MEMA_pipeline/

    ├── README.md

    ├── 00_preprocessing/
    │   └── mema_preprocessing.py

    ├── 01_dataset_builder/
    │   └── build_mema_all_subjects_continuous_labeled_h5_fast.py

    ├── 02_geometry_exploration/
    │   └── analyze_raw_eeg_geometry.py

    ├── 03_trial_dynamics/
    │   └── analyze_single_trial_geometry.py

    └── 04_condition_geometry/
        └── analyze_subject_condition_geometry.py

------------------------------------------------------------------------

# 1. EEG Preprocessing

Script:

    00_preprocessing/mema_preprocessing.py

Purpose:

Convert raw MEMA Data2 TXT recordings into cleaned EEG signals.

Input format:

-   Columns 1-32: EEG/EOG channels
-   Column 33: sample index
-   Column 34: task/rest event marker

Channel configuration:

-   30 EEG channels
-   2 EOG channels

Default preprocessing:

-   50 Hz notch filtering
-   0.1 Hz high-pass filtering
-   40 Hz low-pass filtering
-   Resampling from 500 Hz to 200 Hz
-   EEG global normalization with separate EOG normalization

------------------------------------------------------------------------

# 2. Continuous HDF5 Dataset Construction

Script:

    01_dataset_builder/build_mema_all_subjects_continuous_labeled_h5_fast.py

Purpose:

Construct continuous labeled EEG datasets in HDF5 format.

Output:

    data/
    └── subject_XX/
        ├── preprocessed_eeg
        ├── event
        ├── task_id
        ├── task_labels
        ├── task_bounds
        └── sample_index

Labels:

    [attention,
     valence,
     arousal,
     dominance]

The dataset preserves continuous recordings, aligns task labels to EEG
time points, and retains rest periods.

------------------------------------------------------------------------

# 3. Raw EEG Geometry Exploration

Script:

    02_geometry_exploration/analyze_raw_eeg_geometry.py

Purpose:

Initial exploration of whether continuous EEG activity occupies a
structured low-dimensional space.

Pipeline:

    EEG signal
        |
    Sliding windows
        |
    Flattened features
        |
    PCA embedding
        |
    Geometry analysis

Analyses include:

-   PCA explained variance
-   participation ratio
-   kNN neighborhood consistency
-   velocity
-   curvature

------------------------------------------------------------------------

# 4. Single-Trial Neural Trajectory Analysis

Script:

    03_trial_dynamics/analyze_single_trial_geometry.py

Purpose:

Study how brain activity evolves during a complete behavioral trial.

Trial stages:

    start
     |
    video_clip
     |
    self_assessment
     |
    rest

Outputs:

-   PCA trajectories
-   temporal evolution
-   local neighborhood stability
-   velocity and curvature

------------------------------------------------------------------------

# 5. Condition-Specific Geometry Analysis

Script:

    04_condition_geometry/analyze_subject_condition_geometry.py

Purpose:

Compare geometry under different behavioral conditions.

Conditions:

    task:
        video_clip

    nontask:
        start + self_assessment + rest

Important design choice:

Continuous segments are analyzed independently to avoid artificial jumps
caused by stage boundaries.

Analyses:

-   PCA geometry
-   intrinsic dimensionality
-   kNN Jaccard consistency
-   trajectory velocity
-   curvature
-   optional MLE intrinsic dimension

------------------------------------------------------------------------

# Analysis Philosophy

The pipeline views EEG as a continuous dynamical system:

    continuous EEG activity

            ↓

    high-dimensional neural state

            ↓

    low-dimensional geometry

            ↓

    trajectory and dynamics

The main questions are:

1.  Does EEG occupy a structured low-dimensional space?
2.  Are neural states locally continuous over time?
3.  Do different conditions occupy different geometric structures?
4.  How does brain activity transition between states?

------------------------------------------------------------------------

# Relation to Later EEG Foundation Model Analysis

The MEMA pipeline provides the early foundation for later EEG
embedding-space studies:

    Raw EEG geometry
            |
            v
    State-space representation
            |
            v
    Foundation model embeddings
            |
            v
    Cross-subject adaptation
            |
            v
    Flow matching and individualized state transitions

------------------------------------------------------------------------

# Requirements

Dependencies:

    numpy
    scipy
    pandas
    scikit-learn
    matplotlib
    h5py
    mne

Recommended:

    Python >= 3.10

------------------------------------------------------------------------

# Notes

-   Store preprocessing configuration together with generated datasets.
-   Normalization choices may influence downstream geometry.
-   PCA geometry analyses are exploratory and depend on correct event
    alignment.
