#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Single-trial raw EEG geometry analysis for MEMA stage-v2 HDF5.

Each analysis unit is one complete trial:
    start + video_clip + self_assessment + rest

For each subject and trial, this script computes:
    1. PCA cumulative explained variance
    2. PC1-PC2 scatter colored by event / stage_id / within-trial time
    3. PCA trajectory over time
    4. kNN neighborhood consistency using Jaccard overlap
    5. Velocity and curvature in standardized PCA space

Default geometry space:
    standardized PCA scores using the first --geom-dims PCs.

This keeps the geometry metrics from being dominated only by PC1 scale.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


STAGE_NAME_DEFAULT = ["start", "video_clip", "self_assessment", "rest"]
LABEL_NAME_DEFAULT = ["attention", "valence", "arousal", "dominance"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Single-trial PCA, kNN, velocity and curvature for MEMA raw EEG."
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default="/mnt/dataset4/lingqy/mema/output_stage_v2",
        help="Directory containing stage-v2 MEMA HDF5 files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/lqy/projects/mema/output/single_trial_geometry",
        help="Output directory.",
    )
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        default=[1],
        help="Subject IDs, for example --subjects 1 or --subjects 1 2 3.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        nargs="+",
        default=list(range(12)),
        help="Trial IDs, 0-based. Trial 0 means Trial 1.",
    )
    parser.add_argument(
        "--sfreq",
        type=float,
        default=200.0,
        help="Sampling frequency after preprocessing. Default: 200 Hz.",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=1.0,
        help="Window length in seconds.",
    )
    parser.add_argument(
        "--stride-sec",
        type=float,
        default=0.5,
        help="Stride length in seconds.",
    )
    parser.add_argument(
        "--channels",
        type=str,
        choices=["eeg", "all"],
        default="eeg",
        help="Use first 30 EEG channels, or all 32 channels including EOG.",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=50,
        help="Number of PCA components to fit.",
    )
    parser.add_argument(
        "--geom-dims",
        type=int,
        default=20,
        help="Number of standardized PCA dimensions used for kNN and smoothness.",
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=15,
        help="k for kNN neighborhood consistency.",
    )
    parser.add_argument(
        "--knn-deltas",
        type=int,
        nargs="+",
        default=[1, 2, 5, 10, 20, 50],
        help="Window-index deltas for kNN Jaccard overlap.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random state for randomized PCA.",
    )

    return parser.parse_args()


def ensure_dirs(output_dir: Path):
    for sub in ["figures", "csv", "metadata"]:
        (output_dir / sub).mkdir(parents=True, exist_ok=True)


def decode_str_array(arr):
    out = []
    for x in arr:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return out


def find_h5_file(input_dir: Path, subject_id: int) -> Path:
    patterns = [
        f"mema_subject{subject_id}_continuous_labeled.h5",
        f"mema_subject{subject_id:02d}_continuous_labeled.h5",
        f"*subject{subject_id}*.h5",
        f"*subject{subject_id:02d}*.h5",
    ]

    for pat in patterns:
        matches = sorted(input_dir.glob(pat))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"Cannot find HDF5 file for subject {subject_id} under {input_dir}."
    )


def get_subject_group(h5_file: h5py.File, subject_id: int):
    if "data" not in h5_file:
        raise KeyError("HDF5 file does not contain group 'data'.")

    root = h5_file["data"]
    candidates = [
        f"subject_{subject_id:02d}",
        f"subject_{subject_id}",
        f"subject{subject_id:02d}",
        f"subject{subject_id}",
    ]

    for name in candidates:
        if name in root:
            return root[name], name

    keys = list(root.keys())
    if len(keys) == 1:
        return root[keys[0]], keys[0]

    raise KeyError(
        f"Cannot find subject group for subject {subject_id}. "
        f"Available groups under data/: {keys}"
    )


def load_subject(h5_path: Path, subject_id: int):
    with h5py.File(h5_path, "r") as f:
        group, group_name = get_subject_group(f, subject_id)

        required = [
            "preprocessed_eeg",
            "event",
            "task_id",
            "stage_id",
            "stage_trial_id",
            "stage_bounds",
            "task_labels",
            "sample_index",
        ]
        missing = [k for k in required if k not in group]
        if missing:
            raise KeyError(f"Missing required datasets: {missing}")

        data = {
            "group_name": group_name,
            "preprocessed_eeg": group["preprocessed_eeg"][:].astype(np.float32),
            "event": group["event"][:].squeeze().astype(np.int8),
            "task_id": group["task_id"][:].squeeze().astype(np.int16),
            "stage_id": group["stage_id"][:].squeeze().astype(np.int8),
            "stage_trial_id": group["stage_trial_id"][:].squeeze().astype(np.int16),
            "stage_bounds": group["stage_bounds"][:].astype(np.int64),
            "task_labels": group["task_labels"][:].astype(np.int8),
            "sample_index": group["sample_index"][:].astype(np.int64),
        }

        if "stage_names" in group:
            data["stage_names"] = decode_str_array(group["stage_names"][:])
        else:
            data["stage_names"] = STAGE_NAME_DEFAULT

        if "label_names" in group:
            data["label_names"] = decode_str_array(group["label_names"][:])
        else:
            data["label_names"] = LABEL_NAME_DEFAULT

        if "electrodes" in group:
            data["electrodes"] = decode_str_array(group["electrodes"][:])
        else:
            data["electrodes"] = None

    return data


def choose_channels(n_channels: int, mode: str):
    if mode == "eeg":
        if n_channels < 30:
            raise ValueError(f"Expected at least 30 channels, got {n_channels}.")
        return np.arange(30)
    if mode == "all":
        return np.arange(n_channels)
    raise ValueError(f"Unknown channel mode: {mode}")


def get_complete_trial_bounds(stage_bounds: np.ndarray, trial_id: int):
    if trial_id < 0 or trial_id >= stage_bounds.shape[0]:
        raise ValueError(
            f"trial_id={trial_id} out of range. "
            f"stage_bounds has {stage_bounds.shape[0]} trials."
        )

    bounds = stage_bounds[trial_id]  # [4, 2]
    starts = bounds[:, 0]
    stops = bounds[:, 1]
    valid = (starts >= 0) & (stops > starts)

    if not np.any(valid):
        raise ValueError(f"No valid stage bounds for trial_id={trial_id}.")

    trial_start = int(np.min(starts[valid]))
    trial_stop = int(np.max(stops[valid]))

    if trial_stop <= trial_start:
        raise ValueError(
            f"Invalid trial bounds for trial_id={trial_id}: "
            f"[{trial_start}, {trial_stop})"
        )

    return trial_start, trial_stop


def build_trial_window_matrix(
    eeg,
    event,
    task_id,
    stage_id,
    stage_trial_id,
    task_labels,
    stage_names,
    sfreq,
    trial_start,
    trial_stop,
    channel_idx,
    window_sec,
    stride_sec,
    trial_id,
):
    win_len = int(round(window_sec * sfreq))
    stride = int(round(stride_sec * sfreq))

    if win_len <= 0 or stride <= 0:
        raise ValueError("window-sec and stride-sec must be positive.")

    if trial_stop - trial_start < win_len:
        raise ValueError(
            f"Trial segment too short: length={trial_stop - trial_start}, "
            f"win_len={win_len}"
        )

    starts = np.arange(trial_start, trial_stop - win_len + 1, stride, dtype=np.int64)
    centers = starts + win_len // 2

    n_channels = len(channel_idx)
    n_features = n_channels * win_len
    X = np.empty((len(starts), n_features), dtype=np.float32)

    for i, s in enumerate(starts):
        seg = eeg[channel_idx, s : s + win_len]
        X[i] = seg.reshape(-1)

    center_stage = stage_id[centers]
    center_event = event[centers]
    center_task_id = task_id[centers]
    center_stage_trial_id = stage_trial_id[centers]

    labels = np.full((len(starts), 4), -1, dtype=np.int8)
    valid_trial = (center_stage_trial_id >= 0) & (
        center_stage_trial_id < task_labels.shape[0]
    )
    labels[valid_trial] = task_labels[center_stage_trial_id[valid_trial]]

    within_trial_time_sec = (starts - trial_start) / sfreq
    absolute_time_sec = starts / sfreq

    stage_name_values = []
    for s in center_stage:
        if 0 <= int(s) < len(stage_names):
            stage_name_values.append(stage_names[int(s)])
        else:
            stage_name_values.append("unknown")

    window_info = pd.DataFrame(
        {
            "window_index": np.arange(len(starts), dtype=np.int64),
            "start_sample": starts,
            "center_sample": centers,
            "absolute_time_sec": absolute_time_sec,
            "within_trial_time_sec": within_trial_time_sec,
            "event": center_event,
            "task_id": center_task_id,
            "stage_id": center_stage,
            "stage_name": stage_name_values,
            "stage_trial_id": center_stage_trial_id,
            "target_trial_id": trial_id,
            "is_target_trial": center_stage_trial_id == trial_id,
            "attention": labels[:, 0],
            "valence": labels[:, 1],
            "arousal": labels[:, 2],
            "dominance": labels[:, 3],
        }
    )

    return X, window_info


def fit_pca(X, n_components, random_state=0):
    n_components = min(n_components, X.shape[0] - 1, X.shape[1])
    if n_components < 2:
        raise ValueError(f"Too few samples for PCA: X shape={X.shape}")

    pca = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=random_state,
    )
    Z = pca.fit_transform(X)
    return pca, Z


def first_dim_above(cum, threshold):
    idx = np.searchsorted(cum, threshold)
    if idx >= len(cum):
        return np.nan
    return int(idx + 1)


def safe_sum(arr, n):
    if len(arr) == 0:
        return np.nan
    return float(np.sum(arr[: min(n, len(arr))]))


def pca_summary(pca):
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    eig = pca.explained_variance_
    participation_ratio = (np.sum(eig) ** 2) / (np.sum(eig ** 2) + 1e-12)

    return {
        "pca_explained_var_pc1": float(evr[0]) if len(evr) > 0 else np.nan,
        "pca_explained_var_pc2": float(evr[1]) if len(evr) > 1 else np.nan,
        "pca_explained_var_top5": safe_sum(evr, 5),
        "pca_explained_var_top10": safe_sum(evr, 10),
        "pca_explained_var_top20": safe_sum(evr, 20),
        "pca_d80": first_dim_above(cum, 0.80),
        "pca_d90": first_dim_above(cum, 0.90),
        "pca_d95": first_dim_above(cum, 0.95),
        "pca_d99": first_dim_above(cum, 0.99),
        "pca_participation_ratio": float(participation_ratio),
    }


def standardized_pca_space(Z, geom_dims):
    dims = min(geom_dims, Z.shape[1])
    G = Z[:, :dims].copy()
    scale = np.std(G, axis=0, ddof=1)
    scale[scale < 1e-12] = 1.0
    G = G / scale
    return G


def compute_velocity_curvature(G):
    n = G.shape[0]
    velocity_col = np.full(n, np.nan, dtype=np.float32)
    curvature_col = np.full(n, np.nan, dtype=np.float32)

    if n >= 2:
        velocity = np.linalg.norm(np.diff(G, axis=0), axis=1)
        velocity_col[1:] = velocity.astype(np.float32)

    if n >= 3:
        curvature = np.linalg.norm(G[2:] - 2 * G[1:-1] + G[:-2], axis=1)
        curvature_col[1:-1] = curvature.astype(np.float32)

    return velocity_col, curvature_col


def compute_knn_jaccard(G, k, deltas):
    n = G.shape[0]
    if n < 3:
        return pd.DataFrame()

    k_eff = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean")
    nn.fit(G)
    neighbors = nn.kneighbors(G, return_distance=False)

    cleaned = []
    for i in range(n):
        row = [int(x) for x in neighbors[i] if int(x) != i]
        cleaned.append(row[:k_eff])

    rows = []
    for delta in deltas:
        if delta <= 0 or delta >= n:
            continue
        vals = []
        for i in range(n - delta):
            a = set(cleaned[i])
            b = set(cleaned[i + delta])
            union = len(a | b)
            if union == 0:
                vals.append(np.nan)
            else:
                vals.append(len(a & b) / union)
        vals = np.asarray(vals, dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        rows.append(
            {
                "delta": int(delta),
                "delta_sec": float(delta),
                "knn_k": int(k_eff),
                "mean_jaccard": float(np.mean(vals)),
                "median_jaccard": float(np.median(vals)),
                "std_jaccard": float(np.std(vals)),
                "n_pairs": int(len(vals)),
            }
        )

    return pd.DataFrame(rows)


def get_stage_transition_times(window_info: pd.DataFrame):
    stage = window_info["stage_id"].to_numpy()
    time = window_info["within_trial_time_sec"].to_numpy()
    idx = np.where(stage[1:] != stage[:-1])[0] + 1
    return time[idx]


def save_pca_variance_plot(pca, out_path, title):
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)

    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, len(cum) + 1), cum, marker="o", markersize=3)
    for y in [0.80, 0.90, 0.95]:
        plt.axhline(y, linestyle="--", linewidth=1)
    plt.xlabel("Number of PCA components")
    plt.ylabel("Cumulative explained variance")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_pca_scatter(Z, window_info, out_path, title, color_col, label=None):
    if color_col not in window_info.columns:
        print(f"Skip scatter because {color_col} is not in window_info.")
        return

    c = window_info[color_col].to_numpy()

    plt.figure(figsize=(7, 6))
    sc = plt.scatter(Z[:, 0], Z[:, 1], c=c, s=16, alpha=0.85, cmap="viridis")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    cbar = plt.colorbar(sc)
    cbar.set_label(label if label else color_col)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_trajectory_plot(Z, window_info, out_path, title):
    t = window_info["within_trial_time_sec"].to_numpy()

    plt.figure(figsize=(7, 6))
    plt.plot(Z[:, 0], Z[:, 1], linewidth=0.8, alpha=0.75)
    sc = plt.scatter(Z[:, 0], Z[:, 1], c=t, s=16, alpha=0.85, cmap="viridis")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    cbar = plt.colorbar(sc)
    cbar.set_label("within_trial_time_sec")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_knn_plot(knn_df, out_path, title):
    if knn_df.empty:
        return

    plt.figure(figsize=(7, 5))
    plt.plot(knn_df["delta"], knn_df["mean_jaccard"], marker="o", label="mean")
    plt.plot(knn_df["delta"], knn_df["median_jaccard"], marker="s", label="median")
    plt.xlabel("Delta in window index")
    plt.ylabel("kNN Jaccard overlap")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_velocity_curvature_plot(window_info, out_path, title):
    t = window_info["within_trial_time_sec"].to_numpy()
    v = window_info["velocity"].to_numpy()
    c = window_info["curvature"].to_numpy()
    transition_times = get_stage_transition_times(window_info)

    plt.figure(figsize=(9, 5))
    plt.plot(t, v, label="velocity", linewidth=1.0)
    plt.plot(t, c, label="curvature", linewidth=1.0)

    for tt in transition_times:
        plt.axvline(tt, alpha=0.25, linewidth=0.8)

    plt.xlabel("Within-trial time (s)")
    plt.ylabel("Standardized PCA trajectory norm")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def summarize_smoothness(window_info):
    v = window_info["velocity"].to_numpy(dtype=float)
    c = window_info["curvature"].to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    c = c[np.isfinite(c)]

    out = {}
    if len(v):
        out.update(
            {
                "velocity_mean": float(np.mean(v)),
                "velocity_median": float(np.median(v)),
                "velocity_std": float(np.std(v)),
                "velocity_max": float(np.max(v)),
            }
        )
    else:
        out.update(
            {
                "velocity_mean": np.nan,
                "velocity_median": np.nan,
                "velocity_std": np.nan,
                "velocity_max": np.nan,
            }
        )

    if len(c):
        out.update(
            {
                "curvature_mean": float(np.mean(c)),
                "curvature_median": float(np.median(c)),
                "curvature_std": float(np.std(c)),
                "curvature_max": float(np.max(c)),
            }
        )
    else:
        out.update(
            {
                "curvature_mean": np.nan,
                "curvature_median": np.nan,
                "curvature_std": np.nan,
                "curvature_max": np.nan,
            }
        )

    return out


def analyze_subject_trial(subject_id, trial_id, args, output_dir: Path):
    h5_path = find_h5_file(Path(args.input_dir), subject_id)
    data = load_subject(h5_path, subject_id)

    eeg = data["preprocessed_eeg"]
    channel_idx = choose_channels(eeg.shape[0], args.channels)
    trial_start, trial_stop = get_complete_trial_bounds(data["stage_bounds"], trial_id)

    X, window_info = build_trial_window_matrix(
        eeg=eeg,
        event=data["event"],
        task_id=data["task_id"],
        stage_id=data["stage_id"],
        stage_trial_id=data["stage_trial_id"],
        task_labels=data["task_labels"],
        stage_names=data["stage_names"],
        sfreq=args.sfreq,
        trial_start=trial_start,
        trial_stop=trial_stop,
        channel_idx=channel_idx,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        trial_id=trial_id,
    )

    print(
        f"[Subject {subject_id:02d} Trial {trial_id + 1:02d}] "
        f"bounds=[{trial_start}, {trial_stop}), X={X.shape}, "
        f"stage_counts={dict(window_info['stage_id'].value_counts().sort_index())}"
    )

    pca, Z = fit_pca(X, args.pca_components, random_state=args.random_state)
    G = standardized_pca_space(Z, args.geom_dims)

    velocity, curvature = compute_velocity_curvature(G)
    window_info["velocity"] = velocity
    window_info["curvature"] = curvature

    knn_df = compute_knn_jaccard(G, args.knn_k, args.knn_deltas)

    prefix = f"subject{subject_id:02d}_trial{trial_id + 1:02d}"
    fig_dir = output_dir / "figures"
    csv_dir = output_dir / "csv"

    evr = pca.explained_variance_ratio_
    evr_df = pd.DataFrame(
        {
            "component": np.arange(1, len(evr) + 1),
            "explained_variance_ratio": evr,
            "cumulative_explained_variance": np.cumsum(evr),
        }
    )
    evr_df.to_csv(csv_dir / f"{prefix}_pca_variance.csv", index=False)

    for i in range(min(Z.shape[1], args.pca_components)):
        window_info[f"pc{i + 1}"] = Z[:, i]
    window_info.to_csv(csv_dir / f"{prefix}_window_pca_scores.csv", index=False)

    if not knn_df.empty:
        knn_df.insert(0, "subject", subject_id)
        knn_df.insert(1, "trial_id", trial_id)
        knn_df.insert(2, "trial_number", trial_id + 1)
        knn_df.to_csv(csv_dir / f"{prefix}_knn_jaccard.csv", index=False)

    save_pca_variance_plot(
        pca,
        fig_dir / f"{prefix}_pca_cumulative_variance.png",
        f"Subject {subject_id:02d} Trial {trial_id + 1:02d} PCA cumulative variance",
    )
    save_pca_scatter(
        Z,
        window_info,
        fig_dir / f"{prefix}_pca_event.png",
        f"Subject {subject_id:02d} Trial {trial_id + 1:02d} PCA colored by event",
        "event",
        "event",
    )
    save_pca_scatter(
        Z,
        window_info,
        fig_dir / f"{prefix}_pca_stage_id.png",
        f"Subject {subject_id:02d} Trial {trial_id + 1:02d} PCA colored by stage_id",
        "stage_id",
        "stage_id",
    )
    save_pca_scatter(
        Z,
        window_info,
        fig_dir / f"{prefix}_pca_time.png",
        f"Subject {subject_id:02d} Trial {trial_id + 1:02d} PCA colored by time",
        "within_trial_time_sec",
        "within_trial_time_sec",
    )
    save_trajectory_plot(
        Z,
        window_info,
        fig_dir / f"{prefix}_pca_trajectory_time.png",
        f"Subject {subject_id:02d} Trial {trial_id + 1:02d} PCA trajectory over time",
    )
    save_knn_plot(
        knn_df,
        fig_dir / f"{prefix}_knn_jaccard.png",
        f"Subject {subject_id:02d} Trial {trial_id + 1:02d} kNN neighborhood consistency",
    )
    save_velocity_curvature_plot(
        window_info,
        fig_dir / f"{prefix}_velocity_curvature.png",
        f"Subject {subject_id:02d} Trial {trial_id + 1:02d} velocity / curvature",
    )

    summary = pca_summary(pca)
    summary.update(summarize_smoothness(window_info))

    if not knn_df.empty:
        for _, row in knn_df.iterrows():
            d = int(row["delta"])
            summary[f"knn_delta{d}_mean"] = float(row["mean_jaccard"])
            summary[f"knn_delta{d}_median"] = float(row["median_jaccard"])

    summary.update(
        {
            "subject": subject_id,
            "trial_id": trial_id,
            "trial_number": trial_id + 1,
            "h5_file": str(h5_path),
            "trial_start": trial_start,
            "trial_stop": trial_stop,
            "trial_duration_sec": (trial_stop - trial_start) / args.sfreq,
            "n_windows": X.shape[0],
            "n_features": X.shape[1],
            "channels": args.channels,
            "window_sec": args.window_sec,
            "stride_sec": args.stride_sec,
            "pca_components_fit": Z.shape[1],
            "geom_dims_used": G.shape[1],
            "knn_k": min(args.knn_k, G.shape[0] - 1),
            "n_stage0_start": int((window_info["stage_id"] == 0).sum()),
            "n_stage1_video": int((window_info["stage_id"] == 1).sum()),
            "n_stage2_assessment": int((window_info["stage_id"] == 2).sum()),
            "n_stage3_rest": int((window_info["stage_id"] == 3).sum()),
        }
    )

    return summary, knn_df


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_dirs(output_dir)

    with open(output_dir / "metadata" / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    summaries = []
    all_knn = []

    for subject_id in args.subjects:
        for trial_id in args.trials:
            try:
                summary, knn_df = analyze_subject_trial(
                    subject_id, trial_id, args, output_dir
                )
                summaries.append(summary)
                if knn_df is not None and not knn_df.empty:
                    all_knn.append(knn_df)
            except Exception as e:
                print(
                    f"[Subject {subject_id:02d} Trial {trial_id + 1:02d}] "
                    f"ERROR: {repr(e)}"
                )

    if summaries:
        summary_df = pd.DataFrame(summaries)
        summary_df.to_csv(
            output_dir / "csv" / "single_trial_geometry_summary.csv",
            index=False,
        )

    if all_knn:
        all_knn_df = pd.concat(all_knn, ignore_index=True)
        all_knn_df.to_csv(
            output_dir / "csv" / "single_trial_knn_jaccard_all.csv",
            index=False,
        )

    print("\nDone. Single-trial geometry analysis saved.")


if __name__ == "__main__":
    main()
