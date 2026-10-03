#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze raw EEG geometry for MEMA continuous labeled HDF5 files.

Core idea:
    continuous EEG window x_t: [C, L]
    raw EEG geometry point z_t = vec(x_t): [C * L]

Metrics:
    1. PCA explained variance
    2. Intrinsic dimension:
        - PCA d90 / d95
        - participation ratio
        - Levina-Bickel MLE intrinsic dimension
    3. Neighborhood consistency:
        - Jaccard overlap between kNN(z_t) and kNN(z_{t+delta})
    4. Local smoothness:
        - velocity ||z_{t+1} - z_t||
        - curvature ||z_{t+1} - 2z_t + z_{t-1}||

Input HDF5 structure expected:
    data/subject_XX/preprocessed_eeg  [32, T]
    data/subject_XX/event             [1, T]
    data/subject_XX/label             [4, T]
    data/subject_XX/task_id           [1, T]
"""

import argparse
import os
from pathlib import Path
import json
import warnings

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def parse_args():
    parser = argparse.ArgumentParser(
        description="Raw EEG geometry analysis for MEMA continuous labeled HDF5."
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default="/mnt/dataset4/lingqy/mema/output",
        help="Directory containing mema_subject*_continuous_labeled.h5 files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/lqy/projects/mema/output/raw_eeg_geometry",
        help="Directory for saving CSV files and figures.",
    )
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        default=list(range(1, 21)),
        help="Subject IDs to analyze. Default: 1 2 ... 20.",
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
        help="Window length in seconds. Default: 1.0.",
    )
    parser.add_argument(
        "--stride-sec",
        type=float,
        default=0.5,
        help="Stride length in seconds. Default: 0.5.",
    )

    parser.add_argument(
        "--channels",
        type=str,
        choices=["eeg", "all"],
        default="eeg",
        help="Use first 30 EEG channels or all 32 channels including EOG. Default: eeg.",
    )
    parser.add_argument(
        "--feature-step",
        type=int,
        default=1,
        help=(
            "Temporal subsampling inside each window before flattening. "
            "1 means use all time points. 2 means use every other sample."
        ),
    )

    parser.add_argument(
        "--event-mode",
        type=str,
        choices=["all", "task", "non_task"],
        default="all",
        help="Analyze all windows, task windows only, or non-task windows only.",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=6000,
        help=(
            "Maximum number of windows per subject for geometry analysis. "
            "Uniformly subsamples windows if too many. Default: 6000."
        ),
    )

    parser.add_argument(
        "--pca-components",
        type=int,
        default=100,
        help="Number of PCA components to compute. Default: 100.",
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=15,
        help="k for kNN neighborhood consistency. Default: 15.",
    )
    parser.add_argument(
        "--mle-ks",
        type=int,
        nargs="+",
        default=[10, 20, 30],
        help="k values for MLE intrinsic dimension. Default: 10 20 30.",
    )
    parser.add_argument(
        "--jaccard-deltas",
        type=int,
        nargs="+",
        default=[1, 2, 5, 10, 20, 50],
        help="Temporal deltas in window index for kNN Jaccard. Default: 1 2 5 10 20 50.",
    )
    parser.add_argument(
        "--smooth-dim",
        type=int,
        default=20,
        help="Number of PCA dimensions used for velocity / curvature. Default: 20.",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random seed for PCA. Default: 0.",
    )

    return parser.parse_args()


def ensure_dirs(output_dir: Path):
    for sub in ["figures", "csv", "timeseries", "metadata"]:
        (output_dir / sub).mkdir(parents=True, exist_ok=True)


def get_subject_group(h5_file: h5py.File, subject_id: int):
    """
    Robustly find subject group.

    Expected:
        data/subject_01
    But we also tolerate subject_1 or the only group under data.
    """
    if "data" not in h5_file:
        raise KeyError("HDF5 file does not contain root group 'data'.")

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


def load_subject_h5(h5_path: Path, subject_id: int):
    with h5py.File(h5_path, "r") as f:
        group, group_name = get_subject_group(f, subject_id)

        eeg = group["preprocessed_eeg"][:].astype(np.float32)
        event = group["event"][:].astype(np.int8).squeeze()
        label = group["label"][:].astype(np.int8)
        task_id = group["task_id"][:].astype(np.int16).squeeze()

        electrodes = None
        if "electrodes" in group:
            electrodes = [
                e.decode("utf-8") if isinstance(e, bytes) else str(e)
                for e in group["electrodes"][:]
            ]

    return {
        "eeg": eeg,
        "event": event,
        "label": label,
        "task_id": task_id,
        "electrodes": electrodes,
        "group_name": group_name,
    }


def choose_channels(n_channels: int, mode: str):
    if mode == "eeg":
        if n_channels < 30:
            raise ValueError(f"Expected at least 30 EEG channels, got {n_channels}.")
        return np.arange(30)
    if mode == "all":
        return np.arange(n_channels)
    raise ValueError(f"Unknown channel mode: {mode}")


def build_window_matrix(
    eeg,
    event,
    label,
    task_id,
    sfreq,
    window_sec,
    stride_sec,
    channel_idx,
    feature_step=1,
    event_mode="all",
    max_windows=None,
):
    """
    Convert continuous EEG to window matrix X.

    Each row:
        X[i] = vec(eeg[channel_idx, start:start+window_len:feature_step])

    Window-level event / task / label:
        use center sample for label and task_id.
        use mean event ratio for event flag.
    """
    n_channels, n_time = eeg.shape

    win_len = int(round(window_sec * sfreq))
    stride = int(round(stride_sec * sfreq))

    if win_len <= 0:
        raise ValueError("Window length must be positive.")
    if stride <= 0:
        raise ValueError("Stride length must be positive.")
    if n_time < win_len:
        raise ValueError(f"Signal too short: T={n_time}, window length={win_len}.")

    starts = np.arange(0, n_time - win_len + 1, stride, dtype=np.int64)
    centers = starts + win_len // 2

    event_ratio = np.array(
        [event[s : s + win_len].mean() for s in starts],
        dtype=np.float32,
    )
    event_w = (event_ratio >= 0.5).astype(np.int8)

    if event_mode == "task":
        keep = event_w == 1
    elif event_mode == "non_task":
        keep = event_w == 0
    else:
        keep = np.ones_like(event_w, dtype=bool)

    starts = starts[keep]
    centers = centers[keep]
    event_ratio = event_ratio[keep]
    event_w = event_w[keep]

    if len(starts) == 0:
        raise ValueError(f"No windows left after event_mode={event_mode}.")

    if max_windows is not None and len(starts) > max_windows:
        # Uniform temporal subsampling to preserve long-range trajectory coverage.
        keep_idx = np.linspace(0, len(starts) - 1, max_windows)
        keep_idx = np.round(keep_idx).astype(np.int64)

        starts = starts[keep_idx]
        centers = centers[keep_idx]
        event_ratio = event_ratio[keep_idx]
        event_w = event_w[keep_idx]

    effective_len = len(range(0, win_len, feature_step))
    n_features = len(channel_idx) * effective_len

    X = np.empty((len(starts), n_features), dtype=np.float32)

    for i, s in enumerate(starts):
        seg = eeg[channel_idx, s : s + win_len : feature_step]
        X[i] = seg.reshape(-1)

    label_center = label[:, centers].T
    task_center = task_id[centers]
    time_sec = starts / sfreq

    window_info = pd.DataFrame(
        {
            "window_index": np.arange(len(starts), dtype=np.int64),
            "start_sample": starts,
            "center_sample": centers,
            "time_sec": time_sec,
            "event": event_w,
            "event_ratio": event_ratio,
            "task_id": task_center,
            "attention": label_center[:, 0],
            "valence": label_center[:, 1],
            "arousal": label_center[:, 2],
            "dominance": label_center[:, 3],
        }
    )

    return X, window_info


def fit_pca(X, n_components, random_state=0):
    n_components = min(n_components, X.shape[0] - 1, X.shape[1])
    if n_components < 2:
        raise ValueError("Need at least 2 PCA components.")

    pca = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=random_state,
    )
    Z = pca.fit_transform(X)

    return pca, Z


def compute_pca_summary(pca):
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)

    def first_dim_above(threshold):
        idx = np.searchsorted(cum, threshold)
        if idx >= len(cum):
            return np.nan
        return int(idx + 1)

    eig = pca.explained_variance_
    participation_ratio = (np.sum(eig) ** 2) / (np.sum(eig ** 2) + 1e-12)

    return {
        "pca_d80": first_dim_above(0.80),
        "pca_d90": first_dim_above(0.90),
        "pca_d95": first_dim_above(0.95),
        "pca_d99": first_dim_above(0.99),
        "pca_participation_ratio": float(participation_ratio),
        "pca_explained_var_pc1": float(evr[0]),
        "pca_explained_var_pc2": float(evr[1]),
        "pca_explained_var_top5": float(np.sum(evr[:5])),
        "pca_explained_var_top10": float(np.sum(evr[:10])),
        "pca_explained_var_top20": float(np.sum(evr[:20])),
    }


def compute_mle_intrinsic_dimension(Z, ks):
    """
    Levina-Bickel style MLE intrinsic dimension.

    For each point:
        m_i^{-1} = mean_{j=1}^{k-1} log(r_k / r_j)

    We compute this in PCA space Z to avoid unstable high-dimensional raw distances.
    """
    max_k = max(ks)
    if Z.shape[0] <= max_k + 1:
        return {f"mle_id_k{k}": np.nan for k in ks}

    nbrs = NearestNeighbors(n_neighbors=max_k + 1, metric="euclidean")
    nbrs.fit(Z)
    distances, _ = nbrs.kneighbors(Z)

    # Remove self distance.
    distances = distances[:, 1:]

    result = {}
    eps = 1e-12

    for k in ks:
        if k < 3 or k > distances.shape[1]:
            result[f"mle_id_k{k}"] = np.nan
            continue

        d = distances[:, :k]
        rk = d[:, [k - 1]]
        logs = np.log((rk + eps) / (d[:, : k - 1] + eps))
        inv_m = np.mean(logs, axis=1)

        m = 1.0 / (inv_m + eps)
        m = m[np.isfinite(m)]

        # Trim extreme values to reduce instability from near-duplicates / outliers.
        if len(m) > 20:
            lo, hi = np.percentile(m, [1, 99])
            m = m[(m >= lo) & (m <= hi)]

        result[f"mle_id_k{k}"] = float(np.mean(m)) if len(m) else np.nan

    return result


def compute_knn_neighbors(Z, k):
    if Z.shape[0] <= k + 1:
        raise ValueError(f"Too few windows for kNN: n={Z.shape[0]}, k={k}.")

    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nbrs.fit(Z)
    _, indices = nbrs.kneighbors(Z)

    # Remove self neighbor at column 0.
    return indices[:, 1:]


def compute_jaccard_over_time(neighbor_indices, deltas):
    n = neighbor_indices.shape[0]
    rows = []

    neighbor_sets = [set(row.tolist()) for row in neighbor_indices]

    for delta in deltas:
        if delta <= 0 or delta >= n:
            rows.append(
                {
                    "delta": delta,
                    "jaccard_mean": np.nan,
                    "jaccard_median": np.nan,
                    "jaccard_std": np.nan,
                    "n_pairs": 0,
                }
            )
            continue

        vals = []
        for i in range(n - delta):
            a = neighbor_sets[i]
            b = neighbor_sets[i + delta]
            union = a | b
            if len(union) == 0:
                continue
            vals.append(len(a & b) / len(union))

        vals = np.asarray(vals, dtype=np.float32)
        rows.append(
            {
                "delta": delta,
                "jaccard_mean": float(np.mean(vals)) if len(vals) else np.nan,
                "jaccard_median": float(np.median(vals)) if len(vals) else np.nan,
                "jaccard_std": float(np.std(vals)) if len(vals) else np.nan,
                "n_pairs": int(len(vals)),
            }
        )

    return pd.DataFrame(rows)


def compute_velocity_curvature(Z, smooth_dim=20):
    """
    Compute local smoothness in standardized PCA trajectory space.

    Why standardize PCA coordinates?
        Raw PC scales differ strongly.
        Standardization makes velocity / curvature less dominated by PC1 amplitude.
    """
    dim = min(smooth_dim, Z.shape[1])
    Zs = Z[:, :dim]

    Zs = StandardScaler().fit_transform(Zs)

    velocity = np.linalg.norm(np.diff(Zs, axis=0), axis=1)

    if Zs.shape[0] >= 3:
        curvature = np.linalg.norm(Zs[2:] - 2 * Zs[1:-1] + Zs[:-2], axis=1)
    else:
        curvature = np.array([], dtype=np.float32)

    return velocity, curvature


def save_pca_variance_plot(pca, out_path, title):
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)

    plt.figure(figsize=(7, 5))
    plt.plot(np.arange(1, len(cum) + 1), cum, marker="o", markersize=3)
    plt.axhline(0.80, linestyle="--", linewidth=1)
    plt.axhline(0.90, linestyle="--", linewidth=1)
    plt.axhline(0.95, linestyle="--", linewidth=1)
    plt.xlabel("Number of PCA components")
    plt.ylabel("Cumulative explained variance")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_pca_scatter(Z, window_info, out_path, title, color_col="event"):
    if color_col not in window_info.columns:
        warnings.warn(f"{color_col} not found in window_info. Skip scatter.")
        return

    color_values = window_info[color_col].to_numpy()

    plt.figure(figsize=(7, 6))
    sc = plt.scatter(
        Z[:, 0],
        Z[:, 1],
        c=color_values,
        s=8,
        alpha=0.75,
        cmap="viridis",
    )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    cbar = plt.colorbar(sc)
    cbar.set_label(color_col)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_pca_trajectory_plot(Z, window_info, out_path, title, max_points=1200):
    """
    Plot a temporally ordered PC1-PC2 trajectory.
    To keep figure readable, uniformly downsample points if needed.
    """
    n = Z.shape[0]
    if n > max_points:
        idx = np.linspace(0, n - 1, max_points)
        idx = np.round(idx).astype(np.int64)
    else:
        idx = np.arange(n)

    plt.figure(figsize=(7, 6))
    plt.plot(Z[idx, 0], Z[idx, 1], linewidth=0.8, alpha=0.8)
    sc = plt.scatter(
        Z[idx, 0],
        Z[idx, 1],
        c=window_info["time_sec"].to_numpy()[idx],
        s=8,
        alpha=0.8,
        cmap="viridis",
    )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    cbar = plt.colorbar(sc)
    cbar.set_label("time_sec")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_jaccard_plot(jaccard_df, out_path, title):
    plt.figure(figsize=(7, 5))
    plt.plot(
        jaccard_df["delta"],
        jaccard_df["jaccard_mean"],
        marker="o",
        label="mean",
    )
    plt.plot(
        jaccard_df["delta"],
        jaccard_df["jaccard_median"],
        marker="s",
        label="median",
    )
    plt.xlabel("Delta in window index")
    plt.ylabel("kNN Jaccard overlap")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_smoothness_plot(window_info, velocity, curvature, out_path, title):
    time = window_info["time_sec"].to_numpy()

    vel_time = time[1:]
    curv_time = time[1:-1]

    plt.figure(figsize=(10, 5))
    plt.plot(vel_time, velocity, linewidth=0.8, label="velocity")
    if len(curvature):
        plt.plot(curv_time, curvature, linewidth=0.8, label="curvature")

    # Lightly mark task windows by vertical background hints.
    # To avoid giant patch lists, only mark event transition points.
    event = window_info["event"].to_numpy()
    transitions = np.where(np.diff(event) != 0)[0]
    for idx in transitions:
        plt.axvline(time[idx], linewidth=0.5, alpha=0.3)

    plt.xlabel("Time (s)")
    plt.ylabel("Standardized PCA trajectory norm")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def analyze_one_subject(subject_id, args, output_dir: Path):
    h5_path = Path(args.input_dir) / f"mema_subject{subject_id}_continuous_labeled.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"Cannot find HDF5 file: {h5_path}")

    print(f"\n[Subject {subject_id:02d}] Loading {h5_path}")
    data = load_subject_h5(h5_path, subject_id)

    eeg = data["eeg"]
    event = data["event"]
    label = data["label"]
    task_id = data["task_id"]

    channel_idx = choose_channels(eeg.shape[0], args.channels)

    print(f"[Subject {subject_id:02d}] Building window matrix...")
    X, window_info = build_window_matrix(
        eeg=eeg,
        event=event,
        label=label,
        task_id=task_id,
        sfreq=args.sfreq,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        channel_idx=channel_idx,
        feature_step=args.feature_step,
        event_mode=args.event_mode,
        max_windows=args.max_windows,
    )

    print(
        f"[Subject {subject_id:02d}] X shape = {X.shape}, "
        f"event ratio = {window_info['event'].mean():.3f}"
    )

    print(f"[Subject {subject_id:02d}] Fitting PCA...")
    pca, Z = fit_pca(
        X,
        n_components=args.pca_components,
        random_state=args.random_state,
    )

    pca_summary = compute_pca_summary(pca)

    print(f"[Subject {subject_id:02d}] Computing MLE intrinsic dimension...")
    mle_summary = compute_mle_intrinsic_dimension(Z, args.mle_ks)

    print(f"[Subject {subject_id:02d}] Computing kNN neighborhood consistency...")
    knn_dim = min(args.pca_components, Z.shape[1])
    neighbor_indices = compute_knn_neighbors(Z[:, :knn_dim], args.knn_k)
    jaccard_df = compute_jaccard_over_time(neighbor_indices, args.jaccard_deltas)
    jaccard_df.insert(0, "subject", subject_id)

    print(f"[Subject {subject_id:02d}] Computing velocity and curvature...")
    velocity, curvature = compute_velocity_curvature(Z, smooth_dim=args.smooth_dim)

    # Timeseries output.
    ts_df = window_info.copy()
    ts_df["subject"] = subject_id
    ts_df["pc1"] = Z[:, 0]
    ts_df["pc2"] = Z[:, 1]

    velocity_col = np.full(len(ts_df), np.nan, dtype=np.float32)
    if len(velocity):
        velocity_col[1:] = velocity
    ts_df["velocity"] = velocity_col

    curvature_col = np.full(len(ts_df), np.nan, dtype=np.float32)
    if len(curvature):
        curvature_col[1:-1] = curvature
    ts_df["curvature"] = curvature_col

    ts_path = output_dir / "timeseries" / f"subject{subject_id:02d}_raw_geometry_timeseries.csv"
    ts_df.to_csv(ts_path, index=False)

    # CSV outputs.
    jaccard_path = output_dir / "csv" / f"subject{subject_id:02d}_jaccard.csv"
    jaccard_df.to_csv(jaccard_path, index=False)

    evr_df = pd.DataFrame(
        {
            "component": np.arange(1, len(pca.explained_variance_ratio_) + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    evr_path = output_dir / "csv" / f"subject{subject_id:02d}_pca_variance.csv"
    evr_df.to_csv(evr_path, index=False)

    # Figures.
    fig_dir = output_dir / "figures"

    save_pca_variance_plot(
        pca,
        fig_dir / f"subject{subject_id:02d}_pca_cumulative_variance.png",
        title=f"Subject {subject_id:02d} raw EEG PCA cumulative variance",
    )

    save_pca_scatter(
        Z,
        window_info,
        fig_dir / f"subject{subject_id:02d}_pca_event.png",
        title=f"Subject {subject_id:02d} raw EEG PCA colored by event",
        color_col="event",
    )

    save_pca_scatter(
        Z,
        window_info,
        fig_dir / f"subject{subject_id:02d}_pca_task_id.png",
        title=f"Subject {subject_id:02d} raw EEG PCA colored by task_id",
        color_col="task_id",
    )

    save_pca_scatter(
        Z,
        window_info,
        fig_dir / f"subject{subject_id:02d}_pca_arousal.png",
        title=f"Subject {subject_id:02d} raw EEG PCA colored by arousal",
        color_col="arousal",
    )

    save_pca_trajectory_plot(
        Z,
        window_info,
        fig_dir / f"subject{subject_id:02d}_pca_trajectory_time.png",
        title=f"Subject {subject_id:02d} raw EEG PCA trajectory over time",
    )

    save_jaccard_plot(
        jaccard_df,
        fig_dir / f"subject{subject_id:02d}_knn_jaccard.png",
        title=f"Subject {subject_id:02d} kNN neighborhood consistency",
    )

    save_smoothness_plot(
        window_info,
        velocity,
        curvature,
        fig_dir / f"subject{subject_id:02d}_velocity_curvature.png",
        title=f"Subject {subject_id:02d} raw EEG local smoothness",
    )

    summary = {
        "subject": subject_id,
        "h5_path": str(h5_path),
        "h5_group": data["group_name"],
        "n_channels_used": int(len(channel_idx)),
        "channels_mode": args.channels,
        "n_windows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "window_sec": args.window_sec,
        "stride_sec": args.stride_sec,
        "feature_step": args.feature_step,
        "event_mode": args.event_mode,
        "task_window_ratio": float(window_info["event"].mean()),
        "velocity_mean": float(np.nanmean(velocity)) if len(velocity) else np.nan,
        "velocity_median": float(np.nanmedian(velocity)) if len(velocity) else np.nan,
        "velocity_std": float(np.nanstd(velocity)) if len(velocity) else np.nan,
        "curvature_mean": float(np.nanmean(curvature)) if len(curvature) else np.nan,
        "curvature_median": float(np.nanmedian(curvature)) if len(curvature) else np.nan,
        "curvature_std": float(np.nanstd(curvature)) if len(curvature) else np.nan,
    }
    summary.update(pca_summary)
    summary.update(mle_summary)

    meta_path = output_dir / "metadata" / f"subject{subject_id:02d}_summary.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[Subject {subject_id:02d}] Done.")

    return summary, jaccard_df


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_dirs(output_dir)

    config_path = output_dir / "metadata" / "run_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    all_summaries = []
    all_jaccards = []

    for subject_id in args.subjects:
        try:
            summary, jaccard_df = analyze_one_subject(subject_id, args, output_dir)
            all_summaries.append(summary)
            all_jaccards.append(jaccard_df)
        except Exception as e:
            print(f"[Subject {subject_id:02d}] ERROR: {repr(e)}")
            continue

    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        summary_path = output_dir / "csv" / "all_subjects_raw_geometry_summary.csv"
        summary_df.to_csv(summary_path, index=False)

        print(f"\nSaved summary CSV:")
        print(summary_path)

    if all_jaccards:
        all_jaccard_df = pd.concat(all_jaccards, ignore_index=True)
        all_jaccard_path = output_dir / "csv" / "all_subjects_jaccard.csv"
        all_jaccard_df.to_csv(all_jaccard_path, index=False)

        print(f"\nSaved Jaccard CSV:")
        print(all_jaccard_path)

    print("\nAll done. Raw EEG geometry map is ready 🧭")


if __name__ == "__main__":
    main()