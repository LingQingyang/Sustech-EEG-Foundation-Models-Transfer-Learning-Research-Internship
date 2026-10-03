#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-subject condition-wise raw EEG geometry analysis for MEMA stage_v2 HDF5.

For each subject, run two analyses separately:
    1) nontask: stage_id in {0, 2, 3} = start + self_assessment + rest
    2) task:    stage_id == 1 = video_clip

Important:
    - Windows are created within each stage segment only.
    - PCA is fitted separately for each subject-condition pair.
    - kNN neighborhoods are computed globally within the condition space.
    - kNN temporal consistency, velocity, and curvature are evaluated only within
      the same continuous segment, so artificial jumps across gaps are avoided.

Outputs per subject-condition:
    figures/
      subject01_task_pca_cumulative_variance.png
      subject01_task_pca_stage_id.png
      subject01_task_pca_trial_id.png
      subject01_task_pca_task_id.png            # for task only, if applicable
      subject01_task_pca_time.png
      subject01_task_pca_trajectory_time.png
      subject01_task_knn_jaccard.png
      subject01_task_velocity_curvature.png

    csv/
      subject01_task_pca_variance.csv
      subject01_task_window_geometry_scores.csv
      subject01_task_knn_jaccard.csv

    csv/subject_condition_geometry_summary.csv
    csv/subject_condition_knn_jaccard_all.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


CONDITION_STAGE_IDS = {
    "task": [1],
    "nontask": [0, 2, 3],
}

DEFAULT_STAGE_NAMES = ["start", "video_clip", "self_assessment", "rest"]
DEFAULT_LABEL_NAMES = ["attention", "valence", "arousal", "dominance"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Per-subject task/nontask raw EEG geometry analysis for MEMA stage_v2 HDF5."
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default="/mnt/dataset4/lingqy/mema/output_stage_v2",
        help="Directory containing MEMA stage_v2 HDF5 files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/lqy/projects/mema/output/subject_condition_geometry",
        help="Output directory.",
    )
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        default=[1],
        help="Subject IDs, e.g. --subjects 1 or --subjects 1 2 3.",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        nargs="+",
        choices=["task", "nontask"],
        default=["nontask", "task"],
        help="Conditions to analyze. Default: nontask task.",
    )

    parser.add_argument("--sfreq", type=float, default=200.0, help="Sampling frequency. Default: 200 Hz.")
    parser.add_argument("--window-sec", type=float, default=1.0, help="Window length in seconds.")
    parser.add_argument("--stride-sec", type=float, default=0.5, help="Stride length in seconds.")
    parser.add_argument(
        "--channels",
        type=str,
        choices=["eeg", "all"],
        default="eeg",
        help="Use first 30 EEG channels or all 32 channels including EOG.",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=100,
        help="Number of PCA components to fit. Will be clipped by sample count and feature dimension.",
    )
    parser.add_argument(
        "--geom-dims",
        type=int,
        default=20,
        help="Use first N standardized PCs for kNN, velocity, curvature, and optional MLE ID.",
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=15,
        help="k for kNN neighborhood consistency.",
    )
    parser.add_argument(
        "--deltas",
        type=int,
        nargs="+",
        default=[1, 2, 5, 10, 20, 50],
        help="Window-index deltas for kNN Jaccard. Delta=1 corresponds to stride-sec.",
    )
    parser.add_argument(
        "--mle-ks",
        type=int,
        nargs="+",
        default=[10, 20, 30],
        help="k values for MLE intrinsic dimension computed in standardized PCA geometry space.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random seed for randomized PCA.",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Optional cap on number of windows per subject-condition. If set, windows are evenly subsampled after extraction.",
    )

    return parser.parse_args()


def ensure_dirs(output_dir: Path):
    for sub in ["figures", "csv", "metadata"]:
        (output_dir / sub).mkdir(parents=True, exist_ok=True)


def decode_str_array(arr) -> List[str]:
    out = []
    for x in arr:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return out


def find_h5_path(input_dir: Path, subject_id: int) -> Path:
    candidates = [
        input_dir / f"mema_subject{subject_id}_continuous_labeled.h5",
        input_dir / f"mema_subject{subject_id:02d}_continuous_labeled.h5",
        input_dir / f"subject{subject_id}_continuous_labeled.h5",
        input_dir / f"subject{subject_id:02d}_continuous_labeled.h5",
    ]
    for path in candidates:
        if path.exists():
            return path

    hits = sorted(input_dir.glob(f"*subject{subject_id}*.h5")) + sorted(input_dir.glob(f"*subject{subject_id:02d}*.h5"))
    if hits:
        return hits[0]

    raise FileNotFoundError(
        f"Cannot find HDF5 for subject {subject_id} in {input_dir}. "
        f"Tried common names and glob patterns."
    )


def get_subject_group(h5_file: h5py.File, subject_id: int):
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
        f"Cannot find subject group for subject {subject_id}. Available groups under data/: {keys}"
    )


def load_subject(h5_path: Path, subject_id: int) -> Dict:
    with h5py.File(h5_path, "r") as f:
        group, group_name = get_subject_group(f, subject_id)

        required = [
            "preprocessed_eeg",
            "event",
            "sample_index",
            "task_labels",
            "task_id",
            "stage_id",
            "stage_trial_id",
            "stage_bounds",
        ]
        missing = [k for k in required if k not in group]
        if missing:
            raise KeyError(f"Missing required datasets in {h5_path}: {missing}")

        data = {
            "h5_path": str(h5_path),
            "group_name": group_name,
            "preprocessed_eeg": group["preprocessed_eeg"][:].astype(np.float32),
            "event": group["event"][:].squeeze().astype(np.int8),
            "sample_index": group["sample_index"][:].astype(np.int64),
            "task_labels": group["task_labels"][:].astype(np.int8),
            "task_id": group["task_id"][:].squeeze().astype(np.int16),
            "stage_id": group["stage_id"][:].squeeze().astype(np.int8),
            "stage_trial_id": group["stage_trial_id"][:].squeeze().astype(np.int16),
            "stage_bounds": group["stage_bounds"][:].astype(np.int64),
        }

        data["stage_names"] = decode_str_array(group["stage_names"][:]) if "stage_names" in group else DEFAULT_STAGE_NAMES
        data["label_names"] = decode_str_array(group["label_names"][:]) if "label_names" in group else DEFAULT_LABEL_NAMES
        data["electrodes"] = decode_str_array(group["electrodes"][:]) if "electrodes" in group else None

    return data


def choose_channels(n_channels: int, mode: str) -> np.ndarray:
    if mode == "eeg":
        if n_channels < 30:
            raise ValueError(f"Expected at least 30 channels, got {n_channels}.")
        return np.arange(30)
    if mode == "all":
        return np.arange(n_channels)
    raise ValueError(f"Unknown channel mode: {mode}")


def collect_segments(stage_bounds: np.ndarray, condition: str, min_len: int) -> List[Dict]:
    stage_ids = CONDITION_STAGE_IDS[condition]
    segments = []
    seg_idx = 0
    n_trials = stage_bounds.shape[0]

    for trial_id in range(n_trials):
        for sid in stage_ids:
            start, stop = stage_bounds[trial_id, sid]
            start = int(start)
            stop = int(stop)
            if stop > start and (stop - start) >= min_len:
                segments.append(
                    {
                        "segment_index": seg_idx,
                        "trial_id": trial_id,
                        "stage_id": int(sid),
                        "start": start,
                        "stop": stop,
                        "duration_sec": None,
                    }
                )
                seg_idx += 1
    return segments


def make_window_index_table(
    segments: List[Dict],
    sfreq: float,
    win_len: int,
    stride: int,
) -> pd.DataFrame:
    rows = []
    global_window = 0
    condition_elapsed = 0.0

    for seg in segments:
        starts = np.arange(seg["start"], seg["stop"] - win_len + 1, stride, dtype=np.int64)
        for local_idx, s in enumerate(starts):
            rows.append(
                {
                    "window_index": global_window,
                    "segment_index": seg["segment_index"],
                    "segment_window_index": local_idx,
                    "segment_start_sample": seg["start"],
                    "segment_stop_sample": seg["stop"],
                    "segment_stage_id": seg["stage_id"],
                    "segment_trial_id": seg["trial_id"],
                    "start_sample": int(s),
                    "center_sample": int(s + win_len // 2),
                    "absolute_time_sec": float(s / sfreq),
                    "within_segment_time_sec": float((s - seg["start"]) / sfreq),
                    "condition_elapsed_sec": float(condition_elapsed),
                }
            )
            global_window += 1
            condition_elapsed += stride / sfreq

    return pd.DataFrame(rows)


def maybe_subsample_windows(window_info: pd.DataFrame, max_windows: int | None) -> pd.DataFrame:
    if max_windows is None or len(window_info) <= max_windows:
        return window_info.reset_index(drop=True)
    keep = np.linspace(0, len(window_info) - 1, max_windows).round().astype(int)
    return window_info.iloc[keep].reset_index(drop=True)


def build_window_matrix(
    eeg: np.ndarray,
    event: np.ndarray,
    task_id: np.ndarray,
    stage_id: np.ndarray,
    stage_trial_id: np.ndarray,
    task_labels: np.ndarray,
    channel_idx: np.ndarray,
    window_info: pd.DataFrame,
    win_len: int,
    subject_id: int,
    condition: str,
) -> Tuple[np.ndarray, pd.DataFrame]:
    n_windows = len(window_info)
    n_features = len(channel_idx) * win_len
    X = np.empty((n_windows, n_features), dtype=np.float32)

    starts = window_info["start_sample"].to_numpy(dtype=np.int64)
    centers = window_info["center_sample"].to_numpy(dtype=np.int64)

    for i, s in enumerate(starts):
        seg = eeg[channel_idx, s : s + win_len]
        X[i] = seg.reshape(-1)

    center_event = event[centers]
    center_task_id = task_id[centers]
    center_stage_id = stage_id[centers]
    center_stage_trial_id = stage_trial_id[centers]

    labels = np.full((n_windows, 4), -1, dtype=np.int8)
    valid_trial = (center_stage_trial_id >= 0) & (center_stage_trial_id < task_labels.shape[0])
    labels[valid_trial] = task_labels[center_stage_trial_id[valid_trial]]

    info = window_info.copy()
    info.insert(0, "condition", condition)
    info.insert(0, "subject", subject_id)
    info["event"] = center_event
    info["task_id"] = center_task_id
    info["stage_id"] = center_stage_id
    info["stage_trial_id"] = center_stage_trial_id
    info["attention"] = labels[:, 0]
    info["valence"] = labels[:, 1]
    info["arousal"] = labels[:, 2]
    info["dominance"] = labels[:, 3]

    return X, info


def fit_pca(X: np.ndarray, n_components: int, random_state: int) -> Tuple[PCA, np.ndarray]:
    n_components = min(n_components, X.shape[0] - 1, X.shape[1])
    if n_components < 2:
        raise ValueError(f"Too few samples for PCA: X shape={X.shape}")

    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=random_state)
    Z = pca.fit_transform(X)
    return pca, Z


def first_dim_above(cum: np.ndarray, threshold: float):
    idx = np.searchsorted(cum, threshold)
    if idx >= len(cum):
        return np.nan
    return int(idx + 1)


def pca_summary(pca: PCA) -> Dict:
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    eig = pca.explained_variance_
    participation_ratio = (np.sum(eig) ** 2) / (np.sum(eig ** 2) + 1e-12)

    def sum_first(n: int):
        return float(np.sum(evr[: min(n, len(evr))]))

    return {
        "pc1": float(evr[0]),
        "pc2": float(evr[1]) if len(evr) > 1 else np.nan,
        "top5": sum_first(5),
        "top10": sum_first(10),
        "top20": sum_first(20),
        "top50": sum_first(50),
        "d80": first_dim_above(cum, 0.80),
        "d90": first_dim_above(cum, 0.90),
        "d95": first_dim_above(cum, 0.95),
        "d99": first_dim_above(cum, 0.99),
        "participation_ratio": float(participation_ratio),
    }


def standardize_pca_space(Z: np.ndarray, geom_dims: int) -> np.ndarray:
    m = min(geom_dims, Z.shape[1])
    G = Z[:, :m].astype(np.float64)
    G = G - np.nanmean(G, axis=0, keepdims=True)
    sd = np.nanstd(G, axis=0, keepdims=True)
    sd[sd < 1e-12] = 1.0
    return G / sd


def compute_velocity_curvature(G: np.ndarray, info: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    info = info.copy()
    n = len(info)
    velocity = np.full(n, np.nan, dtype=np.float64)
    curvature = np.full(n, np.nan, dtype=np.float64)

    for _, idx_series in info.groupby("segment_index", sort=False).groups.items():
        idx = np.array(list(idx_series), dtype=np.int64)
        idx = idx[np.argsort(info.loc[idx, "segment_window_index"].to_numpy())]
        if len(idx) >= 2:
            d1 = np.diff(G[idx], axis=0)
            v = np.linalg.norm(d1, axis=1)
            velocity[idx[1:]] = v
        if len(idx) >= 3:
            d2 = G[idx[2:]] - 2 * G[idx[1:-1]] + G[idx[:-2]]
            c = np.linalg.norm(d2, axis=1)
            curvature[idx[1:-1]] = c

    info["velocity"] = velocity
    info["curvature"] = curvature

    summary = {
        "velocity_mean": float(np.nanmean(velocity)) if np.any(~np.isnan(velocity)) else np.nan,
        "velocity_median": float(np.nanmedian(velocity)) if np.any(~np.isnan(velocity)) else np.nan,
        "velocity_std": float(np.nanstd(velocity)) if np.any(~np.isnan(velocity)) else np.nan,
        "velocity_max": float(np.nanmax(velocity)) if np.any(~np.isnan(velocity)) else np.nan,
        "curvature_mean": float(np.nanmean(curvature)) if np.any(~np.isnan(curvature)) else np.nan,
        "curvature_median": float(np.nanmedian(curvature)) if np.any(~np.isnan(curvature)) else np.nan,
        "curvature_std": float(np.nanstd(curvature)) if np.any(~np.isnan(curvature)) else np.nan,
        "curvature_max": float(np.nanmax(curvature)) if np.any(~np.isnan(curvature)) else np.nan,
    }
    return info, summary


def compute_knn_jaccard(
    G: np.ndarray,
    info: pd.DataFrame,
    k: int,
    deltas: Iterable[int],
) -> pd.DataFrame:
    n = G.shape[0]
    if n < 3:
        return pd.DataFrame()

    k_eff = min(k, n - 1)
    nbrs = NearestNeighbors(n_neighbors=k_eff + 1, algorithm="auto", metric="euclidean")
    nbrs.fit(G)
    _, indices = nbrs.kneighbors(G)
    neigh = indices[:, 1:]  # remove self

    rows = []
    grouped_indices = []
    for _, idx_series in info.groupby("segment_index", sort=False).groups.items():
        idx = np.array(list(idx_series), dtype=np.int64)
        idx = idx[np.argsort(info.loc[idx, "segment_window_index"].to_numpy())]
        grouped_indices.append(idx)

    for delta in deltas:
        vals = []
        for idx in grouped_indices:
            if len(idx) <= delta:
                continue
            for p in range(0, len(idx) - delta):
                a = idx[p]
                b = idx[p + delta]
                A = set(neigh[a].tolist())
                B = set(neigh[b].tolist())
                union = len(A | B)
                if union == 0:
                    continue
                vals.append(len(A & B) / union)

        if vals:
            arr = np.array(vals, dtype=np.float64)
            rows.append(
                {
                    "delta": int(delta),
                    "time_gap_sec": float(delta) * float(info.attrs.get("stride_sec", np.nan)),
                    "k": int(k_eff),
                    "n_pairs": int(len(arr)),
                    "mean_jaccard": float(np.nanmean(arr)),
                    "median_jaccard": float(np.nanmedian(arr)),
                    "std_jaccard": float(np.nanstd(arr)),
                }
            )
        else:
            rows.append(
                {
                    "delta": int(delta),
                    "time_gap_sec": float(delta) * float(info.attrs.get("stride_sec", np.nan)),
                    "k": int(k_eff),
                    "n_pairs": 0,
                    "mean_jaccard": np.nan,
                    "median_jaccard": np.nan,
                    "std_jaccard": np.nan,
                }
            )

    return pd.DataFrame(rows)


def mle_intrinsic_dimension(G: np.ndarray, ks: Iterable[int]) -> Dict:
    n = G.shape[0]
    out = {}
    if n < 5:
        for k in ks:
            out[f"mle_id_k{k}"] = np.nan
        return out

    max_k = min(max(ks), n - 1)
    nbrs = NearestNeighbors(n_neighbors=max_k + 1, algorithm="auto", metric="euclidean")
    nbrs.fit(G)
    distances, _ = nbrs.kneighbors(G)
    # remove self distance
    d = distances[:, 1:]
    eps = 1e-12

    for k in ks:
        k_eff = min(k, d.shape[1])
        if k_eff < 2:
            out[f"mle_id_k{k}"] = np.nan
            continue
        dk = d[:, k_eff - 1]
        local = d[:, : k_eff - 1]
        valid = (dk > eps) & np.all(local > eps, axis=1)
        if not np.any(valid):
            out[f"mle_id_k{k}"] = np.nan
            continue
        logs = np.log((dk[valid, None] + eps) / (local[valid] + eps))
        denom = np.mean(logs, axis=1)
        denom = denom[denom > eps]
        if len(denom) == 0:
            out[f"mle_id_k{k}"] = np.nan
        else:
            ids = 1.0 / denom
            out[f"mle_id_k{k}"] = float(np.nanmean(ids))

    return out


def save_pca_variance_plot(pca: PCA, out_path: Path, title: str):
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)

    plt.figure(figsize=(8, 5.5))
    plt.plot(np.arange(1, len(cum) + 1), cum, marker="o", markersize=3)
    for y in [0.80, 0.90, 0.95]:
        plt.axhline(y, linestyle="--", linewidth=1)
    plt.xlabel("Number of PCA components")
    plt.ylabel("Cumulative explained variance")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_pca_scatter(Z: np.ndarray, info: pd.DataFrame, out_path: Path, title: str, color_col: str):
    if color_col not in info.columns:
        return
    c = info[color_col].to_numpy()

    plt.figure(figsize=(7.5, 6))
    sc = plt.scatter(Z[:, 0], Z[:, 1], c=c, s=10, alpha=0.75, cmap="viridis")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    cbar = plt.colorbar(sc)
    cbar.set_label(color_col)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_trajectory_plot(Z: np.ndarray, info: pd.DataFrame, out_path: Path, title: str):
    time = info["absolute_time_sec"].to_numpy()

    plt.figure(figsize=(7.5, 6))
    # break lines at segment boundaries
    for _, idx_series in info.groupby("segment_index", sort=False).groups.items():
        idx = np.array(list(idx_series), dtype=np.int64)
        idx = idx[np.argsort(info.loc[idx, "segment_window_index"].to_numpy())]
        if len(idx) >= 2:
            plt.plot(Z[idx, 0], Z[idx, 1], linewidth=0.7, alpha=0.55)
    sc = plt.scatter(Z[:, 0], Z[:, 1], c=time, s=10, alpha=0.8, cmap="viridis")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    cbar = plt.colorbar(sc)
    cbar.set_label("absolute_time_sec")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_knn_plot(knn_df: pd.DataFrame, out_path: Path, title: str):
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


def save_velocity_curvature_plot(info: pd.DataFrame, out_path: Path, title: str):
    plt.figure(figsize=(9, 5))

    # plot each segment separately to avoid connecting across gaps
    for _, idx_series in info.groupby("segment_index", sort=False).groups.items():
        idx = np.array(list(idx_series), dtype=np.int64)
        idx = idx[np.argsort(info.loc[idx, "segment_window_index"].to_numpy())]
        x = info.loc[idx, "absolute_time_sec"].to_numpy()
        v = info.loc[idx, "velocity"].to_numpy()
        c = info.loc[idx, "curvature"].to_numpy()
        plt.plot(x, v, linewidth=0.8, alpha=0.85, color="C0")
        plt.plot(x, c, linewidth=0.8, alpha=0.85, color="C1")

    # stage/segment boundaries
    starts = info.groupby("segment_index")["absolute_time_sec"].min().to_numpy()
    stops = info.groupby("segment_index")["absolute_time_sec"].max().to_numpy()
    for x in np.concatenate([starts, stops]):
        plt.axvline(x, linewidth=0.5, alpha=0.18)

    # proxy legend
    plt.plot([], [], color="C0", label="velocity")
    plt.plot([], [], color="C1", label="curvature")
    plt.xlabel("Absolute time (s)")
    plt.ylabel("Standardized PCA trajectory norm")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def analyze_subject_condition(subject_id: int, condition: str, args, output_dir: Path) -> Tuple[Dict, pd.DataFrame]:
    input_dir = Path(args.input_dir)
    h5_path = find_h5_path(input_dir, subject_id)
    data = load_subject(h5_path, subject_id)

    eeg = data["preprocessed_eeg"]
    event = data["event"]
    task_id = data["task_id"]
    stage_id = data["stage_id"]
    stage_trial_id = data["stage_trial_id"]
    task_labels = data["task_labels"]
    stage_bounds = data["stage_bounds"]

    win_len = int(round(args.window_sec * args.sfreq))
    stride = int(round(args.stride_sec * args.sfreq))
    channel_idx = choose_channels(eeg.shape[0], args.channels)

    segments = collect_segments(stage_bounds, condition=condition, min_len=win_len)
    if not segments:
        raise ValueError(f"No valid segments for subject {subject_id}, condition={condition}.")

    window_info = make_window_index_table(segments, sfreq=args.sfreq, win_len=win_len, stride=stride)
    window_info = maybe_subsample_windows(window_info, args.max_windows)
    window_info.attrs["stride_sec"] = args.stride_sec

    if len(window_info) < 3:
        raise ValueError(f"Too few windows for subject {subject_id}, condition={condition}: {len(window_info)}")

    X, info = build_window_matrix(
        eeg=eeg,
        event=event,
        task_id=task_id,
        stage_id=stage_id,
        stage_trial_id=stage_trial_id,
        task_labels=task_labels,
        channel_idx=channel_idx,
        window_info=window_info,
        win_len=win_len,
        subject_id=subject_id,
        condition=condition,
    )
    info.attrs["stride_sec"] = args.stride_sec

    print(
        f"[Subject {subject_id:02d} {condition}] "
        f"segments={len(segments)}, windows={len(info)}, X={X.shape}, "
        f"stage_counts={dict(info['stage_id'].value_counts().sort_index())}"
    )

    pca, Z = fit_pca(X, n_components=args.pca_components, random_state=args.random_state)
    G = standardize_pca_space(Z, geom_dims=args.geom_dims)

    info, smooth_summary = compute_velocity_curvature(G, info)
    knn_df = compute_knn_jaccard(G, info, k=args.knn_k, deltas=args.deltas)
    mle_summary = mle_intrinsic_dimension(G, ks=args.mle_ks)

    prefix = f"subject{subject_id:02d}_{condition}"

    # Save PCA variance CSV
    evr = pca.explained_variance_ratio_
    pd.DataFrame(
        {
            "component": np.arange(1, len(evr) + 1),
            "explained_variance_ratio": evr,
            "cumulative_explained_variance": np.cumsum(evr),
        }
    ).to_csv(output_dir / "csv" / f"{prefix}_pca_variance.csv", index=False)

    # Save window scores CSV
    info_out = info.copy()
    info_out["pc1"] = Z[:, 0]
    info_out["pc2"] = Z[:, 1]
    for j in range(min(10, Z.shape[1])):
        info_out[f"pc{j+1}"] = Z[:, j]
    info_out.to_csv(output_dir / "csv" / f"{prefix}_window_geometry_scores.csv", index=False)

    # Save kNN CSV
    if not knn_df.empty:
        knn_df.insert(0, "condition", condition)
        knn_df.insert(0, "subject", subject_id)
        knn_df.to_csv(output_dir / "csv" / f"{prefix}_knn_jaccard.csv", index=False)

    # Figures
    fig_dir = output_dir / "figures"
    save_pca_variance_plot(
        pca,
        fig_dir / f"{prefix}_pca_cumulative_variance.png",
        title=f"Subject {subject_id:02d} {condition} PCA cumulative variance",
    )
    save_pca_scatter(
        Z,
        info,
        fig_dir / f"{prefix}_pca_stage_id.png",
        title=f"Subject {subject_id:02d} {condition} PCA colored by stage_id",
        color_col="stage_id",
    )
    save_pca_scatter(
        Z,
        info,
        fig_dir / f"{prefix}_pca_trial_id.png",
        title=f"Subject {subject_id:02d} {condition} PCA colored by stage_trial_id",
        color_col="stage_trial_id",
    )
    # Useful for task condition; harmless for nontask but likely constant -1.
    save_pca_scatter(
        Z,
        info,
        fig_dir / f"{prefix}_pca_task_id.png",
        title=f"Subject {subject_id:02d} {condition} PCA colored by task_id",
        color_col="task_id",
    )
    save_pca_scatter(
        Z,
        info,
        fig_dir / f"{prefix}_pca_time.png",
        title=f"Subject {subject_id:02d} {condition} PCA colored by absolute time",
        color_col="absolute_time_sec",
    )
    save_trajectory_plot(
        Z,
        info,
        fig_dir / f"{prefix}_pca_trajectory_time.png",
        title=f"Subject {subject_id:02d} {condition} PCA trajectory over time",
    )
    save_knn_plot(
        knn_df,
        fig_dir / f"{prefix}_knn_jaccard.png",
        title=f"Subject {subject_id:02d} {condition} kNN neighborhood consistency",
    )
    save_velocity_curvature_plot(
        info,
        fig_dir / f"{prefix}_velocity_curvature.png",
        title=f"Subject {subject_id:02d} {condition} velocity / curvature",
    )

    # Summary
    pca_sum = pca_summary(pca)
    summary = {
        "subject": subject_id,
        "condition": condition,
        "h5_path": str(h5_path),
        "n_segments": len(segments),
        "n_windows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_channels_used": len(channel_idx),
        "channels": args.channels,
        "window_sec": args.window_sec,
        "stride_sec": args.stride_sec,
        "pca_components_fit": int(len(evr)),
        "geom_dims_used": int(G.shape[1]),
        "n_stage0_start": int((info["stage_id"] == 0).sum()),
        "n_stage1_video": int((info["stage_id"] == 1).sum()),
        "n_stage2_assessment": int((info["stage_id"] == 2).sum()),
        "n_stage3_rest": int((info["stage_id"] == 3).sum()),
        "duration_min_abs_span": float((info["absolute_time_sec"].max() - info["absolute_time_sec"].min()) / 60.0),
    }
    summary.update({f"pca_{k}": v for k, v in pca_sum.items()})
    summary.update(mle_summary)
    summary.update(smooth_summary)

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
        for condition in args.conditions:
            try:
                summary, knn_df = analyze_subject_condition(subject_id, condition, args, output_dir)
                summaries.append(summary)
                if knn_df is not None and not knn_df.empty:
                    all_knn.append(knn_df)
            except Exception as e:
                print(f"[Subject {subject_id:02d} {condition}] ERROR: {repr(e)}")

    if summaries:
        pd.DataFrame(summaries).to_csv(
            output_dir / "csv" / "subject_condition_geometry_summary.csv",
            index=False,
        )
    if all_knn:
        pd.concat(all_knn, ignore_index=True).to_csv(
            output_dir / "csv" / "subject_condition_knn_jaccard_all.csv",
            index=False,
        )

    print("\nDone. Subject condition geometry analysis saved.")


if __name__ == "__main__":
    main()
