#!/usr/bin/env python3
"""
Experiment A v3: within-subject five-state semantic LDA for ds007554.

Scientific question
-------------------
For each individual subject, can five matched EEG-FM embedding states be
separated across sessions, how many multiclass LD dimensions are actually
needed, and what named pairwise contrasts give those anonymous multiclass LDs
an interpretable semantic decomposition?

States
------
    Baseline, Mental Arithmetic (MA), N-back (NB),
    N-back Arithmetic (NBMA), Full Integrated Task (Full)

Design
------
* Three-fold leave-one-session-out within each subject.
* All centering, SVD, covariance estimation, class means,
  multiclass LDA and pairwise LD axes are fit on the outer-training sessions.
* All available stable 1-second embedding windows are used. Baseline pools the
  pre-task segments from the four selected runs; task states use the full stable
  task context after onset guards. State/session/run/window weights remain balanced.
* Standard five-class LDA yields at most four canonical LD dimensions.
* The headline shrinkage is fixed before evaluation; optional sensitivity values
  are descriptive and are never selected on held-out performance.
* Effective dimensionality is reported by several complementary quantities:
  peak predictive dimension, one-standard-error dimension, permutation-tested
  leading eigen-dimensions, and participation-ratio spectral dimension.
* Ten named pairwise LDs are built in the exact same whitened geometry.
* The four anonymous multiclass LDs are interpreted with:
    1) cosine alignment to each named pairwise LD;
    2) minimum-norm coefficients over the redundant ten-axis dictionary;
    3) pair-contrast SVD right loadings;
    4) five-class centroid profiles along each multiclass LD.
* Near-degenerate LDs should be interpreted as a subspace block rather than as
  individually fixed axes. The script reports eigenvalue gaps and the raw
  matrices needed for block-level interpretation.

Important interpretation boundary
---------------------------------
The ten pairwise LDs are a redundant semantic dictionary, not ten independent
Euclidean dimensions. With a shared within-class covariance, their span is at
most four-dimensional and should match the standard multiclass LDA subspace.

Input HDF5 contract
-------------------
Required datasets:
    embedding, subject_id, session_id, run_id, sample_start, sample_end,
    phase_id, task_id, task_family_id
Optional:
    center_sample, task_names, task_family_names

Default ds007554 task IDs:
    MA=0, NB=1, NBMA=5, Full=6
Override with, for example:
    --task-ids MA=0,NB=1,NBMA=5,Full=6
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import platform
import sys
import time
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_VERSION = "2026-07-19-expA-five-state-semantic-lda-v3.1"

MODEL_PATHS: Dict[str, str] = {
    "EEGMamba": "/mnt/dataset4/fuxy/FMS/FM/EEGMamba/output_emb/ds007554_embeddings.h5",
    "BIOT": "/mnt/dataset4/fuxy/FMS/FM/BIOT/output_emb/ds007554_embeddings.h5",
    "LaBraM": "/mnt/dataset4/fuxy/FMS/FM/LaBraM/output_emb/ds007554_embeddings.h5",
    "CBraMod": "/mnt/dataset4/fuxy/FMS/FM/CBraMod/output_emb/ds007554_embeddings.h5",
    "EEGPT": "/mnt/dataset4/fuxy/FMS/FM/EEGPT/output_emb/ds007554_embeddings.h5",
}

STATE_NAMES = ["Baseline", "MA", "NB", "NBMA", "Full"]
STATE_SHORT = ["B", "MA", "NB", "NBMA", "Full"]
STATE_IDS = {name: i for i, name in enumerate(STATE_NAMES)}
TASK_STATE_NAMES = ["MA", "NB", "NBMA", "Full"]
DEFAULT_TASK_IDS = {"MA": 0, "NB": 1, "NBMA": 5, "Full": 6}
PAIR_SPECS = list(itertools.combinations(range(len(STATE_NAMES)), 2))
REQUIRED_KEYS = (
    "embedding", "subject_id", "session_id", "run_id", "sample_start",
    "sample_end", "phase_id", "task_id", "task_family_id",
)


# =============================================================================
# Utilities
# =============================================================================


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def decode_scalar(v: Any) -> Any:
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return [decode_scalar(x) for x in v.tolist()]
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, dict):
        return {str(k): decode_scalar(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [decode_scalar(x) for x in v]
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v


def json_dump(path: Path | str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(decode_scalar(payload), fh, ensure_ascii=False, indent=1, default=str)


def csv_write(path: Path | str, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            out: Dict[str, Any] = {}
            for key in keys:
                value = decode_scalar(row.get(key, ""))
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, ensure_ascii=False)
                out[key] = value
            writer.writerow(out)


def parse_int_csv(text: str) -> List[int]:
    return [int(x) for x in str(text).replace(" ", "").split(",") if x]


def parse_float_csv(text: str) -> List[float]:
    return [float(x) for x in str(text).replace(" ", "").split(",") if x]


def parse_task_ids(text: str) -> Dict[str, int]:
    out = dict(DEFAULT_TASK_IDS)
    if not text:
        return out
    parsed: Dict[str, int] = {}
    aliases = {
        "MA": "MA", "MENTALARITHMETIC": "MA", "MENTAL_ARITHMETIC": "MA",
        "NB": "NB", "NBACK": "NB", "N-BACK": "NB",
        "NBMA": "NBMA", "NBACKARITHMETIC": "NBMA", "N-BACKARITHMETIC": "NBMA",
        "FULL": "Full", "FULLTASK": "Full", "FULLINTEGRATEDTASK": "Full",
    }
    for item in text.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"--task-ids expects NAME=ID entries, got {item!r}")
        raw_name, raw_id = item.split("=", 1)
        key = raw_name.strip().replace(" ", "").upper()
        if key not in aliases:
            raise ValueError(f"Unknown task alias {raw_name!r}; use MA, NB, NBMA, Full")
        parsed[aliases[key]] = int(raw_id)
    missing = [x for x in TASK_STATE_NAMES if x not in parsed]
    if missing:
        raise ValueError(f"--task-ids must define all four task states; missing {missing}")
    return parsed


def empirical_upper_p(real: float, null: np.ndarray) -> float:
    arr = np.asarray(null, dtype=float)
    arr = arr[np.isfinite(arr)]
    if not np.isfinite(real) or arr.size == 0:
        return float("nan")
    return float((1 + np.sum(arr >= real)) / (arr.size + 1))


def safe_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    return float(np.nanmean(arr)) if np.any(np.isfinite(arr)) else float("nan")


def safe_median(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    return float(np.nanmedian(arr)) if np.any(np.isfinite(arr)) else float("nan")


def safe_std(values: Iterable[float], ddof: int = 1) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= ddof:
        return float("nan")
    return float(np.std(arr, ddof=ddof))


def standard_error(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return float("nan")
    return float(np.std(arr, ddof=1) / np.sqrt(arr.size))


def participation_ratio(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size == 0:
        return 0.0
    denom = float(np.sum(arr * arr))
    return float(np.sum(arr) ** 2 / denom) if denom > 0 else 0.0


def numerical_rank_psd(matrix: np.ndarray, relative_tol: float = 1e-8) -> int:
    eig = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    top = float(np.max(np.abs(eig))) if eig.size else 0.0
    if top <= 0:
        return 0
    return int(np.sum(eig > relative_tol * top))


def pair_name(i: int, j: int) -> str:
    return f"{STATE_NAMES[i]}__vs__{STATE_NAMES[j]}"


# =============================================================================
# Metadata and embedding I/O
# =============================================================================


@dataclass
class Metadata:
    subject: np.ndarray
    session: np.ndarray
    run: np.ndarray
    sample_start: np.ndarray
    sample_end: np.ndarray
    center_sample: np.ndarray
    phase: np.ndarray
    task: np.ndarray
    family: np.ndarray
    sampling_rate: float
    stride_seconds: float
    window_seconds: float
    task_names: Dict[int, str]
    family_names: Dict[int, str]
    attrs: Dict[str, Any]


def read_vector(h5: h5py.File, key: str, dtype=None) -> np.ndarray:
    arr = np.asarray(h5[key][...])
    return arr.astype(dtype) if dtype is not None else arr


def read_name_map(h5: h5py.File, key: str) -> Dict[int, str]:
    if key not in h5:
        return {}
    raw = np.asarray(h5[key][...])
    return {int(i): str(decode_scalar(v)) for i, v in enumerate(raw.tolist())}


def load_metadata(h5: h5py.File, embedding_key: str) -> Metadata:
    missing = [key for key in REQUIRED_KEYS if key not in h5]
    if missing:
        raise KeyError(f"Missing required H5 keys: {missing}")
    n = int(h5[embedding_key].shape[0])
    arrays = {
        "subject": read_vector(h5, "subject_id", np.int64),
        "session": read_vector(h5, "session_id", np.int64),
        "run": read_vector(h5, "run_id", np.int64),
        "sample_start": read_vector(h5, "sample_start", np.int64),
        "sample_end": read_vector(h5, "sample_end", np.int64),
        "phase": read_vector(h5, "phase_id", np.int64),
        "task": read_vector(h5, "task_id", np.int64),
        "family": read_vector(h5, "task_family_id", np.int64),
    }
    arrays["center_sample"] = (
        read_vector(h5, "center_sample", np.int64)
        if "center_sample" in h5
        else ((arrays["sample_start"] + arrays["sample_end"] - 1) // 2).astype(np.int64)
    )
    for name, arr in arrays.items():
        if len(arr) != n:
            raise ValueError(f"Length mismatch for {name}: {len(arr)} vs embedding {n}")

    attrs = {str(k): decode_scalar(v) for k, v in h5.attrs.items()}
    sr = float(attrs.get("sampling_rate", attrs.get("target_sfreq", 200.0)))
    unique_starts = np.unique(arrays["sample_start"])
    if unique_starts.size > 1:
        stride = float(attrs.get("stride_seconds", np.median(np.diff(unique_starts)) / sr))
    else:
        stride = float(attrs.get("stride_seconds", 1.0))
    win = float(attrs.get(
        "window_seconds", np.median(arrays["sample_end"] - arrays["sample_start"]) / sr
    ))
    return Metadata(
        arrays["subject"], arrays["session"], arrays["run"],
        arrays["sample_start"], arrays["sample_end"], arrays["center_sample"],
        arrays["phase"], arrays["task"], arrays["family"], sr, stride, win,
        read_name_map(h5, "task_names"), read_name_map(h5, "task_family_names"), attrs,
    )


def vector_dim(ds: h5py.Dataset, mode: str) -> int:
    shape = tuple(int(x) for x in ds.shape)
    if len(shape) < 2:
        raise ValueError(f"Embedding must have shape (N, ...), got {shape}")
    if mode == "flatten" or len(shape) == 2:
        return int(np.prod(shape[1:]))
    if mode == "mean_structural":
        return int(shape[-1])
    raise ValueError(f"Unknown vectorization mode {mode}")


def vectorize_chunk(chunk: np.ndarray, mode: str) -> np.ndarray:
    x = np.asarray(chunk)
    if mode == "flatten" or x.ndim == 2:
        out = x.reshape(x.shape[0], -1)
    elif mode == "mean_structural":
        axes = tuple(range(1, x.ndim - 1))
        out = np.mean(x, axis=axes) if axes else x.reshape(x.shape[0], -1)
    else:
        raise ValueError(f"Unknown vectorization mode {mode}")
    return np.asarray(out, dtype=np.float64).reshape(out.shape[0], -1)


def read_rows(ds: h5py.Dataset, indices: np.ndarray, mode: str, batch: int) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.int64)
    if idx.size == 0:
        return np.empty((0, vector_dim(ds, mode)), dtype=np.float64)
    order = np.argsort(idx, kind="mergesort")
    sorted_idx = idx[order]
    buffer = np.empty((len(idx), vector_dim(ds, mode)), dtype=np.float64)
    for start in range(0, len(sorted_idx), batch):
        sl = sorted_idx[start:start + batch]
        buffer[start:start + len(sl)] = vectorize_chunk(ds[sl], mode)
    out = np.empty_like(buffer)
    out[order] = buffer
    return out


# =============================================================================
# Segment and state-group construction
# =============================================================================


@dataclass
class Segment:
    subject: int
    session: int
    run: int
    kind: str
    task_id: int
    state_label: int
    state_name: str
    indices: np.ndarray
    onset_local_sec: float
    segment_start_sec: float
    segment_end_sec: float


@dataclass
class GroupInfo:
    subject: int
    session: int
    label: int
    state_name: str
    segments: List[Segment]


@dataclass
class GroupData:
    subject: int
    session: int
    label: int
    state_name: str
    X: np.ndarray
    weights: np.ndarray
    source_runs: List[int]
    n_source_segments: int


@dataclass
class ProjectedGroup:
    subject: int
    session: int
    label: int
    state_name: str
    Z: np.ndarray
    weights: np.ndarray
    source_runs: List[int]


def infer_onset_local_sec(meta: Metadata, idx_sorted: np.ndarray, run_start: int) -> float:
    centers = (meta.center_sample[idx_sorted] - run_start) / meta.sampling_rate
    phase = meta.phase[idx_sorted]
    baseline = centers[phase == 0]
    task = centers[phase == 1]
    if baseline.size == 0 or task.size == 0:
        return float("nan")
    last_baseline = float(np.max(baseline))
    first_task = float(np.min(task))
    if first_task <= last_baseline:
        return float("nan")
    return 0.5 * (last_baseline + first_task)


def build_selected_segments(
    meta: Metadata,
    subject: int,
    task_ids: Mapping[str, int],
    args: argparse.Namespace,
) -> Tuple[List[Segment], List[Dict[str, Any]]]:
    """Use the full stable baseline and task context for the four selected runs.

    No arbitrary 15-second truncation is applied. Baseline starts after the run-start
    guard and ends before onset; task starts after the onset guard and ends before the
    run-end guard. All retained windows remain available to SVD and covariance
    estimation, while downstream hierarchical weights prevent long task contexts from
    dominating the five-state analysis.
    """
    selected_id_to_name = {int(v): k for k, v in task_ids.items()}
    state_label_by_task = {int(task_ids[name]): STATE_NAMES.index(name) for name in TASK_STATE_NAMES}
    rows_by_run: Dict[Tuple[int, int], List[int]] = {}
    subject_rows = np.flatnonzero(meta.subject == subject)
    for row in subject_rows:
        rows_by_run.setdefault((int(meta.session[row]), int(meta.run[row])), []).append(int(row))

    segments: List[Segment] = []
    exclusions: List[Dict[str, Any]] = []
    for (session, run), rows in sorted(rows_by_run.items()):
        idx = np.asarray(rows, dtype=np.int64)
        idx = idx[np.argsort(meta.center_sample[idx], kind="mergesort")]
        task_rows = idx[meta.phase[idx] == 1]
        if task_rows.size == 0:
            continue
        task_id = int(np.bincount(meta.task[task_rows]).argmax())
        if task_id not in selected_id_to_name:
            continue
        run_start_sample = int(np.min(meta.sample_start[idx]))
        onset = infer_onset_local_sec(meta, idx, run_start_sample)
        if not np.isfinite(onset):
            exclusions.append({"subject": subject, "session": session, "run": run,
                               "task_id": task_id, "reason": "onset_not_inferable"})
            continue
        centers = (meta.center_sample[idx] - run_start_sample) / meta.sampling_rate
        run_end = float(np.max((meta.sample_end[idx] - run_start_sample) / meta.sampling_rate))
        baseline_start = float(args.run_start_guard_sec)
        baseline_end = float(onset - args.baseline_guard_sec)
        task_start = float(onset + args.task_guard_sec)
        task_end = float(run_end - args.task_end_guard_sec)
        baseline_sel = idx[(centers >= baseline_start) & (centers < baseline_end) & (meta.phase[idx] == 0)]
        task_sel = idx[(centers >= task_start) & (centers < task_end) & (meta.phase[idx] == 1)]
        if baseline_sel.size < args.min_windows_per_segment:
            exclusions.append({"subject": subject, "session": session, "run": run,
                               "task_id": task_id, "reason": "baseline_too_short",
                               "n_windows": int(baseline_sel.size)})
            continue
        if task_sel.size < args.min_windows_per_segment:
            exclusions.append({"subject": subject, "session": session, "run": run,
                               "task_id": task_id, "reason": "task_too_short",
                               "n_windows": int(task_sel.size)})
            continue
        state_name = selected_id_to_name[task_id]
        task_label = state_label_by_task[task_id]
        segments.append(Segment(subject, session, run, "baseline", task_id, 0, "Baseline",
                                baseline_sel, onset, baseline_start, baseline_end))
        segments.append(Segment(subject, session, run, "task", task_id, task_label, state_name,
                                task_sel, onset, task_start, task_end))
    return segments, exclusions


def make_group_infos(
    segments: Sequence[Segment],
    subject: int,
    task_ids: Mapping[str, int],
) -> Tuple[List[GroupInfo], List[Dict[str, Any]]]:
    """One group per state per session; baseline pools four source runs equally."""
    sessions = sorted({segment.session for segment in segments})
    groups: List[GroupInfo] = []
    qc: List[Dict[str, Any]] = []
    for session in sessions:
        session_segments = [x for x in segments if x.session == session]
        baseline_segments = [x for x in session_segments if x.kind == "baseline"]
        expected_task_ids = {int(v) for v in task_ids.values()}
        baseline_task_ids = {x.task_id for x in baseline_segments}
        if baseline_task_ids != expected_task_ids:
            qc.append({
                "subject": subject, "session": session, "state": "Baseline",
                "status": "missing_selected_baseline_source",
                "found_task_ids": sorted(baseline_task_ids),
                "expected_task_ids": sorted(expected_task_ids),
            })
            continue
        groups.append(GroupInfo(subject, session, 0, "Baseline", baseline_segments))

        complete = True
        for state_name in TASK_STATE_NAMES:
            task_id = int(task_ids[state_name])
            matched = [
                x for x in session_segments
                if x.kind == "task" and x.task_id == task_id and x.state_name == state_name
            ]
            if not matched:
                qc.append({
                    "subject": subject, "session": session, "state": state_name,
                    "status": "missing_task_group", "task_id": task_id,
                })
                complete = False
                break
            groups.append(GroupInfo(
                subject, session, STATE_NAMES.index(state_name), state_name, matched,
            ))
        if not complete:
            groups = [g for g in groups if g.session != session]

    return groups, qc


def load_group_data(
    ds: h5py.Dataset,
    infos: Sequence[GroupInfo],
    mode: str,
    batch: int,
) -> List[GroupData]:
    out: List[GroupData] = []
    for info in infos:
        blocks: List[np.ndarray] = []
        weight_blocks: List[np.ndarray] = []
        n_segments = len(info.segments)
        for segment in info.segments:
            X = read_rows(ds, segment.indices, mode, batch)
            if X.shape[0] == 0:
                continue
            blocks.append(X)
            # Each source segment/run contributes equally inside the state-session group.
            weight_blocks.append(np.full(X.shape[0], 1.0 / (n_segments * X.shape[0])))
        if not blocks:
            continue
        X_all = np.concatenate(blocks, axis=0)
        weights = np.concatenate(weight_blocks)
        weights /= weights.sum()
        out.append(GroupData(
            info.subject, info.session, info.label, info.state_name,
            X_all, weights, [x.run for x in info.segments], n_segments,
        ))
    return out


def validate_complete_groups(groups: Sequence[GroupData]) -> Tuple[bool, Dict[str, Any]]:
    sessions = sorted({g.session for g in groups})
    counts: Dict[int, Dict[int, int]] = {}
    ok = len(sessions) >= 3
    for session in sessions:
        counts[int(session)] = {
            label: sum(g.session == session and g.label == label for g in groups)
            for label in range(len(STATE_NAMES))
        }
        if any(counts[int(session)][label] != 1 for label in range(len(STATE_NAMES))):
            ok = False
    return ok, {"sessions": sessions, "counts_by_session": counts}


# =============================================================================
# Weighted representation and LDA
# =============================================================================


def weighted_mean_rows(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    return np.sum(X * w[:, None], axis=0)


def fit_centered_svd_subspace(
    groups: Sequence[GroupData], rank: int, seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Training-only balanced centering + SVD with strict rank diagnostics."""
    if not groups:
        raise ValueError("No groups supplied to fit_centered_svd_subspace")
    n_groups = len(groups)
    X = np.concatenate([g.X for g in groups], axis=0)
    weights = np.concatenate([g.weights / n_groups for g in groups])
    weights /= weights.sum()
    mu = weighted_mean_rows(X, weights)
    Xw = (X - mu) * np.sqrt(weights[:, None])
    n, d = Xw.shape
    algebraic_max = int(min(max(n - 1, 0), d))
    if rank > algebraic_max:
        raise ValueError(
            f"requested SVD rank {rank} exceeds algebraic maximum {algebraic_max} "
            f"(n_train_windows={n}, flatten_dim={d})"
        )
    r = int(rank)
    total_energy = float(np.sum(Xw.astype(np.float64) ** 2))
    if d <= 4 * r or d <= 512:
        _, singular_all, vt_all = np.linalg.svd(Xw, full_matrices=False)
        singular = singular_all[:r]
        vt = vt_all[:r]
    else:
        rng = np.random.default_rng(seed)
        oversample = min(32, max(d - r, 0))
        omega = rng.standard_normal((d, r + oversample))
        Y = Xw @ omega
        for _ in range(3):
            Q, _ = np.linalg.qr(Y, mode="reduced")
            Y = Xw @ (Xw.T @ Q)
        Q, _ = np.linalg.qr(Y, mode="reduced")
        B = Q.T @ Xw
        _, singular_all, vt_all = np.linalg.svd(B, full_matrices=False)
        singular = singular_all[:r]
        vt = vt_all[:r]
    tol = float(max(n, d) * np.finfo(float).eps * max(float(singular[0]), 1e-30))
    numerical_rank = int(np.sum(singular > tol))
    diagnostics = {
        "flatten_dimension": int(d),
        "n_training_windows": int(n),
        "maximum_algebraic_rank": algebraic_max,
        "requested_svd_rank": int(rank),
        "actual_svd_rank": int(len(singular)),
        "numerical_rank_within_retained": numerical_rank,
        "captured_singular_value_fraction": float(np.sum(singular ** 2) / max(total_energy, 1e-30)),
        "largest_retained_singular_value": float(singular[0]),
        "smallest_retained_singular_value": float(singular[-1]),
        "retained_condition_number": float(singular[0] / max(singular[-1], 1e-30)),
        "numerical_rank_tolerance": tol,
    }
    return mu, vt.T, singular, diagnostics

def project_groups(groups: Sequence[GroupData], mu: np.ndarray, basis: np.ndarray) -> List[ProjectedGroup]:
    return [ProjectedGroup(
        g.subject, g.session, g.label, g.state_name,
        (g.X - mu) @ basis, g.weights.copy(), list(g.source_runs),
    ) for g in groups]


def group_mean(group: ProjectedGroup) -> np.ndarray:
    return weighted_mean_rows(group.Z, group.weights)


@dataclass
class CanonicalLDA:
    classes: np.ndarray
    shrinkage: float
    class_means_z: np.ndarray
    class_means_white: np.ndarray
    within_covariance: np.ndarray
    within_session_covariance: np.ndarray
    between_session_covariance: np.ndarray
    shrunk_covariance: np.ndarray
    cholesky: np.ndarray
    inv_cholesky: np.ndarray
    white_axes: np.ndarray
    z_axes: np.ndarray
    eigenvalues: np.ndarray
    class_centroids_ld: np.ndarray
    axis_anchor_pairs: List[str]

def compute_class_moments(
    groups: Sequence[ProjectedGroup], classes: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Equal-class moments with an explicit session decomposition.

    The total within-class covariance is decomposed exactly into
    within-session window covariance plus between-session displacement of the
    state centroid. The latter is retained deliberately because the target is
    cross-session generalization.
    """
    classes = [int(x) for x in classes]
    r = groups[0].Z.shape[1]
    means = np.zeros((len(classes), r), dtype=float)
    Sw_within = np.zeros((r, r), dtype=float)
    Sw_session = np.zeros((r, r), dtype=float)

    for ci, label in enumerate(classes):
        class_groups = [g for g in groups if g.label == label]
        if not class_groups:
            raise ValueError(f"Class {label} has no groups")
        session_means = np.stack([group_mean(g) for g in class_groups], axis=0)
        class_mean = session_means.mean(axis=0)
        means[ci] = class_mean

        within_c = np.zeros((r, r), dtype=float)
        session_c = np.zeros((r, r), dtype=float)
        for g, session_mean in zip(class_groups, session_means):
            D = g.Z - session_mean
            within_c += (D * g.weights[:, None]).T @ D
            delta = (session_mean - class_mean)[:, None]
            session_c += delta @ delta.T
        within_c /= len(class_groups)
        session_c /= len(class_groups)
        Sw_within += within_c
        Sw_session += session_c

    Sw_within /= len(classes)
    Sw_session /= len(classes)
    Sw = Sw_within + Sw_session

    grand = means.mean(axis=0)
    Sb = np.zeros((r, r), dtype=float)
    for mean in means:
        delta = (mean - grand)[:, None]
        Sb += delta @ delta.T
    Sb /= len(classes)
    return means, Sw, Sb, Sw_within, Sw_session

def fit_canonical_lda(
    groups: Sequence[ProjectedGroup],
    classes: Sequence[int],
    shrinkage: float,
) -> CanonicalLDA:
    means, Sw, Sb, Sw_within, Sw_session = compute_class_moments(groups, classes)
    r = Sw.shape[0]
    scale = float(np.trace(Sw) / max(r, 1))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    target = scale * np.eye(r)
    Sw_s = (1.0 - shrinkage) * Sw + shrinkage * target
    Sw_s += 1e-10 * scale * np.eye(r)

    try:
        L = np.linalg.cholesky(Sw_s)
    except np.linalg.LinAlgError:
        Sw_s += 1e-6 * scale * np.eye(r)
        L = np.linalg.cholesky(Sw_s)
    Li = np.linalg.inv(L)
    white_between = Li @ Sb @ Li.T
    white_between = 0.5 * (white_between + white_between.T)
    eigvals, eigvecs = np.linalg.eigh(white_between)
    order = np.argsort(eigvals)[::-1]
    k = min(len(classes) - 1, r)
    eigvals = np.maximum(eigvals[order[:k]], 0.0)
    U = eigvecs[:, order[:k]]

    means_white = means @ Li.T
    centroids = means_white @ U
    anchors: List[str] = []
    for axis in range(U.shape[1]):
        pair_diffs = np.asarray([
            centroids[j, axis] - centroids[i, axis] for i, j in PAIR_SPECS
        ])
        anchor_idx = int(np.argmax(np.abs(pair_diffs)))
        i, j = PAIR_SPECS[anchor_idx]
        if pair_diffs[anchor_idx] < 0:
            U[:, axis] *= -1
            centroids[:, axis] *= -1
        anchors.append(pair_name(i, j))

    Wz = Li.T @ U
    return CanonicalLDA(
        np.asarray(classes, dtype=np.int64), float(shrinkage), means, means_white,
        Sw, Sw_within, Sw_session, Sw_s, L, Li, U, Wz, eigvals, centroids, anchors,
    )

def whiten_group(group: ProjectedGroup, fit: CanonicalLDA) -> np.ndarray:
    return group.Z @ fit.inv_cholesky.T


def margin_scores(projected: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    distances = np.sum((projected[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    out = np.empty_like(distances)
    for c in range(distances.shape[1]):
        other = np.delete(distances, c, axis=1)
        out[:, c] = np.min(other, axis=1) - distances[:, c]
    return out


def flatten_groups(
    groups: Sequence[ProjectedGroup],
    fit: CanonicalLDA,
    n_dims: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    ws: List[np.ndarray] = []
    sessions: List[np.ndarray] = []
    axes = fit.white_axes[:, :n_dims]
    for group in groups:
        Y = whiten_group(group, fit) @ axes
        Xs.append(Y)
        ys.append(np.full(Y.shape[0], group.label, dtype=np.int64))
        # Every state-session group receives equal total weight.
        ws.append(group.weights / len(groups))
        sessions.append(np.full(Y.shape[0], group.session, dtype=np.int64))
    return (
        np.concatenate(Xs, axis=0), np.concatenate(ys),
        np.concatenate(ws), np.concatenate(sessions),
    )


# =============================================================================
# Metrics and evaluation
# =============================================================================


def weighted_balanced_accuracy(
    y: np.ndarray, pred: np.ndarray, weights: np.ndarray, classes: Sequence[int],
) -> float:
    recalls = []
    for label in classes:
        mask = y == int(label)
        if not np.any(mask):
            continue
        denom = float(np.sum(weights[mask]))
        recalls.append(float(np.sum(weights[mask] * (pred[mask] == int(label))) / denom))
    return float(np.mean(recalls)) if recalls else float("nan")


def confusion_matrix(y: np.ndarray, pred: np.ndarray, classes: Sequence[int]) -> np.ndarray:
    pos = {int(label): i for i, label in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for true, guessed in zip(y, pred):
        if int(true) in pos and int(guessed) in pos:
            cm[pos[int(true)], pos[int(guessed)]] += 1
    return cm


def rank_auc(scores: np.ndarray, positive: np.ndarray) -> float:
    positive = np.asarray(positive, dtype=bool)
    n_pos = int(np.sum(positive))
    n_neg = int(np.sum(~positive))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = np.asarray(scores)[order]
    ranks_sorted = np.arange(1, len(scores) + 1, dtype=float)
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks_sorted[start:end] = np.mean(ranks_sorted[start:end])
        start = end
    ranks = np.empty_like(ranks_sorted)
    ranks[order] = ranks_sorted
    return float((np.sum(ranks[positive]) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def macro_auc(y: np.ndarray, scores: np.ndarray, classes: Sequence[int]) -> float:
    aucs = []
    for col, label in enumerate(classes):
        value = rank_auc(scores[:, col], y == int(label))
        if np.isfinite(value):
            aucs.append(value)
    return float(np.mean(aucs)) if aucs else float("nan")


def evaluate_multiclass(
    fit: CanonicalLDA,
    test_groups: Sequence[ProjectedGroup],
    n_dims: int,
) -> Dict[str, Any]:
    n_dims = int(min(n_dims, fit.white_axes.shape[1]))
    projected, y, weights, _ = flatten_groups(test_groups, fit, n_dims)
    centroids = fit.class_centroids_ld[:, :n_dims]
    scores = margin_scores(projected, centroids)
    pred = fit.classes[np.argmax(scores, axis=1)]
    window_bacc = weighted_balanced_accuracy(y, pred, weights, fit.classes)
    auc = macro_auc(y, scores, fit.classes)
    cm = confusion_matrix(y, pred, fit.classes)

    recalls: Dict[int, float] = {}
    for label in fit.classes:
        mask = y == int(label)
        denom = float(np.sum(weights[mask]))
        recalls[int(label)] = (
            float(np.sum(weights[mask] * (pred[mask] == int(label))) / denom)
            if denom > 0 else float("nan")
        )

    group_true: List[int] = []
    group_pred: List[int] = []
    group_rows: List[Dict[str, Any]] = []
    axes = fit.white_axes[:, :n_dims]
    for group in test_groups:
        mean_score = weighted_mean_rows(whiten_group(group, fit) @ axes, group.weights)
        distances = np.sum((centroids - mean_score[None, :]) ** 2, axis=1)
        guess = int(fit.classes[int(np.argmin(distances))])
        group_true.append(group.label)
        group_pred.append(guess)
        group_rows.append({
            "session": group.session,
            "true_label": group.label,
            "true_state": group.state_name,
            "pred_label": guess,
            "pred_state": STATE_NAMES[guess],
            "correct": int(guess == group.label),
        })
    centroid_accuracy = float(np.mean(np.asarray(group_true) == np.asarray(group_pred)))
    return {
        "n_dims": n_dims,
        "window_balanced_accuracy": window_bacc,
        "window_macro_auc": auc,
        "window_confusion_matrix": cm.tolist(),
        "state_centroid_accuracy": centroid_accuracy,
        "baseline_recall": recalls.get(0, float("nan")),
        "mean_task_recall": safe_mean(recalls.get(k, np.nan) for k in range(1, 5)),
        "class_recalls": {STATE_NAMES[k]: recalls.get(k, float("nan")) for k in range(5)},
        "group_predictions": group_rows,
    }

def build_pair_geometry(fit: CanonicalLDA) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    columns: List[np.ndarray] = []
    unit_columns: List[np.ndarray] = []
    names: List[str] = []
    for i, j in PAIR_SPECS:
        delta = fit.class_means_white[j] - fit.class_means_white[i]
        norm = float(np.linalg.norm(delta))
        unit = delta / norm if norm > 1e-12 else np.zeros_like(delta)
        columns.append(delta)
        unit_columns.append(unit)
        names.append(pair_name(i, j))
    return np.column_stack(columns), np.column_stack(unit_columns), names


def evaluate_pairwise(
    fit: CanonicalLDA,
    test_groups: Sequence[ProjectedGroup],
) -> List[Dict[str, Any]]:
    A, A_unit, names = build_pair_geometry(fit)
    test_by_label = {g.label: g for g in test_groups}
    rows: List[Dict[str, Any]] = []
    for p, (i, j) in enumerate(PAIR_SPECS):
        axis = A_unit[:, p]
        norm = float(np.linalg.norm(A[:, p]))
        if norm <= 1e-12 or i not in test_by_label or j not in test_by_label:
            rows.append({
                "pair_index": p, "pair_name": names[p], "state_i": STATE_NAMES[i],
                "state_j": STATE_NAMES[j], "valid": False,
            })
            continue
        train_i = float(fit.class_means_white[i] @ axis)
        train_j = float(fit.class_means_white[j] @ axis)
        if train_j < train_i:
            axis = -axis
            train_i, train_j = -train_i, -train_j
        threshold = 0.5 * (train_i + train_j)

        y_parts: List[np.ndarray] = []
        score_parts: List[np.ndarray] = []
        weight_parts: List[np.ndarray] = []
        centroid_scores: Dict[int, float] = {}
        test_delta_means: Dict[int, np.ndarray] = {}
        for label in (i, j):
            group = test_by_label[label]
            Ywhite = whiten_group(group, fit)
            score = Ywhite @ axis
            y_parts.append(np.full(score.shape[0], label, dtype=np.int64))
            score_parts.append(score)
            weight_parts.append(group.weights / 2.0)
            centroid_scores[label] = float(np.sum(score * group.weights))
            test_delta_means[label] = weighted_mean_rows(Ywhite, group.weights)
        y = np.concatenate(y_parts)
        score = np.concatenate(score_parts)
        weights = np.concatenate(weight_parts)
        pred = np.where(score >= threshold, j, i)
        bacc = weighted_balanced_accuracy(y, pred, weights, [i, j])
        auc = rank_auc(score, y == j)
        centroid_correct = float(
            (centroid_scores[i] < threshold) and (centroid_scores[j] >= threshold)
        )
        train_delta = fit.class_means_white[j] - fit.class_means_white[i]
        test_delta = test_delta_means[j] - test_delta_means[i]
        cross_distance = float(train_delta @ test_delta)
        cosine = float(
            train_delta @ test_delta
            / max(np.linalg.norm(train_delta) * np.linalg.norm(test_delta), 1e-12)
        )
        rows.append({
            "pair_index": p,
            "pair_name": names[p],
            "state_i": STATE_NAMES[i],
            "state_j": STATE_NAMES[j],
            "valid": True,
            "train_center_i": train_i,
            "train_center_j": train_j,
            "threshold": threshold,
            "window_balanced_accuracy": bacc,
            "window_auc": auc,
            "state_centroid_pair_accuracy": centroid_correct,
            "train_test_cross_distance": cross_distance,
            "train_test_delta_cosine": cosine,
        })
    return rows


def semantic_decomposition(
    fit: CanonicalLDA,
    degeneracy_gap: float,
) -> Dict[str, Any]:
    A, A_unit, names = build_pair_geometry(fit)
    U = fit.white_axes
    signed_cosine = A_unit.T @ U
    absolute_cosine = np.abs(signed_cosine)
    squared_alignment = signed_cosine ** 2
    coefficients = np.linalg.pinv(A, rcond=1e-10) @ U
    reconstruction = A @ coefficients
    reconstruction_error = np.linalg.norm(U - reconstruction, axis=0)

    pair_u, singular, pair_vt = np.linalg.svd(A, full_matrices=False)
    k = min(U.shape[1], pair_u.shape[1])
    pair_u = pair_u[:, :k]
    singular = singular[:k]
    pair_loadings = pair_vt[:k].T

    overlap_singular = np.linalg.svd(U[:, :k].T @ pair_u[:, :k], compute_uv=False)
    overlap_singular = np.clip(overlap_singular, -1.0, 1.0)
    principal_angles_deg = np.degrees(np.arccos(overlap_singular))

    blocks: List[List[int]] = []
    if len(fit.eigenvalues):
        start = 0
        for idx in range(len(fit.eigenvalues) - 1):
            value = float(fit.eigenvalues[idx])
            next_value = float(fit.eigenvalues[idx + 1])
            rel_gap = (value - next_value) / max(value, 1e-12)
            if rel_gap >= degeneracy_gap:
                blocks.append(list(range(start, idx + 1)))
                start = idx + 1
        blocks.append(list(range(start, len(fit.eigenvalues))))
    block_names = [
        "LD" + (str(block[0] + 1) if len(block) == 1 else f"{block[0] + 1}-{block[-1] + 1}")
        for block in blocks
    ]
    block_energy = np.column_stack([
        np.sum(squared_alignment[:, block], axis=1) for block in blocks
    ]) if blocks else np.empty((len(names), 0))

    return {
        "pair_names": names,
        "pair_matrix": A,
        "pair_unit_matrix": A_unit,
        "signed_cosine_alignment": signed_cosine,
        "absolute_cosine_alignment": absolute_cosine,
        "squared_alignment": squared_alignment,
        "minimum_norm_coefficients": coefficients,
        "coefficient_reconstruction_error": reconstruction_error,
        "pair_svd_singular_values": singular,
        "pair_svd_left_axes": pair_u,
        "pair_svd_right_loadings": pair_loadings,
        "principal_angles_deg": principal_angles_deg,
        "axis_anchor_pairs": list(fit.axis_anchor_pairs),
        "degenerate_blocks": blocks,
        "block_names": block_names,
        "block_pair_energy": block_energy,
    }

# =============================================================================
# Shrinkage selection and structured permutation null
# =============================================================================


def evaluate_shrinkage_sensitivity(
    train_groups: Sequence[ProjectedGroup],
    test_groups: Sequence[ProjectedGroup],
    classes: Sequence[int],
    shrinkages: Sequence[float],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for lam in shrinkages:
        fit = fit_canonical_lda(train_groups, classes, float(lam))
        for d in range(1, fit.white_axes.shape[1] + 1):
            metrics = evaluate_multiclass(fit, test_groups, d)
            rows.append({
                "shrinkage": float(lam),
                "n_dims": d,
                "window_balanced_accuracy": metrics["window_balanced_accuracy"],
                "window_macro_auc": metrics["window_macro_auc"],
                "state_centroid_accuracy": metrics["state_centroid_accuracy"],
                "baseline_recall": metrics["baseline_recall"],
                "mean_task_recall": metrics["mean_task_recall"],
                "spectral_participation_ratio": participation_ratio(fit.eigenvalues),
            })
    return rows

def permute_labels_by_session(
    groups: Sequence[ProjectedGroup], rng: np.random.Generator,
) -> List[ProjectedGroup]:
    out: List[ProjectedGroup] = []
    for session in sorted({g.session for g in groups}):
        session_groups = sorted([g for g in groups if g.session == session], key=lambda x: x.label)
        if len(session_groups) != len(STATE_NAMES):
            raise ValueError("Structured permutation expects exactly one group per state per session")
        permuted_labels = rng.permutation(np.arange(len(STATE_NAMES)))
        for group, new_label in zip(session_groups, permuted_labels):
            out.append(replace(
                group,
                label=int(new_label),
                state_name=STATE_NAMES[int(new_label)],
            ))
    return out


def run_permutation_null(
    all_projected: Sequence[ProjectedGroup],
    heldout_session: int,
    classes: Sequence[int],
    shrinkage: float,
    n_permutations: int,
    seed: int,
) -> Dict[str, Any]:
    max_dims = min(len(classes) - 1, all_projected[0].Z.shape[1])
    null_bacc = np.full((n_permutations, max_dims), np.nan)
    null_centroid_acc = np.full((n_permutations, max_dims), np.nan)
    null_eigen = np.full((n_permutations, max_dims), np.nan)
    null_pair_bacc = np.full((n_permutations, len(PAIR_SPECS)), np.nan)
    null_pair_cross = np.full((n_permutations, len(PAIR_SPECS)), np.nan)
    rng = np.random.default_rng(seed)
    started = time.perf_counter()

    for perm_idx in range(n_permutations):
        permuted = permute_labels_by_session(all_projected, rng)
        train = [g for g in permuted if g.session != heldout_session]
        test = [g for g in permuted if g.session == heldout_session]
        fit = fit_canonical_lda(train, classes, shrinkage)
        null_eigen[perm_idx, :len(fit.eigenvalues)] = fit.eigenvalues
        for d in range(1, max_dims + 1):
            metrics = evaluate_multiclass(fit, test, d)
            null_bacc[perm_idx, d - 1] = metrics["window_balanced_accuracy"]
            null_centroid_acc[perm_idx, d - 1] = metrics["state_centroid_accuracy"]
        pair_rows = evaluate_pairwise(fit, test)
        for row in pair_rows:
            if row.get("valid"):
                p = int(row["pair_index"])
                null_pair_bacc[perm_idx, p] = row["window_balanced_accuracy"]
                null_pair_cross[perm_idx, p] = row["train_test_cross_distance"]

    elapsed = time.perf_counter() - started
    return {
        "window_bacc": null_bacc,
        "centroid_accuracy": null_centroid_acc,
        "eigenvalues": null_eigen,
        "pair_bacc": null_pair_bacc,
        "pair_cross_distance": null_pair_cross,
        "elapsed_seconds": elapsed,
        "seconds_per_permutation": elapsed / max(n_permutations, 1),
    }

def covariance_diagnostics(fit: CanonicalLDA) -> Dict[str, Any]:
    total = 0.5 * (fit.within_covariance + fit.within_covariance.T)
    within = 0.5 * (fit.within_session_covariance + fit.within_session_covariance.T)
    session = 0.5 * (fit.between_session_covariance + fit.between_session_covariance.T)
    eig_total = np.linalg.eigvalsh(total)[::-1]
    eig_within = np.linalg.eigvalsh(within)[::-1]
    eig_session, vec_session = np.linalg.eigh(session)
    order = np.argsort(eig_session)[::-1]
    eig_session = eig_session[order]
    vec_session = vec_session[:, order]

    session_white = fit.inv_cholesky @ session @ fit.inv_cholesky.T
    session_white = 0.5 * (session_white + session_white.T)
    eig_sw, vec_sw = np.linalg.eigh(session_white)
    order_sw = np.argsort(eig_sw)[::-1]
    eig_sw = eig_sw[order_sw]
    vec_sw = vec_sw[:, order_sw]
    alignment = (fit.white_axes.T @ vec_sw) ** 2

    trace_total = float(np.trace(total))
    trace_within = float(np.trace(within))
    trace_session = float(np.trace(session))
    return {
        "total_eigenvalues": eig_total,
        "within_session_eigenvalues": eig_within,
        "between_session_eigenvalues": eig_session,
        "between_session_white_eigenvalues": eig_sw,
        "ld_to_session_mode_squared_alignment": alignment,
        "trace_total": trace_total,
        "trace_within_session": trace_within,
        "trace_between_session": trace_session,
        "between_session_trace_fraction": trace_session / max(trace_total, 1e-12),
        "total_numerical_rank": numerical_rank_psd(total),
        "within_session_numerical_rank": numerical_rank_psd(within),
        "between_session_numerical_rank": numerical_rank_psd(session),
    }


# =============================================================================
# Plotting and serialization helpers
# =============================================================================


def plot_heatmap(
    matrix: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    path: Path,
    title: str,
    fmt: str = ".2f",
) -> None:
    matrix = np.asarray(matrix, dtype=float)
    fig_w = max(5.5, 0.9 * len(col_labels) + 2.5)
    fig_h = max(4.0, 0.42 * len(row_labels) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    image = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title, fontsize=10)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    if matrix.size <= 80:
        finite = matrix[np.isfinite(matrix)]
        midpoint = float(np.nanmedian(finite)) if finite.size else 0.0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                if np.isfinite(value):
                    ax.text(
                        j, i, format(value, fmt), ha="center", va="center",
                        fontsize=7, color="white" if value < midpoint else "black",
                    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_multiclass_dims(rows: Sequence[Mapping[str, Any]], path: Path, title: str) -> None:
    x = [int(r["n_dims"]) for r in rows]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(x, [r["window_balanced_accuracy"] for r in rows], "o-", label="window bACC")
    ax.plot(x, [r["state_centroid_accuracy"] for r in rows], "s--", label="state-centroid accuracy")
    ax.axhline(0.2, linestyle=":", linewidth=1, label="chance")
    ax.set_xticks(x)
    ax.set_xlabel("Number of multiclass LD dimensions")
    ax.set_ylabel("Held-out performance")
    ax.set_ylim(0, 1.02)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_eigenvalues(eigenvalues: np.ndarray, null: Optional[np.ndarray], path: Path, title: str) -> None:
    values = np.asarray(eigenvalues, dtype=float)
    x = np.arange(1, len(values) + 1)
    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    ax.plot(x, values, "o-", label="real")
    if null is not None and np.size(null):
        null = np.asarray(null, dtype=float)
        median = np.nanmedian(null, axis=0)
        p95 = np.nanpercentile(null, 95, axis=0)
        ax.plot(x, median[:len(x)], "s--", label="null median")
        ax.plot(x, p95[:len(x)], "^:", label="null 95%")
    ax.set_xticks(x)
    ax.set_xlabel("Multiclass LD index")
    ax.set_ylabel("Generalized eigenvalue")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_fold_scatter(
    fit: CanonicalLDA,
    test_groups: Sequence[ProjectedGroup],
    path: Path,
    title: str,
) -> None:
    if fit.white_axes.shape[1] < 2:
        return
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    for group in sorted(test_groups, key=lambda x: x.label):
        projected = whiten_group(group, fit) @ fit.white_axes[:, :2]
        ax.scatter(projected[:, 0], projected[:, 1], s=11, alpha=0.45, label=group.state_name)
        center = weighted_mean_rows(projected, group.weights)
        ax.scatter([center[0]], [center[1]], s=80, marker="X")
        ax.text(center[0], center[1], group.state_name, fontsize=7)
    ax.set_xlabel("LD1")
    ax.set_ylabel("LD2")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=6, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def matrix_rows(
    matrix: np.ndarray,
    row_names: Sequence[str],
    col_names: Sequence[str],
    value_name: str,
) -> List[Dict[str, Any]]:
    rows = []
    for i, row_name in enumerate(row_names):
        for j, col_name in enumerate(col_names):
            rows.append({"row": row_name, "column": col_name, value_name: float(matrix[i, j])})
    return rows


# =============================================================================
# Fold, subject and model drivers
# =============================================================================


def run_fold_rank(
    groups: Sequence[GroupData],
    heldout_session: int,
    rank: int,
    args: argparse.Namespace,
    seed: int,
    null_seed: int,
    outdir: Path,
) -> Dict[str, Any]:
    train_raw = [g for g in groups if g.session != heldout_session]
    test_raw = [g for g in groups if g.session == heldout_session]
    if len(test_raw) != len(STATE_NAMES):
        raise RuntimeError(f"Held-out session {heldout_session} does not contain five complete state groups")

    mu, svd_basis, svd_singular, svd_diag = fit_centered_svd_subspace(train_raw, rank, seed)
    all_projected = project_groups(groups, mu, svd_basis)
    train = [g for g in all_projected if g.session != heldout_session]
    test = [g for g in all_projected if g.session == heldout_session]

    shrinkage = float(args.main_shrinkage)
    fit = fit_canonical_lda(train, list(range(5)), shrinkage)
    semantic = semantic_decomposition(fit, args.degeneracy_gap)
    covdiag = covariance_diagnostics(fit)

    dimension_rows: List[Dict[str, Any]] = []
    for n_dims in range(1, fit.white_axes.shape[1] + 1):
        row = evaluate_multiclass(fit, test, n_dims)
        train_eval = evaluate_multiclass(fit, train, n_dims)
        row["train_window_balanced_accuracy"] = float(train_eval["window_balanced_accuracy"])
        row["train_window_macro_auc"] = float(train_eval["window_macro_auc"])
        row["train_state_centroid_accuracy"] = float(train_eval["state_centroid_accuracy"])
        row["train_test_bacc_gap"] = float(train_eval["window_balanced_accuracy"] - row["window_balanced_accuracy"])
        class_recalls = row.pop("class_recalls")
        for state, value in class_recalls.items():
            row[f"recall_{state}"] = value
        row.update({
            "heldout_session": int(heldout_session),
            "svd_rank_requested": int(rank),
            "svd_rank_effective": int(svd_basis.shape[1]),
            "svd_captured_fraction": float(svd_diag["captured_singular_value_fraction"]),
            "shrinkage": shrinkage,
        })
        dimension_rows.append(row)

    pair_rows = evaluate_pairwise(fit, test)
    for row in pair_rows:
        row.update({
            "heldout_session": int(heldout_session),
            "svd_rank_requested": int(rank),
            "svd_rank_effective": int(svd_basis.shape[1]),
            "svd_captured_fraction": float(svd_diag["captured_singular_value_fraction"]),
            "shrinkage": shrinkage,
        })

    null = None
    if args.n_permutations > 0:
        null = run_permutation_null(
            all_projected, heldout_session, list(range(5)), shrinkage,
            args.n_permutations, null_seed,
        )
        log(
            f"      null: {args.n_permutations} permutations in "
            f"{null['elapsed_seconds']:.2f}s "
            f"({null['seconds_per_permutation']:.4f}s/perm)"
        )
        for row in dimension_rows:
            d = int(row["n_dims"]) - 1
            row["window_bacc_permutation_p_fold"] = empirical_upper_p(
                float(row["window_balanced_accuracy"]), null["window_bacc"][:, d]
            )
            row["state_centroid_accuracy_permutation_p_fold"] = empirical_upper_p(
                float(row["state_centroid_accuracy"]), null["centroid_accuracy"][:, d]
            )
        for row in pair_rows:
            if row.get("valid"):
                p = int(row["pair_index"])
                row["window_bacc_permutation_p_fold"] = empirical_upper_p(
                    float(row["window_balanced_accuracy"]), null["pair_bacc"][:, p]
                )
                row["cross_distance_permutation_p_fold"] = empirical_upper_p(
                    float(row["train_test_cross_distance"]),
                    null["pair_cross_distance"][:, p],
                )

    eigen_rows = []
    for axis, value in enumerate(fit.eigenvalues, start=1):
        next_value = fit.eigenvalues[axis] if axis < len(fit.eigenvalues) else float("nan")
        relative_gap = (
            float((value - next_value) / max(value, 1e-12))
            if np.isfinite(next_value) else float("nan")
        )
        eigen_rows.append({
            "heldout_session": int(heldout_session),
            "svd_rank_requested": int(rank),
            "ld_axis": axis,
            "axis_anchor_pair": fit.axis_anchor_pairs[axis - 1],
            "eigenvalue": float(value),
            "relative_gap_to_next": relative_gap,
            "near_degenerate_with_next": bool(
                np.isfinite(relative_gap) and relative_gap < args.degeneracy_gap
            ),
            "permutation_p_fold": empirical_upper_p(
                float(value), null["eigenvalues"][:, axis - 1]
            ) if null is not None else float("nan"),
        })

    centroid_rows = []
    test_centroids_white = np.stack([
        weighted_mean_rows(whiten_group(g, fit), g.weights)
        for g in sorted(test, key=lambda x: x.label)
    ])
    test_centroids_ld = test_centroids_white @ fit.white_axes
    for label, state in enumerate(STATE_NAMES):
        for axis in range(fit.white_axes.shape[1]):
            centroid_rows.append({
                "state_label": label,
                "state_name": state,
                "ld_axis": axis + 1,
                "axis_anchor_pair": fit.axis_anchor_pairs[axis],
                "train_centroid": float(fit.class_centroids_ld[label, axis]),
                "heldout_centroid": float(test_centroids_ld[label, axis]),
            })

    sensitivity_values = sorted(set(
        [float(args.main_shrinkage)] + list(args.sensitivity_shrinkage_grid)
    ))
    sensitivity_rows = evaluate_shrinkage_sensitivity(
        train, test, list(range(5)), sensitivity_values
    )

    fold_dir = ensure_dir(outdir / f"heldout-session-{heldout_session}" / f"svd-rank-{rank}")
    csv_write(fold_dir / "multiclass_dimension_metrics.csv", [
        {k: v for k, v in row.items() if k not in ("group_predictions", "window_confusion_matrix")}
        for row in dimension_rows
    ])
    csv_write(fold_dir / "multiclass_group_predictions.csv", [
        dict(pred, n_dims=row["n_dims"])
        for row in dimension_rows for pred in row["group_predictions"]
    ])
    csv_write(fold_dir / "pairwise_metrics.csv", pair_rows)
    csv_write(fold_dir / "eigenvalue_spectrum.csv", eigen_rows)
    csv_write(fold_dir / "class_centroid_profiles.csv", centroid_rows)
    csv_write(fold_dir / "shrinkage_sensitivity.csv", sensitivity_rows)
    csv_write(fold_dir / "svd_diagnostics.csv", [svd_diag])

    pair_names = semantic["pair_names"]
    ld_names = [f"LD{k}" for k in range(1, fit.white_axes.shape[1] + 1)]
    csv_write(fold_dir / "pair_signed_cosine_alignment.csv", matrix_rows(
        semantic["signed_cosine_alignment"], pair_names, ld_names, "signed_cosine"
    ))
    csv_write(fold_dir / "pair_absolute_cosine_alignment.csv", matrix_rows(
        semantic["absolute_cosine_alignment"], pair_names, ld_names, "absolute_cosine"
    ))
    csv_write(fold_dir / "pair_squared_alignment.csv", matrix_rows(
        semantic["squared_alignment"], pair_names, ld_names, "squared_alignment"
    ))
    csv_write(fold_dir / "pair_block_energy.csv", matrix_rows(
        semantic["block_pair_energy"], pair_names, semantic["block_names"], "block_energy"
    ))
    csv_write(fold_dir / "pair_minimum_norm_coefficients.csv", matrix_rows(
        semantic["minimum_norm_coefficients"], pair_names, ld_names, "coefficient"
    ))
    csv_write(fold_dir / "pair_svd_right_loadings.csv", matrix_rows(
        semantic["pair_svd_right_loadings"], pair_names,
        [f"PairSVD{k}" for k in range(1, semantic["pair_svd_right_loadings"].shape[1] + 1)],
        "loading",
    ))

    session_mode_names = [f"SessionMode{k}" for k in range(1, covdiag["ld_to_session_mode_squared_alignment"].shape[1] + 1)]
    csv_write(fold_dir / "ld_session_covariance_alignment.csv", matrix_rows(
        covdiag["ld_to_session_mode_squared_alignment"], ld_names, session_mode_names,
        "squared_alignment",
    ))
    csv_write(fold_dir / "covariance_diagnostics.csv", [{
        k: v for k, v in covdiag.items() if np.isscalar(v)
    }])
    covariance_spectrum_rows = []
    n_cov = max(
        len(covdiag["total_eigenvalues"]),
        len(covdiag["within_session_eigenvalues"]),
        len(covdiag["between_session_eigenvalues"]),
    )
    for idx in range(n_cov):
        covariance_spectrum_rows.append({
            "mode": idx + 1,
            "total_eigenvalue": float(covdiag["total_eigenvalues"][idx]),
            "within_session_eigenvalue": float(covdiag["within_session_eigenvalues"][idx]),
            "between_session_eigenvalue": float(covdiag["between_session_eigenvalues"][idx]),
        })
    csv_write(fold_dir / "covariance_spectrum.csv", covariance_spectrum_rows)

    np.savez_compressed(
        fold_dir / "lowdim_geometry.npz",
        centered_svd_singular_values=svd_singular,
        svd_diagnostics_json=np.array([json.dumps(decode_scalar(svd_diag))]),
        multiclass_eigenvalues=fit.eigenvalues,
        multiclass_white_axes=fit.white_axes,
        multiclass_svd_coordinates_axes=fit.z_axes,
        train_class_means_white=fit.class_means_white,
        train_class_centroids_ld=fit.class_centroids_ld,
        heldout_class_centroids_ld=test_centroids_ld,
        within_covariance_total=fit.within_covariance,
        within_session_covariance=fit.within_session_covariance,
        between_session_covariance=fit.between_session_covariance,
        covariance_total_eigenvalues=covdiag["total_eigenvalues"],
        covariance_within_session_eigenvalues=covdiag["within_session_eigenvalues"],
        covariance_between_session_eigenvalues=covdiag["between_session_eigenvalues"],
        ld_to_session_mode_squared_alignment=covdiag["ld_to_session_mode_squared_alignment"],
        pair_matrix_white=semantic["pair_matrix"],
        pair_unit_matrix_white=semantic["pair_unit_matrix"],
        pair_signed_cosine_alignment=semantic["signed_cosine_alignment"],
        pair_absolute_cosine_alignment=semantic["absolute_cosine_alignment"],
        pair_squared_alignment=semantic["squared_alignment"],
        pair_block_energy=semantic["block_pair_energy"],
        pair_minimum_norm_coefficients=semantic["minimum_norm_coefficients"],
        pair_svd_singular_values=semantic["pair_svd_singular_values"],
        pair_svd_right_loadings=semantic["pair_svd_right_loadings"],
        principal_angles_deg=semantic["principal_angles_deg"],
    )
    if null is not None:
        np.savez_compressed(
            fold_dir / "permutation_null_arrays.npz",
            window_bacc=null["window_bacc"],
            centroid_accuracy=null["centroid_accuracy"],
            eigenvalues=null["eigenvalues"],
            pair_bacc=null["pair_bacc"],
            pair_cross_distance=null["pair_cross_distance"],
        )

    plot_multiclass_dims(
        dimension_rows, fold_dir / "multiclass_performance_vs_ld_dimension.png",
        f"Held-out session {heldout_session}, centered-SVD rank {rank}",
    )
    pair_matrix = np.full((5, 5), np.nan)
    for row in pair_rows:
        if row.get("valid"):
            i = STATE_NAMES.index(row["state_i"])
            j = STATE_NAMES.index(row["state_j"])
            pair_matrix[i, j] = pair_matrix[j, i] = row["window_balanced_accuracy"]
    np.fill_diagonal(pair_matrix, 1.0)
    plot_heatmap(
        pair_matrix, STATE_SHORT, STATE_SHORT,
        fold_dir / "pairwise_bacc_matrix.png", "Held-out pairwise bACC",
    )
    plot_heatmap(
        semantic["absolute_cosine_alignment"], pair_names, ld_names,
        fold_dir / "pair_absolute_cosine_alignment.png",
        "Absolute alignment of named pairwise LDs with multiclass LDs",
    )
    if semantic["block_pair_energy"].shape[1]:
        plot_heatmap(
            semantic["block_pair_energy"], pair_names, semantic["block_names"],
            fold_dir / "pair_block_energy.png",
            "Rotation-invariant pair energy in LD blocks",
        )
    plot_heatmap(
        semantic["minimum_norm_coefficients"], pair_names, ld_names,
        fold_dir / "pair_minimum_norm_coefficients.png",
        "Minimum-norm semantic decomposition of multiclass LDs",
    )
    plot_heatmap(
        fit.class_centroids_ld, STATE_SHORT, ld_names,
        fold_dir / "train_class_centroid_profiles.png",
        "Training class-centroid profiles in multiclass LD space",
    )
    plot_eigenvalues(
        fit.eigenvalues,
        null["eigenvalues"] if null is not None else None,
        fold_dir / "multiclass_eigenvalue_spectrum.png",
        "Multiclass generalized eigenvalue spectrum",
    )
    plot_fold_scatter(
        fit, test, fold_dir / "heldout_ld1_ld2_scatter.png",
        f"Held-out session {heldout_session}: five states",
    )

    return {
        "heldout_session": int(heldout_session),
        "svd_rank_requested": int(rank),
        "svd_rank_effective": int(svd_basis.shape[1]),
        "shrinkage": shrinkage,
        "centered_svd_singular_values": svd_singular.tolist(),
        "svd_diagnostics": svd_diag,
        "multiclass_eigenvalues": fit.eigenvalues.tolist(),
        "spectral_participation_ratio_fold": participation_ratio(fit.eigenvalues),
        "axis_anchor_pairs": list(fit.axis_anchor_pairs),
        "principal_angles_deg_pair_span_vs_multiclass": semantic["principal_angles_deg"].tolist(),
        "semantic_reconstruction_error": semantic["coefficient_reconstruction_error"].tolist(),
        "covariance_diagnostics": {
            k: decode_scalar(v) for k, v in covdiag.items() if np.isscalar(v)
        },
        "dimension_metrics": dimension_rows,
        "pairwise_metrics": pair_rows,
        "eigenvalue_rows": eigen_rows,
        "centroid_profiles": centroid_rows,
        "_null_arrays": null,
    }

def summarize_effective_dimension(
    dimension_summary_rows: Sequence[Mapping[str, Any]],
    eigen_summary_rows: Sequence[Mapping[str, Any]],
    alpha: float,
) -> Dict[str, Any]:
    ordered = sorted(dimension_summary_rows, key=lambda x: int(x["n_dims"]))
    if not ordered:
        return {
            "predictive_peak_dimension": 0,
            "predictive_one_se_dimension": 0,
            "spectral_significant_leading_dimensions": 0,
            "spectral_participation_ratio": 0.0,
        }

    peak_value = max(float(row["mean_window_bacc"]) for row in ordered)
    peak_candidates = [
        int(row["n_dims"]) for row in ordered
        if abs(float(row["mean_window_bacc"]) - peak_value) <= 1e-12
    ]
    d_peak = min(peak_candidates)
    peak_row = next(row for row in ordered if int(row["n_dims"]) == d_peak)
    peak_se = float(peak_row.get("se_window_bacc", np.nan))
    threshold = peak_value - peak_se if np.isfinite(peak_se) else peak_value
    d_1se = min(
        int(row["n_dims"]) for row in ordered
        if float(row["mean_window_bacc"]) >= threshold
    )

    eig_ordered = sorted(eigen_summary_rows, key=lambda x: int(x["ld_axis"]))
    significant_leading = 0
    for row in eig_ordered:
        if float(row.get("pooled_permutation_p", np.nan)) <= alpha:
            significant_leading += 1
        else:
            break
    mean_eigen = [float(row["mean_eigenvalue"]) for row in eig_ordered]
    return {
        "predictive_peak_dimension": d_peak,
        "predictive_peak_bacc": peak_value,
        "predictive_peak_bacc_se": peak_se,
        "predictive_one_se_dimension": d_1se,
        "predictive_one_se_threshold": threshold,
        "first_dimension_significant_vs_null_uncorrected": next((
            int(row["n_dims"]) for row in ordered
            if float(row.get("pooled_window_bacc_p", np.nan)) <= alpha
        ), 0),
        "first_dimension_significant_vs_null_maxstat": next((
            int(row["n_dims"]) for row in ordered
            if float(row.get("pooled_window_bacc_maxstat_p", np.nan)) <= alpha
        ), 0),
        "spectral_significant_leading_dimensions": significant_leading,
        "spectral_participation_ratio": participation_ratio(mean_eigen),
        "interpretation_note": (
            "Do not collapse these quantities into one exact integer: predictive and "
            "spectral effective dimensionality answer complementary questions."
        ),
    }

def run_subject(
    model: str,
    ds: h5py.Dataset,
    meta: Metadata,
    subject: int,
    task_ids: Mapping[str, int],
    args: argparse.Namespace,
    model_dir: Path,
) -> Dict[str, Any]:
    subject_dir = ensure_dir(model_dir / f"sub-{subject:03d}")
    segments, exclusions = build_selected_segments(meta, subject, task_ids, args)
    infos, group_qc = make_group_infos(segments, subject, task_ids)
    csv_write(subject_dir / "segment_exclusions.csv", exclusions)
    csv_write(subject_dir / "group_qc.csv", group_qc)
    groups = load_group_data(ds, infos, args.vectorization, args.read_batch_size)
    complete, inventory = validate_complete_groups(groups)
    json_dump(subject_dir / "state_group_inventory.json", inventory)
    if not complete:
        return {
            "model": model, "subject_id": subject, "status": "incomplete_state_groups",
            "inventory": inventory,
        }

    sessions = sorted({g.session for g in groups})
    expected_folds = len(sessions)
    flatten_dim = int(groups[0].X.shape[1])
    log(f"  sub-{subject:03d}: sessions={sessions}, five complete state groups/session")

    # Rank feasibility is fold-specific. An unavailable rank is recorded and
    # skipped; it is never silently truncated and never aborts the subject/model.
    availability_rows: List[Dict[str, Any]] = []
    fold_results: List[Dict[str, Any]] = []
    for heldout in sessions:
        train_groups = [g for g in groups if g.session != heldout]
        n_train_windows = int(sum(len(g.X) for g in train_groups))
        algebraic_max = int(min(max(n_train_windows - 1, 0), flatten_dim))
        feasible_ranks = [int(r) for r in args.svd_rank_list if int(r) <= algebraic_max]
        unavailable_ranks = [int(r) for r in args.svd_rank_list if int(r) > algebraic_max]
        if unavailable_ranks:
            log(
                f"    heldout session {heldout}: mathematically unavailable ranks "
                f"{unavailable_ranks}; algebraic maximum={algebraic_max}"
            )
        for rank in args.svd_rank_list:
            feasible = int(rank) <= algebraic_max
            availability_rows.append({
                "model": model,
                "subject_id": int(subject),
                "heldout_session": int(heldout),
                "svd_rank_requested": int(rank),
                "available": bool(feasible),
                "reason": "available" if feasible else "exceeds_algebraic_maximum",
                "n_training_windows": n_train_windows,
                "flatten_dimension": flatten_dim,
                "maximum_algebraic_rank": algebraic_max,
            })
        for rank in feasible_ranks:
            log(f"    heldout session {heldout}, centered-SVD rank {rank}")
            fold_results.append(run_fold_rank(
                groups, heldout, rank, args,
                args.seed + 100000 * subject + 1000 * heldout + rank,
                args.seed + 100000 * subject + rank,
                subject_dir,
            ))
    csv_write(subject_dir / "svd_rank_availability.csv", availability_rows)

    summary_rows: List[Dict[str, Any]] = []
    pair_summary_rows: List[Dict[str, Any]] = []
    eigen_summary_rows_all: List[Dict[str, Any]] = []
    effective_dimension_by_rank: Dict[str, Any] = {}
    rank_completion: Dict[str, Any] = {}

    for rank in args.svd_rank_list:
        rank_folds = [f for f in fold_results if f["svd_rank_requested"] == rank]
        n_available_folds = len(rank_folds)
        complete_outer_cv = n_available_folds == expected_folds
        rank_completion[str(rank)] = {
            "n_expected_folds": expected_folds,
            "n_available_folds": n_available_folds,
            "complete_outer_cv": complete_outer_cv,
            "formal_inference_available": bool(complete_outer_cv),
        }
        if not rank_folds:
            effective_dimension_by_rank[str(rank)] = {
                "status": "unavailable_all_folds",
                **rank_completion[str(rank)],
            }
            continue

        nulls = [f.get("_null_arrays") for f in rank_folds]
        # Formal pooled permutation inference is only reported for complete LOSO CV.
        have_null = complete_outer_cv and all(n is not None for n in nulls) and bool(nulls)
        pooled_null = None
        if have_null:
            pooled_null = {
                key: np.mean(np.stack([n[key] for n in nulls], axis=0), axis=0)
                for key in (
                    "window_bacc", "centroid_accuracy", "eigenvalues",
                    "pair_bacc", "pair_cross_distance",
                )
            }
            null_dir = ensure_dir(subject_dir / f"svd-rank-{rank}")
            np.savez_compressed(
                null_dir / "subject_level_pooled_nulls.npz", **pooled_null
            )

        rank_dimension_rows: List[Dict[str, Any]] = []
        for d in range(1, 5):
            metrics = [
                next(row for row in f["dimension_metrics"] if row["n_dims"] == d)
                for f in rank_folds if any(row["n_dims"] == d for row in f["dimension_metrics"])
            ]
            if not metrics:
                continue
            real_mean = safe_mean(m["window_balanced_accuracy"] for m in metrics)
            row = {
                "model": model,
                "subject_id": subject,
                "svd_rank_requested": rank,
                "n_dims": d,
                "n_folds": len(metrics),
                "n_expected_folds": expected_folds,
                "n_available_folds": n_available_folds,
                "complete_outer_cv": bool(complete_outer_cv),
                "formal_inference_available": bool(pooled_null is not None),
                "mean_window_bacc": real_mean,
                "sd_window_bacc": safe_std(m["window_balanced_accuracy"] for m in metrics),
                "se_window_bacc": standard_error(m["window_balanced_accuracy"] for m in metrics),
                "min_window_bacc": float(np.min([m["window_balanced_accuracy"] for m in metrics])),
                "mean_window_auc": safe_mean(m["window_macro_auc"] for m in metrics),
                "mean_train_window_bacc": safe_mean(m.get("train_window_balanced_accuracy", np.nan) for m in metrics),
                "mean_train_window_auc": safe_mean(m.get("train_window_macro_auc", np.nan) for m in metrics),
                "mean_train_test_bacc_gap": safe_mean(m.get("train_test_bacc_gap", np.nan) for m in metrics),
                "mean_state_centroid_accuracy": safe_mean(m["state_centroid_accuracy"] for m in metrics),
                "mean_baseline_recall": safe_mean(m["baseline_recall"] for m in metrics),
                "mean_task_recall": safe_mean(m["mean_task_recall"] for m in metrics),
            }
            for state in STATE_NAMES:
                row[f"mean_recall_{state}"] = safe_mean(
                    m.get(f"recall_{state}", np.nan) for m in metrics
                )
            if pooled_null is not None:
                row["pooled_window_bacc_p"] = empirical_upper_p(
                    real_mean, pooled_null["window_bacc"][:, d - 1]
                )
                row["pooled_window_bacc_maxstat_p"] = empirical_upper_p(
                    real_mean, np.nanmax(pooled_null["window_bacc"], axis=1)
                )
                centroid_real = row["mean_state_centroid_accuracy"]
                row["pooled_state_centroid_accuracy_p"] = empirical_upper_p(
                    centroid_real, pooled_null["centroid_accuracy"][:, d - 1]
                )
            rank_dimension_rows.append(row)
            summary_rows.append(row)

        rank_pair_rows: List[Dict[str, Any]] = []
        for pair_idx, (i, j) in enumerate(PAIR_SPECS):
            rows = [
                next(r for r in f["pairwise_metrics"] if r["pair_index"] == pair_idx)
                for f in rank_folds
            ]
            valid = [r for r in rows if r.get("valid")]
            if not valid:
                continue
            real_bacc = safe_mean(r["window_balanced_accuracy"] for r in valid)
            real_cross = safe_mean(r["train_test_cross_distance"] for r in valid)
            row = {
                "model": model,
                "subject_id": subject,
                "svd_rank_requested": rank,
                "n_expected_folds": expected_folds,
                "n_available_folds": n_available_folds,
                "complete_outer_cv": bool(complete_outer_cv),
                "formal_inference_available": bool(pooled_null is not None),
                "pair_index": pair_idx,
                "pair_name": pair_name(i, j),
                "mean_window_bacc": real_bacc,
                "min_window_bacc": float(np.min([r["window_balanced_accuracy"] for r in valid])),
                "mean_window_auc": safe_mean(r["window_auc"] for r in valid),
                "mean_state_centroid_pair_accuracy": safe_mean(r["state_centroid_pair_accuracy"] for r in valid),
                "mean_train_test_cross_distance": real_cross,
                "mean_train_test_delta_cosine_auxiliary": safe_mean(
                    r["train_test_delta_cosine"] for r in valid
                ),
            }
            if pooled_null is not None:
                row["pooled_window_bacc_p"] = empirical_upper_p(
                    real_bacc, pooled_null["pair_bacc"][:, pair_idx]
                )
                row["pooled_cross_distance_p"] = empirical_upper_p(
                    real_cross, pooled_null["pair_cross_distance"][:, pair_idx]
                )
            rank_pair_rows.append(row)
            pair_summary_rows.append(row)

        rank_eigen_rows: List[Dict[str, Any]] = []
        for axis in range(1, 5):
            vals = [
                float(f["multiclass_eigenvalues"][axis - 1])
                for f in rank_folds if len(f["multiclass_eigenvalues"]) >= axis
            ]
            if not vals:
                continue
            real_mean = safe_mean(vals)
            row = {
                "model": model,
                "subject_id": subject,
                "svd_rank_requested": rank,
                "n_expected_folds": expected_folds,
                "n_available_folds": n_available_folds,
                "complete_outer_cv": bool(complete_outer_cv),
                "formal_inference_available": bool(pooled_null is not None),
                "ld_axis": axis,
                "mean_eigenvalue": real_mean,
                "sd_eigenvalue": safe_std(vals),
                "se_eigenvalue": standard_error(vals),
            }
            if pooled_null is not None:
                row["pooled_permutation_p"] = empirical_upper_p(
                    real_mean, pooled_null["eigenvalues"][:, axis - 1]
                )
            rank_eigen_rows.append(row)
            eigen_summary_rows_all.append(row)

        effective = summarize_effective_dimension(
            rank_dimension_rows, rank_eigen_rows, args.dimension_alpha
        )
        effective.update(rank_completion[str(rank)])
        effective["status"] = "complete" if complete_outer_cv else "partial_exploratory"
        effective_dimension_by_rank[str(rank)] = effective

    complete_ranks = [
        int(r) for r in args.svd_rank_list
        if rank_completion.get(str(r), {}).get("complete_outer_cv", False)
    ]
    available_ranks = [
        int(r) for r in args.svd_rank_list
        if rank_completion.get(str(r), {}).get("n_available_folds", 0) > 0
    ]
    candidate_ranks = complete_ranks or available_ranks
    main_rank = (
        min(candidate_ranks, key=lambda x: abs(x - args.main_svd_rank))
        if candidate_ranks else None
    )
    csv_write(subject_dir / "multiclass_across_folds.csv", summary_rows)
    csv_write(subject_dir / "pairwise_across_folds.csv", pair_summary_rows)
    csv_write(subject_dir / "eigenvalues_across_folds.csv", eigen_summary_rows_all)
    csv_write(subject_dir / "effective_dimension_summary.csv", [
        dict(svd_rank_requested=int(rank), **metrics)
        for rank, metrics in effective_dimension_by_rank.items()
    ])

    serializable_folds = []
    for fold in fold_results:
        clean = {k: v for k, v in fold.items() if not k.startswith("_")}
        serializable_folds.append(clean)

    summary = {
        "script_version": SCRIPT_VERSION,
        "model": model,
        "subject_id": subject,
        "status": "ok" if fold_results else "no_feasible_svd_rank",
        "sessions": sessions,
        "task_ids": dict(task_ids),
        "requested_main_svd_rank": args.main_svd_rank,
        "main_svd_rank": main_rank,
        "main_shrinkage": args.main_shrinkage,
        "rank_completion": rank_completion,
        "effective_dimension_by_svd_rank": effective_dimension_by_rank,
        "main_rank_effective_dimension": (
            effective_dimension_by_rank.get(str(main_rank), {}) if main_rank is not None else {}
        ),
        "baseline_limitation": (
            "Baseline pools pre-task segments from four upcoming task identities; "
            "task-conditioned preparation is not tested in Experiment A."
        ),
        "multiclass_across_folds": summary_rows,
        "pairwise_across_folds": pair_summary_rows,
        "eigenvalues_across_folds": eigen_summary_rows_all,
        "fold_results": serializable_folds,
    }
    json_dump(subject_dir / "subject_summary.json", summary)
    return summary

def subject_is_eligible(
    meta: Metadata,
    subject: int,
    task_ids: Mapping[str, int],
    args: argparse.Namespace,
) -> Tuple[bool, Dict[str, Any]]:
    segments, exclusions = build_selected_segments(meta, subject, task_ids, args)
    infos, group_qc = make_group_infos(segments, subject, task_ids)
    sessions = sorted({g.session for g in infos})
    counts = {
        int(session): {
            label: sum(g.session == session and g.label == label for g in infos)
            for label in range(5)
        }
        for session in sessions
    }
    ok = len(sessions) >= 3 and all(
        all(counts[session][label] == 1 for label in range(5)) for session in sessions
    )
    return ok, {
        "subject_id": int(subject),
        "eligible": bool(ok),
        "sessions": sessions,
        "counts_by_session": counts,
        "n_segments": len(segments),
        "n_exclusions": len(exclusions),
        "n_group_qc_issues": len(group_qc),
    }


def choose_subjects(
    meta: Metadata,
    task_ids: Mapping[str, int],
    args: argparse.Namespace,
) -> Tuple[List[int], List[int], List[Dict[str, Any]]]:
    details = []
    eligible = []
    for subject in sorted(int(x) for x in np.unique(meta.subject)):
        ok, detail = subject_is_eligible(meta, subject, task_ids, args)
        details.append(detail)
        if ok:
            eligible.append(subject)
    if args.subjects:
        chosen = parse_int_csv(args.subjects)
        missing = [s for s in chosen if s not in set(int(x) for x in np.unique(meta.subject))]
        ineligible = [s for s in chosen if s not in set(eligible)]
        if missing:
            raise ValueError(f"Requested subjects absent from H5: {missing}")
        if ineligible:
            raise ValueError(f"Requested subjects fail five-state completeness QC: {ineligible}")
        return chosen, eligible, details
    if not eligible:
        raise RuntimeError("No subject has three complete sessions for all five states")
    rng = np.random.default_rng(args.subject_seed)
    k = min(args.n_subjects, len(eligible))
    chosen = sorted(int(x) for x in rng.choice(eligible, size=k, replace=False))
    return chosen, eligible, details


def run_model(model: str, path: str, args: argparse.Namespace, root: Path) -> Dict[str, Any]:
    model_dir = ensure_dir(root / model)
    log(f"model {model}: {path}")
    with h5py.File(path, "r") as h5:
        meta = load_metadata(h5, args.embedding_key)
        ds = h5[args.embedding_key]
        task_ids = parse_task_ids(args.task_ids)
        chosen, eligible, eligibility = choose_subjects(meta, task_ids, args)
        csv_write(model_dir / "subject_eligibility.csv", eligibility)
        log(f"  task IDs: {task_ids}")
        if meta.task_names:
            log(f"  H5 task names: {meta.task_names}")
        log(f"  eligible subjects ({len(eligible)}): {eligible}")
        log(f"  chosen subjects: {chosen} [seed={args.subject_seed}]")

        subjects = []
        for subject in chosen:
            subjects.append(run_subject(model, ds, meta, subject, task_ids, args, model_dir))

    cross_subject_rows = []
    for rank in args.svd_rank_list:
        for d in range(1, 5):
            rows = []
            for subject_summary in subjects:
                if subject_summary.get("status") != "ok":
                    continue
                matches = [
                    r for r in subject_summary["multiclass_across_folds"]
                    if r["svd_rank_requested"] == rank
                    and r["n_dims"] == d
                    and bool(r.get("complete_outer_cv", False))
                ]
                rows.extend(matches)
            if rows:
                cross_subject_rows.append({
                    "model": model,
                    "svd_rank_requested": rank,
                    "n_dims": d,
                    "n_subjects": len(rows),
                    "mean_window_bacc": safe_mean(r["mean_window_bacc"] for r in rows),
                    "min_subject_window_bacc": float(np.min([r["mean_window_bacc"] for r in rows])),
                    "max_subject_window_bacc": float(np.max([r["mean_window_bacc"] for r in rows])),
                    "mean_state_centroid_accuracy": safe_mean(r["mean_state_centroid_accuracy"] for r in rows),
                })
    csv_write(model_dir / "cross_subject_multiclass_summary.csv", cross_subject_rows)
    summary = {
        "script_version": SCRIPT_VERSION,
        "model": model,
        "embedding_path": path,
        "eligible_subjects": eligible,
        "chosen_subjects": chosen,
        "subjects": subjects,
        "cross_subject_multiclass": cross_subject_rows,
    }
    json_dump(model_dir / "model_summary.json", summary)
    return summary


# =============================================================================
# CLI
# =============================================================================


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Within-subject five-state multiclass/pairwise semantic LDA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--models", nargs="+", default=["CBraMod"])
    parser.add_argument("--model-path", action="append", default=[], help="MODEL=/path/file.h5")
    parser.add_argument("--embedding-key", default="embedding")
    parser.add_argument("--vectorization", choices=["flatten", "mean_structural"], default="flatten")
    parser.add_argument("--task-ids", default="MA=0,NB=1,NBMA=5,Full=6")

    parser.add_argument("--n-subjects", type=int, default=3)
    parser.add_argument("--subjects", default="", help="Explicit override, e.g. 3,11,24")
    parser.add_argument("--subject-seed", type=int, default=20260718)

    parser.add_argument("--baseline-guard-sec", type=float, default=2.0)
    parser.add_argument("--task-guard-sec", type=float, default=10.0)
    parser.add_argument("--task-end-guard-sec", type=float, default=5.0)
    parser.add_argument("--run-start-guard-sec", type=float, default=5.0)
    parser.add_argument("--min-windows-per-segment", type=int, default=10)

    parser.add_argument("--svd-ranks", "--ranks", dest="svd_ranks", default="100,200,300,500")
    parser.add_argument("--main-svd-rank", "--main-rank", dest="main_svd_rank", type=int, default=200)
    parser.add_argument("--main-shrinkage", type=float, default=0.9)
    parser.add_argument("--sensitivity-shrinkage", default="0.7,0.99")
    parser.add_argument("--n-permutations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--degeneracy-gap", type=float, default=0.10)
    parser.add_argument("--dimension-alpha", type=float, default=0.05)

    parser.add_argument("--read-batch-size", type=int, default=1024)
    parser.add_argument(
        "--outdir", type=Path,
        default=Path("/mnt/dataset4/yinuo/FM_flow/dataset/expA_five_state_semantic_lda_v3_1"),
    )
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    args.svd_rank_list = parse_int_csv(args.svd_ranks)
    args.sensitivity_shrinkage_grid = parse_float_csv(args.sensitivity_shrinkage)
    if not args.svd_rank_list:
        raise ValueError("--svd-ranks is empty")
    if not (0.0 <= args.main_shrinkage <= 1.0):
        raise ValueError("--main-shrinkage must lie in [0, 1]")
    if any(not (0.0 <= x <= 1.0) for x in args.sensitivity_shrinkage_grid):
        raise ValueError("--sensitivity-shrinkage values must lie in [0, 1]")

    root = ensure_dir(args.outdir)
    paths = dict(MODEL_PATHS)
    for override in args.model_path:
        if "=" not in override:
            raise ValueError(f"--model-path expects MODEL=/path, got {override!r}")
        model, path = override.split("=", 1)
        paths[model] = path

    json_dump(root / "arguments.json", {
        "script_version": SCRIPT_VERSION,
        "arguments": vars(args),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "h5py": h5py.__version__,
        "started_at": now(),
    })

    summaries = []
    failures = []
    for model in args.models:
        if model not in paths:
            failures.append({"model": model, "error": "no path configured"})
            continue
        if not Path(paths[model]).exists():
            failures.append({"model": model, "error": f"missing file {paths[model]}"})
            continue
        try:
            summaries.append(run_model(model, paths[model], args, root))
        except Exception as exc:  # noqa: BLE001
            failures.append({
                "model": model,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            })
            log(f"model {model} FAILED: {exc!r}")
            if args.fail_fast:
                raise

    json_dump(root / "summary.json", {
        "script_version": SCRIPT_VERSION,
        "finished_at": now(),
        "models": summaries,
        "failures": failures,
    })
    log(f"done; {len(summaries)} model(s) ok, {len(failures)} failed")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
