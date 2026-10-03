#!/usr/bin/env python3
"""
Experiment B: five-state within/global slow-feature analysis for ds007554.

Scientific question
-------------------
Within one subject, do EEG foundation-model embeddings preserve reproducible
slow temporal structure inside each of five states, what low-dimensional slow
subspace is required for each state and for complete baseline-to-task runs, and
how much of each anonymous global slow direction is explained by the named
within-state slow subspaces?

States
------
    Baseline, Mental Arithmetic (MA), N-back (NB),
    N-back Arithmetic (NBMA), Full Integrated Task (Full)

Design
------
* Three-fold leave-one-session-out within each subject.
* All centering, balanced SVD, covariance estimation, SFA fitting,
  vector-wise within-state-stratified temporal-permutation nulls, auxiliary r99
  diagnostics, and within/global overlap analysis are fit on the two outer-training
  sessions only.
* The ds007554 embeddings used here already have 1 s windows and 1 s stride, so
  the raw grid is the non-overlapping main grid. No duplicate no-overlap run is
  required unless a future embedding file has stride shorter than its window.
* Adjacencies never cross runs or recording gaps.
* Five named within-state SFA systems are fit independently:
      B_SF01, MA_SF01, NB_SF01, NBMA_SF01, Full_SF01, ...
* One global SFA system is fit on complete selected-task runs and therefore
  includes the true Baseline -> Task adjacency whenever it exists on the grid.
* Cross-fold fixed-top-k principal-angle stability is the primary evidence for
  reproducible slow subspaces. The r99 temporal-order statistic is auxiliary and
  never selects the B-to-C handoff.
* Held-out sessions only validate frozen axes.
* All within-state subspaces are mapped into the common global covariance metric
  before projector overlaps are computed.
* Global directions are interpreted by state-subspace overlap, union explained
  fraction, leave-one-state-out unique contribution, minimum-norm coefficients,
  and near-degenerate block overlap.
* Version 5.0 exports a fold-local, self-contained B-to-C handoff for the frozen
  Global-SFA space. Its rank is a pre-registered fixed k followed only by
  non-chaining degeneracy-block completion. The handoff also contains the exact
  train-only transform, all subject-window coordinates, and the metadata ledger
  required for Experiment C.

Important boundary
------------------
The 99% quantity is not explained variance. SFA features are unit-variance by
construction. It is the cumulative fraction of detectable temporal-order slow
excess over a vector-wise temporal-permutation null stratified by the five-state
label inside each contiguous run segment. The same row permutation is applied to
every coordinate, so the null commutes exactly with any invertible
reparameterization of the fixed ambient space. It preserves each within-state
multivariate point cloud while destroying within-state temporal order. It does not
by itself test cross-coordinate coordination; reproducibility is assessed by
cross-fold fixed-top-k subspace stability.

Typical smoke run
-----------------
python3 run_experiment_B_five_state_within_global_sfa_v5_1.py \
  --models CBraMod --n-subjects 1 --svd-ranks 8 --main-svd-rank 8 \
  --grids no_overlap --n-train-shuffle 10 --n-heldout-shuffle 20 \
  --outdir /mnt/dataset4/yinuo/FM_flow/dataset/expB_five_state_sfa_v5_1_smoke \
  --fail-fast
"""

from __future__ import annotations

import argparse
import hashlib
import csv
import itertools
import json
import math
import os
import platform
import re
import sys
import tempfile
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_VERSION = "2026-07-30-expB-five-state-within-global-sfa-v5.1-resumable-slow-primary"

MODEL_PATHS: Dict[str, str] = {
    "EEGMamba": "/mnt/dataset4/fuxy/FMS/FM/EEGMamba/output_emb/ds007554_embeddings.h5",
    "BIOT": "/mnt/dataset4/fuxy/FMS/FM/BIOT/output_emb/ds007554_embeddings.h5",
    "LaBraM": "/mnt/dataset4/fuxy/FMS/FM/LaBraM/output_emb/ds007554_embeddings.h5",
    "CBraMod": "/mnt/dataset4/fuxy/FMS/FM/CBraMod/output_emb/ds007554_embeddings.h5",
    "EEGPT": "/mnt/dataset4/fuxy/FMS/FM/EEGPT/output_emb/ds007554_embeddings.h5",
}

STATE_NAMES = ["Baseline", "MA", "NB", "NBMA", "Full"]
STATE_SHORT = ["B", "MA", "NB", "NBMA", "Full"]
TASK_STATE_NAMES = ["MA", "NB", "NBMA", "Full"]
DEFAULT_TASK_IDS = {"MA": 0, "NB": 1, "NBMA": 5, "Full": 6}
REQUIRED_KEYS = (
    "embedding", "subject_id", "session_id", "run_id", "sample_start",
    "sample_end", "phase_id", "task_id", "task_family_id",
)


# =============================================================================
# General utilities
# =============================================================================


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{now()}] {message}", flush=True)


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_name(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def decode_scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [decode_scalar(x) for x in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): decode_scalar(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [decode_scalar(x) for x in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


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


def csv_read(path: Path | str) -> List[Dict[str, str]]:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def npz_write_atomic(path: Path | str, **payload: Any) -> None:
    """Write an NPZ atomically so Experiment C never sees a partial handoff."""
    target = Path(path)
    ensure_dir(target.parent)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(target.parent), prefix=target.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as fh:
            np.savez(fh, **payload)
        os.replace(tmp_name, target)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def parse_int_csv(text: str) -> List[int]:
    return [int(x) for x in str(text).replace(" ", "").split(",") if x]


def parse_float_csv(text: str) -> List[float]:
    return [float(x) for x in str(text).replace(" ", "").split(",") if x]


def parse_str_csv(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_task_ids(text: str) -> Dict[str, int]:
    aliases = {
        "MA": "MA", "MENTALARITHMETIC": "MA", "MENTAL_ARITHMETIC": "MA",
        "NB": "NB", "NBACK": "NB", "N-BACK": "NB",
        "NBMA": "NBMA", "NBACKARITHMETIC": "NBMA", "N-BACKARITHMETIC": "NBMA",
        "FULL": "Full", "FULLTASK": "Full", "FULLINTEGRATEDTASK": "Full",
    }
    parsed: Dict[str, int] = {}
    for item in str(text).split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"--task-ids expects NAME=ID entries, got {item!r}")
        raw_name, raw_id = item.split("=", 1)
        key = raw_name.strip().replace(" ", "").upper()
        if key not in aliases:
            raise ValueError(f"Unknown task alias {raw_name!r}")
        parsed[aliases[key]] = int(raw_id)
    missing = [x for x in TASK_STATE_NAMES if x not in parsed]
    if missing:
        raise ValueError(f"--task-ids must define all task states; missing {missing}")
    return parsed


def empirical_lower_p(real: float, null: np.ndarray) -> float:
    arr = np.asarray(null, dtype=float)
    arr = arr[np.isfinite(arr)]
    if not np.isfinite(real) or arr.size == 0:
        return float("nan")
    return float((1 + np.sum(arr <= real)) / (arr.size + 1))


def safe_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    return float(np.nanmean(arr)) if np.any(np.isfinite(arr)) else float("nan")


def safe_median(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    return float(np.nanmedian(arr)) if np.any(np.isfinite(arr)) else float("nan")


def symmetrize(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)


def orthonormalize_columns(A: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[1] == 0:
        return np.empty((A.shape[0] if A.ndim == 2 else 0, 0), dtype=float)
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    if s.size == 0:
        return np.empty((A.shape[0], 0), dtype=float)
    keep = s > tol * max(float(s[0]), 1.0)
    return U[:, keep]


def principal_angles_deg(Q1: np.ndarray, Q2: np.ndarray) -> np.ndarray:
    Q1 = orthonormalize_columns(Q1)
    Q2 = orthonormalize_columns(Q2)
    if Q1.shape[1] == 0 or Q2.shape[1] == 0:
        return np.array([], dtype=float)
    s = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    return np.degrees(np.arccos(s))


def make_degenerate_blocks(gammas: np.ndarray, n_axes: int, relative_gap: float) -> List[List[int]]:
    """Partition an ordered spectrum without single-linkage chaining.

    A new axis joins the current block only when the *entire* prospective block
    remains within ``relative_gap`` from its minimum to maximum eigenvalue.
    This protects genuinely near-degenerate eigenspaces while preventing a
    smooth spectrum from being chained into one giant block merely because
    every adjacent gap is small.  The rule depends only on generalized
    eigenvalues and is therefore invariant under invertible coordinate changes.
    """
    g = np.asarray(gammas, dtype=np.float64).reshape(-1)
    n = min(int(n_axes), len(g))
    if n <= 0:
        return []
    blocks: List[List[int]] = [[0]]
    block_min = block_max = float(g[0])
    for j in range(1, n):
        value = float(g[j])
        candidate_min = min(block_min, value)
        candidate_max = max(block_max, value)
        span = abs(candidate_max - candidate_min) / max(
            abs(candidate_min), abs(candidate_max), 1e-12
        )
        if span < float(relative_gap):
            blocks[-1].append(j)
            block_min, block_max = candidate_min, candidate_max
        else:
            blocks.append([j])
            block_min = block_max = value
    return blocks


def sha256_file(path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# =============================================================================
# HDF5 embedding I/O
# =============================================================================


def read_vector(h5: h5py.File, key: str, dtype=None) -> np.ndarray:
    arr = np.asarray(h5[key][...])
    return arr.astype(dtype) if dtype is not None else arr


def read_name_map(h5: h5py.File, key: str) -> Dict[int, str]:
    if key not in h5:
        return {}
    raw = np.asarray(h5[key][...])
    return {int(i): str(decode_scalar(v)) for i, v in enumerate(raw.tolist())}


def find_subject_rows(
    subject_ds: h5py.Dataset,
    subjects: Sequence[int],
    batch_size: int = 65536,
) -> np.ndarray:
    """Locate explicitly requested subjects with one bounded subject-id scan.

    The original B implementation loaded every metadata column for the complete
    H5 even when ``--subjects`` named only a few subjects.  This helper narrows
    the rows first, so smoke and the planned 3-subject run avoid an unnecessary
    all-dataset inventory pass.  The scientific sample set is unchanged.
    """
    requested = np.asarray(sorted(set(int(x) for x in subjects)), dtype=np.int64)
    if requested.size == 0:
        return np.empty(0, dtype=np.int64)
    chosen: List[np.ndarray] = []
    n = int(subject_ds.shape[0])
    for start in range(0, n, int(batch_size)):
        end = min(start + int(batch_size), n)
        block = np.asarray(subject_ds[start:end], dtype=np.int64).reshape(-1)
        local = np.flatnonzero(np.isin(block, requested))
        if len(local):
            chosen.append(local.astype(np.int64) + start)
    return np.concatenate(chosen) if chosen else np.empty(0, dtype=np.int64)


def vector_dim(ds: h5py.Dataset, mode: str) -> int:
    shape = tuple(int(v) for v in ds.shape)
    if len(shape) < 2:
        raise ValueError(f"Embedding must have shape (N,...), got {shape}")
    if mode == "flatten" or len(shape) == 2:
        return int(np.prod(shape[1:]))
    if mode == "mean_structural":
        return int(shape[-1])
    raise ValueError(mode)


def vectorize_chunk(chunk: np.ndarray, mode: str) -> np.ndarray:
    x = np.asarray(chunk)
    if mode == "flatten" or x.ndim == 2:
        out = x.reshape(x.shape[0], -1)
    elif mode == "mean_structural":
        axes = tuple(range(1, x.ndim - 1))
        out = np.mean(x, axis=axes) if axes else x.reshape(x.shape[0], -1)
    else:
        raise ValueError(mode)
    return np.asarray(out, dtype=np.float32).reshape(out.shape[0], -1)


def iter_embedding_chunks(
    ds: h5py.Dataset,
    global_indices: np.ndarray,
    mode: str,
    batch_size: int,
) -> Iterator[Tuple[int, int, np.ndarray]]:
    idx = np.asarray(global_indices, dtype=np.int64)
    if idx.ndim != 1:
        raise ValueError("indices must be 1D")
    if idx.size and np.any(np.diff(idx) < 0):
        raise ValueError("indices must be sorted")
    for start in range(0, len(idx), batch_size):
        end = min(start + batch_size, len(idx))
        yield start, end, vectorize_chunk(ds[idx[start:end]], mode)


@dataclass
class Metadata:
    global_index: np.ndarray
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


def load_metadata(
    h5: h5py.File,
    embedding_key: str,
    row_indices: Optional[np.ndarray] = None,
) -> Metadata:
    missing = [k for k in REQUIRED_KEYS if k not in h5]
    if missing:
        raise KeyError(f"Missing required H5 keys: {missing}")
    n_total = int(h5[embedding_key].shape[0])
    if row_indices is None:
        idx = np.arange(n_total, dtype=np.int64)
    else:
        idx = np.unique(np.asarray(row_indices, dtype=np.int64))
        idx.sort()
        if idx.ndim != 1 or np.any(idx < 0) or np.any(idx >= n_total):
            raise ValueError("Invalid metadata row_indices")
        if len(idx) == 0:
            raise ValueError("No metadata rows selected")

    def selected_vector(key: str) -> np.ndarray:
        return np.asarray(h5[key][idx], dtype=np.int64).reshape(-1)

    arrays = {
        "subject": selected_vector("subject_id"),
        "session": selected_vector("session_id"),
        "run": selected_vector("run_id"),
        "sample_start": selected_vector("sample_start"),
        "sample_end": selected_vector("sample_end"),
        "phase": selected_vector("phase_id"),
        "task": selected_vector("task_id"),
        "family": selected_vector("task_family_id"),
    }
    arrays["center_sample"] = (
        selected_vector("center_sample")
        if "center_sample" in h5
        else ((arrays["sample_start"] + arrays["sample_end"] - 1) // 2).astype(np.int64)
    )
    for name, arr in arrays.items():
        if len(arr) != len(idx):
            raise ValueError(f"Length mismatch for {name}: {len(arr)} vs {len(idx)}")
    attrs = {str(k): decode_scalar(v) for k, v in h5.attrs.items()}
    sr = float(attrs.get("sampling_rate", attrs.get("target_sfreq", 200.0)))
    # Infer stride within runs, not across unrelated run starts.
    diffs: List[float] = []
    keys = np.stack([arrays["subject"], arrays["session"], arrays["run"]], axis=1)
    for key in np.unique(keys, axis=0):
        mask = np.all(keys == key[None, :], axis=1)
        starts = np.sort(arrays["sample_start"][mask])
        if len(starts) > 1:
            d = np.diff(starts)
            diffs.extend(d[d > 0].tolist())
    inferred_stride = float(np.median(diffs) / sr) if diffs else 1.0
    stride = float(attrs.get("stride_seconds", inferred_stride))
    inferred_window = float(np.median(arrays["sample_end"] - arrays["sample_start"]) / sr)
    window = float(attrs.get("window_seconds", inferred_window))
    return Metadata(
        idx, arrays["subject"], arrays["session"], arrays["run"],
        arrays["sample_start"], arrays["sample_end"], arrays["center_sample"],
        arrays["phase"], arrays["task"], arrays["family"], sr, stride, window,
        read_name_map(h5, "task_names"), read_name_map(h5, "task_family_names"), attrs,
    )


# =============================================================================
# Run validation and raw/no-overlap grids
# =============================================================================


@dataclass
class RunRecord:
    subject: int
    session: int
    run: int
    task: int
    family: int
    all_indices: np.ndarray
    raw_segments: List[np.ndarray]
    no_overlap_segments: List[np.ndarray]
    valid: bool
    reasons: List[str] = field(default_factory=list)


@dataclass
class GridData:
    name: str
    global_indices: np.ndarray
    subject: np.ndarray
    session: np.ndarray
    run: np.ndarray
    phase: np.ndarray
    task: np.ndarray
    family: np.ndarray
    sample_start: np.ndarray
    sample_end: np.ndarray
    center_sample: np.ndarray
    run_to_segments: Dict[Tuple[int, int, int], List[np.ndarray]]
    decimation_factor: int
    effective_stride_seconds: float
    effective_overlap_fraction: float


def group_global_indices(meta: Metadata) -> Dict[Tuple[int, int, int], np.ndarray]:
    out: Dict[Tuple[int, int, int], List[int]] = {}
    for i, key in enumerate(zip(meta.subject.tolist(), meta.session.tolist(), meta.run.tolist())):
        k = (int(key[0]), int(key[1]), int(key[2]))
        out.setdefault(k, []).append(i)
    return {k: np.asarray(v, dtype=np.int64) for k, v in out.items()}


def split_by_stride(
    meta: Metadata,
    ordered_indices: np.ndarray,
    expected_stride_samples: float,
    tolerance_samples: float,
) -> List[np.ndarray]:
    idx = np.asarray(ordered_indices, dtype=np.int64)
    if len(idx) < 2:
        return []
    starts = meta.sample_start[idx].astype(float)
    breaks = np.flatnonzero(np.abs(np.diff(starts) - expected_stride_samples) > tolerance_samples) + 1
    return [x for x in np.split(idx, breaks) if len(x) >= 2]


def validate_selected_runs(
    meta: Metadata,
    selected_task_ids: Sequence[int],
    args: argparse.Namespace,
) -> Tuple[List[RunRecord], List[Dict[str, Any]]]:
    selected = set(int(x) for x in selected_task_ids)
    raw_stride_samples = meta.stride_seconds * meta.sampling_rate
    tol = max(args.stride_tolerance_sec * meta.sampling_rate, 1e-6)
    decimation = max(1, int(math.ceil(meta.window_seconds / max(meta.stride_seconds, 1e-12) - 1e-9)))
    records: List[RunRecord] = []
    qc: List[Dict[str, Any]] = []
    for (subject, session, run), idx0 in sorted(group_global_indices(meta).items()):
        idx = idx0[np.argsort(meta.sample_start[idx0], kind="mergesort")]
        task_rows = idx[meta.phase[idx] == 1]
        task_ids = np.unique(meta.task[task_rows]) if len(task_rows) else np.array([], dtype=int)
        if len(task_ids) != 1 or int(task_ids[0]) not in selected:
            continue
        family_ids = np.unique(meta.family[task_rows])
        reasons: List[str] = []
        phases = meta.phase[idx]
        if np.any(np.diff(meta.sample_start[idx]) <= 0):
            reasons.append("non_monotone_start")
        if not np.all(np.isin(phases, [0, 1])):
            reasons.append("invalid_phase")
        transitions = np.diff(phases)
        if np.any(transitions < 0) or int(np.sum(transitions == 1)) != 1:
            reasons.append("not_one_clean_baseline_to_task_transition")
        if int(np.sum(phases == 0)) < args.min_phase_windows_raw:
            reasons.append("baseline_too_short")
        if int(np.sum(phases == 1)) < args.min_phase_windows_raw:
            reasons.append("task_too_short")
        raw_segments = split_by_stride(meta, idx, raw_stride_samples, tol)
        no_segments: List[np.ndarray] = []
        for seg in raw_segments:
            dec = seg[args.decimation_offset % decimation::decimation]
            if len(dec) >= 2:
                no_segments.extend(split_by_stride(
                    meta, dec, decimation * raw_stride_samples,
                    max(tol, args.stride_tolerance_sec * meta.sampling_rate),
                ))
        if sum(max(0, len(x) - 1) for x in raw_segments) < args.min_adjacencies_per_run:
            reasons.append("too_few_raw_adjacencies")
        if (
            "no_overlap" in args.grid_list
            and sum(max(0, len(x) - 1) for x in no_segments) < args.min_adjacencies_per_run
        ):
            reasons.append("too_few_no_overlap_adjacencies")
        valid = len(reasons) == 0
        rec = RunRecord(
            int(subject), int(session), int(run), int(task_ids[0]),
            int(family_ids[0]) if len(family_ids) == 1 else -1,
            np.asarray(idx, dtype=np.int64), raw_segments, no_segments, valid, reasons,
        )
        records.append(rec)
        qc.append({
            "subject_id": subject, "session_id": session, "run_id": run,
            "task_id": int(task_ids[0]), "valid": valid,
            "reasons": reasons, "n_raw_windows": int(sum(len(x) for x in raw_segments)),
            "n_no_overlap_windows": int(sum(len(x) for x in no_segments)),
        })
    return records, qc


def build_subject_grid(
    meta: Metadata,
    records: Sequence[RunRecord],
    subject: int,
    grid_name: str,
    args: argparse.Namespace,
) -> GridData:
    chosen = [r for r in records if r.valid and r.subject == subject]
    if not chosen:
        raise ValueError(f"No valid selected runs for subject {subject}")
    # RunRecord segments index the compact Metadata arrays.  Convert them to
    # true H5 row indices only after the subject grid has been assembled.
    all_local: List[np.ndarray] = []
    segs_local: List[Tuple[Tuple[int, int, int], np.ndarray]] = []
    for rec in chosen:
        segs = rec.raw_segments if grid_name == "raw" else rec.no_overlap_segments
        key = (rec.subject, rec.session, rec.run)
        if grid_name == "raw":
            # Preserve every validated run window in the B/C ledger. SFA itself
            # still uses only contiguous segments of length >=2.
            all_local.append(np.asarray(rec.all_indices, dtype=np.int64))
        else:
            for seg in segs:
                all_local.append(seg)
        for seg in segs:
            segs_local.append((key, seg))
    local_indices = np.unique(np.concatenate(all_local)).astype(np.int64)
    local_indices.sort()
    global_indices = meta.global_index[local_indices]
    pos_map = np.full(len(meta.global_index), -1, dtype=np.int64)
    pos_map[local_indices] = np.arange(len(local_indices), dtype=np.int64)
    run_to_segments: Dict[Tuple[int, int, int], List[np.ndarray]] = defaultdict(list)
    for key, seg in segs_local:
        pos = pos_map[seg]
        if np.any(pos < 0):
            raise RuntimeError("grid position map failed")
        run_to_segments[key].append(np.asarray(pos, dtype=np.int64))
    d = 1 if grid_name == "raw" else max(
        1, int(math.ceil(meta.window_seconds / max(meta.stride_seconds, 1e-12) - 1e-9))
    )
    eff_stride = d * meta.stride_seconds
    return GridData(
        grid_name, global_indices,
        meta.subject[local_indices], meta.session[local_indices], meta.run[local_indices],
        meta.phase[local_indices], meta.task[local_indices], meta.family[local_indices],
        meta.sample_start[local_indices], meta.sample_end[local_indices],
        meta.center_sample[local_indices], dict(run_to_segments), d, eff_stride,
        max(0.0, 1.0 - eff_stride / max(meta.window_seconds, 1e-12)),
    )


def state_label_for_positions(grid: GridData, task_ids: Mapping[str, int]) -> np.ndarray:
    labels = np.full(len(grid.global_indices), -1, dtype=np.int64)
    labels[grid.phase == 0] = 0
    for name in TASK_STATE_NAMES:
        labels[(grid.phase == 1) & (grid.task == int(task_ids[name]))] = STATE_NAMES.index(name)
    return labels


def contiguous_subsegments(segment: np.ndarray, keep: np.ndarray) -> List[np.ndarray]:
    seg = np.asarray(segment, dtype=np.int64)
    kept = seg[np.asarray(keep, dtype=bool)]
    if len(kept) < 2:
        return []
    # Positions in a run segment are consecutive in the grid array only if no
    # unrelated run rows are interleaved. Use membership index inside seg.
    loc = np.flatnonzero(keep)
    breaks = np.flatnonzero(np.diff(loc) != 1) + 1
    pieces_loc = np.split(loc, breaks)
    return [seg[p] for p in pieces_loc if len(p) >= 2]


def collect_mode_segments(
    grid: GridData,
    sessions: Sequence[int],
    task_ids: Mapping[str, int],
    state_name: Optional[str],
) -> Tuple[List[np.ndarray], List[Tuple[int, int, int]]]:
    allowed = set(int(x) for x in sessions)
    labels = state_label_for_positions(grid, task_ids)
    target = STATE_NAMES.index(state_name) if state_name is not None else None
    segments: List[np.ndarray] = []
    keys: List[Tuple[int, int, int]] = []
    for key, run_segments in sorted(grid.run_to_segments.items()):
        if key[1] not in allowed:
            continue
        for seg in run_segments:
            if state_name is None:
                if len(seg) >= 2:
                    segments.append(seg)
                    keys.append(key)
            else:
                pieces = contiguous_subsegments(seg, labels[seg] == target)
                for piece in pieces:
                    segments.append(piece)
                    keys.append(key)
    return segments, keys


def subject_is_eligible(
    records: Sequence[RunRecord],
    subject: int,
    task_ids: Mapping[str, int],
    expected_sessions: int,
) -> Tuple[bool, Dict[str, Any]]:
    selected = set(task_ids.values())
    good = [r for r in records if r.valid and r.subject == subject and r.task in selected]
    sessions = sorted(set(r.session for r in good))
    counts = {
        int(q): {int(t): sum(r.session == q and r.task == t for r in good) for t in selected}
        for q in sessions
    }
    ok = len(sessions) == int(expected_sessions) and all(
        all(counts[q][t] == 1 for t in selected) for q in sessions
    )
    return ok, {"subject_id": subject, "eligible": ok, "sessions": sessions, "counts": counts}


def choose_subjects(
    records: Sequence[RunRecord],
    task_ids: Mapping[str, int],
    args: argparse.Namespace,
) -> Tuple[List[int], List[int], List[Dict[str, Any]]]:
    subjects = sorted(set(r.subject for r in records))
    details, eligible = [], []
    for s in subjects:
        ok, detail = subject_is_eligible(records, s, task_ids, args.expected_sessions)
        details.append(detail)
        if ok:
            eligible.append(s)
    if args.subjects:
        chosen = parse_int_csv(args.subjects)
        bad = [x for x in chosen if x not in eligible]
        if bad:
            raise ValueError(f"Requested subjects are not eligible: {bad}")
        return chosen, eligible, details
    if not eligible:
        raise RuntimeError("No subject has all four selected tasks in the expected sessions")
    rng = np.random.default_rng(args.subject_seed)
    k = min(args.n_subjects, len(eligible))
    chosen = sorted(int(x) for x in rng.choice(eligible, size=k, replace=False))
    return chosen, eligible, details


# =============================================================================
# Balanced training weights and streaming centered SVD
# =============================================================================


def balanced_state_weights(
    grid: GridData,
    sessions: Sequence[int],
    task_ids: Mapping[str, int],
) -> np.ndarray:
    """Equal state -> equal segment -> equal window weights."""
    weights = np.zeros(len(grid.global_indices), dtype=np.float64)
    active_states: List[Tuple[str, List[np.ndarray]]] = []
    for state in STATE_NAMES:
        segs, _ = collect_mode_segments(grid, sessions, task_ids, state)
        if segs:
            active_states.append((state, segs))
    if len(active_states) != len(STATE_NAMES):
        missing = [x for x in STATE_NAMES if x not in {a for a, _ in active_states}]
        raise ValueError(f"Missing state segments in training fold: {missing}")
    for _, segs in active_states:
        for seg in segs:
            weights[seg] += 1.0 / (len(active_states) * len(segs) * len(seg))
    weights /= weights.sum()
    return weights


def state_point_weights(
    grid: GridData,
    sessions: Sequence[int],
    task_ids: Mapping[str, int],
    state_name: str,
) -> np.ndarray:
    weights = np.zeros(len(grid.global_indices), dtype=np.float64)
    segs, _ = collect_mode_segments(grid, sessions, task_ids, state_name)
    if not segs:
        raise ValueError(f"No segments for {state_name}")
    for seg in segs:
        weights[seg] += 1.0 / (len(segs) * len(seg))
    weights /= weights.sum()
    return weights


def streaming_weighted_mean(
    ds: h5py.Dataset,
    indices: np.ndarray,
    weights: np.ndarray,
    mode: str,
    batch_size: int,
) -> np.ndarray:
    d = vector_dim(ds, mode)
    mu = np.zeros(d, dtype=np.float64)
    total = 0.0
    for start, end, chunk in iter_embedding_chunks(ds, indices, mode, batch_size):
        w = weights[start:end]
        if not np.isfinite(chunk).all():
            raise ValueError("Non-finite embedding in weighted mean")
        mu += np.sum(chunk.astype(np.float64) * w[:, None], axis=0)
        total += float(np.sum(w))
    if total <= 0:
        raise ValueError("zero SVD weight")
    return mu / total


def streaming_randomized_svd(
    ds: h5py.Dataset,
    indices: np.ndarray,
    weights: np.ndarray,
    mean: np.ndarray,
    mode: str,
    rank: int,
    batch_size: int,
    seed: int,
    tempdir: Path,
    oversamples: int,
    power_iterations: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    n = len(indices)
    d = vector_dim(ds, mode)
    algebraic_max = int(min(n - 1, d))
    if rank > algebraic_max:
        raise ValueError(f"requested SVD rank {rank} exceeds algebraic maximum {algebraic_max} (n={n}, d={d})")
    k = int(rank)
    if k < 1:
        raise ValueError("invalid SVD rank")
    ell = int(min(n, d, k + max(0, oversamples)))
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((d, ell)).astype(np.float32)
    path = tempdir / f"svd_range_{os.getpid()}_{seed}.dat"
    Y = np.memmap(path, mode="w+", dtype=np.float32, shape=(n, ell))
    total_ss = 0.0
    for start, end, chunk in iter_embedding_chunks(ds, indices, mode, batch_size):
        Xc = chunk - mean[None, :]
        sw = np.sqrt(np.maximum(weights[start:end], 0)).astype(np.float32)
        A = Xc * sw[:, None]
        Y[start:end] = A @ omega
        total_ss += float(np.sum(A.astype(np.float64) ** 2))
    Y.flush()
    del omega
    for it in range(max(0, power_iterations)):
        Z = np.zeros((d, ell), dtype=np.float64)
        for start, end, chunk in iter_embedding_chunks(ds, indices, mode, batch_size):
            A = (chunk - mean[None, :]) * np.sqrt(np.maximum(weights[start:end], 0))[:, None]
            Z += A.T.astype(np.float64) @ np.asarray(Y[start:end], dtype=np.float64)
        Zq, _ = np.linalg.qr(Z, mode="reduced")
        Zq = Zq.astype(np.float32)
        for start, end, chunk in iter_embedding_chunks(ds, indices, mode, batch_size):
            A = (chunk - mean[None, :]) * np.sqrt(np.maximum(weights[start:end], 0))[:, None]
            Y[start:end] = A @ Zq
        Y.flush()
        log(f"      centered SVD power iteration {it + 1}/{power_iterations}")
    Q, _ = np.linalg.qr(np.asarray(Y), mode="reduced")
    Q = Q.astype(np.float32, copy=False)
    del Y
    try:
        path.unlink()
    except OSError:
        pass
    B = np.zeros((Q.shape[1], d), dtype=np.float64)
    for start, end, chunk in iter_embedding_chunks(ds, indices, mode, batch_size):
        A = (chunk - mean[None, :]) * np.sqrt(np.maximum(weights[start:end], 0))[:, None]
        B += Q[start:end].T.astype(np.float64) @ A.astype(np.float64)
    _, singular, vt = np.linalg.svd(B, full_matrices=False)
    singular = singular[:k]
    # Keep the final reducer in float64.  The range finder may use float32, but
    # exact support-restricted SFA should not throw away another ~9 digits at
    # the point where covariance conditioning matters most.
    vt = np.asarray(vt[:k], dtype=np.float64)
    return singular, vt, {
        "n_rows": n, "original_dim": d, "rank": k, "maximum_algebraic_rank": algebraic_max,
        "smallest_retained_singular_value": float(singular[-1]),
        "largest_retained_singular_value": float(singular[0]),
        "retained_condition_number": float(singular[0] / max(singular[-1], 1e-30)),
        "captured_fraction": float(np.sum(singular ** 2) / max(total_ss, 1e-30)),
    }


def project_grid(
    ds: h5py.Dataset,
    grid: GridData,
    mean: np.ndarray,
    vt: np.ndarray,
    mode: str,
    batch_size: int,
) -> np.ndarray:
    out = np.empty((len(grid.global_indices), vt.shape[0]), dtype=np.float64)
    for start, end, chunk in iter_embedding_chunks(ds, grid.global_indices, mode, batch_size):
        out[start:end] = (
            np.asarray(chunk, dtype=np.float64) - np.asarray(mean, dtype=np.float64)[None, :]
        ) @ np.asarray(vt, dtype=np.float64).T
    return out


# =============================================================================
# SFA moments, fits, nulls, and held-out evaluation
# =============================================================================


def weighted_covariance(coords: np.ndarray, weights: np.ndarray, rank: int) -> Tuple[np.ndarray, np.ndarray]:
    w = np.asarray(weights, dtype=np.float64)
    if len(w) != len(coords):
        raise ValueError("weight length mismatch")
    total = float(np.sum(w))
    if total <= 0:
        raise ValueError("zero covariance weight")
    X = np.asarray(coords[:, :rank], dtype=np.float64)
    mu = np.sum(X * w[:, None], axis=0) / total
    D = X - mu[None, :]
    cov = (D * w[:, None]).T @ D / total
    return mu, symmetrize(cov)


def require_exact_spd(cov: np.ndarray, name: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """Validate an exact SPD covariance without changing its geometry.

    No identity shrinkage, diagonal loading, eigenvalue clipping, pseudoinverse,
    or jitter is permitted.  Positive definiteness is invariant under congruence,
    so an infeasible rank is reported rather than repaired in a coordinate-dependent
    way.  The returned whitening matrix W satisfies W.T @ cov @ W = I.
    """
    cov = symmetrize(np.asarray(cov, dtype=np.float64))
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"{name} must be square, got {cov.shape}")
    if not np.all(np.isfinite(cov)):
        raise ValueError(f"{name} contains non-finite values")
    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError as exc:
        eig = np.linalg.eigvalsh(cov)
        raise ValueError(
            f"{name} is not positive definite at the requested rank; "
            f"min_eigenvalue={float(eig[0]):.6g}, "
            f"max_eigenvalue={float(eig[-1]):.6g}. "
            "Exact coordinate-free SFA forbids identity loading; lower the SVD rank."
        ) from exc
    eig = np.linalg.eigvalsh(cov)
    eye = np.eye(cov.shape[0], dtype=np.float64)
    whitener = np.linalg.solve(chol.T, eye)
    residual = float(
        np.linalg.norm(whitener.T @ cov @ whitener - eye, ord="fro")
        / max(np.sqrt(cov.shape[0]), 1.0)
    )
    if not np.isfinite(residual) or residual > 1e-7:
        raise ValueError(
            f"{name} exact whitening residual is too large: {residual:.6g}"
        )
    return eig, whitener, residual


def pair_covariance(
    coords: np.ndarray,
    segments: Sequence[np.ndarray],
    rank: int,
    keys: Optional[Sequence[Tuple[int, int, int]]] = None,
    group_by_run: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if not segments:
        raise ValueError("no adjacency segments")
    cov = np.zeros((rank, rank), dtype=np.float64)
    total_groups = 0
    n_pairs = 0
    if group_by_run:
        if keys is None or len(keys) != len(segments):
            raise ValueError("run keys required")
        grouped: Dict[Tuple[int, int, int], List[np.ndarray]] = defaultdict(list)
        for key, seg in zip(keys, segments):
            grouped[key].append(seg)
        for run_segments in grouped.values():
            run_cov = np.zeros((rank, rank), dtype=np.float64)
            run_pairs = 0
            for seg in run_segments:
                D = coords[seg[1:], :rank].astype(np.float64) - coords[seg[:-1], :rank].astype(np.float64)
                run_cov += D.T @ D
                run_pairs += len(D)
            if run_pairs > 0:
                cov += run_cov / run_pairs
                total_groups += 1
                n_pairs += run_pairs
    else:
        for seg in segments:
            D = coords[seg[1:], :rank].astype(np.float64) - coords[seg[:-1], :rank].astype(np.float64)
            if len(D):
                cov += (D.T @ D) / len(D)
                total_groups += 1
                n_pairs += len(D)
    if total_groups <= 0:
        raise ValueError("no valid adjacency pairs")
    return symmetrize(cov / total_groups), {
        "n_segments": len(segments), "n_groups": total_groups, "n_pairs": n_pairs,
    }


def vectorwise_temporal_permutation_surrogate_coords(
    coords: np.ndarray,
    segments: Sequence[np.ndarray],
    rank: int,
    rng: np.random.Generator,
    strata: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Destroy within-stratum temporal order with shared row permutations.

    Every coordinate receives the same permutation.  For Global mode, ``strata``
    is the five-state label, so Baseline and Task point clouds are never mixed by
    the null.  This removes the artificial advantage caused by permuting across a
    genuine Baseline-to-Task mean step.  Within-state modes are unchanged because
    their segments are already homogeneous.

    For every invertible ambient map ``X_new = X @ A`` and identical RNG state,

        surrogate(X_new) = surrogate(X) @ A.

    Thus the phase-stratified null remains exactly coordinate-equivariant.
    """
    source = np.asarray(coords[:, :rank], dtype=np.float64)
    surrogate = source.copy()
    strata_arr = None if strata is None else np.asarray(strata)
    if strata_arr is not None and len(strata_arr) != len(source):
        raise ValueError("strata length mismatch")

    for seg_raw in segments:
        seg = np.asarray(seg_raw, dtype=np.int64)
        if len(seg) < 3:
            continue
        if strata_arr is None:
            blocks = [seg]
        else:
            labels = strata_arr[seg]
            breaks = np.flatnonzero(labels[1:] != labels[:-1]) + 1
            blocks = [b for b in np.split(seg, breaks) if len(b)]
        for block in blocks:
            length = len(block)
            if length < 3:
                continue
            permutation = rng.permutation(length)
            if np.array_equal(permutation, np.arange(length)):
                permutation = np.roll(permutation, 1)
            surrogate[block] = source[block][permutation]
    return surrogate


def permuted_pair_covariance(
    coords: np.ndarray,
    segments: Sequence[np.ndarray],
    rank: int,
    rng: np.random.Generator,
    keys: Optional[Sequence[Tuple[int, int, int]]] = None,
    group_by_run: bool = False,
    strata: Optional[np.ndarray] = None,
) -> np.ndarray:
    surrogate = vectorwise_temporal_permutation_surrogate_coords(
        coords, segments, rank, rng, strata=strata
    )
    return pair_covariance(surrogate, segments, rank, keys, group_by_run)[0]


@dataclass
class SFAGeometry:
    ambient_rank: int
    support_rank: int
    support_tolerance: float
    support_basis: np.ndarray
    covariance_regularized_support: np.ndarray
    covariance_eigenvalues_regularized: np.ndarray
    support_whitener: np.ndarray
    whitening_full: np.ndarray


@dataclass
class SFAFit:
    mode_name: str
    rank: int
    mean: np.ndarray
    sigma_x: np.ndarray
    sigma_x_regularized: np.ndarray
    sigma_delta: np.ndarray
    covariance_eigenvalues: np.ndarray
    whitening: np.ndarray
    gammas: np.ndarray
    directions_z: np.ndarray
    white_axes: np.ndarray
    pair_info: Dict[str, Any]
    geometry: SFAGeometry


def prepare_sfa_geometry(sigma_x: np.ndarray) -> SFAGeometry:
    """Prepare exact SFA geometry on the empirical covariance support.

    Rank-deficient within-state covariances are not repaired with an identity
    metric. Instead SFA is solved on the intrinsic quotient by null(Cx), using
    the positive empirical support only. Mathematically this support and the
    finite generalized eigensystem are preserved by congruence. The numerical
    tolerance is recorded so folds near machine-rank boundaries remain auditable.
    """
    sigma_x = symmetrize(np.asarray(sigma_x, dtype=np.float64))
    if sigma_x.ndim != 2 or sigma_x.shape[0] != sigma_x.shape[1]:
        raise ValueError(f"state covariance must be square, got {sigma_x.shape}")
    if not np.all(np.isfinite(sigma_x)):
        raise ValueError("state covariance contains non-finite values")
    raw_eig, raw_Q = np.linalg.eigh(sigma_x)
    order = np.argsort(raw_eig)[::-1]
    raw_eig, raw_Q = raw_eig[order], raw_Q[:, order]
    leading = max(float(raw_eig[0]), 1e-30)
    tol = max(sigma_x.shape) * np.finfo(float).eps * leading * 100.0
    support = raw_eig > tol
    support_rank = int(np.sum(support))
    if support_rank < 1:
        raise ValueError(
            "state covariance has zero empirical support rank under the recorded "
            f"tolerance {tol:.6g}"
        )
    Qs = raw_Q[:, support]
    Cx_s = symmetrize(Qs.T @ sigma_x @ Qs)
    eig_s, Ws, residual = require_exact_spd(Cx_s, "state covariance on empirical support")
    return SFAGeometry(
        ambient_rank=int(sigma_x.shape[0]),
        support_rank=support_rank,
        support_tolerance=float(tol),
        support_basis=Qs,
        covariance_regularized_support=Cx_s.copy(),
        covariance_eigenvalues_regularized=eig_s[::-1].copy(),
        support_whitener=Ws,
        whitening_full=Qs @ Ws,
    )

def sfa_eigensystem_in_geometry(
    sigma_delta: np.ndarray,
    geometry: SFAGeometry,
    return_vectors: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Solve the derivative generalized eigensystem in exact Cx geometry."""
    Cd = symmetrize(np.asarray(sigma_delta, dtype=np.float64))
    if Cd.shape != (geometry.ambient_rank, geometry.ambient_rank):
        raise ValueError(
            f"derivative covariance shape {Cd.shape} does not match ambient "
            f"SVD rank {geometry.ambient_rank}"
        )
    W = geometry.whitening_full
    H = symmetrize(W.T @ Cd @ W)
    if return_vectors:
        gamma, V = np.linalg.eigh(H)
        order = np.argsort(gamma)
        gamma, V = gamma[order], V[:, order]
        U = W @ V
        for j in range(U.shape[1]):
            pivot = int(np.argmax(np.abs(U[:, j])))
            if U[pivot, j] < 0:
                U[:, j] *= -1
                V[:, j] *= -1
        return gamma, U, V
    gamma = np.linalg.eigvalsh(H)
    gamma.sort()
    return gamma, None, None


def solve_sfa(
    mode_name: str,
    mean: np.ndarray,
    sigma_x: np.ndarray,
    sigma_delta: np.ndarray,
    pair_info: Dict[str, Any],
) -> SFAFit:
    """Solve exact support-restricted SFA without identity regularization."""
    try:
        geometry = prepare_sfa_geometry(sigma_x)
    except ValueError as exc:
        raise ValueError(f"{mode_name}: {exc}") from exc
    gamma, U, V = sfa_eigensystem_in_geometry(
        sigma_delta, geometry, return_vectors=True
    )
    assert U is not None and V is not None
    pair_info = dict(pair_info)
    whitening_residual = float(
        np.linalg.norm(
            geometry.whitening_full.T @ sigma_x @ geometry.whitening_full
            - np.eye(geometry.support_rank),
            ord="fro",
        ) / max(np.sqrt(geometry.support_rank), 1.0)
    )
    pair_info.update({
        "sample_support_rank": geometry.support_rank,
        "support_rank_tolerance": geometry.support_tolerance,
        "ambient_svd_rank": geometry.ambient_rank,
        "identifiable_sf_axes": geometry.support_rank,
        "covariance_geometry": "exact_empirical_support_no_loading",
        "whitening_residual_fro_per_sqrt_dim": whitening_residual,
    })
    raw_eig = np.linalg.eigvalsh(symmetrize(sigma_x))[::-1]
    return SFAFit(
        mode_name=mode_name,
        rank=geometry.support_rank,
        mean=mean,
        sigma_x=sigma_x,
        sigma_x_regularized=sigma_x.copy(),
        sigma_delta=sigma_delta,
        covariance_eigenvalues=raw_eig,
        whitening=geometry.whitening_full,
        gammas=gamma,
        directions_z=U,
        white_axes=V,
        pair_info=pair_info,
        geometry=geometry,
    )

def fit_real_mode(
    coords: np.ndarray,
    grid: GridData,
    rank: int,
    sessions: Sequence[int],
    task_ids: Mapping[str, int],
    state_name: Optional[str],
) -> Tuple[SFAFit, List[np.ndarray], List[Tuple[int, int, int]], np.ndarray]:
    mode_name = state_name if state_name is not None else "Global"
    if state_name is None:
        point_weights = balanced_state_weights(grid, sessions, task_ids)
        segments, keys = collect_mode_segments(grid, sessions, task_ids, None)
        group_by_run = True
    else:
        point_weights = state_point_weights(grid, sessions, task_ids, state_name)
        segments, keys = collect_mode_segments(grid, sessions, task_ids, state_name)
        group_by_run = False
    mean, Cx = weighted_covariance(coords, point_weights, rank)
    Cd, pair_info = pair_covariance(coords, segments, rank, keys, group_by_run)
    fit = solve_sfa(mode_name, mean, Cx, Cd, pair_info)
    return fit, segments, keys, point_weights


def training_shuffle_null(
    coords: np.ndarray,
    fit: SFAFit,
    segments: Sequence[np.ndarray],
    keys: Sequence[Tuple[int, int, int]],
    n_shuffle: int,
    seed: int,
    group_by_run: bool,
    strata: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Vector-wise time-permutation null in the fixed exact geometry.

    Surrogates are generated in the full ambient SVD space and solved using the
    frozen real-data covariance geometry. The shared row permutation commutes
    exactly with every invertible ambient coordinate transformation.
    """
    ambient_rank = int(fit.geometry.ambient_rank)
    support_rank = int(fit.geometry.support_rank)
    out = np.full((n_shuffle, support_rank), np.nan, dtype=np.float64)
    rng = np.random.default_rng(seed)
    for b in range(n_shuffle):
        Cd = permuted_pair_covariance(
            coords, segments, ambient_rank, rng, keys, group_by_run, strata=strata
        )
        gamma, _, _ = sfa_eigensystem_in_geometry(
            Cd, fit.geometry, return_vectors=False
        )
        out[b, :len(gamma)] = gamma
    return out

def select_99_temporal_order_slow_advantage(
    gammas: np.ndarray,
    null: np.ndarray,
    coverage: float,
    axis_alpha: float,
) -> Dict[str, Any]:
    if null.size == 0:
        return {
            "null_median": np.full_like(gammas, np.nan),
            "advantage": np.zeros_like(gammas),
            "cumulative_fraction": np.zeros_like(gammas),
            "lower_p": np.full_like(gammas, np.nan),
            "retained_rank_raw": 0,
            "retained_rank": 0,
            "significant_advantage": np.zeros_like(gammas),
        }
    null_median = np.nanmedian(null, axis=0)[:len(gammas)]
    advantage = np.maximum(0.0, 1.0 - gammas / np.maximum(null_median, 1e-30))
    total = float(np.sum(advantage))
    cumulative = np.cumsum(advantage) / total if total > 0 else np.zeros_like(advantage)
    retained_raw = int(np.searchsorted(cumulative, coverage, side="left") + 1) if total > 0 else 0
    retained_raw = min(retained_raw, len(gammas))
    p = np.array([empirical_lower_p(float(g), null[:, k]) for k, g in enumerate(gammas)])
    significant_advantage = np.where(p <= axis_alpha, advantage, 0.0)
    sig_total = float(np.sum(significant_advantage))
    sig_cumulative = (
        np.cumsum(significant_advantage) / sig_total
        if sig_total > 0 else np.zeros_like(significant_advantage)
    )
    retained = int(np.searchsorted(sig_cumulative, coverage, side="left") + 1) if sig_total > 0 else 0
    retained = min(retained, len(gammas))
    return {
        "null_median": null_median,
        "advantage": advantage,
        "cumulative_fraction": cumulative,
        "lower_p": p,
        "significant_advantage": significant_advantage,
        "significant_cumulative_fraction": sig_cumulative,
        "retained_rank_raw": retained_raw,
        "retained_rank": retained,
    }


def rayleigh_frozen_axes(
    coords: np.ndarray,
    axes_z: np.ndarray,
    point_weights: np.ndarray,
    segments: Sequence[np.ndarray],
    keys: Sequence[Tuple[int, int, int]],
    group_by_run: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if axes_z.shape[1] == 0:
        return np.array([]), np.array([]), np.array([])
    Y = (coords[:, :axes_z.shape[0]] - 0.0) @ axes_z
    w = point_weights / max(float(point_weights.sum()), 1e-30)
    mu = np.sum(Y * w[:, None], axis=0)
    denom = np.sum(((Y - mu[None, :]) ** 2) * w[:, None], axis=0)
    numer = np.zeros(axes_z.shape[1], dtype=np.float64)
    n_groups = 0
    if group_by_run:
        grouped: Dict[Tuple[int, int, int], List[np.ndarray]] = defaultdict(list)
        for key, seg in zip(keys, segments):
            grouped[key].append(seg)
        for run_segments in grouped.values():
            values = []
            for seg in run_segments:
                D = Y[seg[1:]] - Y[seg[:-1]]
                if len(D):
                    values.append(D ** 2)
            if values:
                numer += np.mean(np.concatenate(values, axis=0), axis=0)
                n_groups += 1
    else:
        for seg in segments:
            D = Y[seg[1:]] - Y[seg[:-1]]
            if len(D):
                numer += np.mean(D ** 2, axis=0)
                n_groups += 1
    numer /= max(n_groups, 1)
    return numer / np.maximum(denom, 1e-30), numer, denom


def heldout_shuffle_null(
    coords: np.ndarray,
    axes_z: np.ndarray,
    point_weights: np.ndarray,
    segments: Sequence[np.ndarray],
    keys: Sequence[Tuple[int, int, int]],
    n_shuffle: int,
    seed: int,
    group_by_run: bool,
    strata: Optional[np.ndarray] = None,
) -> np.ndarray:
    if axes_z.shape[1] == 0:
        return np.empty((n_shuffle, 0))
    w = point_weights / max(float(point_weights.sum()), 1e-30)
    # Freeze the held-out point-cloud denominator. A temporal-order null should
    # alter adjacency, not the covariance metric against which jumps are judged.
    Y_real = np.asarray(coords[:, :axes_z.shape[0]], dtype=np.float64) @ axes_z
    mu_real = np.sum(Y_real * w[:, None], axis=0)
    denom_real = np.sum(((Y_real - mu_real[None, :]) ** 2) * w[:, None], axis=0)
    out = np.full((n_shuffle, axes_z.shape[1]), np.nan)
    rng = np.random.default_rng(seed)
    for b in range(n_shuffle):
        surrogate = vectorwise_temporal_permutation_surrogate_coords(
            coords, segments, axes_z.shape[0], rng, strata=strata
        )
        Y = surrogate @ axes_z
        numer = np.zeros(axes_z.shape[1], dtype=np.float64)
        n_groups = 0
        if group_by_run:
            grouped: Dict[Tuple[int, int, int], List[np.ndarray]] = defaultdict(list)
            for key, seg in zip(keys, segments):
                grouped[key].append(seg)
            for run_segments in grouped.values():
                vals = []
                for seg in run_segments:
                    D = Y[seg[1:]] - Y[seg[:-1]]
                    if len(D): vals.append(D ** 2)
                if vals:
                    numer += np.mean(np.concatenate(vals, axis=0), axis=0)
                    n_groups += 1
        else:
            for seg in segments:
                D = Y[seg[1:]] - Y[seg[:-1]]
                if len(D):
                    numer += np.mean(D ** 2, axis=0)
                    n_groups += 1
        numer /= max(n_groups, 1)
        out[b] = numer / np.maximum(denom_real, 1e-30)
    return out


def common_metric_coordinates(directions_z: np.ndarray, common_fit: SFAFit) -> np.ndarray:
    """Map ambient z-directions into Euclidean coordinates under global Cx metric."""
    if directions_z.shape[1] == 0:
        return np.empty((common_fit.sigma_x.shape[0], 0), dtype=float)
    eig, Q = np.linalg.eigh(symmetrize(common_fit.sigma_x_regularized))
    eig = np.maximum(eig, 0.0)
    Csqrt = (Q * np.sqrt(eig)[None, :]) @ Q.T
    return Csqrt @ directions_z


# =============================================================================
# Within/global semantic overlap
# =============================================================================


def semantic_overlap(
    global_fit: SFAFit,
    global_rank: int,
    within_fits: Mapping[str, SFAFit],
    within_ranks: Mapping[str, int],
    degeneracy_gap: float,
    max_coefficient_global_axes: int = 20,
) -> Dict[str, Any]:
    """Interpret Global-SFA axes using the named within-state subspaces.

    The v5.0 implementation repeatedly recomputed the same large SVD once for
    every global axis and state.  This implementation performs one SVD for the
    full dictionary, one leave-one-state-out SVD per state, and reuses the
    global covariance square root.  The algebra is identical; only the
    execution graph changes.
    """
    eig, Q = np.linalg.eigh(symmetrize(global_fit.sigma_x_regularized))
    eig = np.maximum(eig, 0.0)
    Csqrt = (Q * np.sqrt(eig)[None, :]) @ Q.T

    G = Csqrt @ np.asarray(global_fit.directions_z[:, :global_rank], dtype=np.float64)
    Qg = orthonormalize_columns(G)
    state_axes: Dict[str, np.ndarray] = {}
    state_Q: Dict[str, np.ndarray] = {}
    dictionary_cols: List[np.ndarray] = []
    dictionary_names: List[str] = []
    for state in STATE_NAMES:
        fit = within_fits[state]
        r = int(within_ranks[state])
        A = Csqrt @ np.asarray(fit.directions_z[:, :r], dtype=np.float64)
        state_axes[state] = A
        state_Q[state] = orthonormalize_columns(A)
        for k in range(A.shape[1]):
            col = A[:, k]
            n = float(np.linalg.norm(col))
            dictionary_cols.append(col / n if n > 1e-12 else col)
            dictionary_names.append(f"{STATE_SHORT[STATE_NAMES.index(state)]}_SF{k + 1:03d}")

    D = (
        np.column_stack(dictionary_cols)
        if dictionary_cols
        else np.empty((global_fit.sigma_x.shape[0], 0), dtype=np.float64)
    )
    if D.shape[1]:
        U, s, Vh = np.linalg.svd(D, full_matrices=False)
        keep = s > 1e-10 * max(float(s[0]), 1.0)
        Qunion = U[:, keep]
        n_coeff = int(global_rank)
        if int(max_coefficient_global_axes) > 0:
            n_coeff = min(n_coeff, int(max_coefficient_global_axes))
        if np.any(keep) and n_coeff > 0:
            coefficients = (
                Vh[keep].T / s[keep][None, :]
            ) @ (U[:, keep].T @ G[:, :n_coeff])
            reconstruction = D @ coefficients
            reconstruction_error_head = np.linalg.norm(
                G[:, :n_coeff] - reconstruction, axis=0
            )
        else:
            coefficients = np.empty((D.shape[1], 0), dtype=np.float64)
            reconstruction_error_head = np.empty(0, dtype=np.float64)
    else:
        Qunion = np.empty((global_fit.sigma_x.shape[0], 0), dtype=np.float64)
        n_coeff = 0
        coefficients = np.empty((0, 0), dtype=np.float64)
        reconstruction_error_head = np.empty(0, dtype=np.float64)

    union_explained = (
        np.sum((Qunion.T @ G) ** 2, axis=0)
        if Qunion.shape[1]
        else np.zeros(global_rank, dtype=np.float64)
    )
    reconstruction_error = np.full(global_rank, np.nan, dtype=np.float64)
    reconstruction_error[:len(reconstruction_error_head)] = reconstruction_error_head

    # The leave-one-state-out spaces depend only on the omitted state, not on
    # the global axis.  Compute each of the five bases once.
    Qminus_by_state: Dict[str, np.ndarray] = {}
    for state in STATE_NAMES:
        others = [
            state_axes[s]
            for s in STATE_NAMES
            if s != state and state_axes[s].shape[1]
        ]
        Qminus_by_state[state] = (
            orthonormalize_columns(np.column_stack(others))
            if others
            else np.empty((G.shape[0], 0), dtype=np.float64)
        )

    axis_rows: List[Dict[str, Any]] = []
    unique_rows: List[Dict[str, Any]] = []
    coefficient_rows: List[Dict[str, Any]] = []
    for j in range(global_rank):
        g = G[:, j]
        for state in STATE_NAMES:
            Qc = state_Q[state]
            marginal = float(np.sum((Qc.T @ g) ** 2)) if Qc.shape[1] else 0.0
            Qminus = Qminus_by_state[state]
            minus = float(np.sum((Qminus.T @ g) ** 2)) if Qminus.shape[1] else 0.0
            unique = max(0.0, float(union_explained[j]) - minus)
            axis_rows.append({
                "global_axis": f"G_SF{j + 1:03d}", "global_axis_index": j + 1,
                "state": state, "marginal_overlap": marginal,
                "union_explained_fraction": float(union_explained[j]),
                "residual_fraction": max(0.0, 1.0 - float(union_explained[j])),
            })
            unique_rows.append({
                "global_axis": f"G_SF{j + 1:03d}", "global_axis_index": j + 1,
                "state": state, "leave_one_state_out_unique": unique,
            })
        if j < coefficients.shape[1]:
            for m, name in enumerate(dictionary_names):
                coefficient_rows.append({
                    "named_within_axis": name,
                    "global_axis": f"G_SF{j + 1:03d}",
                    "minimum_norm_coefficient": float(coefficients[m, j]),
                })

    block_rows: List[Dict[str, Any]] = []
    blocks = make_degenerate_blocks(global_fit.gammas, global_rank, degeneracy_gap)
    for bidx, block in enumerate(blocks, start=1):
        Qb = orthonormalize_columns(G[:, block])
        for state in STATE_NAMES:
            Qc = state_Q[state]
            overlap = (
                float(np.sum((Qc.T @ Qb) ** 2) / max(Qb.shape[1], 1))
                if Qc.shape[1] and Qb.shape[1] else 0.0
            )
            block_rows.append({
                "block_index": bidx,
                "global_axes": ",".join(f"G_SF{k + 1:03d}" for k in block),
                "block_dimension": len(block), "state": state,
                "normalized_projector_overlap": overlap,
            })

    angles = principal_angles_deg(Qg, Qunion)
    return {
        "global_axis_state_overlap": axis_rows,
        "global_axis_unique_contribution": unique_rows,
        "minimum_norm_coefficients": coefficient_rows,
        "coefficient_global_axes_computed": int(coefficients.shape[1]),
        "block_overlap": block_rows,
        "dictionary_names": dictionary_names,
        "dictionary_matrix": D,
        "global_common_axes": G,
        "state_common_axes": state_axes,
        "union_basis": Qunion,
        "principal_angles_deg": angles,
        "reconstruction_error": reconstruction_error,
        "union_explained": union_explained,
        "blocks": blocks,
    }


# =============================================================================
# Experiment B -> C handoff
# =============================================================================


B_TO_C_SCHEMA_VERSION = "expB-global-slow-space-handoff-v4"


def block_complete_prefix_rank(
    gammas: np.ndarray,
    requested_rank: int,
    relative_gap: float,
) -> Tuple[int, List[List[int]], Optional[int]]:
    """Extend a pre-registered fixed-k prefix to a complete spectral block."""
    g = np.asarray(gammas, dtype=np.float64).reshape(-1)
    blocks = make_degenerate_blocks(g, len(g), relative_gap)
    r = int(min(max(requested_rank, 0), len(g)))
    if r == 0:
        return 0, blocks, None
    boundary_axis = r - 1
    for block_index, block_axes in enumerate(blocks):
        if boundary_axis in block_axes:
            return int(max(block_axes) + 1), blocks, int(block_index)
    raise RuntimeError("requested slow prefix was not found in the degeneracy blocks")


def export_b_to_c_handoff(
    outdir: Path,
    model: str,
    subject: int,
    grid: GridData,
    heldout_session: int,
    train_sessions: Sequence[int],
    task_ids: Mapping[str, int],
    svd_rank_requested: int,
    svd_rank_effective: int,
    embedding_mean: np.ndarray,
    centered_svd_vt: np.ndarray,
    coords: np.ndarray,
    global_fit: SFAFit,
    global_selection: Mapping[str, Any],
    degeneracy_gap: float,
    handoff_top_k: int,
) -> Dict[str, Any]:
    """Export the frozen global slow space without changing Experiment B.

    The coordinate map is

        x -> z = (x - embedding_mean) @ centered_svd_vt.T
             -> y = (z - global_sfa_mean) @ global_sfa_directions.

    Experiment C receives a pre-registered fixed leading-k slow prefix.  The
    r99 temporal-order statistic is exported only as an auxiliary diagnostic and
    never selects the C subspace.  If fixed k cuts through a genuinely
    near-degenerate gamma block, the full slow block is retained.  The fast
    control is completed independently in C; inability to match its boundary
    exactly must not erase an otherwise valid primary slow-space analysis.
    """
    retained_rank = int(global_selection.get("retained_rank", 0))
    requested_fixed_k = int(min(max(int(handoff_top_k), 1), int(global_fit.rank)))
    candidate_rank, blocks, boundary_block = block_complete_prefix_rank(
        global_fit.gammas, requested_fixed_k, degeneracy_gap
    )
    suffix_rank_to_block = {
        int(len(global_fit.gammas) - min(block)): i
        for i, block in enumerate(blocks) if block
    }
    fast_boundary_block = suffix_rank_to_block.get(int(candidate_rank))
    if candidate_rank > int(global_fit.rank):
        raise RuntimeError("C handoff candidate rank exceeds global SFA support")
    if centered_svd_vt.shape[0] != svd_rank_effective:
        raise ValueError("SVD basis row count does not match effective rank")
    if global_fit.directions_z.shape[0] != svd_rank_effective:
        raise ValueError("Global SFA directions do not match the SVD coordinates")
    if coords.shape[1] < svd_rank_effective:
        raise ValueError("Projected subject coordinates are narrower than the fold rank")

    axes = np.asarray(
        global_fit.directions_z[:, :candidate_rank], dtype=np.float64
    )
    centered = (
        np.asarray(coords[:, :svd_rank_effective], dtype=np.float64)
        - np.asarray(global_fit.mean, dtype=np.float64)[None, :]
    )
    slow = np.asarray(centered @ axes, dtype=np.float32)
    state = state_label_for_positions(grid, task_ids)
    if np.any(state < 0):
        bad = np.flatnonzero(state < 0)[:10].tolist()
        raise ValueError(f"Invalid five-state labels in C handoff at positions {bad}")
    if not np.all(np.isfinite(slow)):
        raise FloatingPointError("Non-finite global slow coordinates in C handoff")

    train_mask = np.isin(grid.session, np.asarray(train_sessions, dtype=np.int64))
    heldout_mask = grid.session == int(heldout_session)
    train_counts = np.bincount(state[train_mask], minlength=5)
    heldout_counts = np.bincount(state[heldout_mask], minlength=5)
    if np.any(train_counts == 0):
        raise ValueError(f"B-to-C training ledger misses a state: {train_counts.tolist()}")
    if np.any(heldout_counts == 0):
        raise ValueError(f"B-to-C held-out ledger misses a state: {heldout_counts.tolist()}")
    if np.any(train_mask & heldout_mask) or not np.all(train_mask | heldout_mask):
        raise RuntimeError("B-to-C train/held-out session ledger is inconsistent")

    transform_path = outdir / "B_to_C_transform.npz"
    coordinates_path = outdir / "B_to_C_coordinates.npz"
    manifest_path = outdir / "B_to_C_manifest.json"

    npz_write_atomic(
        transform_path,
        embedding_mean=np.asarray(embedding_mean, dtype=np.float64),
        centered_svd_vt=np.asarray(
            centered_svd_vt[:svd_rank_effective], dtype=np.float32
        ),
        global_sfa_mean_svd=np.asarray(global_fit.mean, dtype=np.float64),
        global_sfa_directions_svd_candidate=np.asarray(axes, dtype=np.float64),
        global_sfa_directions_svd_full=np.asarray(global_fit.directions_z, dtype=np.float64),
        global_sfa_gammas_full=np.asarray(global_fit.gammas, dtype=np.float64),
        candidate_axis_indices=np.arange(candidate_rank, dtype=np.int64),
        retained_rank_r99=np.asarray([retained_rank], dtype=np.int64),
        handoff_rank_fixed_topk_requested=np.asarray([requested_fixed_k], dtype=np.int64),
        candidate_rank_fixed_topk_block_complete=np.asarray([candidate_rank], dtype=np.int64),
        candidate_rank_block_complete=np.asarray([candidate_rank], dtype=np.int64),
    )
    npz_write_atomic(
        coordinates_path,
        global_slow_coordinates=np.asarray(slow, dtype=np.float32),
        svd_coordinates=np.asarray(coords[:, :svd_rank_effective], dtype=np.float32),
        global_index=np.asarray(grid.global_indices, dtype=np.int64),
        subject_id=np.asarray(grid.subject, dtype=np.int64),
        session_id=np.asarray(grid.session, dtype=np.int64),
        run_id=np.asarray(grid.run, dtype=np.int64),
        phase_id=np.asarray(grid.phase, dtype=np.int64),
        task_id=np.asarray(grid.task, dtype=np.int64),
        task_family_id=np.asarray(grid.family, dtype=np.int64),
        state_label=np.asarray(state, dtype=np.int64),
        sample_start=np.asarray(grid.sample_start, dtype=np.int64),
        sample_end=np.asarray(grid.sample_end, dtype=np.int64),
        center_sample=np.asarray(grid.center_sample, dtype=np.int64),
        is_training_session=np.asarray(train_mask, dtype=np.uint8),
        is_heldout_session=np.asarray(heldout_mask, dtype=np.uint8),
    )

    transform_sha256 = sha256_file(transform_path)
    coordinates_sha256 = sha256_file(coordinates_path)
    selected_block = blocks[boundary_block] if boundary_block is not None else []
    selected_fast_block = blocks[fast_boundary_block] if fast_boundary_block is not None else []
    manifest = {
        "schema_version": B_TO_C_SCHEMA_VERSION,
        "script_version": SCRIPT_VERSION,
        "scientific_status": (
            "unavailable_fixed_topk_saturates_ambient"
            if candidate_rank >= int(svd_rank_effective)
            else ("available_fixed_topk" if candidate_rank > 0
                  else "unavailable_no_fixed_topk_candidate")
        ),
        "model": model,
        "subject_id": int(subject),
        "grid": grid.name,
        "heldout_session": int(heldout_session),
        "training_sessions": [int(x) for x in train_sessions],
        "svd_rank_requested": int(svd_rank_requested),
        "svd_rank_effective": int(svd_rank_effective),
        "global_sfa_support_rank": int(global_fit.rank),
        "covariance_geometry": "exact_empirical_support_no_loading",
        "temporal_null": "vectorwise_within_state_stratified_time_permutation",
        "temporal_null_h0": "within-state windows are exchangeable inside each contiguous run segment",
        "primary_B_inference": "cross_fold_fixed_topk_subspace_stability",
        "ambient_svd_rank": int(svd_rank_effective),
        "retained_rank_r99": retained_rank,
        "r99_role": "auxiliary_temporal_order_diagnostic_not_used_for_C",
        "handoff_selection_policy": "pre_registered_fixed_topk_then_slow_prefix_nonchaining_block_completion",
        "handoff_rank_fixed_topk_requested": requested_fixed_k,
        "candidate_rank_fixed_topk_block_complete": candidate_rank,
        "candidate_rank_block_complete": candidate_rank,
        "block_completion_added_axes": int(candidate_rank - requested_fixed_k),
        "degeneracy_relative_gap": float(degeneracy_gap),
        "boundary_degenerate_block_zero_based": selected_block,
        "fast_suffix_boundary_degenerate_block_zero_based": selected_fast_block,
        "slow_prefix_boundary_complete": bool(boundary_block is not None),
        "fast_suffix_equal_rank_boundary_complete": bool(fast_boundary_block is not None),
        "rank_matched_boundary_complete": bool(
            boundary_block is not None and fast_boundary_block is not None
        ),
        "all_degenerate_blocks_zero_based": blocks,
        "state_names": list(STATE_NAMES),
        "task_ids": {str(k): int(v) for k, v in task_ids.items()},
        "n_windows": int(len(grid.global_indices)),
        "n_training_windows": int(np.sum(train_mask)),
        "n_heldout_windows": int(np.sum(heldout_mask)),
        "state_counts_all": {
            STATE_NAMES[c]: int(np.sum(state == c)) for c in range(len(STATE_NAMES))
        },
        "state_counts_training": {
            STATE_NAMES[c]: int(np.sum((state == c) & train_mask))
            for c in range(len(STATE_NAMES))
        },
        "state_counts_heldout": {
            STATE_NAMES[c]: int(np.sum((state == c) & heldout_mask))
            for c in range(len(STATE_NAMES))
        },
        "files": {
            "transform": transform_path.name,
            "coordinates": coordinates_path.name,
        },
        "file_sha256": {
            "transform": transform_sha256,
            "coordinates": coordinates_sha256,
        },
        "coordinate_formula": (
            "z=(x-embedding_mean)@centered_svd_vt.T; "
            "global_slow=(z-global_sfa_mean_svd)@"
            "global_sfa_directions_svd_candidate"
        ),
        "leakage_boundary": (
            "embedding mean, SVD basis, global SFA mean/directions, r99 selection, "
            "and degenerate-block completion use the two training sessions only; "
            "the held-out session is projected by the frozen map"
        ),
        "notes": [
            "Only the Global SFA system is handed to Experiment C; within-state SFA systems remain B diagnostics.",
            "The candidate is a prefix because global gammas are ordered from slowest to fastest.",
            "The r99 boundary is extended only when needed to keep a near-degenerate gamma block intact.",
            "The coordinate file is the authoritative window ledger for C and preserves original H5 global indices.",
            "The full fold-local SVD coordinates and full Global-SFA basis are exported for ambient, PCA-prefix, fast-suffix, and random rank-matched controls.",
            "SFA uses exact covariance geometry on each mode's empirical support, with no identity loading or jitter.",
            "Training selection uses a vector-wise temporal-permutation null that is exactly equivariant under invertible ambient coordinate changes.",
        ],
    }
    json_dump(manifest_path, manifest)
    return manifest


# =============================================================================
# Plots
# =============================================================================


def fixed_topk_overlap_rows(
    global_fit: SFAFit,
    within_fits: Mapping[str, SFAFit],
    topk_values: Sequence[int],
) -> List[Dict[str, Any]]:
    """Fixed-k overlap with the global metric factorized only once."""
    rows: List[Dict[str, Any]] = []
    eig, Q = np.linalg.eigh(symmetrize(global_fit.sigma_x_regularized))
    eig = np.maximum(eig, 0.0)
    Csqrt = (Q * np.sqrt(eig)[None, :]) @ Q.T
    Gfull = Csqrt @ np.asarray(global_fit.directions_z, dtype=np.float64)
    global_cache: Dict[int, np.ndarray] = {}
    for state in STATE_NAMES:
        sf = within_fits[state]
        Sfull = Csqrt @ np.asarray(sf.directions_z, dtype=np.float64)
        state_cache: Dict[int, np.ndarray] = {}
        for requested in topk_values:
            k = int(min(requested, Gfull.shape[1], Sfull.shape[1]))
            if k < 1:
                continue
            if k not in global_cache:
                global_cache[k] = orthonormalize_columns(Gfull[:, :k])
            if k not in state_cache:
                state_cache[k] = orthonormalize_columns(Sfull[:, :k])
            G, S = global_cache[k], state_cache[k]
            singular = np.linalg.svd(G.T @ S, compute_uv=False)
            singular = np.clip(singular, 0.0, 1.0)
            angles = np.degrees(np.arccos(singular))
            overlap = float(np.sum(singular ** 2) / max(k, 1))
            rows.append({
                "state": state, "top_k_requested": int(requested), "top_k_effective": k,
                "projector_overlap_per_dimension": overlap,
                "mean_principal_angle_deg": float(np.mean(angles)) if len(angles) else float("nan"),
                "max_principal_angle_deg": float(np.max(angles)) if len(angles) else float("nan"),
            })
    return rows


def plot_spectrum(rows: Sequence[Mapping[str, Any]], path: Path, title: str, n_show: int = 20) -> None:
    rows = sorted(rows, key=lambda x: int(x["axis_index"]))[:n_show]
    if not rows:
        return
    x = np.array([int(r["axis_index"]) for r in rows])
    real = np.array([float(r["gamma_train"]) for r in rows])
    null = np.array([float(r["time_permutation_null_median_train"]) for r in rows])
    retained = np.array([bool(r["retained_99"]) for r in rows])
    fig, ax = plt.subplots(figsize=(6.0, 3.7))
    ax.plot(x, real, "o-", label="real")
    ax.plot(x, null, "s--", label="time-permutation median")
    if np.any(retained):
        ax.scatter(x[retained], real[retained], s=80, marker="o", facecolors="none", label="retained 99%")
    ax.set_xlabel("Slow-feature index")
    ax.set_ylabel("gamma / Rjump (lower is slower)")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_overlap(
    rows: Sequence[Mapping[str, Any]], path: Path, title: str, n_show: int = 30
) -> None:
    rows = list(rows)
    if not rows:
        return
    axes = sorted(set(str(r["global_axis"]) for r in rows))[:max(int(n_show), 1)]
    rows = [r for r in rows if str(r["global_axis"]) in set(axes)]
    matrix = np.zeros((len(axes), len(STATE_NAMES)))
    for r in rows:
        i = axes.index(str(r["global_axis"]))
        j = STATE_NAMES.index(str(r["state"]))
        matrix[i, j] = float(r["marginal_overlap"])
    fig, ax = plt.subplots(figsize=(6.0, max(3.0, 0.45 * len(axes) + 1.5)))
    image = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(STATE_SHORT)))
    ax.set_xticklabels(STATE_SHORT)
    ax.set_yticks(np.arange(len(axes)))
    ax.set_yticklabels(axes)
    ax.set_title(title, fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


# =============================================================================
# Fold / subject / model drivers
# =============================================================================


def b_rank_fingerprint(
    ds: h5py.Dataset,
    grid_name: str,
    subject: int,
    heldout_session: int,
    rank_req: int,
    task_ids: Mapping[str, int],
    args: argparse.Namespace,
) -> str:
    dataset_path = Path(str(ds.file.filename))
    try:
        stat = dataset_path.stat()
        dataset_identity = {
            "path": str(dataset_path.resolve()),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
    except OSError:
        dataset_identity = {"path": str(dataset_path)}
    payload = {
        "script_version": SCRIPT_VERSION,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dataset": dataset_identity,
        "model": str(getattr(args, "current_model", "unknown")),
        "subject": int(subject),
        "grid": str(grid_name),
        "heldout_session": int(heldout_session),
        "rank": int(rank_req),
        "task_ids": {str(k): int(v) for k, v in task_ids.items()},
        "vectorization": str(args.vectorization),
        "min_phase_windows_raw": int(args.min_phase_windows_raw),
        "min_adjacencies_per_run": int(args.min_adjacencies_per_run),
        "decimation_offset": int(args.decimation_offset),
        "stride_tolerance_sec": float(args.stride_tolerance_sec),
        "raw_only_main_rank": bool(args.raw_only_main_rank),
        "svd_oversamples": int(args.svd_oversamples),
        "svd_power_iterations": int(args.svd_power_iterations),
        "slow_advantage_coverage": float(args.slow_advantage_coverage),
        "training_axis_alpha": float(args.training_axis_alpha),
        "max_retained_axes": int(args.max_retained_axes),
        "degeneracy_gap": float(args.degeneracy_gap),
        "c_handoff_top_k": int(args.c_handoff_top_k),
        "disable_c_handoff": bool(args.disable_c_handoff),
        "top_k": [int(x) for x in args.topk_list],
        "n_train_shuffle": int(args.n_train_shuffle),
        "n_heldout_shuffle": int(args.n_heldout_shuffle),
        "raw_n_train_shuffle": int(args.raw_n_train_shuffle),
        "raw_n_heldout_shuffle": int(args.raw_n_heldout_shuffle),
        "seed": int(args.seed),
        "max_coefficient_global_axes": int(args.max_coefficient_global_axes),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def terminal_marker_matches(path: Path, fingerprint: str) -> bool:
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        return (
            str(payload.get("script_version", "")) == SCRIPT_VERSION
            and str(payload.get("rank_fingerprint", "")) == str(fingerprint)
        )
    except Exception:
        return False


def load_completed_rank_result(outdir: Path) -> Dict[str, Any]:
    summary_path = outdir / "fold_rank_summary.json"
    npz_path = outdir / "slow_geometry_and_nulls.npz"
    required = [
        summary_path,
        npz_path,
        outdir / "slow_spectra.csv",
        outdir / "heldout_slowness.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Incomplete resumed B rank; missing {missing}")
    with summary_path.open("r", encoding="utf-8") as fh:
        summary = json.load(fh)
    with np.load(npz_path, allow_pickle=False) as z:
        vt = np.asarray(z["centered_svd_vt"], dtype=np.float64)
        fits: Dict[str, Any] = {}
        retained: Dict[str, int] = {}
        heldout_nulls: Dict[str, np.ndarray] = {}
        for state in STATE_NAMES:
            tag = safe_name(state)
            directions = np.asarray(z[f"{tag}_directions_z"], dtype=np.float64)
            fits[state] = type("ResumedSFAFit", (), {
                "directions_z": directions,
                "rank": int(directions.shape[1]),
            })()
            retained[state] = int(np.asarray(z[f"{tag}_retained_rank_99"]).reshape(-1)[0])
            heldout_nulls[state] = np.asarray(
                z[f"{tag}_heldout_time_permutation_null"], dtype=np.float64
            )
        global_directions = np.asarray(z["global_directions_z"], dtype=np.float64)
        fits["Global"] = type("ResumedSFAFit", (), {
            "directions_z": global_directions,
            "rank": int(global_directions.shape[1]),
        })()
        retained["Global"] = int(np.asarray(z["global_retained_rank_99"]).reshape(-1)[0])
        heldout_nulls["Global"] = np.asarray(
            z["Global_heldout_time_permutation_null"], dtype=np.float64
        )
    return {
        "summary": summary,
        "spectrum_rows": csv_read(outdir / "slow_spectra.csv"),
        "heldout_rows": csv_read(outdir / "heldout_slowness.csv"),
        "overlap_rows": csv_read(outdir / "global_axis_state_overlap.csv"),
        "unique_rows": csv_read(outdir / "global_axis_unique_contribution.csv"),
        "block_rows": csv_read(outdir / "global_block_state_overlap.csv"),
        "runtime": {
            "vt": vt,
            "fits": fits,
            "retained": retained,
            "heldout_time_permutation_nulls": heldout_nulls,
        },
    }


def _run_single_rank(
    grid: GridData,
    subject: int,
    heldout_session: int,
    task_ids: Mapping[str, int],
    args: argparse.Namespace,
    subject_dir: Path,
    grid_name: str,
    train_sessions: Sequence[int],
    coords_max: np.ndarray,
    singular: np.ndarray,
    vt: np.ndarray,
    svd_info: Mapping[str, Any],
    unavailable_ranks: Sequence[int],
    mean: np.ndarray,
    seed0: int,
    rank_req: int,
) -> Dict[str, Any]:
        rank = int(rank_req)
        coords = coords_max[:, :rank]
        outdir = ensure_dir(
            subject_dir / f"grid-{grid_name}" / f"heldout-session-{heldout_session}" / f"svd-rank-{rank_req}"
        )
        log(f"    sub-{subject:03d} grid={grid_name} heldout={heldout_session} SVD-rank={rank_req}")

        fits: Dict[str, SFAFit] = {}
        selections: Dict[str, Dict[str, Any]] = {}
        train_segments: Dict[str, List[np.ndarray]] = {}
        train_keys: Dict[str, List[Tuple[int, int, int]]] = {}
        spectrum_rows: List[Dict[str, Any]] = []
        heldout_rows: List[Dict[str, Any]] = []
        heldout_time_permutation_nulls: Dict[str, np.ndarray] = {}

        mode_order = STATE_NAMES + ["Global"]
        temporal_strata = state_label_for_positions(grid, task_ids)
        if np.any(temporal_strata < 0):
            raise ValueError("Invalid state labels for phase-stratified temporal null")
        n_train_shuffle = args.n_train_shuffle if grid_name == "no_overlap" else args.raw_n_train_shuffle
        n_heldout_shuffle = args.n_heldout_shuffle if grid_name == "no_overlap" else args.raw_n_heldout_shuffle

        for mode_index, mode_name in enumerate(mode_order):
            state_name = None if mode_name == "Global" else mode_name
            fit, segs, keys, _ = fit_real_mode(
                coords, grid, rank, train_sessions, task_ids, state_name,
            )
            fits[mode_name] = fit
            train_segments[mode_name] = segs
            train_keys[mode_name] = keys
            null = training_shuffle_null(
                coords, fit, segs, keys, n_train_shuffle,
                seed0 + rank_req * 100 + mode_index * 10000 + 17,
                state_name is None,
                strata=temporal_strata,
            ) if n_train_shuffle > 0 else np.empty((0, fit.rank))
            selection = select_99_temporal_order_slow_advantage(
                fit.gammas, null, args.slow_advantage_coverage, args.training_axis_alpha
            )
            selections[mode_name] = selection
            r99 = int(min(selection["retained_rank"], args.max_retained_axes, fit.rank))
            selection["retained_rank"] = r99
            selection["saturation_status"] = retained_subspace_status(
                r99, int(fit.rank), int(rank)
            )

            for k in range(fit.rank):
                spectrum_rows.append({
                    "mode": mode_name,
                    "axis_name": f"{'G' if mode_name == 'Global' else STATE_SHORT[STATE_NAMES.index(mode_name)]}_SF{k + 1:03d}",
                    "axis_index": k + 1,
                    "gamma_train": float(fit.gammas[k]),
                    "time_permutation_null_median_train": float(selection["null_median"][k]) if len(selection["null_median"]) > k else float("nan"),
                    "temporal_order_slow_advantage": float(selection["advantage"][k]) if len(selection["advantage"]) > k else 0.0,
                    "cumulative_temporal_order_slow_advantage": float(selection["cumulative_fraction"][k]) if len(selection["cumulative_fraction"]) > k else 0.0,
                    "significant_temporal_order_slow_advantage": float(selection["significant_advantage"][k]) if len(selection["significant_advantage"]) > k else 0.0,
                    "significant_cumulative_temporal_order_slow_advantage": float(selection["significant_cumulative_fraction"][k]) if len(selection["significant_cumulative_fraction"]) > k else 0.0,
                    "time_permutation_lower_tail_p_train": float(selection["lower_p"][k]) if len(selection["lower_p"]) > k else float("nan"),
                    "retained_99": bool(k < r99),
                    "retained_rank_99_raw_temporal_order_advantage": int(selection["retained_rank_raw"]),
                    "retained_rank_99": r99,
                    "n_train_time_permutations": int(n_train_shuffle),
                    "ambient_svd_rank": int(rank),
                    "sample_support_rank": int(fit.pair_info.get("sample_support_rank", fit.rank)),
                    "identifiable_sf_axes": int(fit.rank),
                    "retained_subspace_status": selection["saturation_status"],
                    "null_interpretation": (
                        "Shared row permutations are restricted to contiguous blocks of the "
                        "same five-state label. They preserve each within-state multivariate "
                        "point cloud and test temporal ordering, not cross-coordinate coordination."
                    ),
                })

            # Held-out evaluation uses frozen train directions and train-selected axes.
            test_sessions = [heldout_session]
            if state_name is None:
                test_weights = balanced_state_weights(grid, test_sessions, task_ids)
                test_segs, test_keys = collect_mode_segments(grid, test_sessions, task_ids, None)
                by_run = True
            else:
                test_weights = state_point_weights(grid, test_sessions, task_ids, state_name)
                test_segs, test_keys = collect_mode_segments(grid, test_sessions, task_ids, state_name)
                by_run = False
            axes = fit.directions_z[:, :r99]
            real_rjump, numer, denom = rayleigh_frozen_axes(
                coords, axes, test_weights, test_segs, test_keys, by_run,
            )
            hnull = heldout_shuffle_null(
                coords, axes, test_weights, test_segs, test_keys, n_heldout_shuffle,
                seed0 + rank_req * 100 + mode_index * 10000 + 913,
                by_run,
                strata=temporal_strata,
            ) if n_heldout_shuffle > 0 else np.empty((0, r99))
            heldout_time_permutation_nulls[mode_name] = hnull
            for k in range(r99):
                heldout_rows.append({
                    "mode": mode_name,
                    "axis_name": f"{'G' if mode_name == 'Global' else STATE_SHORT[STATE_NAMES.index(mode_name)]}_SF{k + 1:03d}",
                    "axis_index": k + 1,
                    "heldout_rjump": float(real_rjump[k]),
                    "heldout_time_permutation_median": float(np.nanmedian(hnull[:, k])) if hnull.size else float("nan"),
                    "heldout_time_permutation_lower_tail_p": empirical_lower_p(float(real_rjump[k]), hnull[:, k]) if hnull.size else float("nan"),
                    "heldout_numerator": float(numer[k]),
                    "heldout_variance": float(denom[k]),
                    "n_heldout_time_permutations": int(n_heldout_shuffle),
                })

            if not args.skip_plots:
                plot_spectrum(
                    [r for r in spectrum_rows if r["mode"] == mode_name],
                    outdir / f"slow_spectrum_{safe_name(mode_name)}.png",
                    f"{mode_name}: train real vs vector-wise time-permutation surrogate",
                )

        within_ranks = {state: int(selections[state]["retained_rank"]) for state in STATE_NAMES}
        global_rank = int(selections["Global"]["retained_rank"])
        topk_overlap = fixed_topk_overlap_rows(
            fits["Global"], {state: fits[state] for state in STATE_NAMES}, args.topk_list
        )
        overlap = semantic_overlap(
            fits["Global"], global_rank,
            {state: fits[state] for state in STATE_NAMES},
            within_ranks, args.degeneracy_gap,
            args.max_coefficient_global_axes,
        )

        csv_write(outdir / "slow_spectra.csv", spectrum_rows)
        csv_write(outdir / "heldout_slowness.csv", heldout_rows)
        csv_write(outdir / "global_axis_state_overlap.csv", overlap["global_axis_state_overlap"])
        csv_write(outdir / "global_within_fixed_topk_overlap.csv", topk_overlap)
        csv_write(outdir / "global_axis_unique_contribution.csv", overlap["global_axis_unique_contribution"])
        csv_write(outdir / "global_axis_minimum_norm_coefficients.csv", overlap["minimum_norm_coefficients"])
        csv_write(outdir / "global_block_state_overlap.csv", overlap["block_overlap"])
        if not args.skip_plots:
            plot_overlap(
                overlap["global_axis_state_overlap"],
                outdir / "global_axis_state_overlap.png",
                "Global slow axes explained by named within-state subspaces",
                n_show=args.plot_max_global_axes,
            )

        npz_payload: Dict[str, Any] = {
            "centered_svd_singular_values": singular[:rank],
            # Keep exact B fitting in float64, but store this large reducer matrix
            # in float32 as in v5.0 to avoid an avoidable disk-volume increase.
            "centered_svd_vt": np.asarray(vt[:rank], dtype=np.float32),
            "global_gammas": fits["Global"].gammas,
            "global_directions_z": fits["Global"].directions_z,
            "global_retained_rank_99": np.array([global_rank]),
            "within_union_basis_common": overlap["union_basis"],
            "global_axes_common": overlap["global_common_axes"],
            "principal_angles_global_vs_within_union_deg": overlap["principal_angles_deg"],
            "global_reconstruction_error": overlap["reconstruction_error"],
            "global_union_explained": overlap["union_explained"],
        }
        for state in STATE_NAMES:
            tag = safe_name(state)
            npz_payload[f"{tag}_gammas"] = fits[state].gammas
            npz_payload[f"{tag}_directions_z"] = fits[state].directions_z
            npz_payload[f"{tag}_retained_rank_99"] = np.array([within_ranks[state]])
            npz_payload[f"{tag}_axes_common"] = overlap["state_common_axes"][state]
            npz_payload[f"{tag}_heldout_time_permutation_null"] = heldout_time_permutation_nulls[state]
        npz_payload["Global_heldout_time_permutation_null"] = heldout_time_permutation_nulls["Global"]
        npz_write_atomic(outdir / "slow_geometry_and_nulls.npz", **npz_payload)

        if args.disable_c_handoff:
            c_handoff = {
                "schema_version": B_TO_C_SCHEMA_VERSION,
                "scientific_status": "disabled_by_cli",
            }
        else:
            c_handoff = export_b_to_c_handoff(
                outdir=outdir,
                model=str(getattr(args, "current_model", "unknown")),
                subject=subject,
                grid=grid,
                heldout_session=heldout_session,
                train_sessions=train_sessions,
                task_ids=task_ids,
                svd_rank_requested=rank_req,
                svd_rank_effective=rank,
                embedding_mean=mean,
                centered_svd_vt=vt[:rank],
                coords=coords,
                global_fit=fits["Global"],
                global_selection=selections["Global"],
                degeneracy_gap=args.degeneracy_gap,
                handoff_top_k=args.c_handoff_top_k,
            )
            log(
                f"      B->C handoff: fixed_k={c_handoff['handoff_rank_fixed_topk_requested']} "
                f"candidate={c_handoff['candidate_rank_fixed_topk_block_complete']} "
                f"r99_aux={c_handoff['retained_rank_r99']} windows={c_handoff['n_windows']}"
            )

        summary = {
            "script_version": SCRIPT_VERSION,
            "subject_id": subject,
            "grid": grid_name,
            "heldout_session": heldout_session,
            "training_sessions": train_sessions,
            "svd_rank_requested": rank_req,
            "svd_rank_effective": rank,
            "svd_info": dict(svd_info, requested_ranks=list(args.svd_rank_list), unavailable_ranks=unavailable_ranks),
            "decimation_factor": grid.decimation_factor,
            "effective_stride_seconds": grid.effective_stride_seconds,
            "effective_overlap_fraction": grid.effective_overlap_fraction,
            "retained_rank_99": {**within_ranks, "Global": global_rank},
            "retained_subspace_status": {
                mode: selections[mode]["saturation_status"] for mode in mode_order
            },
            "identifiable_sf_axes": {mode: int(fits[mode].rank) for mode in mode_order},
            "fixed_topk_overlap": topk_overlap,
            "global_vs_within_union_principal_angles_deg": overlap["principal_angles_deg"].tolist(),
            "global_union_explained": overlap["union_explained"].tolist(),
            "global_reconstruction_error": overlap["reconstruction_error"].tolist(),
            "minimum_norm_coefficient_global_axes_computed": int(overlap["coefficient_global_axes_computed"]),
            "slow_advantage_coverage": args.slow_advantage_coverage,
            "training_axis_alpha": args.training_axis_alpha,
            "cov_shrinkage": 0.0,
            "covariance_geometry": "exact_empirical_support_no_loading",
            "temporal_null": "vectorwise_within_state_stratified_time_permutation",
            "primary_B_inference": "cross_fold_fixed_topk_subspace_stability",
            "B_to_C_handoff": c_handoff,
            "notes": [
                "Axis selection is training-only. Held-out time-permutation p-values validate frozen axes and do not alter retained_rank_99.",
                "The vector-wise temporal-permutation null is exactly coordinate-equivariant and phase-stratified; it tests within-state temporal order, not cross-coordinate coordination.",
                "Cross-fold fixed-top-k principal-angle stability is the primary evidence for reproducible slow subspaces.",
                "r99 is auxiliary and never selects the B-to-C handoff rank.",
                "A retained subspace marked saturated_ambient is mathematically non-informative for mode-specific full-subspace stability.",
                "Marginal state overlaps are not additive because within-state subspaces may overlap.",
                "Leave-one-state-out unique contribution measures the loss of union projection after removing one state dictionary.",
            ],
        }
        json_dump(outdir / "fold_rank_summary.json", summary)
        return {
            "summary": summary,
            "spectrum_rows": spectrum_rows,
            "heldout_rows": heldout_rows,
            "overlap_rows": overlap["global_axis_state_overlap"],
            "unique_rows": overlap["global_axis_unique_contribution"],
            "block_rows": overlap["block_overlap"],
            "runtime": {
                "vt": vt[:rank].copy(),
                "fits": fits,
                "retained": {**within_ranks, "Global": global_rank},
                "heldout_time_permutation_nulls": heldout_time_permutation_nulls,
            },
        }


def run_fold_grid(
    ds: h5py.Dataset,
    grid: GridData,
    subject: int,
    heldout_session: int,
    task_ids: Mapping[str, int],
    args: argparse.Namespace,
    subject_dir: Path,
    grid_name: str,
) -> List[Dict[str, Any]]:
    sessions = sorted(set(int(x) for x in grid.session))
    train_sessions = [x for x in sessions if x != heldout_session]
    train_weights_full = balanced_state_weights(grid, train_sessions, task_ids)
    train_positions = np.flatnonzero(train_weights_full > 0)
    train_global_indices = grid.global_indices[train_positions]
    train_weights = train_weights_full[train_positions]
    train_weights /= train_weights.sum()

    flatten_dim = vector_dim(ds, args.vectorization)
    algebraic_max = int(min(len(train_positions) - 1, flatten_dim))
    feasible_ranks = [int(r) for r in args.svd_rank_list if int(r) <= algebraic_max]
    unavailable_ranks = [int(r) for r in args.svd_rank_list if int(r) > algebraic_max]
    if unavailable_ranks:
        log(f"    mathematically unavailable SVD ranks {unavailable_ranks}; algebraic maximum={algebraic_max}")
    if not feasible_ranks:
        return [{"summary": {
            "status": "no_feasible_svd_rank",
            "grid": grid_name,
            "heldout_session": heldout_session,
            "maximum_algebraic_rank": algebraic_max,
            "requested_ranks": list(args.svd_rank_list),
        }}]
    ranks_to_run = list(feasible_ranks)
    if grid_name == "raw" and args.raw_only_main_rank:
        ranks_to_run = [min(feasible_ranks, key=lambda x: abs(x - args.main_svd_rank))]

    # Fast resume path: if every requested rank already has a matching terminal
    # marker, return before rereading the H5 file or recomputing the fold SVD.
    if args.resume:
        resumed_results: List[Dict[str, Any]] = []
        all_terminal = True
        for rank_req in ranks_to_run:
            rank_outdir = (
                subject_dir / f"grid-{grid_name}" / f"heldout-session-{heldout_session}"
                / f"svd-rank-{int(rank_req)}"
            )
            fingerprint = b_rank_fingerprint(
                ds, grid_name, subject, heldout_session, int(rank_req), task_ids, args
            )
            done_marker = rank_outdir / "DONE.json"
            unavailable_marker = rank_outdir / "RANK_UNAVAILABLE.json"
            if terminal_marker_matches(done_marker, fingerprint):
                log(
                    f"    [RESUME DONE] sub-{subject:03d} grid={grid_name} "
                    f"heldout={heldout_session} M={rank_req} (pre-SVD)"
                )
                resumed_results.append(load_completed_rank_result(rank_outdir))
            elif terminal_marker_matches(unavailable_marker, fingerprint):
                with unavailable_marker.open("r", encoding="utf-8") as fh:
                    payload = json.load(fh)
                log(
                    f"    [RESUME UNAVAILABLE] sub-{subject:03d} grid={grid_name} "
                    f"heldout={heldout_session} M={rank_req} (pre-SVD)"
                )
                resumed_results.append({"summary": payload})
            else:
                all_terminal = False
                break
        if all_terminal:
            return resumed_results

    max_rank = max(feasible_ranks)
    seed0 = args.seed + 100000 * subject + 1000 * heldout_session + (0 if grid_name == "no_overlap" else 50000000)
    with tempfile.TemporaryDirectory(dir=args.tempdir, prefix="expB_svd_") as tmp:
        tmpdir = Path(tmp)
        mean = streaming_weighted_mean(
            ds, train_global_indices, train_weights, args.vectorization, args.read_batch_size,
        )
        singular, vt, svd_info = streaming_randomized_svd(
            ds, train_global_indices, train_weights, mean, args.vectorization,
            max_rank, args.read_batch_size, seed0, tmpdir,
            args.svd_oversamples, args.svd_power_iterations,
        )
        coords_max = project_grid(ds, grid, mean, vt, args.vectorization, args.read_batch_size)

    fold_results: List[Dict[str, Any]] = []

    for rank_req in ranks_to_run:
        rank = int(rank_req)
        outdir = ensure_dir(
            subject_dir / f"grid-{grid_name}" / f"heldout-session-{heldout_session}" / f"svd-rank-{rank_req}"
        )
        fingerprint = b_rank_fingerprint(
            ds, grid_name, subject, heldout_session, rank_req, task_ids, args
        )
        done_marker = outdir / "DONE.json"
        unavailable_marker = outdir / "RANK_UNAVAILABLE.json"
        if args.resume and terminal_marker_matches(done_marker, fingerprint):
            log(
                f"    [RESUME DONE] sub-{subject:03d} grid={grid_name} "
                f"heldout={heldout_session} M={rank_req}"
            )
            fold_results.append(load_completed_rank_result(outdir))
            continue
        if args.resume and terminal_marker_matches(unavailable_marker, fingerprint):
            with unavailable_marker.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            log(
                f"    [RESUME UNAVAILABLE] sub-{subject:03d} grid={grid_name} "
                f"heldout={heldout_session} M={rank_req}"
            )
            fold_results.append({"summary": payload})
            continue
        for marker in (done_marker, unavailable_marker):
            if marker.exists():
                marker.unlink()
        try:
            result = _run_single_rank(
                grid=grid,
                subject=subject,
                heldout_session=heldout_session,
                task_ids=task_ids,
                args=args,
                subject_dir=subject_dir,
                grid_name=grid_name,
                train_sessions=train_sessions,
                coords_max=coords_max,
                singular=singular,
                vt=vt,
                svd_info=svd_info,
                unavailable_ranks=unavailable_ranks,
                mean=mean,
                seed0=seed0,
                rank_req=rank_req,
            )
        except Exception as exc:  # noqa: BLE001
            if not is_coordinate_geometry_unavailable(exc):
                raise
            payload = {
                "script_version": SCRIPT_VERSION,
                "rank_fingerprint": fingerprint,
                "status": "unavailable_exact_coordinate_geometry_at_rank",
                "subject_id": int(subject),
                "grid": grid_name,
                "heldout_session": int(heldout_session),
                "svd_rank_requested": int(rank_req),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "finished_at": now(),
            }
            json_dump(unavailable_marker, payload)
            fold_results.append({"summary": payload})
            log(
                f"    [RANK UNAVAILABLE] sub-{subject:03d} grid={grid_name} "
                f"heldout={heldout_session} M={rank_req}: {exc}"
            )
            continue
        json_dump(done_marker, {
            "script_version": SCRIPT_VERSION,
            "rank_fingerprint": fingerprint,
            "subject_id": int(subject),
            "grid": grid_name,
            "heldout_session": int(heldout_session),
            "svd_rank_requested": int(rank_req),
            "finished_at": now(),
        })
        fold_results.append(result)
    return fold_results


def retained_subspace_status(
    retained_rank: int,
    support_rank: int,
    ambient_rank: int,
) -> str:
    if retained_rank <= 0:
        return "unavailable_no_detected_temporal_order_advantage"
    if retained_rank >= ambient_rank:
        return "saturated_ambient"
    if retained_rank >= support_rank:
        return "saturated_support"
    return "informative"


def _cross_fold_subspace_metrics(
    fit1: SFAFit,
    fit2: SFAFit,
    vt1: np.ndarray,
    vt2: np.ndarray,
    rank1: int,
    rank2: int,
) -> Dict[str, Any]:
    Q1 = orthonormalize_columns(fit1.directions_z[:, :rank1])
    Q2 = orthonormalize_columns(fit2.directions_z[:, :rank2])
    cross_svd = vt1 @ vt2.T
    singular = np.linalg.svd(Q1.T @ cross_svd @ Q2, compute_uv=False)
    singular = np.clip(singular, 0, 1)
    angles = np.degrees(np.arccos(singular))
    shared_energy = float(np.sum(singular ** 2))
    return {
        "mean_principal_angle_deg": float(np.mean(angles)) if len(angles) else float("nan"),
        "max_principal_angle_deg": float(np.max(angles)) if len(angles) else float("nan"),
        "angles_deg": angles.tolist(),
        "shared_projector_energy": shared_energy,
        "coverage_a_to_b": shared_energy / max(rank1, 1),
        "coverage_b_to_a": shared_energy / max(rank2, 1),
        "overlap_per_min_dimension": shared_energy / max(min(rank1, rank2), 1),
    }


def cross_fold_fixed_topk_stability(
    fold_results: Sequence[Mapping[str, Any]],
    mode: str,
    topk_values: Sequence[int],
) -> List[Dict[str, Any]]:
    """Primary mode-specific stability diagnostic using fixed leading k axes."""
    rows: List[Dict[str, Any]] = []
    for a, b in itertools.combinations(fold_results, 2):
        ra, rb = a["runtime"], b["runtime"]
        fit1, fit2 = ra["fits"][mode], rb["fits"][mode]
        ambient1, ambient2 = int(ra["vt"].shape[0]), int(rb["vt"].shape[0])
        support1, support2 = int(fit1.rank), int(fit2.rank)
        for requested in topk_values:
            k = int(requested)
            base = {
                "stability_type": "fixed_topk",
                "mode": mode,
                "heldout_session_a": a["summary"]["heldout_session"],
                "heldout_session_b": b["summary"]["heldout_session"],
                "top_k_requested": k,
                "rank_a": k,
                "rank_b": k,
                "support_rank_a": support1,
                "support_rank_b": support2,
                "ambient_svd_rank_a": ambient1,
                "ambient_svd_rank_b": ambient2,
            }
            if k < 1 or k > min(support1, support2):
                rows.append({
                    **base,
                    "status": "unavailable_support",
                    "informative_mode_specific": False,
                })
                continue
            status = "informative"
            if k >= ambient1 or k >= ambient2:
                status = "saturated_ambient"
            elif k >= support1 or k >= support2:
                status = "saturated_support"
            metrics = _cross_fold_subspace_metrics(
                fit1, fit2, ra["vt"], rb["vt"], k, k
            )
            rows.append({
                **base,
                "status": status,
                "informative_mode_specific": status == "informative",
                # Only saturated_ambient is vacuous by construction: there Q1 and Q2
                # are square orthogonal, so sigma(Q1' M Q2) = sigma(M) and the angles
                # reduce to the mode-independent angle between the two fold SVD bases.
                # saturated_support still spans a mode-specific empirical support.
                "mathematically_vacuous": status == "saturated_ambient",
                "support_subspace_is_mode_specific": status != "saturated_ambient",
                **metrics,
            })
    return rows


def cross_fold_r99_stability(
    fold_results: Sequence[Mapping[str, Any]],
    mode: str,
) -> List[Dict[str, Any]]:
    """Auxiliary r99 stability with explicit ambient/support saturation flags."""
    rows: List[Dict[str, Any]] = []
    for a, b in itertools.combinations(fold_results, 2):
        ra, rb = a["runtime"], b["runtime"]
        fit1, fit2 = ra["fits"][mode], rb["fits"][mode]
        r1, r2 = int(ra["retained"][mode]), int(rb["retained"][mode])
        ambient1, ambient2 = int(ra["vt"].shape[0]), int(rb["vt"].shape[0])
        support1, support2 = int(fit1.rank), int(fit2.rank)
        status_a = retained_subspace_status(r1, support1, ambient1)
        status_b = retained_subspace_status(r2, support2, ambient2)
        if "unavailable" in status_a or "unavailable" in status_b:
            combined = "unavailable"
        elif status_a == "saturated_ambient" or status_b == "saturated_ambient":
            combined = "saturated_ambient"
        elif status_a == "saturated_support" or status_b == "saturated_support":
            combined = "saturated_support"
        else:
            combined = "informative"
        base = {
            "stability_type": "r99_temporal_order_slow_excess",
            "mode": mode,
            "heldout_session_a": a["summary"]["heldout_session"],
            "heldout_session_b": b["summary"]["heldout_session"],
            "rank_a": r1,
            "rank_b": r2,
            "support_rank_a": support1,
            "support_rank_b": support2,
            "ambient_svd_rank_a": ambient1,
            "ambient_svd_rank_b": ambient2,
            "status_a": status_a,
            "status_b": status_b,
            "status": combined,
            "informative_mode_specific": combined == "informative",
            "mathematically_vacuous": combined == "saturated_ambient",
            "support_subspace_is_mode_specific": combined not in ("saturated_ambient", "unavailable"),
        }
        if r1 <= 0 or r2 <= 0:
            rows.append(base)
            continue
        rows.append({
            **base,
            **_cross_fold_subspace_metrics(
                fit1, fit2, ra["vt"], rb["vt"], r1, r2
            ),
        })
    return rows

def aggregate_subject_results(
    model: str,
    subject: int,
    all_results: Sequence[Mapping[str, Any]],
    subject_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    valid_results = [
        r for r in all_results
        if r.get("summary", {}).get("status", "ok") == "ok"
    ]
    summary_rows: List[Dict[str, Any]] = []
    pooled_rows: List[Dict[str, Any]] = []
    stability_topk_rows: List[Dict[str, Any]] = []
    stability_r99_rows: List[Dict[str, Any]] = []
    for grid_name in args.grid_list:
        ranks = sorted(set(
            int(r["summary"]["svd_rank_requested"])
            for r in valid_results if r["summary"]["grid"] == grid_name
        ))
        for rank in ranks:
            folds = [
                r for r in valid_results
                if r["summary"]["grid"] == grid_name
                and int(r["summary"]["svd_rank_requested"]) == rank
            ]
            for mode in STATE_NAMES + ["Global"]:
                retained = [int(f["summary"]["retained_rank_99"][mode]) for f in folds]
                support = [int(f["summary"]["identifiable_sf_axes"][mode]) for f in folds]
                ambient = [int(f["summary"]["svd_rank_effective"]) for f in folds]
                statuses = [
                    retained_subspace_status(r, s, a)
                    for r, s, a in zip(retained, support, ambient)
                ]
                summary_rows.append({
                    "model": model, "subject_id": subject, "grid": grid_name,
                    "svd_rank_requested": rank, "mode": mode,
                    "n_folds": len(folds),
                    "mean_retained_rank_99": float(np.mean(retained)) if retained else float("nan"),
                    "median_retained_rank_99": float(np.median(retained)) if retained else float("nan"),
                    "min_retained_rank_99": int(np.min(retained)) if retained else 0,
                    "max_retained_rank_99": int(np.max(retained)) if retained else 0,
                    "mean_identifiable_support_rank": float(np.mean(support)) if support else float("nan"),
                    "n_informative": int(sum(x == "informative" for x in statuses)),
                    "n_saturated_support": int(sum(x == "saturated_support" for x in statuses)),
                    "n_saturated_ambient": int(sum(x == "saturated_ambient" for x in statuses)),
                    "n_unavailable": int(sum(x.startswith("unavailable") for x in statuses)),
                    "fold_statuses": statuses,
                })
                # Pool the same time-permutation surrogate index across folds for
                # axes available in every fold.
                max_common = min(retained) if retained else 0
                for k in range(max_common):
                    real = []
                    nulls = []
                    for f in folds:
                        matches = [
                            x for x in f["heldout_rows"]
                            if x["mode"] == mode and int(x["axis_index"]) == k + 1
                        ]
                        if not matches:
                            continue
                        real.append(float(matches[0]["heldout_rjump"]))
                        nulls.append(np.asarray(f["runtime"]["heldout_time_permutation_nulls"][mode])[:, k])
                    if real and nulls:
                        n = min(len(x) for x in nulls)
                        pooled_null = np.mean(
                            np.stack([x[:n] for x in nulls], axis=0), axis=0
                        )
                        mean_real = float(np.mean(real))
                        pooled_rows.append({
                            "model": model, "subject_id": subject, "grid": grid_name,
                            "svd_rank_requested": rank, "mode": mode,
                            "axis_index": k + 1,
                            "axis_name": f"{'G' if mode == 'Global' else STATE_SHORT[STATE_NAMES.index(mode)]}_SF{k + 1:03d}",
                            "mean_heldout_rjump": mean_real,
                            "pooled_time_permutation_median": float(np.median(pooled_null)),
                            "subject_level_time_permutation_lower_tail_p": empirical_lower_p(mean_real, pooled_null),
                            "n_folds": len(real), "n_pooled_time_permutations": n,
                            "null_interpretation": (
                                "Tests within-state temporal-order slow structure against "
                                "vector-wise exchangeable samples stratified by state."
                            ),
                        })
            if folds:
                for mode in STATE_NAMES + ["Global"]:
                    stability_topk_rows.extend(
                        cross_fold_fixed_topk_stability(folds, mode, args.topk_list)
                    )
                    stability_r99_rows.extend(
                        cross_fold_r99_stability(folds, mode)
                    )

    csv_write(subject_dir / "effective_slow_dimensions_across_folds.csv", summary_rows)
    csv_write(subject_dir / "heldout_temporal_order_slowness_pooled_across_folds.csv", pooled_rows)
    csv_write(subject_dir / "slow_subspace_cross_fold_stability_topk.csv", stability_topk_rows)
    csv_write(subject_dir / "slow_subspace_cross_fold_stability_r99.csv", stability_r99_rows)
    # Backward-friendly headline file now contains the informative fixed-top-k diagnostic.
    csv_write(subject_dir / "slow_subspace_cross_fold_stability.csv", stability_topk_rows)
    available_ranks = sorted(set(
        int(r["summary"]["svd_rank_requested"]) for r in valid_results
    ))
    main_rank = (
        min(available_ranks, key=lambda x: abs(x - args.main_svd_rank))
        if available_ranks else None
    )
    main_grid = args.grid_list[0] if args.grid_list else "raw"
    summary = {
        "script_version": SCRIPT_VERSION,
        "model": model, "subject_id": subject,
        "status": "ok" if valid_results else "no_feasible_svd_rank",
        "main_grid": main_grid, "main_svd_rank": main_rank,
        "effective_dimensions": summary_rows,
        "pooled_heldout_temporal_order_slowness": pooled_rows,
        "cross_fold_stability_topk": stability_topk_rows,
        "cross_fold_stability_r99": stability_r99_rows,
        "null_interpretation": (
            "The vector-wise temporal-permutation null is stratified by the five-state "
            "label inside each contiguous run segment. It preserves each within-state "
            "multivariate point cloud and destroys only within-state temporal order. The "
            "same row permutation is applied to all coordinates, so the null commutes "
            "exactly with invertible reparameterizations of the fixed ambient space."
        ),
        "primary_inference": "cross_fold_fixed_topk_subspace_stability",
    }
    json_dump(subject_dir / "subject_summary.json", summary)
    return summary

def is_coordinate_geometry_unavailable(exc: BaseException) -> bool:
    """Classify expected numerical infeasibility without hiding coding errors."""
    message = str(exc).lower()
    tokens = (
        "exact whitening residual",
        "not positive definite at the requested rank",
        "state covariance on empirical support",
        "no positive empirical covariance support",
        "no valid adjacency pairs",
        "support rank",
    )
    return isinstance(exc, (ValueError, FloatingPointError, np.linalg.LinAlgError)) and any(
        token in message for token in tokens
    )


def run_subject(
    model: str,
    ds: h5py.Dataset,
    meta: Metadata,
    records: Sequence[RunRecord],
    subject: int,
    task_ids: Mapping[str, int],
    args: argparse.Namespace,
    model_dir: Path,
) -> Dict[str, Any]:
    subject_dir = ensure_dir(model_dir / f"sub-{subject:03d}")
    all_results: List[Dict[str, Any]] = []
    for grid_name in args.grid_list:
        grid = build_subject_grid(meta, records, subject, grid_name, args)
        inventory = {
            "grid": grid_name, "n_windows": len(grid.global_indices),
            "sessions": sorted(set(int(x) for x in grid.session)),
            "runs": len(grid.run_to_segments),
            "decimation_factor": grid.decimation_factor,
            "effective_stride_seconds": grid.effective_stride_seconds,
            "effective_overlap_fraction": grid.effective_overlap_fraction,
        }
        json_dump(subject_dir / f"grid_{grid_name}_inventory.json", inventory)
        sessions = inventory["sessions"]
        if len(sessions) != int(args.expected_sessions):
            raise RuntimeError(
                f"Subject {subject} expected {args.expected_sessions} sessions, got {sessions}"
            )
        heldout_sessions = args.heldout_session_values or sessions
        invalid = [x for x in heldout_sessions if x not in sessions]
        if invalid:
            raise ValueError(f"Requested held-out sessions {invalid} not in {sessions}")
        for heldout in heldout_sessions:
            try:
                all_results.extend(run_fold_grid(
                    ds, grid, subject, int(heldout), task_ids, args, subject_dir, grid_name,
                ))
            except Exception as exc:  # noqa: BLE001
                if not is_coordinate_geometry_unavailable(exc):
                    raise
                unavailable_dir = ensure_dir(
                    subject_dir / f"grid-{grid_name}" / f"heldout-session-{int(heldout)}"
                )
                payload = {
                    "script_version": SCRIPT_VERSION,
                    "status": "unavailable_exact_coordinate_geometry",
                    "subject_id": int(subject),
                    "grid": grid_name,
                    "heldout_session": int(heldout),
                    "requested_ranks": list(args.svd_rank_list),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "finished_at": now(),
                }
                json_dump(unavailable_dir / "FOLD_UNAVAILABLE.json", payload)
                all_results.append({"summary": payload})
                log(
                    f"    [UNAVAILABLE] sub-{subject:03d} grid={grid_name} "
                    f"heldout={int(heldout)}: {exc}"
                )
    return aggregate_subject_results(model, subject, all_results, subject_dir, args)


def parse_model_rank_overrides(values: Sequence[str]) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"--model-svd-ranks expects MODEL=r1,r2,..., got {item!r}")
        model, ranks = item.split("=", 1)
        parsed = parse_int_csv(ranks)
        if not parsed:
            raise ValueError(f"Empty rank list for model {model!r}")
        out[model.strip()] = parsed
    return out


def parse_model_overrides(values: Sequence[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"--model-path expects MODEL=/path, got {item}")
        model, path = item.split("=", 1)
        out[model] = path
    return out


def run_model(model: str, path: str, args: argparse.Namespace, root: Path) -> Dict[str, Any]:
    model_dir = ensure_dir(root / model)
    model_args = argparse.Namespace(**vars(args))
    model_args.current_model = model
    model_args.svd_rank_list = list(
        args.model_svd_rank_values.get(model, args.svd_rank_list)
    )
    log(f"model {model}: {path}; ranks={model_args.svd_rank_list}")
    with h5py.File(path, "r") as h5:
        ds = h5[model_args.embedding_key]
        requested_subjects = parse_int_csv(model_args.subjects) if model_args.subjects else []
        if requested_subjects:
            log(f"  locate explicit subjects {requested_subjects}")
            row_indices = find_subject_rows(
                h5["subject_id"], requested_subjects, model_args.metadata_scan_batch_size
            )
            meta = load_metadata(h5, model_args.embedding_key, row_indices=row_indices)
            metadata_scope = "explicit_subject_subset"
        else:
            meta = load_metadata(h5, model_args.embedding_key)
            metadata_scope = "complete_dataset"
        task_ids = parse_task_ids(model_args.task_ids)
        records, run_qc = validate_selected_runs(meta, list(task_ids.values()), model_args)
        csv_write(model_dir / "selected_run_qc.csv", run_qc)
        chosen, eligible, subject_qc = choose_subjects(records, task_ids, model_args)
        csv_write(model_dir / "subject_eligibility.csv", subject_qc)
        log(f"  window={meta.window_seconds:.3f}s stride={meta.stride_seconds:.3f}s")
        log(f"  task IDs={task_ids}")
        log(f"  eligible subjects={eligible}")
        log(f"  chosen subjects={chosen}")
        subjects = [
            run_subject(model, ds, meta, records, s, task_ids, model_args, model_dir)
            for s in chosen
        ]
    summary = {
        "script_version": SCRIPT_VERSION,
        "model": model, "embedding_path": path,
        "requested_svd_ranks": model_args.svd_rank_list,
        "chosen_subjects": chosen, "eligible_subjects": eligible,
        "subjects": subjects, "task_ids": task_ids,
        "metadata_scope": metadata_scope,
        "metadata_rows_loaded": int(len(meta.global_index)),
    }
    json_dump(model_dir / "model_summary.json", summary)
    return summary


def self_test() -> Dict[str, Any]:
    """Regression-test exact SFA and the temporal null under random GL maps."""
    rng = np.random.default_rng(20260730)

    def one_case(X: np.ndarray, weights: np.ndarray, label: str) -> Dict[str, float]:
        n, m = X.shape
        segments = [
            np.arange(i, min(i + 40, n), dtype=np.int64)
            for i in range(0, n, 40)
            if min(i + 40, n) - i >= 3
        ]
        keys = [(0, 0, i) for i in range(len(segments))]
        mean, Cx = weighted_covariance(X, weights, m)
        Cd, pair_info = pair_covariance(X, segments, m, keys, False)
        fit = solve_sfa(label, mean, Cx, Cd, pair_info)

        q1, _ = np.linalg.qr(rng.normal(size=(m, m)))
        q2, _ = np.linalg.qr(rng.normal(size=(m, m)))
        scales = np.geomspace(0.35, 3.5, m)
        A = q1 @ np.diag(scales) @ q2.T
        X2 = X @ A
        mean2, Cx2 = weighted_covariance(X2, weights, m)
        Cd2, pair_info2 = pair_covariance(X2, segments, m, keys, False)
        fit2 = solve_sfa(label + "_transformed", mean2, Cx2, Cd2, pair_info2)
        if fit.rank != fit2.rank:
            raise AssertionError(
                f"{label}: support rank changed under GL map: {fit.rank} vs {fit2.rank}"
            )

        gamma_error = float(np.max(np.abs(fit.gammas - fit2.gammas)))
        mapped = A @ fit2.directions_z
        eig, q = np.linalg.eigh(symmetrize(Cx))
        eig = np.maximum(eig, 0.0)
        Csqrt = (q * np.sqrt(eig)[None, :]) @ q.T
        G1 = orthonormalize_columns(Csqrt @ fit.directions_z)
        G2 = orthonormalize_columns(Csqrt @ mapped)
        singular = np.linalg.svd(G1.T @ G2, compute_uv=False)
        subspace_error = float(np.max(np.abs(singular - 1.0)))

        seed = 9317
        S1 = vectorwise_temporal_permutation_surrogate_coords(
            X, segments, m, np.random.default_rng(seed)
        )
        S2 = vectorwise_temporal_permutation_surrogate_coords(
            X2, segments, m, np.random.default_rng(seed)
        )
        surrogate_error = float(np.max(np.abs(S2 - S1 @ A)))
        null1 = training_shuffle_null(X, fit, segments, keys, 8, 771, False)
        null2 = training_shuffle_null(X2, fit2, segments, keys, 8, 771, False)
        null_gamma_error = float(np.max(np.abs(null1 - null2)))
        axes1 = fit.directions_z[:, :min(3, fit.rank)]
        axes2 = fit2.directions_z[:, :min(3, fit2.rank)]
        hnull1 = heldout_shuffle_null(
            X, axes1, weights, segments, keys, 8, 1881, False
        )
        hnull2 = heldout_shuffle_null(
            X2, axes2, weights, segments, keys, 8, 1881, False
        )
        heldout_null_error = float(np.max(np.abs(hnull1 - hnull2)))
        return {
            "support_rank": float(fit.rank),
            "ambient_rank": float(m),
            "transform_condition_number": float(np.linalg.cond(A)),
            "max_gamma_error": gamma_error,
            "max_subspace_singular_value_error": subspace_error,
            "max_surrogate_equivariance_error": surrogate_error,
            "max_null_gamma_error": null_gamma_error,
            "max_heldout_null_error": heldout_null_error,
        }

    n, m = 240, 8
    weights = rng.uniform(0.2, 1.0, size=n)
    weights /= weights.sum()
    full = rng.normal(size=(n, m)) @ rng.normal(size=(m, m))
    full += 0.05 * rng.normal(size=(n, m))
    singular_latent_rank = 5
    singular = (
        rng.normal(size=(n, singular_latent_rank))
        @ rng.normal(size=(singular_latent_rank, m))
    )

    cases = {
        "full_rank": one_case(full, weights, "self_test_full"),
        "rank_deficient": one_case(singular, weights, "self_test_singular"),
    }

    # Regression: adjacent gaps may all be small while the total spectral span
    # is large. The non-chaining rule must split such a smooth spectrum.
    chaining_spectrum = np.asarray([1.00, 1.06, 1.12, 1.18, 1.24, 1.30])
    chaining_blocks = make_degenerate_blocks(chaining_spectrum, 6, 0.10)
    if len(chaining_blocks) <= 1:
        raise AssertionError("degeneracy block construction still exhibits chaining")
    slow_rank, _, slow_block_i = block_complete_prefix_rank(
        chaining_spectrum, 2, 0.10
    )
    if slow_block_i is None or slow_rank < 2:
        raise AssertionError("slow-prefix block-complete rank selection failed")

    # Regression: phase-stratification must preserve the Baseline and Task point
    # clouds separately, while remaining exactly equivariant under a GL map.
    step = np.vstack([
        rng.normal(loc=0.0, scale=0.1, size=(30, m)),
        rng.normal(loc=3.0, scale=0.1, size=(30, m)),
    ])
    step_seg = [np.arange(60, dtype=np.int64)]
    step_strata = np.r_[np.zeros(30, dtype=np.int64), np.ones(30, dtype=np.int64)]
    q1, _ = np.linalg.qr(rng.normal(size=(m, m)))
    q2, _ = np.linalg.qr(rng.normal(size=(m, m)))
    A_step = q1 @ np.diag(np.geomspace(0.5, 2.0, m)) @ q2.T
    seed_step = 5519
    step_surr = vectorwise_temporal_permutation_surrogate_coords(
        step, step_seg, m, np.random.default_rng(seed_step), strata=step_strata
    )
    step_surr_2 = vectorwise_temporal_permutation_surrogate_coords(
        step @ A_step, step_seg, m, np.random.default_rng(seed_step), strata=step_strata
    )
    step_equivariance_error = float(np.max(np.abs(step_surr_2 - step_surr @ A_step)))
    baseline_cloud_error = float(np.max(np.abs(
        np.sort(step_surr[:30], axis=0) - np.sort(step[:30], axis=0)
    )))
    task_cloud_error = float(np.max(np.abs(
        np.sort(step_surr[30:], axis=0) - np.sort(step[30:], axis=0)
    )))
    tolerances = {
        "gamma": 2e-8,
        "subspace": 2e-8,
        "surrogate": 2e-12,
        "null_gamma": 2e-7,
        "heldout_null": 2e-7,
    }
    for label, result in cases.items():
        if not (
            result["max_gamma_error"] < tolerances["gamma"]
            and result["max_subspace_singular_value_error"] < tolerances["subspace"]
            and result["max_surrogate_equivariance_error"] < tolerances["surrogate"]
            and result["max_null_gamma_error"] < tolerances["null_gamma"]
            and result["max_heldout_null_error"] < tolerances["heldout_null"]
        ):
            raise AssertionError({"case": label, "result": result, "tolerances": tolerances})
    if int(cases["rank_deficient"]["support_rank"]) >= m:
        raise AssertionError("rank-deficient self-test did not exercise support restriction")
    if step_equivariance_error > tolerances["surrogate"]:
        raise AssertionError("phase-stratified surrogate equivariance failed")
    if baseline_cloud_error > 1e-12 or task_cloud_error > 1e-12:
        raise AssertionError("phase-stratified surrogate mixed state point clouds")

    return {
        "status": "passed",
        "covariance_geometry": "exact_empirical_support_no_loading",
        "temporal_null": "vectorwise_within_state_stratified_time_permutation",
        "handoff_selection": "fixed_topk_slow_prefix_nonchaining_block_complete",
        "slow_prefix_block_complete_rank_test": slow_rank,
        "degeneracy_nonchaining_blocks": chaining_blocks,
        "phase_stratified_step_test": {
            "max_surrogate_equivariance_error": step_equivariance_error,
            "baseline_point_cloud_error": baseline_cloud_error,
            "task_point_cloud_error": task_cloud_error,
        },
        "cases": cases,
        "tolerances": tolerances,
    }


# =============================================================================
# CLI
# =============================================================================


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Five-state within/global SFA with semantic projector overlap",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--models", nargs="+", default=["CBraMod"])
    p.add_argument("--model-path", action="append", default=[])
    p.add_argument("--embedding-key", default="embedding")
    p.add_argument("--vectorization", choices=["flatten", "mean_structural"], default="flatten")
    p.add_argument("--task-ids", default="MA=0,NB=1,NBMA=5,Full=6")

    p.add_argument("--n-subjects", type=int, default=3)
    p.add_argument("--expected-sessions", type=int, default=3)
    p.add_argument("--heldout-sessions", default="", help="Optional comma-separated held-out subset")
    p.add_argument("--subjects", default="")
    p.add_argument("--subject-seed", type=int, default=20260718)
    p.add_argument(
        "--metadata-scan-batch-size", type=int, default=65536,
        help="Batch size for locating explicitly requested subjects before metadata loading",
    )

    p.add_argument("--grids", default="raw")
    p.add_argument("--raw-only-main-rank", action="store_true", default=False)
    p.add_argument("--decimation-offset", type=int, default=0)
    p.add_argument("--stride-tolerance-sec", type=float, default=1e-4)
    p.add_argument("--min-phase-windows-raw", type=int, default=10)
    p.add_argument("--min-adjacencies-per-run", type=int, default=5)

    p.add_argument("--svd-ranks", default="100,200,300,500")
    p.add_argument(
        "--model-svd-ranks", action="append", default=[],
        help="Optional MODEL=r1,r2,... override; repeat once per model (e.g. BIOT=100,200,256)",
    )
    p.add_argument("--main-svd-rank", type=int, default=200)
    p.add_argument("--svd-oversamples", type=int, default=8)
    p.add_argument("--svd-power-iterations", type=int, default=1)
    p.add_argument(
        "--cov-shrinkage", type=float, default=0.0,
        help="DEPRECATED compatibility flag; exact coordinate-free SFA requires 0",
    )
    p.add_argument("--self-test", action="store_true")

    p.add_argument(
        "--slow-advantage-coverage", "--temporal-order-slow-excess-coverage",
        dest="slow_advantage_coverage", type=float, default=0.99,
        help="Cumulative temporal-order slow-excess coverage used for r99 selection",
    )
    p.add_argument("--training-axis-alpha", type=float, default=0.05)
    p.add_argument("--max-retained-axes", type=int, default=500)
    p.add_argument("--degeneracy-gap", type=float, default=0.10)
    p.add_argument(
        "--c-handoff-top-k", type=int, default=20,
        help=("Pre-registered leading Global-SFA rank exported to C before "
              "non-chaining degeneracy-block completion; independent of r99"),
    )
    p.add_argument(
        "--disable-c-handoff", action="store_true",
        help="Do not export the additive frozen Global-SFA interface for Experiment C",
    )
    p.add_argument("--top-k", default="1,2,3,5,10,20,50,100")
    p.add_argument(
        "--max-coefficient-global-axes", type=int, default=20,
        help="Limit the auxiliary minimum-norm coefficient table to the leading Global-SFA axes; 0 means all",
    )
    p.add_argument("--skip-plots", action="store_true", help="Skip PNG generation; all numerical tables are still written")
    p.add_argument("--plot-max-global-axes", type=int, default=30)
    p.add_argument("--resume", action="store_true", help="Resume completed or numerically unavailable B ranks")
    p.add_argument(
        "--n-train-shuffle", "--n-train-time-permutation", "--n-train-coordinate-shift",
        dest="n_train_shuffle", type=int, default=100,
    )
    p.add_argument(
        "--n-heldout-shuffle", "--n-heldout-time-permutation", "--n-heldout-coordinate-shift",
        dest="n_heldout_shuffle", type=int, default=200,
    )
    # DEPRECATED. These existed when the raw overlapping grid was only a cheap
    # sensitivity analysis. The ds007554 embeddings have 1 s windows at 1 s stride,
    # so raw IS the non-overlapping main grid and must not silently get fewer
    # surrogates than the main counts above. Default None means "inherit".
    p.add_argument(
        "--raw-n-train-shuffle", type=int, default=None,
        help="DEPRECATED override for the raw grid; defaults to --n-train-shuffle",
    )
    p.add_argument(
        "--raw-n-heldout-shuffle", type=int, default=None,
        help="DEPRECATED override for the raw grid; defaults to --n-heldout-shuffle",
    )

    p.add_argument("--read-batch-size", type=int, default=512)
    p.add_argument("--tempdir", type=Path, default=Path("/mnt/dataset4/yinuo/FM_flow/dataset/.expB_five_state_sfa_v5_1_tmp"))
    p.add_argument(
        "--outdir", type=Path,
        default=Path("/mnt/dataset4/yinuo/FM_flow/dataset/expB_five_state_within_global_sfa_v5_1"),
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fail-fast", action="store_true")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    if args.self_test:
        print(json.dumps(decode_scalar(self_test()), indent=2))
        return
    if abs(float(args.cov_shrinkage)) > 0.0:
        raise ValueError(
            "Exact coordinate-free SFA requires --cov-shrinkage 0. "
            "Identity loading is not invariant under general invertible coordinate changes."
        )
    args.svd_rank_list = parse_int_csv(args.svd_ranks)
    args.model_svd_rank_values = parse_model_rank_overrides(args.model_svd_ranks)
    args.heldout_session_values = parse_int_csv(args.heldout_sessions)
    args.topk_list = parse_int_csv(args.top_k)
    if int(args.c_handoff_top_k) < 1:
        raise ValueError("--c-handoff-top-k must be positive")
    if int(args.max_coefficient_global_axes) < 0 or int(args.plot_max_global_axes) < 1:
        raise ValueError("coefficient and plot caps must be non-negative/positive")
    args.grid_list = parse_str_csv(args.grids)
    if not args.svd_rank_list:
        raise ValueError("--svd-ranks is empty")
    if any(x not in {"raw", "no_overlap"} for x in args.grid_list):
        raise ValueError("--grids supports raw,no_overlap")
    if not (0 < args.slow_advantage_coverage <= 1):
        raise ValueError("--slow-advantage-coverage must be in (0,1]")
    if not (0 < args.training_axis_alpha < 1):
        raise ValueError("--training-axis-alpha must be in (0,1)")

    # B selects retained slow axes only when the empirical lower-tail p-value
    # passes training_axis_alpha.  With B surrogate draws, the smallest possible
    # p-value is 1/(B+1).  If that floor exceeds alpha, a nonzero C handoff is
    # mathematically impossible regardless of the data.  Reject that setup
    # instead of silently exporting candidate_rank=0.
    train_p_floor = 1.0 / (int(args.n_train_shuffle) + 1)
    if not args.disable_c_handoff and train_p_floor > float(args.training_axis_alpha) + 1e-15:
        min_required = int(math.ceil(1.0 / float(args.training_axis_alpha)) - 1)
        raise ValueError(
            f"C handoff requires --n-train-shuffle >= {min_required} for "
            f"--training-axis-alpha={args.training_axis_alpha:g}; got "
            f"{args.n_train_shuffle}, whose empirical p-value floor is "
            f"{train_p_floor:.4g}. Increase the surrogate count or pass "
            f"--disable-c-handoff for a B-only diagnostic run."
        )

    # Resolve the deprecated raw-grid surrogate counts. Inheriting by default
    # removes the silent detection-floor drop (1/21, 1/51) that the old hard-coded
    # 20/50 caused whenever --grids raw was used without explicit overrides.
    for scope in ("train", "heldout"):
        raw_attr, main_attr = f"raw_n_{scope}_shuffle", f"n_{scope}_shuffle"
        if getattr(args, raw_attr) is None:
            setattr(args, raw_attr, getattr(args, main_attr))
        elif getattr(args, raw_attr) != getattr(args, main_attr):
            log(
                f"WARNING: --raw-n-{scope}-shuffle={getattr(args, raw_attr)} differs from "
                f"--n-{scope}-shuffle={getattr(args, main_attr)}; the raw grid will use a "
                f"different surrogate count and its p-value floor will not be comparable"
            )
    for scope, count in (("train", args.n_train_shuffle), ("heldout", args.n_heldout_shuffle)):
        log(
            f"vector-time-permutation surrogates ({scope}): {count} "
            f"-> lower-tail p floor {1.0 / (count + 1):.4g}"
        )
    args.temporal_permutation_counts = {
        "train": int(args.n_train_shuffle),
        "heldout": int(args.n_heldout_shuffle),
        "raw_train": int(args.raw_n_train_shuffle),
        "raw_heldout": int(args.raw_n_heldout_shuffle),
        "train_lower_tail_p_floor": 1.0 / (int(args.n_train_shuffle) + 1),
        "heldout_lower_tail_p_floor": 1.0 / (int(args.n_heldout_shuffle) + 1),
    }
    ensure_dir(args.outdir)
    ensure_dir(args.tempdir)
    paths = dict(MODEL_PATHS)
    paths.update(parse_model_overrides(args.model_path))
    json_dump(args.outdir / "arguments.json", {
        "script_version": SCRIPT_VERSION,
        "arguments": vars(args),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__, "h5py": h5py.__version__,
        "started_at": now(),
    })
    summaries, failures = [], []
    for model in args.models:
        if model not in paths or not Path(paths[model]).exists():
            failures.append({"model": model, "error": f"missing path {paths.get(model)}"})
            continue
        try:
            summaries.append(run_model(model, paths[model], args, args.outdir))
        except Exception as exc:  # noqa: BLE001
            failures.append({
                "model": model, "error": repr(exc), "traceback": traceback.format_exc(),
            })
            log(f"model {model} FAILED: {exc!r}")
            if args.fail_fast:
                raise
    json_dump(args.outdir / "summary.json", {
        "script_version": SCRIPT_VERSION,
        "finished_at": now(), "models": summaries, "failures": failures,
    })
    log(f"done; {len(summaries)} model(s) ok, {len(failures)} failed")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
