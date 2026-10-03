#!/usr/bin/env python3
"""
Strict cross-subject LD/QD/Tensor experiment for CogBCI all-session 5 s embeddings.

Scientific design
=================
The experiment uses Rest-EO only and seven global classes:
    0 Rest-EO
    1 NBack-0
    2 NBack-1
    3 NBack-2
    4 MATB-easy
    5 MATB-medium
    6 MATB-difficult

Every outer subject-heldout fold owns its reducer.  The fold mean and randomized-SVD
basis are fitted only from that fold's training subjects.  Held-out subjects do not
participate in centering, SVD, class means, covariances, alpha/ridge selection, or
readout fitting.  They enter only through the frozen outer-train transform and final
evaluation.

To keep the formal run practical, raw 58,000-dimensional embeddings are never loaded
as one giant train matrix.  Each fold:
  1) selects a bounded, subject/class-balanced outer-training sample;
  2) fits its train-only SVD on a smaller balanced basis sample;
  3) streams the selected training rows and all held-out rows through the frozen basis;
  4) caches only the M-dimensional coordinates;
  5) performs all nested alpha/ridge selection and final LD/QD/Tensor evaluation in cache.

The bounded training sample is an explicit scientific protocol, not an accidental
shortcut.  It prevents subjects/classes with more windows from dominating.  Formal
high-rank runs use arithmetic pooled-covariance shrinkage, so empirical class
covariances may be singular while every fitted QDA covariance remains positive
definite.  The complete held-out fold is evaluated.

Representations
---------------
1. global7
   Seven-class pooled-covariance LDA/QDA with Rest-EO as reference, giving six
   Rest-relative scores ell and q.
2. nback3_dedicated
   NBack-only three-class LDA/QDA, with NBack-0 as reference, giving two scores.
3. matb3_dedicated
   MATB-only three-class LDA/QDA, with MATB-easy as reference, giving two scores.

For every representation:
    r = q - ell
    F0 = standardized ell
    F1 = standardized q
    F2 = standardized ell direct-sum standardized r
    F3 = F2 direct-sum standardized vec(ell outer r)

Evaluations
-----------
A. global7 representation -> seven-class readout
B. global7 representation -> NBack three-class readout
C. global7 representation -> MATB three-class readout
D. dedicated NBack representation -> NBack three-class readout
E. dedicated MATB representation -> MATB three-class readout
F. strict subset audit using the actual seven-class readout.

Engineering choices
-------------------
- Strict outer-test isolation, including the reducer.
- One fold-specific reduced cache, resumable independently.
- Arithmetic pooled-covariance shrinkage at positive alpha; no pseudoinverse.
- Closed-form class-balanced multiclass ridge readout with sample-size-normalized loss.
- Generalized-eigenbasis QDA scoring reused across alpha values.
- Metadata-only dimension preflight before any fold SVD, exact duplicate-subject audit,
  cache fingerprints, reducer diagnostics, and explicit alpha=1 arm-collapse reporting.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
from scipy import linalg, stats
from sklearn.utils.extmath import randomized_svd

SCRIPT_VERSION = "4.2.1-highrank-memory-resume-paired-audit"

MODEL_PATHS: Dict[str, str] = {
    "CBraMod": "/mnt/dataset4/fuxy/FMS/FM/CBraMod/output_emb_5s/cogbci_allsessions_5sflat_embeddings.h5",
    "LaBraM": "/mnt/dataset4/fuxy/FMS/FM/LaBraM/output_emb_5s/cogbci_allsessions_5sflat_embeddings.h5",
    "EEGPT": "/mnt/dataset4/fuxy/FMS/FM/EEGPT/output_emb_5s/cogbci_allsessions_5sflat_embeddings.h5",
    "EEGMamba": "/mnt/dataset4/fuxy/FMS/FM/EEGMamba/output_emb_5s/cogbci_allsessions_5sflat_embeddings.h5",
    "BIOT": "/mnt/dataset4/fuxy/FMS/FM/BIOT/output_emb_5s/cogbci_allsessions_5sflat_embeddings.h5",
}

DEFAULT_MODEL_DIMS: Dict[str, int] = {
    "CBraMod": 500,
    "LaBraM": 500,
    "EEGPT": 500,
    "EEGMamba": 500,
    "BIOT": 256,
}

GLOBAL_CLASS_NAMES = [
    "Rest-EO", "NBack-0", "NBack-1", "NBack-2",
    "MATB-easy", "MATB-medium", "MATB-difficult",
]
NBACK_CLASS_NAMES = ["NBack-0", "NBack-1", "NBack-2"]
MATB_CLASS_NAMES = ["MATB-easy", "MATB-medium", "MATB-difficult"]
ARMS = ("F0_LD", "F1_QD", "F2_LD_plus_QResidual", "F3_Tensor")
REPRESENTATIONS = ("global7", "nback3_dedicated", "matb3_dedicated")
SCENARIOS = (
    "global7_to_7class",
    "global7_to_nback3",
    "global7_to_matb3",
    "nback3_dedicated",
    "matb3_dedicated",
)


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{now()}] {message}", flush=True)


class StepTimer:
    def __init__(self, label: str):
        self.label = label
        self.t0 = 0.0

    def __enter__(self):
        self.t0 = time.perf_counter()
        log(f"BEGIN {self.label}")
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.perf_counter() - self.t0
        status = "FAILED" if exc_type is not None else "END"
        log(f"{status} {self.label}: {dt:.2f} s")
        return False


def parse_list(text: str, cast) -> List:
    return [cast(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_model_dims(text: str) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                f"Invalid --model-dims item {item!r}; expected Model:dimension"
            )
        name, value = item.split(":", 1)
        name = name.strip()
        dim = int(value.strip())
        if dim < 1:
            raise ValueError(f"Invalid dimension for {name}: {dim}")
        mapping[name] = dim
    return mapping


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_json(x) -> str:
    return json.dumps(x, ensure_ascii=False)


def write_csv(path: str | Path, rows: List[Dict]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        # Never leave a stale table from an earlier, incompatible run.
        path.unlink(missing_ok=True)
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def read_csv(path: str | Path) -> List[Dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def stable_payload_sha256(payload: Dict) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def marker_matches_current_script(
    path: str | Path, expected_run_signature: Optional[str] = None
) -> bool:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return False
    if str(payload.get("script_version", "")) != SCRIPT_VERSION:
        return False
    if expected_run_signature is not None:
        return str(payload.get("run_signature", "")) == str(expected_run_signature)
    return True


def fold_run_signature(
    model: str,
    source_path: str,
    outer_fold: int,
    train_subjects: Sequence[int],
    test_subjects: Sequence[int],
    requested_dim: int,
    args: argparse.Namespace,
) -> str:
    stat = os.stat(source_path)
    payload = {
        "script_version": SCRIPT_VERSION,
        "model": str(model),
        "source_path": str(source_path),
        "source_size": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
        "outer_fold": int(outer_fold),
        "train_subjects": [int(x) for x in train_subjects],
        "test_subjects": [int(x) for x in test_subjects],
        "M_requested": int(requested_dim),
        "seed": int(args.seed),
        "embedding_key": str(args.embedding_key),
        "label_key": str(args.label_key),
        "task_key": str(args.task_key),
        "subject_key": str(args.subject_key),
        "session_key": str(args.session_key),
        "rest_raw_label": int(args.rest_raw_label),
        "alpha_grid": [float(x) for x in args.alpha_grid_values],
        "ridge_grid": [float(x) for x in args.ridge_grid_values],
        "cov_interpolation": str(args.cov_interpolation),
        "train_cap_per_subject_class": int(args.train_cap_per_subject_class),
        "basis_cap_per_subject_class": int(args.basis_cap_per_subject_class),
        "inner_max_per_subject_class": int(args.inner_max_per_subject_class),
        "sampling_mode": str(args.sampling_mode),
        "train_blocks_per_session": int(args.train_blocks_per_session),
        "basis_blocks_per_session": int(args.basis_blocks_per_session),
        "basis_max_gap_rows": int(args.basis_max_gap_rows),
        "basis_max_span_rows": int(args.basis_max_span_rows),
        "project_max_gap_rows": int(args.project_max_gap_rows),
        "project_max_span_rows": int(args.project_max_span_rows),
        "randomized_n_iter": int(args.randomized_n_iter),
        "randomized_oversamples": int(args.randomized_oversamples),
        "degenerate_tol": float(args.degenerate_tol),
    }
    return stable_payload_sha256(payload)



def as_float(value) -> float:
    try:
        if value is None or str(value).strip() == "":
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def finite_mean(values: Iterable[float]) -> float:
    a = np.asarray(list(values), dtype=float)
    a = a[np.isfinite(a)]
    return float(np.mean(a)) if len(a) else float("nan")


def finite_sem(values: Iterable[float]) -> float:
    a = np.asarray(list(values), dtype=float)
    a = a[np.isfinite(a)]
    return float(np.std(a, ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0


def read_vector(f: h5py.File, key: str) -> np.ndarray:
    if key not in f:
        raise KeyError(f"Missing H5 key {key}; available={list(f.keys())}")
    return np.asarray(f[key][:]).reshape(-1)


def flat_dim(ds: h5py.Dataset) -> int:
    return int(np.prod(ds.shape[1:], dtype=np.int64))


def read_embedding_rows(
    ds: h5py.Dataset,
    indices: np.ndarray,
    batch_size: int = 4096,
    dtype=np.float32,
) -> np.ndarray:
    """Read sorted H5 rows in bounded batches, matching the validated LDA pipeline."""
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError("indices must be one-dimensional")
    if len(indices) == 0:
        return np.empty((0, flat_dim(ds)), dtype=dtype)
    if np.any(np.diff(indices) < 0):
        raise ValueError("indices must be sorted")

    n = len(indices)
    d = flat_dim(ds)
    out = np.empty((n, d), dtype=dtype)
    for start in range(0, n, int(batch_size)):
        end = min(start + int(batch_size), n)
        idx = indices[start:end]
        block = np.asarray(ds[idx])
        out[start:end] = block.reshape(len(idx), d).astype(dtype, copy=False)
    return out


def make_global_labels(raw_label: np.ndarray, task_id: np.ndarray, rest_raw_label: int = 1) -> np.ndarray:
    raw = np.asarray(raw_label, dtype=np.int64)
    task = np.asarray(task_id, dtype=np.int64)
    y = np.full(len(raw), -1, dtype=np.int64)
    y[(task == 0) & (raw == int(rest_raw_label))] = 0
    y[raw == 10] = 1
    y[raw == 11] = 2
    y[raw == 12] = 3
    y[raw == 20] = 4
    y[raw == 21] = 5
    y[raw == 22] = 6
    return y


def split_subjects(subjects: Sequence[int], n_folds: int, seed: int) -> List[List[int]]:
    a = np.asarray(sorted(int(s) for s in subjects), dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(a)
    return [sorted(x.astype(int).tolist()) for x in np.array_split(a, int(n_folds))]


def make_subject_kfold_groups_from_counts(
    subjects: Sequence[int],
    subject_counts: Dict[int, int],
    n_folds: int,
    seed: int,
) -> List[List[int]]:
    """Greedy sample-balanced subject folds, matching the validated Experiment A logic."""
    subjects = [int(s) for s in subjects]
    if n_folds < 2 or n_folds > len(subjects):
        raise ValueError(
            f"Invalid n_folds={n_folds} for {len(subjects)} subjects"
        )
    rng = np.random.default_rng(seed)
    shuffled = list(subjects)
    rng.shuffle(shuffled)
    ordered = sorted(
        shuffled,
        key=lambda s: int(subject_counts[int(s)]),
        reverse=True,
    )
    groups: List[List[int]] = [[] for _ in range(n_folds)]
    totals = [0 for _ in range(n_folds)]
    for s in ordered:
        j = int(np.argmin(totals))
        groups[j].append(int(s))
        totals[j] += int(subject_counts[int(s)])
    groups = [sorted(g) for g in groups]
    groups = sorted(groups, key=lambda g: (min(g), len(g)))
    return groups


@dataclass
class FoldReducedCache:
    R_train: np.ndarray
    y_train: np.ndarray
    subject_train: np.ndarray
    global_index_train: np.ndarray
    R_test: np.ndarray
    y_test: np.ndarray
    subject_test: np.ndarray
    global_index_test: np.ndarray
    n_components: int
    method: str
    cache_dir: Path


def choose_balanced_rows(
    selected_indices: np.ndarray,
    y_selected: np.ndarray,
    subject_selected: np.ndarray,
    session_selected: np.ndarray,
    allowed_subjects: Sequence[int],
    cap_per_subject_class: int,
    seed: int,
    sampling_mode: str = "block_spread",
    blocks_per_session: int = 4,
) -> np.ndarray:
    """Choose deterministic subject/class-balanced rows, spread across sessions.

    ``block_spread`` partitions every session/class ledger into several temporal
    strata and takes one short contiguous block from each stratum.  It retains most
    of the HDF5 locality of the old one-block sampler while avoiding a single long
    autocorrelated run of windows.  ``contiguous`` preserves the v4.1 behavior.
    """
    allowed = set(int(x) for x in allowed_subjects)
    cap = int(cap_per_subject_class)
    rng = np.random.default_rng(seed)
    chosen: List[np.ndarray] = []

    if sampling_mode not in {"block_spread", "contiguous"}:
        raise ValueError(f"Unknown sampling_mode={sampling_mode!r}")

    def pick_from_ledger(local: np.ndarray, quota: int) -> np.ndarray:
        local = np.asarray(local, dtype=np.int64)
        quota = int(min(max(quota, 0), len(local)))
        if quota <= 0:
            return np.empty(0, dtype=np.int64)
        if quota == len(local):
            return local.copy()
        if sampling_mode == "contiguous":
            start = int(rng.integers(0, len(local) - quota + 1))
            return local[start:start + quota]

        n_blocks = int(min(max(1, blocks_per_session), quota, len(local)))
        block_sizes = np.full(n_blocks, quota // n_blocks, dtype=np.int64)
        block_sizes[: quota % n_blocks] += 1
        edges = np.floor(np.linspace(0, len(local), n_blocks + 1)).astype(np.int64)
        pieces: List[np.ndarray] = []
        for b in range(n_blocks):
            lo, hi = int(edges[b]), int(edges[b + 1])
            segment = local[lo:hi]
            need = int(min(block_sizes[b], len(segment)))
            if need <= 0:
                continue
            if need == len(segment):
                pieces.append(segment)
            else:
                start = int(rng.integers(0, len(segment) - need + 1))
                pieces.append(segment[start:start + need])
        picked = np.unique(np.concatenate(pieces)) if pieces else np.empty(0, dtype=np.int64)
        if len(picked) < quota:
            remaining = np.setdiff1d(local, picked, assume_unique=False)
            need = min(quota - len(picked), len(remaining))
            if need > 0:
                extra = np.sort(rng.choice(remaining, size=need, replace=False))
                picked = np.concatenate([picked, extra])
        return np.sort(picked.astype(np.int64))

    for subject in sorted(allowed):
        for cls in range(7):
            mask_sc = (subject_selected == subject) & (y_selected == cls)
            local_sc = np.flatnonzero(mask_sc)
            if len(local_sc) == 0:
                raise ValueError(f"No rows for training subject={subject}, class={GLOBAL_CLASS_NAMES[cls]}")
            if cap <= 0 or len(local_sc) <= cap:
                chosen.append(selected_indices[local_sc])
                continue

            sessions = sorted(np.unique(session_selected[local_sc]).astype(int).tolist())
            base = cap // len(sessions)
            remainder = cap % len(sessions)
            pieces: List[np.ndarray] = []
            for j, session in enumerate(sessions):
                local = local_sc[session_selected[local_sc] == session]
                quota = base + (1 if j < remainder else 0)
                quota = min(quota, len(local))
                if quota <= 0:
                    continue
                take = pick_from_ledger(local, quota)
                pieces.append(selected_indices[take])

            picked = np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)
            # Sessions with too few rows can leave unused quota. Fill from the remaining
            # subject/class ledger without replacement.
            if len(picked) < cap:
                remaining = np.setdiff1d(selected_indices[local_sc], picked, assume_unique=False)
                need = min(cap - len(picked), len(remaining))
                if need > 0:
                    extra_pos = np.sort(rng.choice(len(remaining), size=need, replace=False))
                    picked = np.concatenate([picked, remaining[extra_pos]])
            chosen.append(np.sort(picked.astype(np.int64)))

    out = np.unique(np.concatenate(chosen).astype(np.int64))
    out.sort()
    return out


def require_sorted_unique_indices(indices: np.ndarray, label: str) -> np.ndarray:
    arr = np.asarray(indices, dtype=np.int64).reshape(-1)
    if len(arr) == 0:
        raise ValueError(f"{label} cannot be empty")
    if np.any(arr[1:] <= arr[:-1]):
        raise ValueError(
            f"{label} must be strictly increasing and duplicate-free; "
            "silent reordering would break feature/label alignment"
        )
    return arr


def weighted_mean_bounded_memory(
    X: np.ndarray, weights: np.ndarray, chunk_rows: int = 128
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float64)
    if len(X) != len(weights):
        raise ValueError("weighted mean row mismatch")
    mean64 = np.zeros(X.shape[1], dtype=np.float64)
    for start in range(0, len(X), int(chunk_rows)):
        end = min(start + int(chunk_rows), len(X))
        mean64 += weights[start:end] @ X[start:end].astype(np.float64)
    return mean64.astype(np.float32)


def frobenius_sq_bounded_memory(X: np.ndarray, chunk_rows: int = 128) -> float:
    X = np.asarray(X, dtype=np.float32)
    total = 0.0
    for start in range(0, len(X), int(chunk_rows)):
        end = min(start + int(chunk_rows), len(X))
        block64 = X[start:end].astype(np.float64)
        total += float(np.sum(block64 * block64, dtype=np.float64))
    return total


def read_rows_by_merged_slices(
    ds: h5py.Dataset,
    indices: np.ndarray,
    max_gap_rows: int,
    max_span_rows: int,
) -> np.ndarray:
    """Read sparse rows through bounded contiguous HDF5 slices."""
    indices = require_sorted_unique_indices(indices, "read indices")
    groups: List[np.ndarray] = []
    start = 0
    for i in range(1, len(indices)):
        gap = int(indices[i] - indices[i - 1])
        span = int(indices[i] - indices[start] + 1)
        if gap > int(max_gap_rows) or span > int(max_span_rows):
            groups.append(indices[start:i])
            start = i
    groups.append(indices[start:])

    d = flat_dim(ds)
    out = np.empty((len(indices), d), dtype=np.float32)
    cursor = 0
    for group in groups:
        lo, hi = int(group[0]), int(group[-1]) + 1
        block = np.asarray(ds[lo:hi], dtype=np.float32).reshape(hi - lo, d)
        out[cursor:cursor + len(group)] = block[group - lo]
        cursor += len(group)
    log(f"basis read: requested_rows={len(indices)} slices={len(groups)} D={d}")
    return out


def project_rows_to_npy(
    ds: h5py.Dataset,
    indices: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
    out_path: Path,
    max_gap_rows: int,
    max_span_rows: int,
) -> None:
    """Stream selected raw rows into a compact reduced-coordinate .npy memmap."""
    indices = require_sorted_unique_indices(indices, "projection indices")
    m = int(components.shape[0])
    d = flat_dim(ds)
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(len(indices), m))

    groups: List[np.ndarray] = []
    start = 0
    for i in range(1, len(indices)):
        gap = int(indices[i] - indices[i - 1])
        span = int(indices[i] - indices[start] + 1)
        if gap > int(max_gap_rows) or span > int(max_span_rows):
            groups.append(indices[start:i])
            start = i
    groups.append(indices[start:])

    cursor = 0
    next_report = 5000
    for group in groups:
        lo, hi = int(group[0]), int(group[-1]) + 1
        raw = np.asarray(ds[lo:hi], dtype=np.float32).reshape(hi - lo, d)
        X = raw[group - lo]
        X -= mean
        out[cursor:cursor + len(group)] = X @ components.T
        cursor += len(group)
        if cursor >= next_report:
            log(f"projected {cursor}/{len(indices)} rows -> M={m}")
            next_report += 5000
    out.flush()
    del out
    if cursor != len(indices):
        raise RuntimeError(f"Projection stopped at {cursor}/{len(indices)} rows")


def cache_meta_matches(meta_path: Path, expected: Dict) -> bool:
    if not meta_path.exists():
        return False
    try:
        current = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return all(current.get(k) == v for k, v in expected.items())


def array_sha256(values: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(values))
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
    h.update(arr.view(np.uint8).tobytes())
    return h.hexdigest()


def representation_counts(y_global: np.ndarray, representation: str) -> np.ndarray:
    y_global = np.asarray(y_global, dtype=np.int64)
    if representation == "global7":
        return np.bincount(y_global, minlength=7).astype(np.int64)
    if representation == "nback3_dedicated":
        mask, y = local_labels(y_global, np.arange(1, 4))
        del mask
        return np.bincount(y, minlength=3).astype(np.int64)
    if representation == "matb3_dedicated":
        mask, y = local_labels(y_global, np.arange(4, 7))
        del mask
        return np.bincount(y, minlength=3).astype(np.int64)
    raise KeyError(representation)


def covariance_rank_cap_from_counts(counts: np.ndarray, cov_mode: str) -> int:
    counts = np.asarray(counts, dtype=np.int64)
    if np.any(counts <= 0):
        return 0
    if cov_mode == "arithmetic":
        # Equal-class pooled covariance rank is at most sum_c (n_c - 1).
        return int(np.sum(counts - 1))
    if cov_mode == "geometric":
        # Geometric interpolation needs every empirical class covariance SPD.
        return int(np.min(counts) - 1)
    raise ValueError(cov_mode)


def fold_dimension_preflight(
    model: str,
    info: Dict,
    outer_fold: int,
    train_subjects: Sequence[int],
    test_subjects: Sequence[int],
    requested_dim: int,
    args: argparse.Namespace,
) -> Dict:
    selected_indices = np.asarray(info["_selected_indices"], dtype=np.int64)
    y_selected = np.asarray(info["_y_selected"], dtype=np.int64)
    subject_selected = np.asarray(info["_subject_selected"], dtype=np.int64)
    session_selected = np.asarray(info["_session_selected"], dtype=np.int64)

    train_indices = choose_balanced_rows(
        selected_indices, y_selected, subject_selected, session_selected,
        train_subjects, args.train_cap_per_subject_class,
        args.seed + 1000003 * int(outer_fold) + 17,
        args.sampling_mode, args.train_blocks_per_session,
    )
    basis_indices = choose_balanced_rows(
        selected_indices, y_selected, subject_selected, session_selected,
        train_subjects, args.basis_cap_per_subject_class,
        args.seed + 1000003 * int(outer_fold) + 31,
        args.sampling_mode, args.basis_blocks_per_session,
    )
    test_mask = np.isin(subject_selected, np.asarray(test_subjects, dtype=np.int64))
    test_indices = selected_indices[test_mask]

    position = {int(g): i for i, g in enumerate(selected_indices.tolist())}
    train_pos = np.asarray([position[int(g)] for g in train_indices], dtype=np.int64)
    y_train = y_selected[train_pos]
    subject_train = subject_selected[train_pos]
    inner_folds = split_subjects(
        train_subjects, args.cross_inner_folds, args.seed + 1000 * int(outer_fold)
    )

    rep_caps: Dict[str, int] = {}
    rep_details: Dict[str, Dict] = {}
    for rep in REPRESENTATIONS:
        outer_counts = representation_counts(y_train, rep)
        inner_caps: List[int] = []
        inner_counts: List[List[int]] = []
        for val_subjects in inner_folds:
            inner_train = ~np.isin(subject_train, np.asarray(val_subjects, dtype=np.int64))
            counts = representation_counts(y_train[inner_train], rep)
            inner_counts.append(counts.astype(int).tolist())
            inner_caps.append(covariance_rank_cap_from_counts(counts, args.cov_interpolation))
        outer_cap = covariance_rank_cap_from_counts(outer_counts, args.cov_interpolation)
        rep_caps[rep] = int(min([outer_cap] + inner_caps))
        rep_details[rep] = {
            "outer_counts": outer_counts.astype(int).tolist(),
            "outer_covariance_rank_cap": int(outer_cap),
            "inner_counts": inner_counts,
            "inner_covariance_rank_caps": [int(x) for x in inner_caps],
        }

    structural_caps = {
        "flat_dim": int(info["flat_dim"]),
        "train_centered_rank_cap": int(len(train_indices) - 1),
        "basis_centered_rank_cap": int(len(basis_indices) - 1),
        **{f"{rep}_covariance_rank_cap": int(rep_caps[rep]) for rep in REPRESENTATIONS},
    }
    maximum = int(min(structural_caps.values()))
    effective = int(min(int(requested_dim), maximum))
    return {
        "model": model,
        "outer_fold": int(outer_fold),
        "requested_M": int(requested_dim),
        "maximum_feasible_M": maximum,
        "effective_M": effective,
        "feasible": int(requested_dim) <= maximum,
        "covariance_interpolation": args.cov_interpolation,
        "n_train": int(len(train_indices)),
        "n_basis": int(len(basis_indices)),
        "n_test": int(len(test_indices)),
        "train_indices_sha256": array_sha256(train_indices),
        "basis_indices_sha256": array_sha256(basis_indices),
        "test_indices_sha256": array_sha256(test_indices),
        "structural_caps": structural_caps,
        "representation_details": rep_details,
    }


def build_or_load_fold_cache(
    model: str,
    path: str,
    info: Dict,
    outer_fold: int,
    train_subjects: List[int],
    test_subjects: List[int],
    requested_dim: int,
    args: argparse.Namespace,
    outdir: Path,
) -> FoldReducedCache:
    cache_dir = ensure_dir(
        outdir / "cache" / model / f"fold_{int(outer_fold)}_M{int(requested_dim)}"
    )
    paths = {
        "R_train": cache_dir / "R_train.npy",
        "y_train": cache_dir / "y_train.npy",
        "subject_train": cache_dir / "subject_train.npy",
        "index_train": cache_dir / "global_index_train.npy",
        "R_test": cache_dir / "R_test.npy",
        "y_test": cache_dir / "y_test.npy",
        "subject_test": cache_dir / "subject_test.npy",
        "index_test": cache_dir / "global_index_test.npy",
        "singular_values": cache_dir / "reducer_singular_values.npy",
        "reducer_diagnostics": cache_dir / "reducer_diagnostics.json",
    }
    meta_path = cache_dir / "cache_metadata.json"

    selected_indices = np.asarray(info["_selected_indices"], dtype=np.int64)
    y_selected = np.asarray(info["_y_selected"], dtype=np.int64)
    subject_selected = np.asarray(info["_subject_selected"], dtype=np.int64)
    session_selected = np.asarray(info["_session_selected"], dtype=np.int64)
    flat_d = int(info["flat_dim"])

    train_indices = choose_balanced_rows(
        selected_indices, y_selected, subject_selected, session_selected,
        train_subjects, args.train_cap_per_subject_class,
        args.seed + 1000003 * int(outer_fold) + 17,
        args.sampling_mode, args.train_blocks_per_session,
    )
    basis_indices = choose_balanced_rows(
        selected_indices, y_selected, subject_selected, session_selected,
        train_subjects, args.basis_cap_per_subject_class,
        args.seed + 1000003 * int(outer_fold) + 31,
        args.sampling_mode, args.basis_blocks_per_session,
    )
    test_mask = np.isin(subject_selected, np.asarray(test_subjects, dtype=np.int64))
    test_indices = selected_indices[test_mask]

    # Map global row -> metadata position once.
    position = {int(g): i for i, g in enumerate(selected_indices.tolist())}
    train_pos = np.asarray([position[int(g)] for g in train_indices], dtype=np.int64)
    test_pos = np.asarray([position[int(g)] for g in test_indices], dtype=np.int64)
    y_train = y_selected[train_pos]
    subject_train = subject_selected[train_pos]
    y_test = y_selected[test_pos]
    subject_test = subject_selected[test_pos]

    preflight = fold_dimension_preflight(
        model, info, outer_fold, train_subjects, test_subjects,
        requested_dim, args,
    )
    max_exact_m = int(preflight["maximum_feasible_M"])
    m = int(preflight["effective_M"])
    if m < 1:
        raise ValueError(
            f"No feasible QDA dimension for {model} fold={outer_fold}; "
            f"caps={preflight['structural_caps']}"
        )
    if m < int(requested_dim):
        message = (
            f"{model} fold={outer_fold}: requested M={requested_dim} exceeds the "
            f"{args.cov_interpolation}-QDA feasible M={m}; "
            f"caps={preflight['structural_caps']}"
        )
        if args.require_requested_dim:
            raise ValueError(message)
        log(f"[DIMENSION CAP] {message}")

    stat = os.stat(path)
    expected = {
        "script_cache_version": "v4.2-pooled-shrinkage-integrity-cache",
        "model": model,
        "outer_fold": int(outer_fold),
        "source_path": path,
        "source_size": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
        "rest_raw_label": int(args.rest_raw_label),
        "embedding_key": str(args.embedding_key),
        "label_key": str(args.label_key),
        "task_key": str(args.task_key),
        "subject_key": str(args.subject_key),
        "session_key": str(args.session_key),
        "train_subjects": [int(x) for x in train_subjects],
        "test_subjects": [int(x) for x in test_subjects],
        "train_cap_per_subject_class": int(args.train_cap_per_subject_class),
        "basis_cap_per_subject_class": int(args.basis_cap_per_subject_class),
        "sampling_mode": str(args.sampling_mode),
        "train_blocks_per_session": int(args.train_blocks_per_session),
        "basis_blocks_per_session": int(args.basis_blocks_per_session),
        "seed": int(args.seed),
        "n_train": int(len(train_indices)),
        "n_test": int(len(test_indices)),
        "n_basis": int(len(basis_indices)),
        "train_indices_sha256": array_sha256(train_indices),
        "basis_indices_sha256": array_sha256(basis_indices),
        "test_indices_sha256": array_sha256(test_indices),
        "flat_dim": flat_d,
        "M_requested": int(requested_dim),
        "M_effective": m,
        "randomized_n_iter": int(args.randomized_n_iter),
        "randomized_oversamples": int(args.randomized_oversamples),
        "covariance_interpolation": str(args.cov_interpolation),
    }
    if args.resume and all(p.exists() for p in paths.values()) and cache_meta_matches(meta_path, expected):
        log(f"[CACHE] reuse strict {model} fold={outer_fold} cache from {cache_dir}")
        return FoldReducedCache(
            R_train=np.load(paths["R_train"], mmap_mode="r"),
            y_train=np.load(paths["y_train"], mmap_mode="r"),
            subject_train=np.load(paths["subject_train"], mmap_mode="r"),
            global_index_train=np.load(paths["index_train"], mmap_mode="r"),
            R_test=np.load(paths["R_test"], mmap_mode="r"),
            y_test=np.load(paths["y_test"], mmap_mode="r"),
            subject_test=np.load(paths["subject_test"], mmap_mode="r"),
            global_index_test=np.load(paths["index_test"], mmap_mode="r"),
            n_components=m,
            method="outer_train_balanced_randomized_svd_cache" if m < flat_d else "outer_train_identity_centered_cache",
            cache_dir=cache_dir,
        )

    tmp_dir = cache_dir.with_name(cache_dir.name + ".building")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "r") as h5:
        ds = h5[args.embedding_key]
        with StepTimer(f"{model} fold={outer_fold} read train-only SVD basis rows={len(basis_indices)}"):
            X_basis = read_rows_by_merged_slices(
                ds, basis_indices, args.basis_max_gap_rows, args.basis_max_span_rows
            )

        basis_pos = np.asarray([position[int(g)] for g in basis_indices], dtype=np.int64)
        y_basis = y_selected[basis_pos]
        with StepTimer(f"{model} fold={outer_fold} fit train-only reducer M={m}"):
            counts = np.bincount(y_basis, minlength=7).astype(np.int64)
            if np.any(counts == 0):
                raise ValueError(f"SVD basis misses a class: counts={counts.tolist()}")
            weights = 1.0 / (7.0 * counts[y_basis].astype(np.float64))
            weights /= np.sum(weights)
            mean = weighted_mean_bounded_memory(X_basis, weights, chunk_rows=128)
            X_basis -= mean
            X_basis *= np.sqrt(weights * len(X_basis)).astype(np.float32)[:, None]
            weighted_frobenius_sq = frobenius_sq_bounded_memory(X_basis, chunk_rows=128)
            if m == flat_d:
                components = np.eye(flat_d, dtype=np.float32)
                singular_values = np.linalg.svd(
                    np.asarray(X_basis, dtype=np.float64), compute_uv=False
                )[:m]
                method = "outer_train_identity_centered_cache"
            else:
                _, singular_values, components = randomized_svd(
                    X_basis,
                    n_components=m,
                    n_iter=int(args.randomized_n_iter),
                    random_state=int(args.seed + 7919 * int(outer_fold)),
                    n_oversamples=int(max(5, args.randomized_oversamples)),
                )
                components = np.asarray(components, dtype=np.float32)
                singular_values = np.asarray(singular_values, dtype=np.float64)
                method = "outer_train_balanced_randomized_svd_cache"
            gram_error = float(np.max(np.abs(
                components.astype(np.float64) @ components.astype(np.float64).T
                - np.eye(m, dtype=np.float64)
            )))
            energy_fraction = (
                float(np.sum(singular_values ** 2) / weighted_frobenius_sq)
                if weighted_frobenius_sq > 0 else float("nan")
            )
            reducer_diagnostics = {
                "model": model,
                "outer_fold": int(outer_fold),
                "method": method,
                "M_requested": int(requested_dim),
                "M_effective": int(m),
                "basis_rows": int(len(basis_indices)),
                "basis_class_counts": counts.astype(int).tolist(),
                "weighted_basis_frobenius_sq": weighted_frobenius_sq,
                "retained_weighted_energy_fraction": energy_fraction,
                "largest_singular_value": float(singular_values[0]),
                "smallest_retained_singular_value": float(singular_values[-1]),
                "retained_singular_value_ratio": float(singular_values[-1] / singular_values[0]),
                "component_orthogonality_max_abs": gram_error,
                "randomized_n_iter": int(args.randomized_n_iter),
                "randomized_oversamples": int(args.randomized_oversamples),
            }
        del X_basis
        gc.collect()

        with StepTimer(f"{model} fold={outer_fold} project balanced train rows={len(train_indices)}"):
            project_rows_to_npy(
                ds, train_indices, mean, components, tmp_dir / "R_train.npy",
                args.project_max_gap_rows, args.project_max_span_rows,
            )
        with StepTimer(f"{model} fold={outer_fold} project all held-out rows={len(test_indices)}"):
            project_rows_to_npy(
                ds, test_indices, mean, components, tmp_dir / "R_test.npy",
                args.project_max_gap_rows, args.project_max_span_rows,
            )

    np.save(tmp_dir / "y_train.npy", y_train.astype(np.int8))
    np.save(tmp_dir / "subject_train.npy", subject_train.astype(np.int16))
    np.save(tmp_dir / "global_index_train.npy", train_indices.astype(np.int64))
    np.save(tmp_dir / "y_test.npy", y_test.astype(np.int8))
    np.save(tmp_dir / "subject_test.npy", subject_test.astype(np.int16))
    np.save(tmp_dir / "global_index_test.npy", test_indices.astype(np.int64))
    np.save(tmp_dir / "reducer_singular_values.npy", np.asarray(singular_values, dtype=np.float64))
    (tmp_dir / "reducer_diagnostics.json").write_text(
        json.dumps(reducer_diagnostics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    expected.update({
        "basis_rows": int(len(basis_indices)),
        "method": method,
        "created_at": now(),
        "outer_test_isolation": (
            "Held-out subjects are absent from mean, SVD basis, class statistics, "
            "hyperparameter selection, and readout fitting."
        ),
        "training_sampling": (
            f"Outer-training rows are capped per subject/class using {args.sampling_mode}; "
            "every held-out selected row is evaluated."
        ),
        "dimension_preflight": preflight,
    })
    (tmp_dir / "cache_metadata.json").write_text(
        json.dumps(expected, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    for old in cache_dir.iterdir():
        if old.is_file():
            old.unlink()
        elif old.is_dir():
            shutil.rmtree(old)
    for item in tmp_dir.iterdir():
        item.replace(cache_dir / item.name)
    tmp_dir.rmdir()
    log(f"[CACHE] built strict {model} fold={outer_fold} cache at {cache_dir}")

    return FoldReducedCache(
        R_train=np.load(paths["R_train"], mmap_mode="r"),
        y_train=np.load(paths["y_train"], mmap_mode="r"),
        subject_train=np.load(paths["subject_train"], mmap_mode="r"),
        global_index_train=np.load(paths["index_train"], mmap_mode="r"),
        R_test=np.load(paths["R_test"], mmap_mode="r"),
        y_test=np.load(paths["y_test"], mmap_mode="r"),
        subject_test=np.load(paths["subject_test"], mmap_mode="r"),
        global_index_test=np.load(paths["index_test"], mmap_mode="r"),
        n_components=m,
        method=method,
        cache_dir=cache_dir,
    )


@dataclass
class ClassStats:
    means: np.ndarray
    covs: np.ndarray
    pooled: np.ndarray
    pooled_chol: np.ndarray
    class_counts: np.ndarray
    lambda_spectra: np.ndarray
    generalized_eigvecs: np.ndarray
    class_names: List[str]


def covariance_unbiased(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    Z = X - X.mean(axis=0)
    C = (Z.T @ Z) / float(len(X) - 1)
    return 0.5 * (C + C.T)


def rank_condition(C: np.ndarray) -> Tuple[int, float, float]:
    vals = np.linalg.eigvalsh(0.5 * (C + C.T))
    scale = max(float(np.max(np.abs(vals))), 1.0)
    tol = np.finfo(float).eps * max(C.shape) * scale
    pos = vals[vals > tol]
    rank = int(len(pos))
    cond = float(np.max(pos) / np.min(pos)) if len(pos) else float("inf")
    logdet = float(np.sum(np.log(pos))) if rank == C.shape[0] else float("nan")
    return rank, cond, logdet


def fit_class_stats(
    R: np.ndarray,
    y: np.ndarray,
    class_names: Sequence[str],
    cov_mode: str,
) -> ClassStats:
    R = np.asarray(R, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    K = len(class_names)
    M = R.shape[1]
    means = np.zeros((K, M), dtype=np.float64)
    covs = np.zeros((K, M, M), dtype=np.float64)
    counts = np.zeros(K, dtype=np.int64)

    for c in range(K):
        Rc = R[y == c]
        counts[c] = len(Rc)
        if len(Rc) < 2:
            raise ValueError(
                f"Need at least two rows for covariance: class={class_names[c]}, n={len(Rc)}"
            )
        if cov_mode == "geometric" and len(Rc) <= M:
            raise ValueError(
                f"Geometric interpolation requires full-rank empirical class covariance: "
                f"class={class_names[c]}, n={len(Rc)}, M={M}"
            )
        means[c] = Rc.mean(axis=0)
        covs[c] = covariance_unbiased(Rc)
        if cov_mode == "geometric":
            try:
                np.linalg.cholesky(covs[c])
            except np.linalg.LinAlgError as exc:
                raise np.linalg.LinAlgError(
                    f"Non-SPD class covariance {class_names[c]} at M={M}"
                ) from exc

    pooled = np.mean(covs, axis=0)
    pooled = 0.5 * (pooled + pooled.T)
    try:
        pooled_chol = np.linalg.cholesky(pooled)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            f"Equal-class pooled covariance is not SPD at M={M}; "
            f"counts={counts.tolist()}"
        ) from exc

    # Cache the generalized covariance spectrum once.  All alpha values reuse it.
    lambdas = np.zeros((K, M), dtype=np.float64)
    eigvecs = np.zeros((K, M, M), dtype=np.float64)
    for c in range(K):
        vals, vecs = linalg.eigh(covs[c], pooled, check_finite=False)
        if not np.all(np.isfinite(vals)):
            raise np.linalg.LinAlgError(
                f"Invalid generalized covariance eigenvalues: {class_names[c]}"
            )
        scale = max(float(np.max(np.abs(vals))), 1.0)
        tol = 100.0 * np.finfo(np.float64).eps * max(M, 1) * scale
        if np.min(vals) < -tol:
            raise np.linalg.LinAlgError(
                f"Materially negative generalized covariance eigenvalue for "
                f"{class_names[c]}: min={float(np.min(vals)):.3e}, tol={tol:.3e}"
            )
        vals = np.maximum(vals, 0.0)
        if cov_mode == "geometric" and np.any(vals <= 0):
            raise np.linalg.LinAlgError(
                f"Geometric interpolation requires positive generalized eigenvalues: "
                f"{class_names[c]}"
            )
        lambdas[c] = vals
        eigvecs[c] = vecs

    return ClassStats(
        means, covs, pooled, pooled_chol, counts,
        lambdas, eigvecs, list(class_names),
    )


def lda_relative_scores(R: np.ndarray, stats: ClassStats) -> np.ndarray:
    beta = linalg.cho_solve((stats.pooled_chol, True), stats.means.T, check_finite=False).T
    constants = -0.5 * np.sum(stats.means * beta, axis=1)
    absolute = np.asarray(R, dtype=np.float64) @ beta.T + constants[None, :]
    return absolute[:, 1:] - absolute[:, [0]]


@dataclass
class QDAModel:
    diagonal_scales: np.ndarray  # K x M in the pooled-generalized eigenbasis
    logdets: np.ndarray
    alpha: float


def qda_diagonal_scales(stats: ClassStats, alpha: float, mode: str) -> np.ndarray:
    alpha = float(alpha)
    lam = stats.lambda_spectra
    if mode == "geometric":
        scales = lam ** (1.0 - alpha)
    elif mode == "arithmetic":
        scales = (1.0 - alpha) * lam + alpha
    else:
        raise ValueError(mode)
    if np.any(scales <= 0) or not np.all(np.isfinite(scales)):
        raise np.linalg.LinAlgError(
            f"Invalid interpolated covariance scales at alpha={alpha}"
        )
    return scales


def fit_qda(stats: ClassStats, alpha: float, mode: str) -> QDAModel:
    scales = qda_diagonal_scales(stats, alpha, mode)
    pooled_logdet = 2.0 * np.sum(np.log(np.diag(stats.pooled_chol)))
    logdets = pooled_logdet + np.sum(np.log(scales), axis=1)
    return QDAModel(scales, logdets, float(alpha))


def qda_relative_scores(R: np.ndarray, stats: ClassStats, model: QDAModel) -> np.ndarray:
    """Exact QDA scores without reconstructing/interverting each interpolated covariance.

    scipy.linalg.eigh(C_c, C_pool) returns V with V^T C_pool V = I and
    V^T C_c V = diag(lambda).  Both geometric and arithmetic interpolations
    remain diagonal in this basis, so Mahalanobis terms are weighted sums.
    """
    R = np.asarray(R, dtype=np.float64)
    K = len(stats.class_names)
    absolute = np.empty((len(R), K), dtype=np.float64)
    for c in range(K):
        delta = R - stats.means[c]
        z = delta @ stats.generalized_eigvecs[c]
        mahal = np.sum(
            (z * z) / model.diagonal_scales[c][None, :],
            axis=1,
        )
        absolute[:, c] = -0.5 * (model.logdets[c] + mahal)
    return absolute[:, 1:] - absolute[:, [0]]


def qda_relative_scores_grid(
    R: np.ndarray,
    stats: ClassStats,
    alpha_grid: Sequence[float],
    mode: str,
) -> Dict[float, np.ndarray]:
    """Score all alpha values after one class-wise projection of R."""
    R = np.asarray(R, dtype=np.float64)
    alphas = np.asarray([float(a) for a in alpha_grid], dtype=np.float64)
    A = len(alphas)
    K = len(stats.class_names)
    N = len(R)
    absolute = np.empty((N, K, A), dtype=np.float64)
    pooled_logdet = 2.0 * np.sum(np.log(np.diag(stats.pooled_chol)))

    for c in range(K):
        delta = R - stats.means[c]
        z2 = (delta @ stats.generalized_eigvecs[c]) ** 2
        lam = stats.lambda_spectra[c][:, None]
        if mode == "geometric":
            scales = lam ** (1.0 - alphas[None, :])
        elif mode == "arithmetic":
            scales = (1.0 - alphas[None, :]) * lam + alphas[None, :]
        else:
            raise ValueError(mode)
        if np.any(scales <= 0) or not np.all(np.isfinite(scales)):
            raise np.linalg.LinAlgError(
                f"Invalid interpolated covariance scales for class={stats.class_names[c]}"
            )
        mahal = z2 @ (1.0 / scales)
        logdet = pooled_logdet + np.sum(np.log(scales), axis=0)
        absolute[:, c, :] = -0.5 * (mahal + logdet[None, :])

    out: Dict[float, np.ndarray] = {}
    for j, alpha in enumerate(alphas.tolist()):
        out[float(alpha)] = absolute[:, 1:, j] - absolute[:, [0], j]
    return out


def native_prediction(relative_scores: np.ndarray) -> np.ndarray:
    scores = np.concatenate([np.zeros((len(relative_scores), 1)), relative_scores], axis=1)
    return np.argmax(scores, axis=1).astype(np.int64)


@dataclass
class SafeStandardizer:
    mean: np.ndarray
    scale: np.ndarray
    degenerate: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray, tol: float) -> "SafeStandardizer":
        X = np.asarray(X, dtype=np.float64)
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        deg = std < float(tol) * (1.0 + np.abs(mean))
        scale = std.copy()
        scale[deg] = 1.0
        return cls(mean, scale, deg)

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = (np.asarray(X, dtype=np.float64) - self.mean) / self.scale
        Z[:, self.degenerate] = 0.0
        return Z


@dataclass
class FeatureBuilder:
    ell: SafeStandardizer
    q: SafeStandardizer
    r: SafeStandardizer
    tensor: SafeStandardizer

    @classmethod
    def fit(cls, ell: np.ndarray, q: np.ndarray, tol: float) -> "FeatureBuilder":
        r = q - ell
        se = SafeStandardizer.fit(ell, tol)
        sq = SafeStandardizer.fit(q, tol)
        sr = SafeStandardizer.fit(r, tol)
        e = se.transform(ell)
        rr = sr.transform(r)
        tensor = np.einsum("ni,nj->nij", e, rr, optimize=True).reshape(len(e), -1)
        st = SafeStandardizer.fit(tensor, tol)
        return cls(se, sq, sr, st)

    def transform(self, ell: np.ndarray, q: np.ndarray) -> Dict[str, np.ndarray]:
        r = q - ell
        e = self.ell.transform(ell)
        qq = self.q.transform(q)
        rr = self.r.transform(r)
        tensor = np.einsum("ni,nj->nij", e, rr, optimize=True).reshape(len(e), -1)
        tt = self.tensor.transform(tensor)
        return {
            "F0_LD": e,
            "F1_QD": qq,
            "F2_LD_plus_QResidual": np.concatenate([e, rr], axis=1),
            "F3_Tensor": np.concatenate([e, rr, tt], axis=1),
        }

    def degenerate_report(self) -> Dict[str, List[int]]:
        return {
            "ell": np.flatnonzero(self.ell.degenerate).astype(int).tolist(),
            "q": np.flatnonzero(self.q.degenerate).astype(int).tolist(),
            "r": np.flatnonzero(self.r.degenerate).astype(int).tolist(),
            "tensor": np.flatnonzero(self.tensor.degenerate).astype(int).tolist(),
        }

    def arm_collapse(self) -> bool:
        """True when q-ell is entirely degenerate and F0/F1/F2/F3 add no geometry."""
        return bool(np.all(self.r.degenerate))


@dataclass
class RidgeReadout:
    coefficients: np.ndarray  # (p+1) x K, final row intercept
    n_classes: int
    ridge_lambda: float

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return X @ self.coefficients[:-1] + self.coefficients[-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.decision_function(X), axis=1).astype(np.int64)

    @property
    def weight_norm(self) -> float:
        return float(np.linalg.norm(self.coefficients[:-1]))


def fit_ridge_readout(X: np.ndarray, y: np.ndarray, n_classes: int, ridge_lambda: float) -> RidgeReadout:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError(f"Missing readout class: counts={counts.tolist()}")
    w = 1.0 / counts[y]
    w *= len(w) / np.sum(w)
    Xa = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    Y = np.eye(n_classes, dtype=np.float64)[y]
    n_eff = float(np.sum(w))
    A = (Xa.T @ (w[:, None] * Xa)) / n_eff
    penalty = np.eye(Xa.shape[1], dtype=np.float64) * float(ridge_lambda)
    penalty[-1, -1] = 0.0
    B = (Xa.T @ (w[:, None] * Y)) / n_eff
    coef = linalg.solve(A + penalty, B, assume_a="sym", check_finite=False)
    return RidgeReadout(coef, int(n_classes), float(ridge_lambda))


def confusion_fixed(y_true: np.ndarray, y_pred: np.ndarray, K: int) -> np.ndarray:
    cm = np.zeros((K, K), dtype=np.int64)
    for a, b in zip(np.asarray(y_true, int), np.asarray(y_pred, int)):
        if 0 <= a < K and 0 <= b < K:
            cm[a, b] += 1
    return cm


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, K: int) -> Dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    recalls = []
    for c in range(K):
        mask = y_true == c
        recalls.append(float(np.mean(y_pred[mask] == c)) if np.any(mask) else float("nan"))
    return {
        "bacc": finite_mean(recalls),
        "accuracy": float(np.mean(y_true == y_pred)),
        "recalls": recalls,
        "confusion_matrix": confusion_fixed(y_true, y_pred, K),
    }


def evaluate_strict_subset(y_true_local: np.ndarray, y_pred_global: np.ndarray, global_start: int) -> Dict:
    y_true_local = np.asarray(y_true_local, dtype=np.int64)
    y_pred_global = np.asarray(y_pred_global, dtype=np.int64)
    expected_global = y_true_local + int(global_start)
    recalls = []
    for c in range(3):
        mask = y_true_local == c
        recalls.append(
            float(np.mean(y_pred_global[mask] == c + global_start))
            if np.any(mask) else float("nan")
        )
    reject = ~np.isin(y_pred_global, np.arange(global_start, global_start + 3))
    cm = np.zeros((3, 4), dtype=np.int64)
    for yt, yp, rj in zip(y_true_local, y_pred_global, reject):
        if rj:
            cm[yt, 3] += 1
        else:
            cm[yt, yp - global_start] += 1
    return {
        "bacc": finite_mean(recalls),
        "accuracy": float(np.mean(y_pred_global == expected_global)) if len(y_true_local) else float("nan"),
        "recalls": recalls,
        "outside_family_rate": float(np.mean(reject)) if len(reject) else float("nan"),
        "confusion_matrix_with_outside_column": cm,
    }


def subject_balanced_score(y: np.ndarray, pred: np.ndarray, subjects: np.ndarray, K: int) -> float:
    vals = []
    for s in sorted(np.unique(subjects).astype(int).tolist()):
        mask = subjects == s
        vals.append(evaluate(y[mask], pred[mask], K)["bacc"])
    return finite_mean(vals)


def scenario_spec(name: str) -> Tuple[str, np.ndarray, int, List[str]]:
    if name == "global7_to_7class":
        return "global7", np.arange(7), 7, GLOBAL_CLASS_NAMES
    if name == "global7_to_nback3":
        return "global7", np.arange(1, 4), 3, NBACK_CLASS_NAMES
    if name == "global7_to_matb3":
        return "global7", np.arange(4, 7), 3, MATB_CLASS_NAMES
    if name == "nback3_dedicated":
        return "nback3_dedicated", np.arange(1, 4), 3, NBACK_CLASS_NAMES
    if name == "matb3_dedicated":
        return "matb3_dedicated", np.arange(4, 7), 3, MATB_CLASS_NAMES
    raise KeyError(name)


def local_labels(y_global: np.ndarray, global_classes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.isin(y_global, global_classes)
    mapping = {int(c): i for i, c in enumerate(global_classes.tolist())}
    y = np.asarray([mapping[int(v)] for v in y_global[mask]], dtype=np.int64)
    return mask, y


def representation_training_data(rep: str, R: np.ndarray, y_global: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if rep == "global7":
        return R, y_global, GLOBAL_CLASS_NAMES
    if rep == "nback3_dedicated":
        mask, y = local_labels(y_global, np.arange(1, 4))
        return R[mask], y, NBACK_CLASS_NAMES
    if rep == "matb3_dedicated":
        mask, y = local_labels(y_global, np.arange(4, 7))
        return R[mask], y, MATB_CLASS_NAMES
    raise KeyError(rep)


def representation_scores(
    rep: str,
    R_train: np.ndarray,
    y_train_global: np.ndarray,
    R_apply: np.ndarray,
    alpha: float,
    cov_mode: str,
) -> Tuple[ClassStats, np.ndarray, np.ndarray, np.ndarray, np.ndarray, FeatureBuilder]:
    Rfit, yfit, names = representation_training_data(rep, R_train, y_train_global)
    stats = fit_class_stats(Rfit, yfit, names, cov_mode)
    ell_fit = lda_relative_scores(Rfit, stats)
    qda = fit_qda(stats, alpha, cov_mode)
    q_fit = qda_relative_scores(Rfit, stats, qda)
    builder = FeatureBuilder.fit(ell_fit, q_fit, tol=1e-10)
    ell_apply = lda_relative_scores(R_apply, stats)
    q_apply = qda_relative_scores(R_apply, stats, qda)
    return stats, ell_fit, q_fit, ell_apply, q_apply, builder


def balanced_subject_class_indices(
    y: np.ndarray,
    subjects: np.ndarray,
    max_per_subject_class: int,
    seed: int,
) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64)
    subjects = np.asarray(subjects, dtype=np.int64)
    cap = int(max_per_subject_class)
    if cap <= 0:
        return np.arange(len(y), dtype=np.int64)

    rng = np.random.default_rng(seed)
    chosen: List[np.ndarray] = []
    for s in sorted(np.unique(subjects).astype(int).tolist()):
        for c in sorted(np.unique(y[subjects == s]).astype(int).tolist()):
            idx = np.flatnonzero((subjects == s) & (y == c))
            if len(idx) > cap:
                idx = np.sort(rng.choice(idx, size=cap, replace=False))
            chosen.append(idx)
    if not chosen:
        raise ValueError("Balanced inner subsample is empty")
    return np.sort(np.concatenate(chosen).astype(np.int64))


@dataclass
class InnerRepCache:
    stats: ClassStats
    names: List[str]
    R_train_score: np.ndarray
    y_train_score: np.ndarray
    R_val_score: np.ndarray
    y_val_score: np.ndarray
    subject_val_score: np.ndarray


def prepare_inner_caches(
    R_outer_train: np.ndarray,
    y_outer_train: np.ndarray,
    subject_outer_train: np.ndarray,
    inner_val_subject_folds: List[List[int]],
    max_per_subject_class: int,
    seed: int,
    cov_mode: str,
) -> List[Dict[str, InnerRepCache]]:
    caches: List[Dict[str, InnerRepCache]] = []

    for inner_i, val_subjects in enumerate(inner_val_subject_folds, start=1):
        log(f"prepare inner fold {inner_i}/{len(inner_val_subject_folds)}; val_subjects={val_subjects}")
        val_mask = np.isin(subject_outer_train, val_subjects)
        tr_mask = ~val_mask

        Rtr_all = R_outer_train[tr_mask]
        Rva_all = R_outer_train[val_mask]
        ytr_all = y_outer_train[tr_mask]
        yva_all = y_outer_train[val_mask]
        str_all = subject_outer_train[tr_mask]
        sva_all = subject_outer_train[val_mask]

        tr_score_idx = balanced_subject_class_indices(
            ytr_all, str_all, max_per_subject_class,
            seed + inner_i * 100003 + 1,
        )
        va_score_idx = balanced_subject_class_indices(
            yva_all, sva_all, max_per_subject_class,
            seed + inner_i * 100003 + 2,
        )

        Rtr_score_all = Rtr_all[tr_score_idx]
        ytr_score_all = ytr_all[tr_score_idx]
        str_score_all = str_all[tr_score_idx]
        Rva_score_all = Rva_all[va_score_idx]
        yva_score_all = yva_all[va_score_idx]
        sva_score_all = sva_all[va_score_idx]

        fold_cache: Dict[str, InnerRepCache] = {}
        for rep in REPRESENTATIONS:
            with StepTimer(f"inner fold {inner_i} fit covariance stats: {rep}"):
                Rfit, yfit, names = representation_training_data(rep, Rtr_all, ytr_all)
                stats = fit_class_stats(Rfit, yfit, names, cov_mode)

            if rep == "global7":
                Rtr_s, ytr_s, str_s = Rtr_score_all, ytr_score_all, str_score_all
                Rva_s, yva_s, sva_s = Rva_score_all, yva_score_all, sva_score_all
            else:
                classes = np.arange(1, 4) if rep == "nback3_dedicated" else np.arange(4, 7)
                tr_family, ytr_s = local_labels(ytr_score_all, classes)
                va_family, yva_s = local_labels(yva_score_all, classes)
                Rtr_s, str_s = Rtr_score_all[tr_family], str_score_all[tr_family]
                Rva_s, sva_s = Rva_score_all[va_family], sva_score_all[va_family]

            fold_cache[rep] = InnerRepCache(
                stats=stats,
                names=list(names),
                R_train_score=Rtr_s,
                y_train_score=ytr_s,
                R_val_score=Rva_s,
                y_val_score=yva_s,
                subject_val_score=sva_s,
            )

        log(
            f"inner fold {inner_i}: score samples train={len(Rtr_score_all)} "
            f"val={len(Rva_score_all)} cap={max_per_subject_class}"
        )
        caches.append(fold_cache)

    return caches


def select_alphas(
    inner_caches: List[Dict[str, InnerRepCache]],
    inner_val_subject_folds: List[List[int]],
    alpha_grid: Sequence[float],
    cov_mode: str,
) -> Tuple[Dict[str, float], List[Dict]]:
    candidates: Dict[str, Dict[float, List[float]]] = {
        rep: {float(a): [] for a in alpha_grid} for rep in REPRESENTATIONS
    }
    rows: List[Dict] = []

    for inner_i, (fold_cache, val_subjects) in enumerate(
        zip(inner_caches, inner_val_subject_folds), start=1
    ):
        for rep in REPRESENTATIONS:
            cache = fold_cache[rep]
            try:
                q_grid = qda_relative_scores_grid(
                    cache.R_val_score,
                    cache.stats,
                    alpha_grid,
                    cov_mode,
                )
                grid_error = ""
            except Exception as exc:
                q_grid = {}
                grid_error = str(exc)

            for alpha in alpha_grid:
                try:
                    if float(alpha) not in q_grid:
                        raise RuntimeError(grid_error or "QDA alpha-grid scoring failed")
                    pred = native_prediction(q_grid[float(alpha)])
                    score = subject_balanced_score(
                        cache.y_val_score,
                        pred,
                        cache.subject_val_score,
                        len(cache.names),
                    )
                    valid = 1
                    error = ""
                    candidates[rep][float(alpha)].append(score)
                except Exception as exc:
                    score = float("nan")
                    valid = 0
                    error = str(exc)
                rows.append({
                    "representation": rep,
                    "inner_fold": inner_i,
                    "validation_subjects": safe_json(val_subjects),
                    "alpha": float(alpha),
                    "subject_balanced_bacc": score,
                    "valid": valid,
                    "error": error,
                    "n_validation_score_samples": int(len(cache.y_val_score)),
                })

    selected: Dict[str, float] = {}
    for rep in REPRESENTATIONS:
        valid: List[Tuple[float, float]] = []
        for alpha in alpha_grid:
            vals = candidates[rep][float(alpha)]
            if len(vals) == len(inner_caches) and np.all(np.isfinite(vals)):
                valid.append((float(np.mean(vals)), float(alpha)))
        if not valid:
            raise RuntimeError(f"No valid alpha for {rep}")
        valid.sort(key=lambda x: (x[0], x[1]))
        selected[rep] = valid[-1][1]
    return selected, rows


def select_ridge_lambdas(
    inner_caches: List[Dict[str, InnerRepCache]],
    inner_val_subject_folds: List[List[int]],
    selected_alpha: Dict[str, float],
    ridge_grid: Sequence[float],
    cov_mode: str,
    degenerate_tol: float,
) -> Tuple[Dict[str, Dict[str, float]], List[Dict]]:
    score_store: Dict[Tuple[str, str, float], List[float]] = {}
    rows: List[Dict] = []

    for inner_i, (fold_cache, val_subjects) in enumerate(
        zip(inner_caches, inner_val_subject_folds), start=1
    ):
        rep_cache: Dict[
            str,
            Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]
        ] = {}

        for rep in REPRESENTATIONS:
            cache = fold_cache[rep]
            alpha = selected_alpha[rep]
            qda = fit_qda(cache.stats, alpha, cov_mode)

            ell_tr = lda_relative_scores(cache.R_train_score, cache.stats)
            q_tr = qda_relative_scores(cache.R_train_score, cache.stats, qda)
            ell_va = lda_relative_scores(cache.R_val_score, cache.stats)
            q_va = qda_relative_scores(cache.R_val_score, cache.stats, qda)

            builder = FeatureBuilder.fit(ell_tr, q_tr, degenerate_tol)
            rep_cache[rep] = (
                builder.transform(ell_tr, q_tr),
                builder.transform(ell_va, q_va),
                cache.y_train_score,
                cache.y_val_score,
                cache.subject_val_score,
            )

        for scenario in SCENARIOS:
            rep, classes, K, _ = scenario_spec(scenario)
            train_features, val_features, ytr_rep, yva_rep, sva_rep = rep_cache[rep]

            if rep == "global7" and scenario != "global7_to_7class":
                tr_mask, ytr = local_labels(ytr_rep, classes)
                va_mask, yva = local_labels(yva_rep, classes)
                Xtr_by_arm = {arm: train_features[arm][tr_mask] for arm in ARMS}
                Xva_by_arm = {arm: val_features[arm][va_mask] for arm in ARMS}
                sva = sva_rep[va_mask]
            else:
                ytr, yva = ytr_rep, yva_rep
                Xtr_by_arm = train_features
                Xva_by_arm = val_features
                sva = sva_rep

            for arm in ARMS:
                for lam in ridge_grid:
                    readout = fit_ridge_readout(Xtr_by_arm[arm], ytr, K, float(lam))
                    pred = readout.predict(Xva_by_arm[arm])
                    score = subject_balanced_score(yva, pred, sva, K)
                    score_store.setdefault((scenario, arm, float(lam)), []).append(score)
                    rows.append({
                        "scenario": scenario,
                        "representation": rep,
                        "inner_fold": inner_i,
                        "validation_subjects": safe_json(val_subjects),
                        "arm": arm,
                        "ridge_lambda": float(lam),
                        "subject_balanced_bacc": score,
                        "n_train_score_samples": int(len(ytr)),
                        "n_validation_score_samples": int(len(yva)),
                    })

    selected: Dict[str, Dict[str, float]] = {s: {} for s in SCENARIOS}
    for scenario in SCENARIOS:
        for arm in ARMS:
            valid: List[Tuple[float, float]] = []
            for lam in ridge_grid:
                vals = score_store.get((scenario, arm, float(lam)), [])
                if len(vals) == len(inner_caches) and np.all(np.isfinite(vals)):
                    valid.append((float(np.mean(vals)), float(lam)))
            if not valid:
                raise RuntimeError(f"No ridge candidate for {scenario}/{arm}")
            valid.sort(key=lambda x: (x[0], x[1]))
            selected[scenario][arm] = valid[-1][1]
    return selected, rows


def generalized_lambda_rows(
    stats: ClassStats,
    representation: str,
    base: Dict,
) -> Tuple[List[Dict], List[Dict]]:
    """Use already-computed generalized eigenvalues; do not repeat eigvalsh audits."""
    summary_rows: List[Dict] = []
    spectrum_rows: List[Dict] = []
    for c, name in enumerate(stats.class_names):
        vals = np.asarray(stats.lambda_spectra[c], dtype=np.float64)
        scale = max(float(np.max(np.abs(vals))), 1.0)
        tol = 100.0 * np.finfo(np.float64).eps * max(len(vals), 1) * scale
        positive = vals[vals > tol]
        numerical_rank = int(len(positive))
        positive_condition = (
            float(np.max(positive) / np.min(positive)) if len(positive) else float("inf")
        )
        summary_rows.append({
            **base,
            "representation": representation,
            "class_id": int(c),
            "class_name": name,
            "n_class": int(stats.class_counts[c]),
            "dimension": int(len(vals)),
            "lambda_min": float(np.min(vals)),
            "lambda_q05": float(np.quantile(vals, 0.05)),
            "lambda_median": float(np.median(vals)),
            "lambda_q95": float(np.quantile(vals, 0.95)),
            "lambda_max": float(np.max(vals)),
            "lambda_numerical_rank": numerical_rank,
            "lambda_zero_count": int(len(vals) - numerical_rank),
            "lambda_positive_min": float(np.min(positive)) if len(positive) else float("nan"),
            "relative_condition_positive": positive_condition,
        })
        for j, value in enumerate(vals, start=1):
            spectrum_rows.append({
                **base,
                "representation": representation,
                "class_id": int(c),
                "class_name": name,
                "eigen_index": int(j),
                "generalized_lambda": float(value),
            })
    return summary_rows, spectrum_rows


def pooled_covariance_diagnostic(stats: ClassStats, representation: str, base: Dict) -> Dict:
    rank, cond, logdet = rank_condition(stats.pooled)
    return {
        **base,
        "representation": representation,
        "dimension": int(stats.pooled.shape[0]),
        "pooled_rank": int(rank),
        "pooled_condition_number": float(cond),
        "pooled_logdet": float(logdet),
        "class_counts": safe_json(stats.class_counts.astype(int).tolist()),
    }


def metric_row(
    base: Dict,
    scenario: str,
    representation: str,
    arm: str,
    alpha: float,
    ridge_lambda: float,
    train_met: Dict,
    test_met: Dict,
    readout: RidgeReadout,
    feature_dim: int,
    builder: FeatureBuilder,
    q_minus_ell_max_abs: float,
) -> Dict:
    return {
        **base,
        "scenario": scenario,
        "representation": representation,
        "arm": arm,
        "feature_dim": int(feature_dim),
        "alpha": float(alpha),
        "ridge_lambda": float(ridge_lambda),
        "train_bacc": float(train_met["bacc"]),
        "test_bacc": float(test_met["bacc"]),
        "test_accuracy": float(test_met["accuracy"]),
        "train_test_gap": float(train_met["bacc"] - test_met["bacc"]),
        "weight_norm": readout.weight_norm,
        "class_recalls": safe_json(test_met["recalls"]),
        "confusion_matrix": safe_json(test_met["confusion_matrix"].tolist()),
        "degenerate_features": safe_json(builder.degenerate_report()),
        "arm_collapse": int(builder.arm_collapse()),
        "arm_comparison_vacuous": int(builder.arm_collapse()),
        "q_minus_ell_train_max_abs": float(q_minus_ell_max_abs),
    }


def run_outer_fold(
    model: str,
    cache: FoldReducedCache,
    outer_fold: int,
    train_subjects: List[int],
    test_subjects: List[int],
    M: int,
    args: argparse.Namespace,
    fold_dir: Path,
    run_signature: str,
) -> None:
    log(f"{model} fold={outer_fold} M={M}: load strict fold-specific reduced cache")
    Rtr = np.asarray(cache.R_train, dtype=np.float64)
    Rte = np.asarray(cache.R_test, dtype=np.float64)
    ytr = np.asarray(cache.y_train, dtype=np.int64)
    yte = np.asarray(cache.y_test, dtype=np.int64)
    strn = np.asarray(cache.subject_train, dtype=np.int64)
    ste = np.asarray(cache.subject_test, dtype=np.int64)
    M_effective = int(cache.n_components)
    reducer_method = cache.method
    if set(np.unique(strn).astype(int).tolist()) - set(train_subjects):
        raise RuntimeError("Fold cache contains a non-training subject in R_train")
    if set(np.unique(ste).astype(int).tolist()) - set(test_subjects):
        raise RuntimeError("Fold cache contains a non-test subject in R_test")
    log(
        f"{model} fold={outer_fold}: balanced train={len(Rtr)} all-test={len(Rte)} "
        f"M={M_effective}"
    )

    inner_folds = split_subjects(
        train_subjects, args.cross_inner_folds, args.seed + 1000 * outer_fold
    )
    with StepTimer(f"{model} fold={outer_fold} prepare cached inner covariance models"):
        inner_caches = prepare_inner_caches(
            Rtr, ytr, strn, inner_folds,
            args.inner_max_per_subject_class,
            args.seed + 500000 * outer_fold,
            args.cov_interpolation,
        )

    with StepTimer(f"{model} fold={outer_fold} select alpha"):
        selected_alpha, alpha_rows = select_alphas(
            inner_caches, inner_folds,
            args.alpha_grid_values, args.cov_interpolation,
        )
    log(f"{model} fold={outer_fold}: selected alpha={selected_alpha}")

    with StepTimer(f"{model} fold={outer_fold} select ridge lambdas"):
        selected_ridge, ridge_rows = select_ridge_lambdas(
            inner_caches, inner_folds, selected_alpha,
            args.ridge_grid_values, args.cov_interpolation,
            args.degenerate_tol,
        )
    del inner_caches
    gc.collect()

    base = {
        "script_version": SCRIPT_VERSION,
        "run_signature": str(run_signature),
        "seed": int(args.seed),
        "model": model,
        "outer_fold": int(outer_fold),
        "M_requested": int(M),
        "M_effective": int(M_effective),
        "reducer_method": reducer_method,
        "train_subjects": safe_json(train_subjects),
        "test_subjects": safe_json(test_subjects),
    }
    selected_rows: List[Dict] = []
    for rep, alpha in selected_alpha.items():
        selected_rows.append({
            **base, "parameter_type": "alpha", "representation": rep,
            "scenario": "", "arm": "native_QDA", "selected_value": float(alpha),
        })
    for scenario, arm_map in selected_ridge.items():
        rep, _, _, _ = scenario_spec(scenario)
        for arm, lam in arm_map.items():
            selected_rows.append({
                **base, "parameter_type": "ridge_lambda", "representation": rep,
                "scenario": scenario, "arm": arm, "selected_value": float(lam),
            })

    metrics_rows: List[Dict] = []
    per_subject_rows: List[Dict] = []
    lambda_summary_rows: List[Dict] = []
    spectrum_rows: List[Dict] = []
    pooled_covariance_rows: List[Dict] = []
    representation_diagnostic_rows: List[Dict] = []

    # Fit each representation once on the complete outer-training set.
    rep_bundles: Dict[str, Dict] = {}
    for rep in REPRESENTATIONS:
        alpha = selected_alpha[rep]
        Rfit, yfit, names = representation_training_data(rep, Rtr, ytr)
        stats = fit_class_stats(Rfit, yfit, names, args.cov_interpolation)
        qda = fit_qda(stats, alpha, args.cov_interpolation)
        ell_fit = lda_relative_scores(Rfit, stats)
        q_fit = qda_relative_scores(Rfit, stats, qda)
        builder = FeatureBuilder.fit(ell_fit, q_fit, args.degenerate_tol)
        residual_fit = q_fit - ell_fit
        q_minus_ell_max_abs = float(np.max(np.abs(residual_fit)))
        ell_std_raw = np.std(ell_fit, axis=0)
        residual_std_raw = np.std(residual_fit, axis=0)
        residual_to_ld_ratio = residual_std_raw / np.maximum(
            ell_std_raw, np.finfo(np.float64).tiny
        )
        ell_rms = float(np.sqrt(np.mean(ell_fit * ell_fit)))
        q_rms = float(np.sqrt(np.mean(q_fit * q_fit)))
        residual_rms = float(np.sqrt(np.mean(residual_fit * residual_fit)))
        cancellation_floor = np.finfo(np.float64).eps * max(
            ell_rms + q_rms, np.finfo(np.float64).tiny
        )
        residual_cancellation_snr = float(residual_rms / cancellation_floor)
        if builder.arm_collapse():
            log(
                f"[COLLAPSE] {model} fold={outer_fold} rep={rep} alpha={alpha}: "
                "q-ell is degenerate; F0/F1/F2/F3 are algebraically equivalent "
                "up to zero-padded features, so the arm comparison is vacuous."
            )

        if rep == "global7":
            ell_tr = lda_relative_scores(Rtr, stats)
            q_tr = qda_relative_scores(Rtr, stats, qda)
            ell_te = lda_relative_scores(Rte, stats)
            q_te = qda_relative_scores(Rte, stats, qda)
            train_features = builder.transform(ell_tr, q_tr)
            test_features = builder.transform(ell_te, q_te)
            rep_ytr, rep_yte = ytr, yte
            rep_str, rep_ste = strn, ste
        else:
            classes = np.arange(1, 4) if rep == "nback3_dedicated" else np.arange(4, 7)
            tr_mask, rep_ytr = local_labels(ytr, classes)
            te_mask, rep_yte = local_labels(yte, classes)
            ell_te = lda_relative_scores(Rte[te_mask], stats)
            q_te = qda_relative_scores(Rte[te_mask], stats, qda)
            train_features = builder.transform(ell_fit, q_fit)
            test_features = builder.transform(ell_te, q_te)
            rep_str, rep_ste = strn[tr_mask], ste[te_mask]

        rep_bundles[rep] = {
            "stats": stats, "builder": builder,
            "train_features": train_features, "test_features": test_features,
            "ytr": rep_ytr, "yte": rep_yte,
            "str": rep_str, "ste": rep_ste,
            "q_minus_ell_max_abs": q_minus_ell_max_abs,
        }
        lrows, srows = generalized_lambda_rows(stats, rep, base)
        lambda_summary_rows.extend(lrows)
        spectrum_rows.extend(srows)
        pooled_covariance_rows.append(pooled_covariance_diagnostic(stats, rep, base))
        representation_diagnostic_rows.append({
            **base,
            "representation": rep,
            "selected_alpha": float(alpha),
            "q_minus_ell_train_max_abs": q_minus_ell_max_abs,
            "q_minus_ell_train_rms": residual_rms,
            "ell_train_rms": ell_rms,
            "q_train_rms": q_rms,
            "residual_to_ld_std_ratio_median": float(np.median(residual_to_ld_ratio)),
            "residual_to_ld_std_ratio_min": float(np.min(residual_to_ld_ratio)),
            "residual_to_ld_std_ratio_max": float(np.max(residual_to_ld_ratio)),
            "residual_to_ld_std_ratio_by_axis": safe_json(residual_to_ld_ratio.tolist()),
            "residual_cancellation_snr": residual_cancellation_snr,
            "alpha_distance_from_collapse": float(1.0 - alpha),
            "arm_collapse": int(builder.arm_collapse()),
            "arm_comparison_vacuous": int(builder.arm_collapse()),
            "degenerate_features": safe_json(builder.degenerate_report()),
        })

    global_readouts: Dict[str, RidgeReadout] = {}
    global_test_predictions: Dict[str, np.ndarray] = {}

    for scenario in SCENARIOS:
        rep, classes, K, names = scenario_spec(scenario)
        bundle = rep_bundles[rep]
        builder = bundle["builder"]
        train_features = bundle["train_features"]
        test_features = bundle["test_features"]

        if rep == "global7" and scenario != "global7_to_7class":
            tr_mask, ytr_local = local_labels(ytr, classes)
            te_mask, yte_local = local_labels(yte, classes)
            Xtr_map = {a: train_features[a][tr_mask] for a in ARMS}
            Xte_map = {a: test_features[a][te_mask] for a in ARMS}
            str_local, ste_local = strn[tr_mask], ste[te_mask]
        else:
            ytr_local, yte_local = bundle["ytr"], bundle["yte"]
            Xtr_map, Xte_map = train_features, test_features
            str_local, ste_local = bundle["str"], bundle["ste"]

        for arm in ARMS:
            lam = selected_ridge[scenario][arm]
            readout = fit_ridge_readout(Xtr_map[arm], ytr_local, K, lam)
            pred_tr = readout.predict(Xtr_map[arm])
            pred_te = readout.predict(Xte_map[arm])
            train_met = evaluate(ytr_local, pred_tr, K)
            test_met = evaluate(yte_local, pred_te, K)
            row = metric_row(
                base, scenario, rep, arm, selected_alpha[rep], lam,
                train_met, test_met, readout, Xtr_map[arm].shape[1], builder,
                float(bundle["q_minus_ell_max_abs"]),
            )
            row["subject_balanced_test_bacc"] = subject_balanced_score(yte_local, pred_te, ste_local, K)
            row["class_names"] = safe_json(names)
            metrics_rows.append(row)

            for s in sorted(np.unique(ste_local).astype(int).tolist()):
                m = ste_local == s
                smet = evaluate(yte_local[m], pred_te[m], K)
                per_subject_rows.append({
                    **base, "scenario": scenario, "representation": rep, "arm": arm,
                    "test_subject": int(s), "alpha": selected_alpha[rep],
                    "ridge_lambda": lam, "bacc": smet["bacc"], "accuracy": smet["accuracy"],
                    "class_recalls": safe_json(smet["recalls"]),
                })

            if scenario == "global7_to_7class":
                global_readouts[arm] = readout
                global_test_predictions[arm] = pred_te

    # Strict audit: use the actual seven-class readout, not a retrained task-family readout.
    for family, classes, start, names in (
        ("nback3_strict_from_7class", np.arange(1, 4), 1, NBACK_CLASS_NAMES),
        ("matb3_strict_from_7class", np.arange(4, 7), 4, MATB_CLASS_NAMES),
    ):
        test_mask, ylocal = local_labels(yte, classes)
        ste_local = ste[test_mask]
        for arm in ARMS:
            pred_global = global_test_predictions[arm][test_mask]
            met = evaluate_strict_subset(ylocal, pred_global, start)
            metrics_rows.append({
                **base,
                "scenario": family,
                "representation": "global7",
                "arm": arm,
                "feature_dim": int(rep_bundles["global7"]["test_features"][arm].shape[1]),
                "alpha": selected_alpha["global7"],
                "ridge_lambda": selected_ridge["global7_to_7class"][arm],
                "train_bacc": "",
                "test_bacc": met["bacc"],
                "subject_balanced_test_bacc": finite_mean(
                    evaluate_strict_subset(ylocal[ste_local == s], pred_global[ste_local == s], start)["bacc"]
                    for s in np.unique(ste_local)
                ),
                "test_accuracy": met["accuracy"],
                "outside_family_rate": met["outside_family_rate"],
                "class_recalls": safe_json(met["recalls"]),
                "confusion_matrix": safe_json(met["confusion_matrix_with_outside_column"].tolist()),
                "class_names": safe_json(names + ["outside_family"]),
            })

    # Add fold identity to inner candidate tables.
    alpha_rows = [{**base, **r} for r in alpha_rows]
    ridge_rows = [{**base, **r} for r in ridge_rows]

    write_csv(fold_dir / "outer_metrics.csv", metrics_rows)
    write_csv(fold_dir / "per_subject_metrics.csv", per_subject_rows)
    write_csv(fold_dir / "selected_hyperparameters.csv", selected_rows)
    write_csv(fold_dir / "alpha_selection_candidates.csv", alpha_rows)
    write_csv(fold_dir / "ridge_selection_candidates.csv", ridge_rows)
    write_csv(fold_dir / "generalized_lambda_summary.csv", lambda_summary_rows)
    write_csv(fold_dir / "generalized_lambda_spectra.csv", spectrum_rows)
    write_csv(fold_dir / "pooled_covariance_diagnostics.csv", pooled_covariance_rows)
    write_csv(fold_dir / "representation_diagnostics.csv", representation_diagnostic_rows)
    with (fold_dir / "DONE.json").open("w", encoding="utf-8") as fp:
        json.dump({**base, "finished_at": now()}, fp, indent=2, ensure_ascii=False)

    print(f"[{now()}] {model} fold={outer_fold}: completed and checkpointed", flush=True)
    del Rtr, Rte, rep_bundles
    gc.collect()


def benjamini_hochberg(p_values: Sequence[float]) -> List[float]:
    p = np.asarray(p_values, dtype=np.float64)
    q = np.full(len(p), np.nan, dtype=np.float64)
    valid = np.flatnonzero(np.isfinite(p))
    if len(valid) == 0:
        return q.tolist()
    order = valid[np.argsort(p[valid])]
    ranked = p[order] * len(order) / np.arange(1, len(order) + 1, dtype=np.float64)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q[order] = np.minimum(ranked, 1.0)
    return q.tolist()


def paired_arm_contrasts(per_subject_rows: List[Dict]) -> List[Dict]:
    grouped: Dict[Tuple[str, str, str, str], Dict[str, Dict[int, float]]] = {}
    for row in per_subject_rows:
        key = (
            row.get("model", ""), row.get("scenario", ""),
            row.get("representation", ""), row.get("M_requested", ""),
        )
        arm = str(row.get("arm", ""))
        subject = int(row.get("test_subject", -1))
        value = as_float(row.get("bacc"))
        if subject < 0 or not np.isfinite(value):
            continue
        grouped.setdefault(key, {}).setdefault(arm, {})[subject] = float(value)

    rows: List[Dict] = []
    tol = 1e-12
    for key, arm_maps in grouped.items():
        baseline = arm_maps.get("F0_LD", {})
        for arm in ARMS[1:]:
            comparator = arm_maps.get(arm, {})
            common = sorted(set(baseline) & set(comparator))
            diff = np.asarray(
                [comparator[s] - baseline[s] for s in common], dtype=np.float64
            )
            diff = diff[np.isfinite(diff)]
            wins = int(np.sum(diff > tol))
            losses = int(np.sum(diff < -tol))
            ties = int(len(diff) - wins - losses)
            nonzero = diff[np.abs(diff) > tol]
            if len(nonzero) == 0:
                wilcoxon_p = 1.0
                sign_p = 1.0
            else:
                try:
                    wilcoxon_p = float(stats.wilcoxon(
                        diff, zero_method="wilcox", alternative="two-sided"
                    ).pvalue)
                except Exception:
                    wilcoxon_p = float("nan")
                sign_p = float(stats.binomtest(
                    wins, wins + losses, p=0.5, alternative="two-sided"
                ).pvalue)
            rows.append({
                "model": key[0],
                "scenario": key[1],
                "representation": key[2],
                "M_requested": key[3],
                "contrast": f"{arm}_minus_F0_LD",
                "comparison_arm": arm,
                "n_paired_subjects": int(len(diff)),
                "n_nonzero_subjects": int(len(nonzero)),
                "mean_bacc_difference": float(np.mean(diff)) if len(diff) else float("nan"),
                "median_bacc_difference": float(np.median(diff)) if len(diff) else float("nan"),
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "wilcoxon_p": wilcoxon_p,
                "sign_test_p": sign_p,
                "inference_note": (
                    "paired held-out-subject contrast; subjects in one outer fold share "
                    "the same fitted model, so use as a complementary audit to fold summaries"
                ),
            })
    wilcoxon_q = benjamini_hochberg([as_float(r["wilcoxon_p"]) for r in rows])
    sign_q = benjamini_hochberg([as_float(r["sign_test_p"]) for r in rows])
    for row, wq, sq in zip(rows, wilcoxon_q, sign_q):
        row["wilcoxon_bh_q_global"] = wq
        row["sign_test_bh_q_global"] = sq
    return rows


def aggregate_outputs(outdir: Path) -> None:
    tables = ensure_dir(outdir / "tables")
    names = [
        "outer_metrics.csv", "per_subject_metrics.csv", "selected_hyperparameters.csv",
        "alpha_selection_candidates.csv", "ridge_selection_candidates.csv",
        "generalized_lambda_summary.csv", "generalized_lambda_spectra.csv",
        "pooled_covariance_diagnostics.csv", "representation_diagnostics.csv",
    ]
    aggregated: Dict[str, List[Dict]] = {}
    for name in names:
        rows: List[Dict] = []
        for path in sorted((outdir / "folds").glob(f"*/*/{name}")):
            rows.extend(read_csv(path))
        aggregated[name] = rows
        write_csv(tables / name, rows)

    metrics = aggregated["outer_metrics.csv"]
    groups: Dict[Tuple[str, str, str, str, str], List[Dict]] = {}
    for r in metrics:
        key = (
            r.get("model", ""), r.get("scenario", ""), r.get("representation", ""),
            r.get("arm", ""), r.get("M_requested", ""),
        )
        groups.setdefault(key, []).append(r)
    summary: List[Dict] = []
    for key, rows in groups.items():
        summary.append({
            "model": key[0], "scenario": key[1], "representation": key[2],
            "arm": key[3], "M_requested": key[4], "n_outer_folds": len(rows),
            "test_bacc_mean": finite_mean(as_float(r.get("test_bacc")) for r in rows),
            "test_bacc_sem": finite_sem(as_float(r.get("test_bacc")) for r in rows),
            "subject_balanced_test_bacc_mean": finite_mean(
                as_float(r.get("subject_balanced_test_bacc")) for r in rows
            ),
            "subject_balanced_test_bacc_sem": finite_sem(
                as_float(r.get("subject_balanced_test_bacc")) for r in rows
            ),
            "test_accuracy_mean": finite_mean(as_float(r.get("test_accuracy")) for r in rows),
            "train_test_gap_mean": finite_mean(as_float(r.get("train_test_gap")) for r in rows),
            "outside_family_rate_mean": finite_mean(
                as_float(r.get("outside_family_rate")) for r in rows
            ),
            "arm_collapse_rate": finite_mean(as_float(r.get("arm_collapse")) for r in rows),
        })
    write_csv(tables / "summary_by_model_scenario_arm.csv", summary)
    write_csv(
        tables / "arm_contrast_paired_subject.csv",
        paired_arm_contrasts(aggregated["per_subject_metrics.csv"]),
    )


def inspect_model(path: str, args: argparse.Namespace) -> Dict:
    with h5py.File(path, "r") as f:
        ds = f[args.embedding_key]
        raw = read_vector(f, args.label_key).astype(np.int64)
        task = read_vector(f, args.task_key).astype(np.int64)
        subjects = read_vector(f, args.subject_key).astype(np.int64)
        sessions = read_vector(f, args.session_key).astype(np.int64)
        y = make_global_labels(raw, task, args.rest_raw_label)
        selected = y >= 0
        selected_indices = np.flatnonzero(selected).astype(np.int64)
        y_selected = y[selected].astype(np.int64)
        selected_subjects = subjects[selected].astype(np.int64)
        selected_sessions = sessions[selected].astype(np.int64)
        subject_list = sorted(np.unique(selected_subjects).astype(int).tolist())
        subject_class_rows: List[Dict] = []
        subject_session_class_rows: List[Dict] = []
        for s in subject_list:
            for c in range(7):
                subject_class_rows.append({
                    "subject": int(s),
                    "class_id": int(c),
                    "class_name": GLOBAL_CLASS_NAMES[c],
                    "count": int(np.sum((selected_subjects == s) & (y_selected == c))),
                })
            for session in sorted(np.unique(selected_sessions[selected_subjects == s]).astype(int).tolist()):
                for c in range(7):
                    subject_session_class_rows.append({
                        "subject": int(s),
                        "session": int(session),
                        "class_id": int(c),
                        "count": int(np.sum(
                            (selected_subjects == s)
                            & (selected_sessions == session)
                            & (y_selected == c)
                        )),
                    })
        count_matrix = np.asarray(
            [[r["subject"], r["class_id"], r["count"]] for r in subject_class_rows],
            dtype=np.int64,
        )
        session_count_matrix = np.asarray(
            [[r["subject"], r["session"], r["class_id"], r["count"]]
             for r in subject_session_class_rows],
            dtype=np.int64,
        )
        return {
            "embedding_shape": list(ds.shape),
            "flat_dim": flat_dim(ds),
            "subjects": subject_list,
            "subject_window_counts": {
                int(s): int(np.sum(selected_subjects == int(s))) for s in subject_list
            },
            "class_counts": {GLOBAL_CLASS_NAMES[c]: int(np.sum(y_selected == c)) for c in range(7)},
            "subject_class_counts": subject_class_rows,
            "subject_session_class_counts": subject_session_class_rows,
            "subject_class_count_sha256": array_sha256(count_matrix),
            "subject_session_class_count_sha256": array_sha256(session_count_matrix),
            "_selected_indices": selected_indices,
            "_y_selected": y_selected,
            "_subject_selected": selected_subjects,
            "_session_selected": selected_sessions,
        }


def audit_subject_content_fingerprints(
    model: str,
    path: str,
    info: Dict,
    args: argparse.Namespace,
) -> Tuple[List[Dict], List[List[int]]]:
    n_rows = int(args.duplicate_audit_rows)
    if n_rows <= 0:
        return [], []

    selected_indices = np.asarray(info["_selected_indices"], dtype=np.int64)
    selected_subjects = np.asarray(info["_subject_selected"], dtype=np.int64)
    rows: List[Dict] = []

    def audit_positions(n: int, k: int) -> np.ndarray:
        k = min(int(k), int(n))
        if k <= 0:
            return np.empty(0, dtype=np.int64)
        edge = min(4, k // 2)
        positions = list(range(edge))
        positions.extend(range(max(edge, n - edge), n))
        remaining = k - len(set(positions))
        if remaining > 0:
            positions.extend(
                np.floor(np.linspace(0, n - 1, remaining + 2)[1:-1]).astype(int).tolist()
            )
        return np.asarray(sorted(set(positions))[:k], dtype=np.int64)

    with h5py.File(path, "r") as h5:
        ds = h5[args.embedding_key]
        d = flat_dim(ds)
        for s in info["subjects"]:
            ledger = selected_indices[selected_subjects == int(s)]
            pos = audit_positions(len(ledger), n_rows)
            indices = ledger[pos]
            sample = read_rows_by_merged_slices(
                ds, indices,
                max_gap_rows=int(args.duplicate_audit_max_gap_rows),
                max_span_rows=int(args.duplicate_audit_max_span_rows),
            )
            digest = array_sha256(np.asarray(sample, dtype=np.float32))
            rows.append({
                "model": model,
                "subject": int(s),
                "n_subject_rows": int(len(ledger)),
                "n_audit_rows": int(len(indices)),
                "flat_dim": int(d),
                "audit_indices_sha256": array_sha256(indices),
                "embedding_fingerprint_sha256": digest,
                "sample_mean": float(np.mean(sample, dtype=np.float64)),
                "sample_std": float(np.std(sample, dtype=np.float64)),
            })
            del sample

    by_digest: Dict[str, List[int]] = {}
    for row in rows:
        by_digest.setdefault(str(row["embedding_fingerprint_sha256"]), []).append(int(row["subject"]))
    duplicates = [sorted(v) for v in by_digest.values() if len(v) > 1]
    if duplicates:
        log(f"[DUPLICATE SUBJECT AUDIT] {model}: exact fingerprint collisions={duplicates}")
    else:
        log(f"[DUPLICATE SUBJECT AUDIT] {model}: passed for {len(rows)} subjects")
    return rows, duplicates


def public_inspection(info: Dict) -> Dict:
    return {k: v for k, v in info.items() if not str(k).startswith("_")}


def integration_self_test(seed: int = 0) -> Dict:
    rng = np.random.default_rng(int(seed))
    K, M, n_per_class = 3, 8, 5
    names = [f"c{i}" for i in range(K)]
    y = np.repeat(np.arange(K, dtype=np.int64), n_per_class)
    R = rng.normal(size=(len(y), M))
    R += np.asarray(y, dtype=np.float64)[:, None] * np.linspace(0.05, 0.3, M)[None, :]

    stats = fit_class_stats(R, y, names, "arithmetic")
    ell = lda_relative_scores(R, stats)
    q = qda_relative_scores(R, stats, fit_qda(stats, 0.4, "arithmetic"))
    if not np.all(np.isfinite(q)):
        raise AssertionError("Arithmetic QDA failed with singular empirical class covariances")

    alpha1_q = qda_relative_scores(R, stats, fit_qda(stats, 1.0, "arithmetic"))
    alpha1_error = float(np.max(np.abs(alpha1_q - ell)))
    alpha1_builder = FeatureBuilder.fit(ell, alpha1_q, 1e-10)
    if alpha1_error > 1e-9 or not alpha1_builder.arm_collapse():
        raise AssertionError(f"alpha=1 collapse audit failed: error={alpha1_error}")

    Q, _ = np.linalg.qr(rng.normal(size=(M, M)))
    scales = np.linspace(0.7, 1.4, M)
    transform = Q @ np.diag(scales)
    shift = rng.normal(size=M)
    R2 = R @ transform.T + shift[None, :]
    stats2 = fit_class_stats(R2, y, names, "arithmetic")
    ell2 = lda_relative_scores(R2, stats2)
    q2 = qda_relative_scores(R2, stats2, fit_qda(stats2, 0.4, "arithmetic"))
    lda_invariance_error = float(np.max(np.abs(ell2 - ell)))
    qda_invariance_error = float(np.max(np.abs(q2 - q)))
    if lda_invariance_error > 1e-8 or qda_invariance_error > 1e-8:
        raise AssertionError(
            f"Coordinate invariance failed: LDA={lda_invariance_error}, QDA={qda_invariance_error}"
        )

    X = rng.normal(size=(60, 7))
    yr = np.repeat(np.arange(3, dtype=np.int64), 20)
    readout = fit_ridge_readout(X, yr, 3, 0.1)
    repeated = fit_ridge_readout(np.repeat(X, 5, axis=0), np.repeat(yr, 5), 3, 0.1)
    ridge_duplication_error = float(np.max(np.abs(readout.coefficients - repeated.coefficients)))
    if ridge_duplication_error > 1e-10:
        raise AssertionError(f"Ridge sample-size normalization failed: {ridge_duplication_error}")

    ledger_y: List[int] = []
    ledger_s: List[int] = []
    ledger_session: List[int] = []
    for s in (1, 2):
        for c in range(7):
            for session in (1, 2):
                ledger_y.extend([c] * 20)
                ledger_s.extend([s] * 20)
                ledger_session.extend([session] * 20)
    selected_indices = np.arange(len(ledger_y), dtype=np.int64)
    picked1 = choose_balanced_rows(
        selected_indices, np.asarray(ledger_y), np.asarray(ledger_s),
        np.asarray(ledger_session), [1, 2], 12, 123,
        "block_spread", 3,
    )
    picked2 = choose_balanced_rows(
        selected_indices, np.asarray(ledger_y), np.asarray(ledger_s),
        np.asarray(ledger_session), [1, 2], 12, 123,
        "block_spread", 3,
    )
    if len(picked1) != 2 * 7 * 12 or not np.array_equal(picked1, picked2):
        raise AssertionError("Balanced block-spread sampler is not exact and deterministic")

    return {
        "status": "ok",
        "singular_class_covariance_arithmetic_qda": True,
        "alpha1_max_abs_q_minus_ell": alpha1_error,
        "alpha1_arm_collapse": alpha1_builder.arm_collapse(),
        "lda_coordinate_invariance_max_abs": lda_invariance_error,
        "qda_coordinate_invariance_max_abs": qda_invariance_error,
        "ridge_duplication_invariance_max_abs": ridge_duplication_error,
        "sampler_rows": int(len(picked1)),
        "sampler_sha256": array_sha256(picked1),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Optimized cross-subject LD/QD/Tensor experiment")
    p.add_argument("--models", default="CBraMod,LaBraM,EEGPT,EEGMamba,BIOT")
    p.add_argument("--h5", default="")
    p.add_argument("--model-name", default="")
    p.add_argument("--outdir", required=True)
    p.add_argument("--embedding-key", default="embedding")
    p.add_argument("--label-key", default="label")
    p.add_argument("--task-key", default="task_id")
    p.add_argument("--subject-key", default="subject_id")
    p.add_argument("--session-key", default="session")
    p.add_argument("--rest-raw-label", type=int, default=1, choices=[0, 1])
    p.add_argument("--expected-n-subjects", type=int, default=27)
    p.add_argument("--outer-folds", type=int, default=4)
    p.add_argument("--cross-inner-folds", type=int, default=3)
    p.add_argument(
        "--model-dims",
        default="CBraMod:500,LaBraM:500,EEGPT:500,EEGMamba:500,BIOT:256",
        help="Per-model reducer dimensions as Model:M pairs.",
    )
    p.add_argument(
        "--cross-dims",
        default="",
        help="Optional single fallback dimension, mainly for --h5/--model-name runs.",
    )
    p.add_argument("--alpha-grid", default="0.001,0.01,0.05,0.1,0.25,0.5,0.9,0.99,1")
    p.add_argument("--ridge-grid", default="0.0001,0.001,0.01,0.1,1,10,100")
    p.add_argument("--cov-interpolation", choices=["geometric", "arithmetic"], default="arithmetic")
    p.add_argument("--folds", default="", help="Optional outer-fold IDs, e.g. 1 or 1,2")
    p.add_argument(
        "--train-cap-per-subject-class", type=int, default=96,
        help="Maximum outer-training windows retained per subject and global class; <=0 uses all.",
    )
    p.add_argument(
        "--basis-cap-per-subject-class", type=int, default=24,
        help="Maximum train-only SVD-basis windows per subject and global class.",
    )
    p.add_argument(
        "--sampling-mode", choices=["block_spread", "contiguous"], default="block_spread",
        help="Temporal sampling policy inside each subject/session/class ledger.",
    )
    p.add_argument("--train-blocks-per-session", type=int, default=4)
    p.add_argument("--basis-blocks-per-session", type=int, default=2)
    p.add_argument("--basis-max-gap-rows", type=int, default=16)
    p.add_argument("--basis-max-span-rows", type=int, default=256)
    p.add_argument("--project-max-gap-rows", type=int, default=8)
    p.add_argument("--project-max-span-rows", type=int, default=256)
    p.add_argument("--randomized-n-iter", type=int, default=4)
    p.add_argument("--randomized-oversamples", type=int, default=32)
    p.add_argument(
        "--require-requested-dim", action="store_true",
        help="Fail instead of silently lowering M when an exact inner-fold covariance is infeasible.",
    )
    p.add_argument(
        "--inner-max-per-subject-class",
        type=int,
        default=96,
        help=(
            "Balanced deterministic cap used only for inner alpha/ridge scoring. "
            "Covariances are always fitted on all inner-training windows, and the "
            "outer final fit/evaluation uses all windows. Matching the outer train cap "
            "avoids an unnecessary ridge sample-size mismatch. Set <=0 to disable."
        ),
    )
    p.add_argument("--degenerate-tol", type=float, default=1e-10)
    p.add_argument(
        "--duplicate-audit-rows", type=int, default=16,
        help="Full-vector rows sampled per subject for exact duplicate-content fingerprints; <=0 disables.",
    )
    p.add_argument("--duplicate-audit-max-gap-rows", type=int, default=0)
    p.add_argument("--duplicate-audit-max-span-rows", type=int, default=1)
    p.add_argument("--fail-on-duplicate-subjects", action="store_true")
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.self_test:
        print(json.dumps(integration_self_test(args.seed), indent=2, ensure_ascii=False))
        return

    args.alpha_grid_values = sorted(set(parse_list(args.alpha_grid, float)))
    args.ridge_grid_values = sorted(set(parse_list(args.ridge_grid, float)))
    if not args.alpha_grid_values:
        raise ValueError("--alpha-grid is empty")
    if any((a < 0.0 or a > 1.0) for a in args.alpha_grid_values):
        raise ValueError("--alpha-grid values must lie in [0,1]")
    if args.cov_interpolation == "arithmetic" and any(a <= 0.0 for a in args.alpha_grid_values):
        raise ValueError(
            "Arithmetic pooled-covariance shrinkage requires alpha > 0; remove alpha=0."
        )
    if not args.ridge_grid_values or any(lam < 0.0 for lam in args.ridge_grid_values):
        raise ValueError("--ridge-grid must contain non-negative values")
    if args.train_blocks_per_session < 1 or args.basis_blocks_per_session < 1:
        raise ValueError("Sampling block counts must be positive")
    if args.randomized_n_iter < 0 or args.randomized_oversamples < 1:
        raise ValueError("Invalid randomized SVD settings")

    model_dims = parse_model_dims(args.model_dims)
    fallback_dims = parse_list(args.cross_dims, int) if str(args.cross_dims).strip() else []
    if len(fallback_dims) > 1:
        raise ValueError("--cross-dims accepts at most one fallback dimension")
    fallback_M = fallback_dims[0] if fallback_dims else None

    outdir = ensure_dir(args.outdir)
    ensure_dir(outdir / "folds")
    ensure_dir(outdir / "tables")
    ensure_dir(outdir / "preflight")

    if args.h5:
        if not args.model_name:
            raise ValueError("--model-name required with --h5")
        model_paths = {args.model_name: args.h5}
    else:
        requested = parse_list(args.models, str)
        unknown = [m for m in requested if m not in MODEL_PATHS]
        if unknown:
            raise KeyError(f"Unknown models: {unknown}")
        model_paths = {m: MODEL_PATHS[m] for m in requested}

    resolved_model_dims: Dict[str, int] = {}
    for model in model_paths:
        if model in model_dims:
            resolved_model_dims[model] = int(model_dims[model])
        elif fallback_M is not None:
            resolved_model_dims[model] = int(fallback_M)
        elif model in DEFAULT_MODEL_DIMS:
            resolved_model_dims[model] = int(DEFAULT_MODEL_DIMS[model])
        else:
            raise ValueError(f"No reducer dimension specified for model {model!r}")

    print("=" * 112, flush=True)
    print(f"Cross-subject LD/QD/Tensor [{SCRIPT_VERSION}]", flush=True)
    print(f"Models: {list(model_paths)}", flush=True)
    print(f"Model-specific M={resolved_model_dims}", flush=True)
    print(
        f"covariance={args.cov_interpolation}; alpha={args.alpha_grid_values}; "
        f"ridge={args.ridge_grid_values}; inner cap={args.inner_max_per_subject_class}",
        flush=True,
    )
    print(
        f"sampling={args.sampling_mode} train-blocks/session={args.train_blocks_per_session} "
        f"basis-blocks/session={args.basis_blocks_per_session}; "
        f"randomized SVD n_iter={args.randomized_n_iter}",
        flush=True,
    )
    print(f"Output: {outdir}", flush=True)
    print("=" * 112, flush=True)

    requested_fold_ids = set(parse_list(args.folds, int)) if str(args.folds).strip() else set()
    infos: Dict[str, Dict] = {}
    inspections: Dict[str, Dict] = {}
    duplicate_rows: List[Dict] = []
    duplicate_groups: Dict[str, List[List[int]]] = {}
    reference_subjects: Optional[List[int]] = None
    reference_count_digest: Optional[str] = None
    reference_session_count_digest: Optional[str] = None
    folds: Optional[List[List[int]]] = None

    # First pass: inspect every model and run integrity audits before any expensive fold SVD.
    for model, path in model_paths.items():
        with StepTimer(f"inspect {model} metadata"):
            info = inspect_model(path, args)
        infos[model] = info
        inspections[model] = public_inspection(info)
        log(
            f"{model}: shape={info['embedding_shape']} flat_dim={info['flat_dim']} "
            f"subjects={len(info['subjects'])} class_counts={info['class_counts']}"
        )

        if reference_subjects is None:
            reference_subjects = list(info["subjects"])
            if args.expected_n_subjects > 0 and len(reference_subjects) != args.expected_n_subjects:
                raise ValueError(
                    f"Expected {args.expected_n_subjects} subjects, found {len(reference_subjects)}"
                )
            reference_count_digest = str(info["subject_class_count_sha256"])
            reference_session_count_digest = str(info["subject_session_class_count_sha256"])
            folds = make_subject_kfold_groups_from_counts(
                reference_subjects, info["subject_window_counts"], args.outer_folds, args.seed
            )
            (outdir / "fold_assignments.json").write_text(
                json.dumps({f"fold_{i+1}": x for i, x in enumerate(folds)}, indent=2),
                encoding="utf-8",
            )
            log(f"shared outer folds={folds}")
        else:
            if info["subjects"] != reference_subjects:
                raise ValueError(
                    f"Subject mismatch for {model}: {info['subjects']} vs {reference_subjects}"
                )
            if str(info["subject_class_count_sha256"]) != reference_count_digest:
                raise ValueError(
                    f"Subject/class window-count mismatch for {model}; shared folds would be invalid"
                )
            if str(info["subject_session_class_count_sha256"]) != reference_session_count_digest:
                raise ValueError(
                    f"Subject/session/class window-count mismatch for {model}; sampling would differ"
                )

    assert reference_subjects is not None and folds is not None

    # Metadata-only dimension preflight for every requested model/fold.
    preflight_full: List[Dict] = []
    preflight_csv: List[Dict] = []
    for model, info in infos.items():
        requested_M = int(resolved_model_dims[model])
        for outer_i, test_subjects in enumerate(folds, start=1):
            if requested_fold_ids and outer_i not in requested_fold_ids:
                continue
            train_subjects = [s for s in reference_subjects if s not in set(test_subjects)]
            pf = fold_dimension_preflight(
                model, info, outer_i, train_subjects, test_subjects,
                requested_M, args,
            )
            preflight_full.append(pf)
            preflight_csv.append({
                **{k: v for k, v in pf.items()
                   if k not in {"structural_caps", "representation_details"}},
                "structural_caps": safe_json(pf["structural_caps"]),
                "representation_details": safe_json(pf["representation_details"]),
            })
            status = "PASS" if pf["feasible"] else "FAIL"
            log(
                f"[DIMENSION PREFLIGHT {status}] {model} fold={outer_i}: "
                f"requested M={requested_M}, max={pf['maximum_feasible_M']}"
            )

    write_csv(outdir / "preflight" / "dimension_preflight.csv", preflight_csv)
    (outdir / "preflight" / "dimension_preflight.json").write_text(
        json.dumps(preflight_full, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    infeasible = [r for r in preflight_full if not r["feasible"]]
    if infeasible and args.require_requested_dim:
        compact = [
            (r["model"], r["outer_fold"], r["requested_M"], r["maximum_feasible_M"])
            for r in infeasible
        ]
        raise RuntimeError(f"Requested dimensions failed metadata preflight: {compact}")

    # Only after the dimension gate passes do we touch embedding rows for duplicate fingerprints.
    for model, path in model_paths.items():
        with StepTimer(f"{model} duplicate-subject content audit"):
            model_rows, duplicates = audit_subject_content_fingerprints(
                model, path, infos[model], args
            )
        duplicate_rows.extend(model_rows)
        duplicate_groups[model] = duplicates
    write_csv(outdir / "preflight" / "subject_content_fingerprints.csv", duplicate_rows)
    (outdir / "preflight" / "duplicate_subject_groups.json").write_text(
        json.dumps(duplicate_groups, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    duplicate_failures = {m: g for m, g in duplicate_groups.items() if g}
    if duplicate_failures and args.fail_on_duplicate_subjects:
        raise RuntimeError(
            f"Exact duplicate-subject fingerprints detected: {duplicate_failures}. "
            "Formal cross-subject evaluation is blocked."
        )

    base_metadata = {
        "script_version": SCRIPT_VERSION,
        "created_at": now(),
        "window_length": "5 seconds per H5 row",
        "rest_definition": "EO only (raw label 1)" if args.rest_raw_label == 1 else "EC only",
        "model_paths": model_paths,
        "model_dims_requested": resolved_model_dims,
        "alpha_grid": args.alpha_grid_values,
        "ridge_grid": args.ridge_grid_values,
        "covariance_interpolation": args.cov_interpolation,
        "covariance_regularization": (
            "C_alpha=(1-alpha) C_class + alpha C_pool; alpha>0; no pseudoinverse"
            if args.cov_interpolation == "arithmetic"
            else "geometric covariance interpolation requiring SPD empirical class covariances"
        ),
        "readout": (
            "closed-form class-balanced multiclass ridge with weighted loss normalized "
            "by total sample weight"
        ),
        "reducer_policy": (
            "one strict outer-train-only balanced randomized SVD cache per outer fold; "
            "held-out outer subjects never enter centering or SVD fitting"
        ),
        "inner_selection_scope": (
            "alpha/ridge inner subject folds reuse the fixed outer-train reducer; the reducer is not "
            "refit inside inner folds. Outer-test isolation remains strict."
        ),
        "outer_training_sampling": (
            f"up to {args.train_cap_per_subject_class} windows per training subject/global class; "
            f"{args.sampling_mode} with {args.train_blocks_per_session} blocks/session; "
            "all selected held-out windows are evaluated"
        ),
        "inner_selection_sampling": (
            f"up to {args.inner_max_per_subject_class} windows per subject/class for alpha/ridge "
            "scoring only; full inner-training data fit covariances"
        ),
        "integrity_audits": {
            "duplicate_subject_groups": duplicate_groups,
            "subject_class_count_sha256": reference_count_digest,
            "subject_session_class_count_sha256": reference_session_count_digest,
            "dimension_preflight": preflight_full,
        },
        "inspections_completed": inspections,
        "common_subjects": reference_subjects,
        "outer_subject_folds": folds,
        "scenarios": list(SCENARIOS) + [
            "nback3_strict_from_7class", "matb3_strict_from_7class"
        ],
        "args": {k: v for k, v in vars(args).items() if not k.endswith("_values")},
        "last_updated": now(),
    }
    (outdir / "run_metadata.json").write_text(
        json.dumps(base_metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if args.preflight_only:
        log("Preflight-only mode completed; no fold reducer or classifier was fitted")
        return

    # Second pass: all integrity and dimension checks have passed, so run expensive folds.
    for model, path in model_paths.items():
        info = infos[model]
        M = int(resolved_model_dims[model])
        for outer_i, test_subjects in enumerate(folds, start=1):
            if requested_fold_ids and outer_i not in requested_fold_ids:
                continue
            train_subjects = [s for s in reference_subjects if s not in set(test_subjects)]
            fold_dir = ensure_dir(outdir / "folds" / model / f"fold_{outer_i}")
            done = fold_dir / "DONE.json"
            run_signature = fold_run_signature(
                model, path, outer_i, train_subjects, test_subjects, M, args
            )
            if args.resume and done.exists() and marker_matches_current_script(
                done, run_signature
            ):
                log(f"[RESUME] skip completed {model} fold={outer_i}")
                continue
            if args.resume and done.exists():
                log(f"[RESUME INVALIDATED] rebuilding stale marker {done}")

            with StepTimer(f"{model} fold={outer_i} build/load strict reduced cache"):
                cache = build_or_load_fold_cache(
                    model, path, info, outer_i, train_subjects,
                    test_subjects, M, args, outdir,
                )
            with StepTimer(f"{model} outer fold {outer_i} total"):
                run_outer_fold(
                    model, cache, outer_i, train_subjects,
                    test_subjects, M, args, fold_dir, run_signature,
                )
            del cache
            gc.collect()
            aggregate_outputs(outdir)
        log(f"{model}: all requested folds completed")

    aggregate_outputs(outdir)
    log("All requested models and folds completed")


if __name__ == "__main__":
    main()
