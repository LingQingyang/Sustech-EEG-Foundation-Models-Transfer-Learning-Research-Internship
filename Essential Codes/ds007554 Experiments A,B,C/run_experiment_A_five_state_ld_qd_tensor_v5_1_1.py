#!/usr/bin/env python3
"""Experiment A v5.0: five-state LD/QD/tensor endpoint geometry for ds007554.

Scientific question
-------------------
Within one subject, which geometric order is required to distinguish the five
endpoint states retained in ds007554?

    0 Baseline
    1 Mental Arithmetic (MA)
    2 N-back (NB)
    3 N-back Arithmetic (NBMA)
    4 Full Integrated Task (Full)

The experiment is strictly within-subject and leave-one-session-out.  For every
outer fold, all reducer fitting, class statistics, covariance interpolation,
feature standardization, and ridge selection use the two training sessions only.
The held-out session is used once for final evaluation and structured label-null
calculation.

Rest-relative feature ladder
----------------------------
With Baseline as class 0 and four task states:

    ell in R^4                  pooled-covariance LDA relative scores
    q   in R^4                  interpolated-QDA relative scores
    r = q - ell                 quadratic residual beyond LDA

    F0 = z(ell)                                            4D
    F1 = z(q)                                              4D
    F2 = z(ell) direct-sum z(r)                            8D
    F3 = F2 direct-sum z(vec(z(ell) outer z(r)))          24D

One shared positive alpha is selected by nested session-heldout native-QDA
performance.  Formal high-rank runs use arithmetic pooled-covariance shrinkage,
so empirical class covariances may be singular while every fitted QDA
covariance remains positive definite.  A class-balanced closed-form ridge
readout is selected separately for each scenario and feature arm.

Scenarios
---------
    five_state     Baseline / MA / NB / NBMA / Full
    context2       Baseline vs pooled Task
    task4          MA / NB / NBMA / Full, evaluated only on task windows

Held-out nulls are cheap and leakage-free because the trained predictor is fixed
before any held-out labels are inspected:
    * context2: circularly shift the Baseline/Task phase sequence inside each run
    * task4: permute the four task identities among runs in the held-out session
    * five_state: combine both operations

The null unit is therefore a complete run or within-run phase sequence, never an
individual window.
"""

from __future__ import annotations

import argparse
import itertools
import csv
import gc
import json
import math
import os
import platform
import re
import sys
import time
import traceback
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np
from sklearn.utils.extmath import randomized_svd

from ld_qd_feature_core_v5_0 import (
    ARMS,
    ClassStats,
    FeatureBuilder,
    evaluate_predictions,
    feature_dimensions,
    finite_mean,
    fit_class_stats,
    fit_qda_model,
    fit_ridge_readout,
    lda_relative_scores,
    native_prediction,
    qda_relative_scores,
    rank_condition,
    self_test as core_self_test,
)


SCRIPT_VERSION = "2026-07-30-expA-five-state-ld-qd-tensor-v5.1.1-stable-generalized-covariance-rank-atomic"

MODEL_PATHS: Dict[str, str] = {
    "EEGMamba": "/mnt/dataset4/fuxy/FMS/FM/EEGMamba/output_emb/ds007554_embeddings.h5",
    "BIOT": "/mnt/dataset4/fuxy/FMS/FM/BIOT/output_emb/ds007554_embeddings.h5",
    "LaBraM": "/mnt/dataset4/fuxy/FMS/FM/LaBraM/output_emb/ds007554_embeddings.h5",
    "CBraMod": "/mnt/dataset4/fuxy/FMS/FM/CBraMod/output_emb/ds007554_embeddings.h5",
    "EEGPT": "/mnt/dataset4/fuxy/FMS/FM/EEGPT/output_emb/ds007554_embeddings.h5",
}

STATE_NAMES = ["Baseline", "MA", "NB", "NBMA", "Full"]
TASK_STATE_NAMES = ["MA", "NB", "NBMA", "Full"]
DEFAULT_TASK_IDS = {"MA": 0, "NB": 1, "NBMA": 5, "Full": 6}
SCENARIOS = ("five_state", "context2", "task4")
REQUIRED_KEYS = (
    "embedding",
    "subject_id",
    "session_id",
    "run_id",
    "sample_start",
    "sample_end",
    "phase_id",
    "task_id",
)


# =============================================================================
# Utilities
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


def marker_matches_current_script(path: Path) -> bool:
    try:
        with Path(path).open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        return str(payload.get("script_version", "")) == SCRIPT_VERSION
    except Exception:
        return False


def is_exact_covariance_geometry_unavailable(exc: BaseException) -> bool:
    """Return True only for expected exact-covariance feasibility failures.

    These failures are rank-local scientific unavailability, not programming
    errors.  They must not erase lower-rank results or terminate later models.
    """
    text = f"{type(exc).__name__}: {exc}"
    markers = (
        "Equal-class pooled covariance is not SPD",
        "Empirical class covariance is not positive semidefinite",
        "Whitened empirical class covariance lost positive semidefiniteness",
        "Interpolated covariance is not SPD",
        "No alpha candidate was valid in every inner split",
        "No ridge candidate for scenario=",
    )
    return any(marker in text for marker in markers)


def decode_scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
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


def safe_json(value: Any) -> str:
    return json.dumps(decode_scalar(value), ensure_ascii=False)


def json_dump(path: Path | str, payload: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(decode_scalar(payload), fh, ensure_ascii=False, indent=2, default=str)
    tmp.replace(path)


def atomic_npz(path: Path | str, **payload: np.ndarray) -> None:
    """Atomically write a compressed NPZ so resume never sees a partial file."""
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("wb") as fh:
            np.savez_compressed(fh, **payload)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def csv_write(path: Path | str, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out: Dict[str, Any] = {}
            for key in fields:
                value = decode_scalar(row.get(key, ""))
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, ensure_ascii=False)
                out[key] = value
            writer.writerow(out)
    tmp.replace(path)


def csv_read(path: Path | str) -> List[Dict[str, str]]:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def parse_int_csv(text: str) -> List[int]:
    return [int(x) for x in str(text).replace(" ", "").split(",") if x]


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


def parse_float_csv(text: str) -> List[float]:
    return [float(x) for x in str(text).replace(" ", "").split(",") if x]


def parse_str_csv(text: str) -> List[str]:
    return [x.strip() for x in str(text).replace(",", " ").split() if x.strip()]


def parse_task_ids(text: str) -> Dict[str, int]:
    aliases = {
        "MA": "MA",
        "MENTALARITHMETIC": "MA",
        "MENTAL_ARITHMETIC": "MA",
        "NB": "NB",
        "NBACK": "NB",
        "N-BACK": "NB",
        "NBMA": "NBMA",
        "NBACKARITHMETIC": "NBMA",
        "N-BACKARITHMETIC": "NBMA",
        "FULL": "Full",
        "FULLTASK": "Full",
        "FULLINTEGRATEDTASK": "Full",
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
        raise ValueError(f"--task-ids must define all task states; missing={missing}")
    return parsed


def empirical_upper_p(real: float, null: np.ndarray) -> float:
    null = np.asarray(null, dtype=np.float64)
    null = null[np.isfinite(null)]
    if not np.isfinite(real) or len(null) == 0:
        return float("nan")
    return float((1 + np.sum(null >= real)) / (len(null) + 1))


def as_float(value: Any) -> float:
    try:
        if value is None or str(value).strip() == "":
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


# =============================================================================
# Metadata, run validation, and dense subject loading
# =============================================================================


def read_vector(h5: h5py.File, key: str, dtype=None) -> np.ndarray:
    if key not in h5:
        raise KeyError(f"Missing H5 key {key!r}; available={list(h5.keys())}")
    arr = np.asarray(h5[key][...]).reshape(-1)
    return arr.astype(dtype) if dtype is not None else arr


def flat_dim(ds: h5py.Dataset) -> int:
    if len(ds.shape) < 2:
        raise ValueError(f"embedding must have shape (N,...), got {ds.shape}")
    return int(np.prod(ds.shape[1:], dtype=np.int64))


def read_embedding_rows(
    ds: h5py.Dataset,
    indices: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or len(indices) == 0:
        raise ValueError("indices must be a non-empty one-dimensional array")
    if np.any(np.diff(indices) < 0):
        raise ValueError("indices must be sorted")
    d = flat_dim(ds)
    out = np.empty((len(indices), d), dtype=np.float32)
    for start in range(0, len(indices), int(batch_size)):
        end = min(start + int(batch_size), len(indices))
        idx = indices[start:end]
        block = np.asarray(ds[idx])
        out[start:end] = block.reshape(len(idx), d).astype(np.float32, copy=False)
    if not np.all(np.isfinite(out)):
        raise FloatingPointError("Non-finite embedding values found")
    return out


def find_subject_rows(
    subject_ds: h5py.Dataset,
    subjects: Sequence[int],
    batch_size: int = 65536,
) -> np.ndarray:
    """Find H5 rows for explicit subjects with bounded, visible I/O."""
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
    if not chosen:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(chosen)


@dataclass
class Metadata:
    n_rows: int
    global_index: np.ndarray
    subject: np.ndarray
    session: np.ndarray
    run: np.ndarray
    sample_start: np.ndarray
    sample_end: np.ndarray
    phase: np.ndarray
    task: np.ndarray


@dataclass
class RunRecord:
    subject: int
    session: int
    run: int
    task: int
    indices: np.ndarray
    valid: bool
    reasons: List[str]


@dataclass
class SubjectData:
    model: str
    subject: int
    X: np.ndarray
    global_index: np.ndarray
    session: np.ndarray
    run: np.ndarray
    sample_start: np.ndarray
    sample_end: np.ndarray
    phase: np.ndarray
    task: np.ndarray
    state: np.ndarray
    task_ids: Dict[str, int]


@dataclass
class ModelInventory:
    meta: Metadata
    records: List[RunRecord]
    run_qc: List[Dict[str, Any]]
    subject_qc: List[Dict[str, Any]]
    eligible_subjects: List[int]
    info: Dict[str, Any]


def load_metadata(
    h5: h5py.File,
    embedding_key: str,
    row_indices: Optional[np.ndarray] = None,
) -> Metadata:
    """Load metadata once, optionally only for explicitly requested subjects.

    The v1 smoke path read every metadata column for the complete H5 and then
    repeated the same work inside ``load_subject_data``.  Here an explicit
    subject list first narrows rows using only ``subject_id``; the remaining
    metadata columns are read only for those rows and then reused by every fold.
    """
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

    arrays = {
        "subject": np.asarray(h5["subject_id"][idx], dtype=np.int64),
        "session": np.asarray(h5["session_id"][idx], dtype=np.int64),
        "run": np.asarray(h5["run_id"][idx], dtype=np.int64),
        "sample_start": np.asarray(h5["sample_start"][idx], dtype=np.int64),
        "sample_end": np.asarray(h5["sample_end"][idx], dtype=np.int64),
        "phase": np.asarray(h5["phase_id"][idx], dtype=np.int64),
        "task": np.asarray(h5["task_id"][idx], dtype=np.int64),
    }
    for name, arr in arrays.items():
        if len(arr) != len(idx):
            raise ValueError(f"Length mismatch for {name}: {len(arr)} vs selected rows={len(idx)}")
    return Metadata(n_rows=len(idx), global_index=idx, **arrays)


def group_indices(meta: Metadata) -> Dict[Tuple[int, int, int], np.ndarray]:
    """Vectorized grouping by subject/session/run, returning local metadata rows."""
    if meta.n_rows == 0:
        return {}
    order = np.lexsort((meta.sample_start, meta.run, meta.session, meta.subject))
    keys = np.column_stack((meta.subject[order], meta.session[order], meta.run[order]))
    breaks = np.flatnonzero(np.any(np.diff(keys, axis=0) != 0, axis=1)) + 1
    chunks = np.split(order, breaks)
    out: Dict[Tuple[int, int, int], np.ndarray] = {}
    for chunk in chunks:
        first = int(chunk[0])
        key = (int(meta.subject[first]), int(meta.session[first]), int(meta.run[first]))
        out[key] = np.asarray(chunk, dtype=np.int64)
    return out


def validate_runs(
    meta: Metadata,
    task_ids: Mapping[str, int],
    min_phase_windows: int,
) -> Tuple[List[RunRecord], List[Dict[str, Any]]]:
    selected_task_ids = set(int(x) for x in task_ids.values())
    records: List[RunRecord] = []
    qc_rows: List[Dict[str, Any]] = []
    for (subject, session, run), idx0 in sorted(group_indices(meta).items()):
        idx = idx0[np.argsort(meta.sample_start[idx0], kind="mergesort")]
        task_rows = idx[meta.phase[idx] == 1]
        task_values = np.unique(meta.task[task_rows]) if len(task_rows) else np.array([], dtype=int)
        if len(task_values) != 1 or int(task_values[0]) not in selected_task_ids:
            continue
        reasons: List[str] = []
        phases = meta.phase[idx]
        if np.any(np.diff(meta.sample_start[idx]) <= 0):
            reasons.append("non_monotone_sample_start")
        if not np.all(np.isin(phases, [0, 1])):
            reasons.append("invalid_phase_id")
        transitions = np.diff(phases)
        if np.any(transitions < 0) or int(np.sum(transitions == 1)) != 1:
            reasons.append("not_one_clean_baseline_to_task_transition")
        if int(np.sum(phases == 0)) < int(min_phase_windows):
            reasons.append("baseline_too_short")
        if int(np.sum(phases == 1)) < int(min_phase_windows):
            reasons.append("task_too_short")
        task_id = int(task_values[0])
        valid = len(reasons) == 0
        records.append(
            RunRecord(
                subject=int(subject),
                session=int(session),
                run=int(run),
                task=task_id,
                indices=meta.global_index[idx],
                valid=valid,
                reasons=reasons,
            )
        )
        qc_rows.append({
            "subject_id": int(subject),
            "session_id": int(session),
            "run_id": int(run),
            "task_id": task_id,
            "valid": int(valid),
            "reasons": reasons,
            "n_windows": int(len(idx)),
            "n_baseline_windows": int(np.sum(phases == 0)),
            "n_task_windows": int(np.sum(phases == 1)),
        })
    return records, qc_rows


def eligible_subjects(
    records: Sequence[RunRecord],
    task_ids: Mapping[str, int],
    expected_sessions: int,
) -> Tuple[List[int], List[Dict[str, Any]]]:
    chosen_tasks = sorted(int(x) for x in task_ids.values())
    subjects = sorted(set(r.subject for r in records))
    eligible: List[int] = []
    rows: List[Dict[str, Any]] = []
    for subject in subjects:
        good = [r for r in records if r.valid and r.subject == subject and r.task in chosen_tasks]
        sessions = sorted(set(r.session for r in good))
        counts = {
            int(s): {int(t): sum(r.session == s and r.task == t for r in good) for t in chosen_tasks}
            for s in sessions
        }
        ok = len(sessions) == int(expected_sessions) and all(
            all(counts[s][t] == 1 for t in chosen_tasks) for s in sessions
        )
        if ok:
            eligible.append(int(subject))
        rows.append({
            "subject_id": int(subject),
            "eligible": int(ok),
            "sessions": sessions,
            "counts": counts,
        })
    return eligible, rows


def state_labels(phase: np.ndarray, task: np.ndarray, task_ids: Mapping[str, int]) -> np.ndarray:
    phase = np.asarray(phase, dtype=np.int64)
    task = np.asarray(task, dtype=np.int64)
    y = np.full(len(phase), -1, dtype=np.int64)
    y[phase == 0] = 0
    for class_id, name in enumerate(TASK_STATE_NAMES, start=1):
        y[(phase == 1) & (task == int(task_ids[name]))] = class_id
    return y


def load_subject_data(
    model: str,
    path: str,
    subject: int,
    task_ids: Mapping[str, int],
    meta: Metadata,
    records: Sequence[RunRecord],
    run_qc_rows: Sequence[Mapping[str, Any]],
    read_batch_size: int,
) -> Tuple[SubjectData, List[Dict[str, Any]]]:
    """Load one subject's embeddings using the already validated model inventory."""
    valid_runs = [r for r in records if r.valid and r.subject == int(subject)]
    if not valid_runs:
        raise ValueError(f"No valid selected runs for model={model}, subject={subject}")
    global_indices = np.unique(np.concatenate([r.indices for r in valid_runs])).astype(np.int64)
    global_indices.sort()

    # Inventory metadata may contain only explicitly requested subjects. Build a
    # direct global-row -> local-row map once for exact ledger reconstruction.
    local_of_global = {int(g): i for i, g in enumerate(meta.global_index.tolist())}
    try:
        local_rows = np.asarray([local_of_global[int(g)] for g in global_indices], dtype=np.int64)
    except KeyError as exc:
        raise RuntimeError(f"Inventory is missing subject row {exc.args[0]}") from exc

    phase = meta.phase[local_rows]
    task = meta.task[local_rows]
    y = state_labels(phase, task, task_ids)
    if np.any(y < 0):
        bad = global_indices[y < 0][:10].tolist()
        raise RuntimeError(f"State mapping failed for indices {bad}")

    t0 = time.perf_counter()
    with h5py.File(path, "r") as h5:
        X = read_embedding_rows(h5["embedding"], global_indices, read_batch_size)
    log(
        f"  loaded subject rows={len(global_indices)} D={X.shape[1]} "
        f"in {time.perf_counter() - t0:.2f}s"
    )
    data = SubjectData(
        model=model,
        subject=int(subject),
        X=X,
        global_index=global_indices,
        session=meta.session[local_rows],
        run=meta.run[local_rows],
        sample_start=meta.sample_start[local_rows],
        sample_end=meta.sample_end[local_rows],
        phase=phase,
        task=task,
        state=y,
        task_ids=dict(task_ids),
    )
    subject_qc = [dict(r) for r in run_qc_rows if int(r["subject_id"]) == int(subject)]
    return data, subject_qc


# =============================================================================
# Train-only balanced randomized SVD
# =============================================================================


@dataclass
class Reducer:
    mean: np.ndarray
    components: np.ndarray
    n_components: int
    requested_components: int
    maximum_feasible_components: int
    method: str
    singular_values: np.ndarray
    class_counts: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=np.float64) - self.mean) @ self.components.T


def fit_balanced_reducer(
    X: np.ndarray,
    y: np.ndarray,
    requested_components: int,
    seed: int,
    randomized_n_iter: int,
    randomized_oversamples: int,
) -> Reducer:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    K = len(STATE_NAMES)
    counts = np.bincount(y, minlength=K).astype(np.int64)
    if np.any(counts == 0):
        raise ValueError(f"Reducer training split is missing a state; counts={counts.tolist()}")
    # Arithmetic pooled-covariance shrinkage does not require each class
    # covariance to be full rank.  It only requires the equal-class pooled
    # covariance to be full rank.  Its algebraic rank is at most the total
    # within-class degrees of freedom sum_c (n_c - 1) = N - K.
    pooled_covariance_dof = int(np.sum(counts - 1))
    maximum = int(min(X.shape[1], len(X) - 1, pooled_covariance_dof))
    requested = int(requested_components)
    if requested > maximum:
        raise ValueError(
            f"Requested SVD rank {requested} is infeasible for pooled-shrinkage "
            f"five-class QDA in this split; maximum={maximum}, "
            f"pooled_covariance_dof={pooled_covariance_dof}, "
            f"class_counts={counts.tolist()}"
        )
    if requested < 1:
        raise ValueError("requested_components must be positive")

    weights = 1.0 / (K * counts[y].astype(np.float64))
    weights /= np.sum(weights)
    mean = np.sum(X * weights[:, None], axis=0)
    weighted_centered = (X - mean[None, :]) * np.sqrt(weights * len(X))[:, None]

    if requested == X.shape[1]:
        components = np.eye(X.shape[1], dtype=np.float64)
        singular_values = np.linalg.svd(weighted_centered, compute_uv=False)[:requested]
        method = "identity_centered"
    else:
        _, singular_values, components = randomized_svd(
            weighted_centered,
            n_components=requested,
            n_iter=int(randomized_n_iter),
            random_state=int(seed),
            n_oversamples=int(max(5, randomized_oversamples)),
        )
        components = np.asarray(components, dtype=np.float64)
        singular_values = np.asarray(singular_values, dtype=np.float64)
        method = "balanced_randomized_svd"

    return Reducer(
        mean=np.asarray(mean, dtype=np.float64),
        components=components,
        n_components=requested,
        requested_components=requested,
        maximum_feasible_components=maximum,
        method=method,
        singular_values=singular_values,
        class_counts=counts,
    )


# =============================================================================
# Scenarios and nested selection
# =============================================================================


def scenario_data(
    features: Dict[str, np.ndarray],
    y_state: np.ndarray,
    scenario: str,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, int, List[str], np.ndarray]:
    y_state = np.asarray(y_state, dtype=np.int64)
    if scenario == "five_state":
        mask = np.ones(len(y_state), dtype=bool)
        y = y_state.copy()
        names = STATE_NAMES
    elif scenario == "context2":
        mask = np.ones(len(y_state), dtype=bool)
        y = (y_state > 0).astype(np.int64)
        names = ["Baseline", "Task"]
    elif scenario == "task4":
        mask = y_state > 0
        y = y_state[mask] - 1
        names = TASK_STATE_NAMES
    else:
        raise KeyError(scenario)
    Xmap = {arm: np.asarray(features[arm])[mask] for arm in ARMS}
    return Xmap, y, len(names), list(names), mask


def stable_select_alpha(scores: Dict[float, List[float]], n_splits: int) -> float:
    candidates: List[Tuple[float, float]] = []
    for alpha, values in scores.items():
        arr = np.asarray(values, dtype=np.float64)
        if len(arr) == n_splits and np.all(np.isfinite(arr)):
            candidates.append((float(np.mean(arr)), float(alpha)))
    if not candidates:
        raise RuntimeError("No alpha candidate was valid in every inner split")
    # Ties prefer more pooling and therefore lower variance.
    candidates.sort(key=lambda x: (x[0], x[1]))
    return float(candidates[-1][1])


def stable_select_ridge(
    scores: Dict[Tuple[str, str, float, float], List[float]],
    scenario: str,
    arm: str,
    alpha: float,
    ridge_grid: Sequence[float],
    n_splits: int,
) -> float:
    candidates: List[Tuple[float, float]] = []
    for lam in ridge_grid:
        values = scores.get((scenario, arm, float(alpha), float(lam)), [])
        arr = np.asarray(values, dtype=np.float64)
        if len(arr) == n_splits and np.all(np.isfinite(arr)):
            candidates.append((float(np.mean(arr)), float(lam)))
    if not candidates:
        raise RuntimeError(f"No ridge candidate for scenario={scenario}, arm={arm}, alpha={alpha}")
    # Ties prefer stronger regularization, i.e. larger ridge lambda.
    candidates.sort(key=lambda x: (x[0], x[1]))
    return float(candidates[-1][1])


@dataclass
class SelectionResult:
    alpha: float
    ridge_by_scenario_arm: Dict[str, Dict[str, float]]
    alpha_rows: List[Dict[str, Any]]
    ridge_rows: List[Dict[str, Any]]


def nested_select(
    data: SubjectData,
    outer_train_sessions: Sequence[int],
    requested_rank: int,
    alpha_grid: Sequence[float],
    ridge_grid: Sequence[float],
    covariance_interpolation: str,
    degenerate_tol: float,
    randomized_n_iter: int,
    randomized_oversamples: int,
    seed: int,
) -> SelectionResult:
    sessions = sorted(int(x) for x in outer_train_sessions)
    if len(sessions) != 2:
        raise ValueError(f"Expected exactly two outer-training sessions, got {sessions}")

    alpha_scores: Dict[float, List[float]] = {float(a): [] for a in alpha_grid}
    ridge_scores: Dict[Tuple[str, str, float, float], List[float]] = {}
    alpha_rows: List[Dict[str, Any]] = []
    ridge_rows: List[Dict[str, Any]] = []

    for inner_index, val_session in enumerate(sessions, start=1):
        train_session = [s for s in sessions if s != val_session][0]
        train_mask = data.session == train_session
        val_mask = data.session == val_session
        Xtr, Xva = data.X[train_mask], data.X[val_mask]
        ytr, yva = data.state[train_mask], data.state[val_mask]

        reducer = fit_balanced_reducer(
            Xtr,
            ytr,
            requested_components=requested_rank,
            seed=seed + 1009 * inner_index + 37 * requested_rank,
            randomized_n_iter=randomized_n_iter,
            randomized_oversamples=randomized_oversamples,
        )
        Rtr = reducer.transform(Xtr)
        Rva = reducer.transform(Xva)
        stats = fit_class_stats(Rtr, ytr, STATE_NAMES)
        ell_tr = lda_relative_scores(Rtr, stats)
        ell_va = lda_relative_scores(Rva, stats)

        for alpha in alpha_grid:
            alpha = float(alpha)
            try:
                qda = fit_qda_model(stats, alpha, covariance_interpolation)
                q_tr = qda_relative_scores(Rtr, stats, qda)
                q_va = qda_relative_scores(Rva, stats, qda)
                native_pred = native_prediction(q_va)
                native_metric = evaluate_predictions(yva, native_pred, len(STATE_NAMES))
                alpha_score = float(native_metric["balanced_accuracy"])
                alpha_scores[alpha].append(alpha_score)
                alpha_valid = 1
                alpha_error = ""
            except Exception as exc:  # noqa: BLE001
                alpha_score = float("nan")
                alpha_valid = 0
                alpha_error = repr(exc)
                q_tr = q_va = None

            alpha_rows.append({
                "inner_split": inner_index,
                "train_session": train_session,
                "validation_session": val_session,
                "M_requested": int(requested_rank),
                "M_effective": int(reducer.n_components),
                "maximum_feasible_M": int(reducer.maximum_feasible_components),
                "alpha": alpha,
                "native_qda_validation_bacc_five_state": alpha_score,
                "valid": alpha_valid,
                "error": alpha_error,
            })
            if not alpha_valid:
                continue

            builder = FeatureBuilder.fit(ell_tr, q_tr, degenerate_tol)
            train_features = builder.transform(ell_tr, q_tr)
            val_features = builder.transform(ell_va, q_va)

            for scenario in SCENARIOS:
                Xtr_map, ytr_s, K, names, _ = scenario_data(train_features, ytr, scenario)
                Xva_map, yva_s, _, _, _ = scenario_data(val_features, yva, scenario)
                for arm in ARMS:
                    for lam in ridge_grid:
                        readout = fit_ridge_readout(Xtr_map[arm], ytr_s, K, float(lam))
                        pred = readout.predict(Xva_map[arm])
                        metric = evaluate_predictions(yva_s, pred, K)
                        score = float(metric["balanced_accuracy"])
                        ridge_scores.setdefault(
                            (scenario, arm, alpha, float(lam)), []
                        ).append(score)
                        ridge_rows.append({
                            "inner_split": inner_index,
                            "train_session": train_session,
                            "validation_session": val_session,
                            "M_requested": int(requested_rank),
                            "M_effective": int(reducer.n_components),
                            "scenario": scenario,
                            "class_names": names,
                            "arm": arm,
                            "feature_dim": int(Xtr_map[arm].shape[1]),
                            "alpha": alpha,
                            "ridge_lambda": float(lam),
                            "validation_balanced_accuracy": score,
                            "weight_norm": readout.weight_norm,
                            "degenerate_features": builder.degenerate_report(),
                        })

        del Xtr, Xva, Rtr, Rva
        gc.collect()

    selected_alpha = stable_select_alpha(alpha_scores, len(sessions))
    selected_ridge: Dict[str, Dict[str, float]] = {s: {} for s in SCENARIOS}
    for scenario in SCENARIOS:
        for arm in ARMS:
            selected_ridge[scenario][arm] = stable_select_ridge(
                ridge_scores,
                scenario,
                arm,
                selected_alpha,
                ridge_grid,
                len(sessions),
            )
    return SelectionResult(
        alpha=selected_alpha,
        ridge_by_scenario_arm=selected_ridge,
        alpha_rows=alpha_rows,
        ridge_rows=ridge_rows,
    )


# =============================================================================
# Structured held-out nulls
# =============================================================================


def heldout_run_positions(data: SubjectData, test_mask: np.ndarray) -> Dict[int, np.ndarray]:
    positions: Dict[int, np.ndarray] = {}
    test_indices = np.flatnonzero(test_mask)
    for run_id in sorted(np.unique(data.run[test_mask]).astype(int).tolist()):
        pos = test_indices[data.run[test_indices] == int(run_id)]
        order = np.argsort(data.sample_start[pos], kind="mergesort")
        positions[int(run_id)] = pos[order]
    return positions


def permuted_heldout_labels(
    data: SubjectData,
    test_mask: np.ndarray,
    scenario: str,
    rng: np.random.Generator,
) -> np.ndarray:
    runs = heldout_run_positions(data, test_mask)
    if len(runs) != len(TASK_STATE_NAMES):
        raise ValueError(
            f"Held-out session must contain exactly {len(TASK_STATE_NAMES)} task runs, got {sorted(runs)}"
        )

    run_task_class: Dict[int, int] = {}
    for run_id, pos in runs.items():
        task_states = np.unique(data.state[pos][data.phase[pos] == 1])
        if len(task_states) != 1 or not (1 <= int(task_states[0]) <= 4):
            raise ValueError(f"Run {run_id} has invalid task-state ledger: {task_states.tolist()}")
        run_task_class[run_id] = int(task_states[0])

    run_ids = sorted(runs)
    original_tasks = np.asarray([run_task_class[r] for r in run_ids], dtype=np.int64)
    permuted_tasks = rng.permutation(original_tasks)
    task_map = {run_id: int(task_class) for run_id, task_class in zip(run_ids, permuted_tasks)}

    test_positions = np.flatnonzero(test_mask)
    local_index = {int(pos): i for i, pos in enumerate(test_positions.tolist())}

    if scenario == "task4":
        task_positions = test_positions[data.state[test_positions] > 0]
        out = np.empty(len(task_positions), dtype=np.int64)
        for i, pos in enumerate(task_positions):
            out[i] = task_map[int(data.run[pos])] - 1
        return out

    out_state = np.empty(len(test_positions), dtype=np.int64)
    for run_id, pos in runs.items():
        phase = data.phase[pos].astype(np.int64)
        if len(phase) < 2:
            raise ValueError(f"Run {run_id} too short for circular phase null")
        shift = int(rng.integers(1, len(phase)))
        shifted_phase = np.roll(phase, shift)
        permuted_state = np.where(shifted_phase == 0, 0, task_map[run_id]).astype(np.int64)
        for global_pos, state_value in zip(pos, permuted_state):
            out_state[local_index[int(global_pos)]] = int(state_value)

    if scenario == "five_state":
        return out_state
    if scenario == "context2":
        return (out_state > 0).astype(np.int64)
    raise KeyError(scenario)


def exact_task4_permuted_labels(
    data: SubjectData, test_mask: np.ndarray
) -> List[np.ndarray]:
    """Enumerate the complete 4! run-to-task randomization group."""
    runs = heldout_run_positions(data, test_mask)
    run_ids = sorted(runs)
    if len(run_ids) != 4:
        raise ValueError(f"Held-out session must contain four task runs, got {run_ids}")
    run_task_class: Dict[int, int] = {}
    for run_id, pos in runs.items():
        states = np.unique(data.state[pos][data.phase[pos] == 1])
        if len(states) != 1 or not (1 <= int(states[0]) <= 4):
            raise ValueError(f"Run {run_id} has invalid task-state ledger: {states.tolist()}")
        run_task_class[run_id] = int(states[0])
    original = [run_task_class[r] for r in run_ids]
    test_positions = np.flatnonzero(test_mask)
    task_positions = test_positions[data.state[test_positions] > 0]
    outputs: List[np.ndarray] = []
    for assignment in itertools.permutations(original):
        task_map = {r: int(c) for r, c in zip(run_ids, assignment)}
        outputs.append(np.asarray(
            [task_map[int(data.run[p])] - 1 for p in task_positions], dtype=np.int64
        ))
    return outputs


def structured_null_bacc(
    data: SubjectData,
    test_mask: np.ndarray,
    scenario: str,
    predictions: np.ndarray,
    n_permutations: int,
    seed: int,
) -> np.ndarray:
    if int(n_permutations) <= 0:
        return np.empty(0, dtype=np.float64)
    K = 5 if scenario == "five_state" else (2 if scenario == "context2" else 4)
    if scenario == "task4":
        labels = exact_task4_permuted_labels(data, test_mask)
        return np.asarray([
            evaluate_predictions(y, predictions, K)["balanced_accuracy"] for y in labels
        ], dtype=np.float64)
    rng = np.random.default_rng(seed)
    null = np.empty(int(n_permutations), dtype=np.float64)
    for b in range(int(n_permutations)):
        y_perm = permuted_heldout_labels(data, test_mask, scenario, rng)
        null[b] = evaluate_predictions(y_perm, predictions, K)["balanced_accuracy"]
    return null


# =============================================================================
# Outer fold
# =============================================================================


def covariance_diagnostic_rows(stats: ClassStats, base: Dict[str, Any]) -> Tuple[List[Dict], List[Dict]]:
    covariance_rows: List[Dict] = []
    spectrum_rows: List[Dict] = []
    rank, cond, logdet, eig = rank_condition(stats.pooled)
    covariance_rows.append({
        **base,
        "class_id": "pooled",
        "class_name": "pooled_equal_class",
        "n_class": int(np.sum(stats.class_counts)),
        "rank": rank,
        "dimension": int(stats.pooled.shape[0]),
        "condition_number": cond,
        "logdet": logdet,
        "min_eigenvalue": float(np.min(eig)),
        "max_eigenvalue": float(np.max(eig)),
    })
    for c, name in enumerate(stats.class_names):
        rank, cond, logdet, eig = rank_condition(stats.covs[c])
        covariance_rows.append({
            **base,
            "class_id": c,
            "class_name": name,
            "n_class": int(stats.class_counts[c]),
            "rank": rank,
            "dimension": int(stats.covs[c].shape[0]),
            "condition_number": cond,
            "logdet": logdet,
            "min_eigenvalue": float(np.min(eig)),
            "max_eigenvalue": float(np.max(eig)),
        })
        for j, value in enumerate(stats.lambda_spectra[c], start=1):
            spectrum_rows.append({
                **base,
                "class_id": c,
                "class_name": name,
                "eigen_index": j,
                "generalized_lambda": float(value),
            })
    return covariance_rows, spectrum_rows


def run_outer_fold(
    data: SubjectData,
    heldout_session: int,
    requested_rank: int,
    args: argparse.Namespace,
    fold_dir: Path,
) -> None:
    train_mask = data.session != int(heldout_session)
    test_mask = data.session == int(heldout_session)
    train_sessions = sorted(np.unique(data.session[train_mask]).astype(int).tolist())
    if len(train_sessions) != 2:
        raise ValueError(f"Expected two training sessions, got {train_sessions}")

    selection = nested_select(
        data=data,
        outer_train_sessions=train_sessions,
        requested_rank=requested_rank,
        alpha_grid=args.alpha_grid_values,
        ridge_grid=args.ridge_grid_values,
        covariance_interpolation=args.cov_interpolation,
        degenerate_tol=args.degenerate_tol,
        randomized_n_iter=args.randomized_n_iter,
        randomized_oversamples=args.randomized_oversamples,
        seed=args.seed + 100000 * data.subject + 1000 * int(heldout_session),
    )

    Xtr, Xte = data.X[train_mask], data.X[test_mask]
    ytr, yte = data.state[train_mask], data.state[test_mask]
    train_counts = np.bincount(ytr, minlength=5)
    test_counts = np.bincount(yte, minlength=5)
    if np.any(train_counts == 0):
        raise ValueError(f"Outer training split misses a state: {train_counts.tolist()}")
    if np.any(test_counts == 0):
        raise ValueError(f"Held-out session misses a state: {test_counts.tolist()}")
    reducer = fit_balanced_reducer(
        Xtr,
        ytr,
        requested_components=requested_rank,
        seed=args.seed + 300000 * data.subject + 7919 * int(heldout_session),
        randomized_n_iter=args.randomized_n_iter,
        randomized_oversamples=args.randomized_oversamples,
    )
    Rtr, Rte = reducer.transform(Xtr), reducer.transform(Xte)
    stats = fit_class_stats(Rtr, ytr, STATE_NAMES)
    ell_tr, ell_te = lda_relative_scores(Rtr, stats), lda_relative_scores(Rte, stats)
    qda = fit_qda_model(stats, selection.alpha, args.cov_interpolation)
    q_tr = qda_relative_scores(Rtr, stats, qda)
    q_te = qda_relative_scores(Rte, stats, qda)
    builder = FeatureBuilder.fit(ell_tr, q_tr, args.degenerate_tol)
    train_features = builder.transform(ell_tr, q_tr)
    test_features = builder.transform(ell_te, q_te)

    base = {
        "script_version": SCRIPT_VERSION,
        "model": data.model,
        "subject_id": data.subject,
        "heldout_session": int(heldout_session),
        "training_sessions": train_sessions,
        "M_requested": int(requested_rank),
        "M_effective": int(reducer.n_components),
        "maximum_feasible_M": int(reducer.maximum_feasible_components),
        "reducer_method": reducer.method,
        "selected_alpha": float(selection.alpha),
        "covariance_interpolation": args.cov_interpolation,
    }

    outer_rows: List[Dict[str, Any]] = []
    permutation_rows: List[Dict[str, Any]] = []
    prediction_payload: Dict[str, np.ndarray] = {
        "global_index": data.global_index[test_mask],
        "session_id": data.session[test_mask],
        "run_id": data.run[test_mask],
        "sample_start": data.sample_start[test_mask],
        "sample_end": data.sample_end[test_mask],
        "phase_id": data.phase[test_mask],
        "task_id": data.task[test_mask],
        "state_label": yte,
    }
    null_payload: Dict[str, np.ndarray] = {}

    # Native five-state endpoints, useful as a compact sanity comparison.
    native_lda_pred = native_prediction(ell_te)
    native_qda_pred = native_prediction(q_te)
    for arm_name, pred in (("native_LDA", native_lda_pred), ("native_QDA", native_qda_pred)):
        metric = evaluate_predictions(yte, pred, 5)
        outer_rows.append({
            **base,
            "scenario": "five_state",
            "arm": arm_name,
            "feature_dim": 4,
            "ridge_lambda": "",
            "train_balanced_accuracy": "",
            "test_balanced_accuracy": metric["balanced_accuracy"],
            "test_accuracy": metric["accuracy"],
            "train_test_gap": "",
            "weight_norm": "",
            "class_names": STATE_NAMES,
            "class_recalls": metric["recalls"],
            "confusion_matrix": metric["confusion_matrix"].tolist(),
            "degenerate_features": builder.degenerate_report(),
        })

    for scenario_index, scenario in enumerate(SCENARIOS):
        Xtr_map, ytr_s, K, class_names, train_scenario_mask = scenario_data(
            train_features, ytr, scenario
        )
        Xte_map, yte_s, _, _, test_scenario_mask = scenario_data(
            test_features, yte, scenario
        )
        for arm_index, arm in enumerate(ARMS):
            ridge_lambda = selection.ridge_by_scenario_arm[scenario][arm]
            readout = fit_ridge_readout(Xtr_map[arm], ytr_s, K, ridge_lambda)
            pred_tr = readout.predict(Xtr_map[arm])
            pred_te = readout.predict(Xte_map[arm])
            train_metric = evaluate_predictions(ytr_s, pred_tr, K)
            test_metric = evaluate_predictions(yte_s, pred_te, K)

            null = structured_null_bacc(
                data=data,
                test_mask=test_mask,
                scenario=scenario,
                predictions=pred_te,
                n_permutations=args.n_permutations,
                seed=(
                    args.seed
                    + 900000 * data.subject
                    + 10000 * int(heldout_session)
                    + 100 * scenario_index
                    + arm_index
                ),
            )
            exact_task4 = scenario == "task4"
            p_value = (
                float(np.mean(null >= test_metric["balanced_accuracy"] - 1e-15))
                if exact_task4 and len(null)
                else empirical_upper_p(test_metric["balanced_accuracy"], null)
            )
            null_key = f"{scenario}__{arm}"
            null_payload[null_key] = null

            outer_rows.append({
                **base,
                "scenario": scenario,
                "arm": arm,
                "feature_dim": int(Xtr_map[arm].shape[1]),
                "ridge_lambda": float(ridge_lambda),
                "train_balanced_accuracy": train_metric["balanced_accuracy"],
                "test_balanced_accuracy": test_metric["balanced_accuracy"],
                "test_accuracy": test_metric["accuracy"],
                "train_test_gap": (
                    train_metric["balanced_accuracy"] - test_metric["balanced_accuracy"]
                ),
                "weight_norm": readout.weight_norm,
                "class_names": class_names,
                "class_recalls": test_metric["recalls"],
                "confusion_matrix": test_metric["confusion_matrix"].tolist(),
                "degenerate_features": builder.degenerate_report(),
                "heldout_null_type": {
                    "five_state": "within-run phase circular shift plus within-session task-run permutation",
                    "context2": "within-run phase circular shift",
                    "task4": "within-session task-run permutation",
                }[scenario],
                "n_heldout_permutations": int(len(null)),
                "heldout_permutation_support_size": 24 if exact_task4 else "",
                "heldout_pvalue_method": "exact_full_4_factorial" if exact_task4 else "monte_carlo_plus_one",
                "minimum_attainable_p": (1.0 / 24.0) if exact_task4 else (1.0 / (len(null) + 1.0) if len(null) else float("nan")),
                "heldout_permutation_p": p_value,
                "heldout_null_mean": float(np.mean(null)) if len(null) else float("nan"),
                "heldout_null_q95": float(np.quantile(null, 0.95)) if len(null) else float("nan"),
            })
            permutation_rows.append({
                **base,
                "scenario": scenario,
                "arm": arm,
                "real_test_balanced_accuracy": test_metric["balanced_accuracy"],
                "n_permutations": int(len(null)),
                "permutation_support_size": 24 if exact_task4 else "",
                "pvalue_method": "exact_full_4_factorial" if exact_task4 else "monte_carlo_plus_one",
                "minimum_attainable_p": (1.0 / 24.0) if exact_task4 else (1.0 / (len(null) + 1.0) if len(null) else float("nan")),
                "upper_tail_p": p_value,
                "null_mean": float(np.mean(null)) if len(null) else float("nan"),
                "null_std": float(np.std(null)) if len(null) else float("nan"),
                "null_q05": float(np.quantile(null, 0.05)) if len(null) else float("nan"),
                "null_q50": float(np.quantile(null, 0.50)) if len(null) else float("nan"),
                "null_q95": float(np.quantile(null, 0.95)) if len(null) else float("nan"),
            })

            full_pred = np.full(len(yte), -1, dtype=np.int64)
            full_pred[test_scenario_mask] = pred_te
            prediction_payload[f"pred__{scenario}__{arm}"] = full_pred

    cov_rows, lambda_rows = covariance_diagnostic_rows(stats, base)
    selected_rows: List[Dict[str, Any]] = [{
        **base,
        "parameter_type": "alpha",
        "scenario": "five_state_native_QDA",
        "arm": "native_QDA",
        "selected_value": float(selection.alpha),
    }]
    for scenario in SCENARIOS:
        for arm in ARMS:
            selected_rows.append({
                **base,
                "parameter_type": "ridge_lambda",
                "scenario": scenario,
                "arm": arm,
                "selected_value": float(selection.ridge_by_scenario_arm[scenario][arm]),
            })

    ledger_rows = []
    test_positions = np.flatnonzero(test_mask)
    for local_i, pos in enumerate(test_positions):
        ledger_rows.append({
            "global_index": int(data.global_index[pos]),
            "model": data.model,
            "subject_id": data.subject,
            "heldout_session": int(heldout_session),
            "session_id": int(data.session[pos]),
            "run_id": int(data.run[pos]),
            "sample_start": int(data.sample_start[pos]),
            "sample_end": int(data.sample_end[pos]),
            "phase_id": int(data.phase[pos]),
            "task_id": int(data.task[pos]),
            "state_label": int(data.state[pos]),
            "state_name": STATE_NAMES[int(data.state[pos])],
            "heldout_row_index": local_i,
        })

    csv_write(fold_dir / "outer_metrics.csv", outer_rows)
    csv_write(fold_dir / "heldout_permutation_summary.csv", permutation_rows)
    csv_write(fold_dir / "selected_hyperparameters.csv", selected_rows)
    csv_write(
        fold_dir / "alpha_selection_candidates.csv",
        [{**base, **row} for row in selection.alpha_rows],
    )
    csv_write(
        fold_dir / "ridge_selection_candidates.csv",
        [{**base, **row} for row in selection.ridge_rows],
    )
    csv_write(fold_dir / "covariance_diagnostics.csv", cov_rows)
    csv_write(fold_dir / "generalized_lambda_spectra.csv", lambda_rows)
    csv_write(fold_dir / "heldout_ledger.csv", ledger_rows)
    atomic_npz(fold_dir / "heldout_predictions.npz", **prediction_payload)
    atomic_npz(fold_dir / "heldout_null_distributions.npz", **null_payload)

    alpha1_q = qda_relative_scores(
        Rtr, stats, fit_qda_model(stats, 1.0, args.cov_interpolation)
    )
    endpoint_error = float(np.max(np.abs(alpha1_q - ell_tr)))
    summary = {
        **base,
        "status": "ok",
        "train_class_counts": train_counts.tolist(),
        "test_class_counts": test_counts.tolist(),
        "heldout_has_all_five_states": bool(np.all(test_counts > 0)),
        "feature_dimensions": feature_dimensions(4),
        "selected_ridge": selection.ridge_by_scenario_arm,
        "degenerate_features": builder.degenerate_report(),
        "alpha1_max_abs_q_minus_ell_train": endpoint_error,
        "n_heldout_permutations": int(args.n_permutations),
        "notes": [
            "The shared positive alpha is selected only from the two outer-training sessions by two-way session-heldout inner validation.",
            "Arithmetic pooled-covariance shrinkage permits singular empirical class covariances without a pseudoinverse; the equal-class pooled covariance remains the positive-definite anchor.",
            "Held-out structured label nulls operate on complete runs or within-run phase sequences; no windowwise label shuffle is used.",
            "F3 is a structured interaction lift of F2, not an independent 24-dimensional state manifold.",
        ],
    }
    json_dump(fold_dir / "fold_summary.json", summary)
    json_dump(fold_dir / "DONE.json", {**base, "finished_at": now()})
    log(
        f"{data.model} sub-{data.subject:03d} heldout={heldout_session} M={requested_rank} "
        f"complete; alpha={selection.alpha}"
    )


# =============================================================================
# Aggregation and model driver
# =============================================================================


def aggregate_outputs(root: Path) -> None:
    table_names = [
        "outer_metrics.csv",
        "heldout_permutation_summary.csv",
        "selected_hyperparameters.csv",
        "alpha_selection_candidates.csv",
        "ridge_selection_candidates.csv",
        "covariance_diagnostics.csv",
        "generalized_lambda_spectra.csv",
    ]
    tables = ensure_dir(root / "tables")
    aggregated: Dict[str, List[Dict[str, str]]] = {}
    for name in table_names:
        rows: List[Dict[str, str]] = []
        for path in sorted((root / "folds").glob(f"*/*/*/*/{name}")):
            rows.extend(csv_read(path))
        aggregated[name] = rows
        csv_write(tables / name, rows)

    metrics = aggregated["outer_metrics.csv"]
    groups: Dict[Tuple[str, str, str, str, str], List[Dict[str, str]]] = {}
    for row in metrics:
        if row.get("arm", "").startswith("native_"):
            continue
        key = (
            row.get("model", ""),
            row.get("subject_id", ""),
            row.get("M_requested", ""),
            row.get("scenario", ""),
            row.get("arm", ""),
        )
        groups.setdefault(key, []).append(row)

    summary_rows: List[Dict[str, Any]] = []
    for key, rows in groups.items():
        bacc = [as_float(r.get("test_balanced_accuracy")) for r in rows]
        pvals = [as_float(r.get("heldout_permutation_p")) for r in rows]
        summary_rows.append({
            "model": key[0],
            "subject_id": key[1],
            "M_requested": key[2],
            "scenario": key[3],
            "arm": key[4],
            "n_outer_folds": len(rows),
            "test_balanced_accuracy_mean": finite_mean(bacc),
            "test_balanced_accuracy_min": float(np.nanmin(bacc)) if np.any(np.isfinite(bacc)) else float("nan"),
            "test_balanced_accuracy_max": float(np.nanmax(bacc)) if np.any(np.isfinite(bacc)) else float("nan"),
            "mean_heldout_permutation_p": finite_mean(pvals),
            "n_folds_p_le_0_05": int(np.sum(np.asarray(pvals) <= 0.05)),
        })
    csv_write(tables / "summary_by_model_subject_rank_scenario_arm.csv", summary_rows)
    json_dump(root / "result_summary.json", {
        "script_version": SCRIPT_VERSION,
        "finished_at": now(),
        "summary": summary_rows,
    })


def inspect_model(
    model: str,
    path: str,
    task_ids: Mapping[str, int],
    min_phase_windows: int,
    expected_sessions: int,
    requested_subjects: Sequence[int],
) -> ModelInventory:
    """Build one reusable metadata/run inventory per model.

    With an explicit ``--subjects`` list, only ``subject_id`` is scanned across
    the file. All other metadata columns are read for those subjects only. This
    is the normal smoke/formal path for A/B/C integration.
    """
    log(f"Inspect {model}: open H5")
    t0 = time.perf_counter()
    with h5py.File(path, "r") as h5:
        embedding_shape = list(h5["embedding"].shape)
        original_dim = flat_dim(h5["embedding"])
        requested = sorted(set(int(x) for x in requested_subjects))
        if requested:
            log(f"  locate explicit subjects {requested}")
            rows = find_subject_rows(h5["subject_id"], requested)
            if len(rows) == 0:
                raise ValueError(f"None of the requested subjects {requested} occur in {model}")
            observed = sorted(np.unique(np.asarray(h5["subject_id"][rows], dtype=np.int64)).tolist())
            missing = [x for x in requested if x not in observed]
            if missing:
                raise ValueError(f"Requested subjects absent in {model}: {missing}")
            log(f"  load selected metadata rows={len(rows)}")
            meta = load_metadata(h5, "embedding", rows)
            inspection_scope = "explicit_subjects_only"
        else:
            log("  no explicit subjects: load full metadata inventory")
            meta = load_metadata(h5, "embedding", None)
            inspection_scope = "full_file"

        log("  validate selected Baseline-to-Task runs")
        records, run_qc = validate_runs(meta, task_ids, min_phase_windows)
        eligible, subject_qc = eligible_subjects(records, task_ids, expected_sessions)
        info = {
            "model": model,
            "path": path,
            "embedding_shape": embedding_shape,
            "flat_dim": original_dim,
            "inspection_scope": inspection_scope,
            "metadata_rows_loaded": int(meta.n_rows),
            "eligible_subjects": eligible,
            "n_valid_selected_runs": int(sum(r.valid for r in records)),
            "inventory_seconds": float(time.perf_counter() - t0),
        }
    log(
        f"  inventory complete in {info['inventory_seconds']:.2f}s; "
        f"eligible={eligible}"
    )
    return ModelInventory(
        meta=meta,
        records=records,
        run_qc=run_qc,
        subject_qc=subject_qc,
        eligible_subjects=eligible,
        info=info,
    )


def resolve_models(args: argparse.Namespace) -> Dict[str, str]:
    paths = dict(MODEL_PATHS)
    for item in args.model_path:
        if "=" not in item:
            raise ValueError(f"--model-path expects MODEL=/path, got {item!r}")
        model, path = item.split("=", 1)
        paths[model.strip()] = path.strip()
    requested = parse_str_csv(args.models)
    unknown = [m for m in requested if m not in paths]
    if unknown:
        raise KeyError(f"Unknown models: {unknown}")
    return {m: paths[m] for m in requested}


def choose_common_subjects(
    eligible_by_model: Mapping[str, Sequence[int]],
    explicit_subjects: Sequence[int],
    n_subjects: int,
    seed: int,
) -> List[int]:
    common = sorted(set.intersection(*[set(v) for v in eligible_by_model.values()]))
    if explicit_subjects:
        missing = [int(s) for s in explicit_subjects if int(s) not in common]
        if missing:
            raise ValueError(f"Requested subjects are not eligible in every model: {missing}")
        return [int(s) for s in explicit_subjects]
    if not common:
        raise RuntimeError("No subject is eligible in every requested model")
    if n_subjects <= 0 or n_subjects > len(common):
        raise ValueError(f"Invalid --n-subjects={n_subjects}; common eligible={common}")
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(np.asarray(common), size=n_subjects, replace=False).astype(int).tolist())


def run_model(
    model: str,
    path: str,
    subjects: Sequence[int],
    task_ids: Mapping[str, int],
    inventory: ModelInventory,
    args: argparse.Namespace,
    root: Path,
) -> None:
    model_dir = ensure_dir(root / model)
    for subject in subjects:
        log(f"Load {model} subject={subject} from cached inventory")
        data, subject_run_qc = load_subject_data(
            model=model,
            path=path,
            subject=int(subject),
            task_ids=task_ids,
            meta=inventory.meta,
            records=inventory.records,
            run_qc_rows=inventory.run_qc,
            read_batch_size=args.read_batch_size,
        )
        subject_dir = ensure_dir(model_dir / f"sub-{int(subject):03d}")
        csv_write(subject_dir / "selected_run_qc.csv", subject_run_qc)
        sessions = sorted(np.unique(data.session).astype(int).tolist())
        if len(sessions) != args.expected_sessions:
            raise ValueError(f"{model} subject={subject}: expected {args.expected_sessions} sessions, got {sessions}")
        json_dump(subject_dir / "subject_inventory.json", {
            "model": model,
            "subject_id": int(subject),
            "sessions": sessions,
            "n_windows": int(len(data.state)),
            "state_counts": {STATE_NAMES[c]: int(np.sum(data.state == c)) for c in range(5)},
            "run_ids": sorted(np.unique(data.run).astype(int).tolist()),
            "task_ids": task_ids,
            "flat_dim": int(data.X.shape[1]),
        })

        heldout_sessions = (
            [int(x) for x in args.heldout_session_values]
            if args.heldout_session_values else sessions
        )
        invalid_heldout = [x for x in heldout_sessions if x not in sessions]
        if invalid_heldout:
            raise ValueError(
                f"{model} subject={subject}: requested heldout sessions {invalid_heldout} "
                f"not in available sessions {sessions}"
            )

        model_ranks = args.model_svd_rank_values.get(model, args.svd_rank_values)
        for heldout_session in heldout_sessions:
            for requested_rank in model_ranks:
                fold_dir = ensure_dir(
                    root
                    / "folds"
                    / model
                    / f"sub-{int(subject):03d}"
                    / f"heldout-session-{int(heldout_session)}"
                    / f"svd-rank-{int(requested_rank)}"
                )
                done_marker = fold_dir / "DONE.json"
                skipped_marker = fold_dir / "SKIPPED.json"
                if args.resume and done_marker.exists() and marker_matches_current_script(done_marker):
                    log(
                        f"[RESUME] skip {model} sub-{subject:03d} heldout={heldout_session} M={requested_rank}"
                    )
                    continue
                # A scientific-version change or a previously skipped fold must be
                # recomputed.  Remove stale terminal markers before starting so a
                # new success cannot coexist with an old SKIPPED.json.
                for marker in (done_marker, skipped_marker):
                    if marker.exists():
                        marker.unlink()
                try:
                    run_outer_fold(
                        data=data,
                        heldout_session=int(heldout_session),
                        requested_rank=int(requested_rank),
                        args=args,
                        fold_dir=fold_dir,
                    )
                except ValueError as exc:
                    message = str(exc)
                    if "Requested SVD rank" in message and "infeasible" in message:
                        json_dump(fold_dir / "SKIPPED.json", {
                            "script_version": SCRIPT_VERSION,
                            "model": model,
                            "subject_id": int(subject),
                            "heldout_session": int(heldout_session),
                            "M_requested": int(requested_rank),
                            "status": "skipped_infeasible_rank",
                            "reason": message,
                        })
                        log(f"[SKIP] {model} sub-{subject:03d} heldout={heldout_session} M={requested_rank}: {message}")
                        continue
                    raise
                except (np.linalg.LinAlgError, RuntimeError) as exc:
                    if not is_exact_covariance_geometry_unavailable(exc):
                        raise
                    message = str(exc)
                    json_dump(fold_dir / "SKIPPED.json", {
                        "script_version": SCRIPT_VERSION,
                        "model": model,
                        "subject_id": int(subject),
                        "heldout_session": int(heldout_session),
                        "M_requested": int(requested_rank),
                        "status": "skipped_unavailable_exact_covariance_geometry",
                        "error_type": type(exc).__name__,
                        "reason": message,
                    })
                    log(
                        f"[RANK UNAVAILABLE] {model} sub-{subject:03d} "
                        f"heldout={heldout_session} M={requested_rank}: {message}"
                    )
                    continue
        # Aggregate once per subject instead of rescanning every fold (O(N^2) I/O).
        aggregate_outputs(root)
        del data
        gc.collect()


# =============================================================================
# CLI and self-test
# =============================================================================


def integration_self_test(seed: int = 0) -> Dict[str, Any]:
    core = core_self_test(seed)
    rng = np.random.default_rng(seed + 1)
    sessions, runs, phases, tasks, states, sample_start = [], [], [], [], [], []
    cursor = 0
    task_ids = dict(DEFAULT_TASK_IDS)
    for session in (1, 2, 3):
        for run_offset, name in enumerate(TASK_STATE_NAMES):
            run_id = session * 10 + run_offset
            n_base, n_task = 8, 12
            for t in range(n_base + n_task):
                sessions.append(session)
                runs.append(run_id)
                phases.append(0 if t < n_base else 1)
                tasks.append(task_ids[name])
                states.append(0 if t < n_base else TASK_STATE_NAMES.index(name) + 1)
                sample_start.append(cursor)
                cursor += 1
    n = len(states)
    Xsyn = rng.standard_normal((n, 12))
    # Add modest state-dependent mean and covariance structure so every branch is finite.
    state_arr = np.asarray(states, dtype=np.int64)
    for c in range(5):
        mask = state_arr == c
        Xsyn[mask, c % 12] += 0.35 * c
        Xsyn[mask, (c + 1) % 12] *= 0.8 + 0.12 * c
    data = SubjectData(
        model="synthetic",
        subject=1,
        X=Xsyn.astype(np.float32),
        global_index=np.arange(n, dtype=np.int64),
        session=np.asarray(sessions, dtype=np.int64),
        run=np.asarray(runs, dtype=np.int64),
        sample_start=np.asarray(sample_start, dtype=np.int64),
        sample_end=np.asarray(sample_start, dtype=np.int64) + 1,
        phase=np.asarray(phases, dtype=np.int64),
        task=np.asarray(tasks, dtype=np.int64),
        state=np.asarray(states, dtype=np.int64),
        task_ids=task_ids,
    )
    test_mask = data.session == 3
    for scenario in SCENARIOS:
        K = 5 if scenario == "five_state" else (2 if scenario == "context2" else 4)
        expected_len = int(np.sum(test_mask)) if scenario != "task4" else int(np.sum(test_mask & (data.state > 0)))
        y_perm = permuted_heldout_labels(data, test_mask, scenario, rng)
        if len(y_perm) != expected_len:
            raise AssertionError(f"Structured null length failed for {scenario}")
        observed = sorted(np.unique(y_perm).astype(int).tolist())
        if observed != list(range(K)):
            raise AssertionError(f"Structured null classes failed for {scenario}: {observed}")

    task4_predictions = np.zeros(int(np.sum(test_mask & (data.state > 0))), dtype=np.int64)
    task4_exact = structured_null_bacc(data, test_mask, "task4", task4_predictions, 2, seed)
    if len(task4_exact) != 24:
        raise AssertionError(f"task4 exact null must enumerate 4!=24 assignments, got {len(task4_exact)}")

    # One compact end-to-end fold catches split, reducer, QDA, four-arm, null, and output wiring.
    smoke_args = argparse.Namespace(
        alpha_grid_values=[0.5, 1.0],
        ridge_grid_values=[0.1, 1.0],
        cov_interpolation="arithmetic",
        degenerate_tol=1e-10,
        randomized_n_iter=1,
        randomized_oversamples=5,
        seed=seed,
        n_permutations=2,
    )
    with tempfile.TemporaryDirectory(prefix="expA_selftest_") as tmp:
        fold_dir = Path(tmp) / "fold"
        fold_dir.mkdir(parents=True, exist_ok=True)
        run_outer_fold(data, heldout_session=3, requested_rank=5, args=smoke_args, fold_dir=fold_dir)
        required = [
            "outer_metrics.csv",
            "selected_hyperparameters.csv",
            "heldout_permutation_summary.csv",
            "heldout_predictions.npz",
            "fold_summary.json",
            "DONE.json",
        ]
        missing = [name for name in required if not (fold_dir / name).exists()]
        if missing:
            raise AssertionError(f"End-to-end self-test missing outputs: {missing}")
    return {"core": core, "structured_nulls": "passed", "end_to_end_fold": "passed"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Within-subject five-state LD/QD/tensor endpoint geometry for ds007554",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--models", default="BIOT")
    p.add_argument("--model-path", action="append", default=[])
    p.add_argument("--subjects", default="")
    p.add_argument("--n-subjects", type=int, default=1)
    p.add_argument("--subject-seed", type=int, default=20260718)
    p.add_argument("--task-ids", default="MA=0,NB=1,NBMA=5,Full=6")
    p.add_argument("--expected-sessions", type=int, default=3)
    p.add_argument("--heldout-sessions", default="", help="Optional comma-separated subset for smoke runs")
    p.add_argument("--min-phase-windows", type=int, default=10)

    p.add_argument("--svd-ranks", default="100,200,300,500")
    p.add_argument(
        "--model-svd-ranks", action="append", default=[],
        help="Optional MODEL=r1,r2,... override; repeat once per model (e.g. BIOT=100,200,256)",
    )
    p.add_argument("--alpha-grid", default="0.001,0.01,0.05,0.1,0.25,0.5,0.9,1")
    p.add_argument("--ridge-grid", default="0.0001,0.001,0.01,0.1,1,10,100")
    p.add_argument("--cov-interpolation", choices=["geometric", "arithmetic"], default="arithmetic")
    p.add_argument("--degenerate-tol", type=float, default=1e-10)

    p.add_argument("--read-batch-size", type=int, default=512)
    p.add_argument("--randomized-n-iter", type=int, default=2)
    p.add_argument("--randomized-oversamples", type=int, default=20)
    p.add_argument("--n-permutations", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--self-test", action="store_true")
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("/mnt/dataset4/yinuo/FM_flow/dataset/expA_five_state_ld_qd_tensor_v5_0"),
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.self_test:
        print(json.dumps(integration_self_test(args.seed), indent=2, ensure_ascii=False))
        return

    args.svd_rank_values = parse_int_csv(args.svd_ranks)
    args.model_svd_rank_values = parse_model_rank_overrides(args.model_svd_ranks)
    args.heldout_session_values = parse_int_csv(args.heldout_sessions)
    args.alpha_grid_values = sorted(set(parse_float_csv(args.alpha_grid)))
    args.ridge_grid_values = sorted(set(parse_float_csv(args.ridge_grid)))
    if not args.svd_rank_values:
        raise ValueError("--svd-ranks is empty")
    if not args.alpha_grid_values:
        raise ValueError("--alpha-grid is empty")
    if any((a < 0.0 or a > 1.0) for a in args.alpha_grid_values):
        raise ValueError("--alpha-grid values must lie in [0,1]")
    if args.cov_interpolation == "arithmetic" and any(a <= 0.0 for a in args.alpha_grid_values):
        raise ValueError(
            "Arithmetic pooled-covariance shrinkage requires alpha > 0; "
            "remove alpha=0 from --alpha-grid"
        )
    if not args.ridge_grid_values:
        raise ValueError("--ridge-grid is empty")
    if args.n_permutations < 0:
        raise ValueError("--n-permutations must be non-negative")

    task_ids = parse_task_ids(args.task_ids)
    model_paths = resolve_models(args)
    ensure_dir(args.outdir)
    ensure_dir(args.outdir / "folds")

    explicit_subjects = parse_int_csv(args.subjects)
    inspections: Dict[str, Dict[str, Any]] = {}
    inventories: Dict[str, ModelInventory] = {}
    eligible_by_model: Dict[str, List[int]] = {}
    for model, path in model_paths.items():
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing embedding file for {model}: {path}")
        inventory = inspect_model(
            model,
            path,
            task_ids,
            args.min_phase_windows,
            args.expected_sessions,
            explicit_subjects,
        )
        inventories[model] = inventory
        inspections[model] = inventory.info
        eligible_by_model[model] = inventory.eligible_subjects
        model_dir = ensure_dir(args.outdir / model)
        csv_write(model_dir / "all_selected_run_qc.csv", inventory.run_qc)
        csv_write(model_dir / "subject_eligibility.csv", inventory.subject_qc)
        log(
            f"{model}: shape={inventory.info['embedding_shape']} "
            f"flat_dim={inventory.info['flat_dim']} "
            f"eligible_subjects={inventory.eligible_subjects}"
        )

    subjects = choose_common_subjects(
        eligible_by_model,
        explicit_subjects,
        args.n_subjects,
        args.subject_seed,
    )
    log(f"Selected common subjects: {subjects}")

    metadata = {
        "script_version": SCRIPT_VERSION,
        "created_at": now(),
        "model_paths": model_paths,
        "inspections": inspections,
        "selected_subjects": subjects,
        "task_ids": task_ids,
        "state_names": STATE_NAMES,
        "scenarios": SCENARIOS,
        "arms": ARMS,
        "feature_dimensions": feature_dimensions(4),
        "outer_protocol": "within-subject leave-one-session-out",
        "inner_protocol": "two-way session-heldout within the two outer-training sessions",
        "alpha_selection": "shared alpha selected by native five-state QDA balanced accuracy",
        "heldout_nulls": {
            "five_state": "within-run phase circular shift plus within-session task-run permutation",
            "context2": "within-run phase circular shift",
            "task4": "within-session task-run permutation",
        },
        "args": vars(args),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "h5py": h5py.__version__,
    }
    json_dump(args.outdir / "run_metadata.json", metadata)

    failures: List[Dict[str, Any]] = []
    for model, path in model_paths.items():
        try:
            run_model(
                model, path, subjects, task_ids,
                inventories[model], args, args.outdir,
            )
        except Exception as exc:  # noqa: BLE001
            failures.append({
                "model": model,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            })
            log(f"{model} FAILED: {exc!r}")
            if args.fail_fast:
                raise
    aggregate_outputs(args.outdir)
    json_dump(args.outdir / "run_failures.json", failures)
    log(f"Done; failures={len(failures)}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
