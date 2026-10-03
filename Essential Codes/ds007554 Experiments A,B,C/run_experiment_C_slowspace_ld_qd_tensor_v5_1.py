#!/usr/bin/env python3
"""Experiment C: semantic readout of the frozen Global-SFA space from Experiment B.

Scientific role
===============
A diagnoses endpoint geometry in the train-only reduced embedding space.
B fits slow directions without labels in the SFA objective; known state labels only
stratify modes and the temporal-order null. C asks how much endpoint semantics
survives inside B's frozen Global-SFA space.

For every B outer fold, C reads only the additive B-to-C handoff produced by
Experiment B v5.0.  B is never refitted or re-selected by C.  Within the frozen
slow coordinates, C fits the same four rest-relative feature arms as A:

    F0 = standardized ell
    F1 = standardized q
    F2 = standardized ell direct-sum standardized r,  r = q - ell
    F3 = F2 direct-sum standardized vec(ell outer r)

Three readouts are reported from the same five-state discriminant map:

    five_state : Baseline / MA / NB / NBMA / Full
    context2   : Baseline vs pooled Task
    task4      : MA / NB / NBMA / Full

The B map is trained on the two outer-training sessions and frozen before C.
C hyperparameters are selected by swapping those two sessions as inner
train/validation splits.  The untouched outer session is used once for final
metrics and structured run-level label nulls.

The primary C candidate is a pre-registered fixed leading-k Global-SFA prefix,
followed only by non-chaining degeneracy-block completion. The auxiliary r99
temporal-order statistic never selects the C subspace. Formal
high-rank runs use arithmetic pooled-covariance shrinkage with alpha > 0.
Individual empirical class covariances may therefore be singular; feasibility
requires the equal-class pooled covariance to be positive definite in every
training session.  If the full B prefix is algebraically infeasible, C
deterministically uses the largest complete slow-eigenvalue block prefix that
is jointly feasible for the rank-matched sources.  This is a feasibility
truncation, never a label-performance choice, and is reported explicitly.
"""

from __future__ import annotations

import argparse
import itertools
import hashlib
import csv
import gc
import json
import math
import platform
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ld_qd_feature_core_v5_0 import (
    ARMS,
    ClassStats,
    FeatureBuilder,
    evaluate_predictions,
    feature_dimensions,
    fit_class_stats,
    fit_qda_model,
    fit_ridge_readout,
    lda_relative_scores,
    native_prediction,
    qda_relative_scores,
    rank_condition,
    self_test as core_self_test,
)

SCRIPT_VERSION = "2026-07-30-expC-global-slowspace-ld-qd-tensor-v5.1-source-specific-ranks"
EXPECTED_B_SCHEMA = "expB-global-slow-space-handoff-v4"
STATE_NAMES = ["Baseline", "MA", "NB", "NBMA", "Full"]
TASK_STATE_NAMES = ["MA", "NB", "NBMA", "Full"]
SCENARIOS = ("five_state", "context2", "task4")
CHANCE_BY_SCENARIO = {"five_state": 0.20, "context2": 0.50, "task4": 0.25}


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
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(decode_scalar(payload), fh, ensure_ascii=False, indent=2, default=str)
    tmp.replace(path)


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
                if isinstance(value, (list, dict)):
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


def parse_float_csv(text: str) -> List[float]:
    return [float(x) for x in str(text).replace(" ", "").split(",") if x]


def parse_str_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).replace(",", " ").split() if x.strip()]


def as_float(value: Any) -> float:
    try:
        if value is None or str(value).strip() == "":
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def finite_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if len(arr) else float("nan")


def finite_sem(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0


def empirical_upper_p(real: float, null: np.ndarray) -> float:
    arr = np.asarray(null, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if not np.isfinite(real) or len(arr) == 0:
        return float("nan")
    return float((1 + np.sum(arr >= real)) / (len(arr) + 1))


def atomic_npz(path: Path | str, **payload: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=path.name + ".", suffix=".npz", delete=False
    ) as fh:
        tmp = Path(fh.name)
    try:
        np.savez_compressed(tmp, **payload)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


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
# B handoff loading and candidate-rank policy
# =============================================================================


@dataclass
class HandoffData:
    manifest_path: Path
    coordinates_path: Path
    transform_path: Path
    manifest: Dict[str, Any]
    model: str
    subject: int
    grid: str
    heldout_session: int
    training_sessions: List[int]
    svd_rank_requested: int
    svd_rank_effective: int
    candidate_rank_b: int
    handoff_rank_fixed_topk_requested: int
    handoff_selection_policy: str
    retained_rank_r99: int
    slow: np.ndarray
    ambient: np.ndarray
    global_sfa_mean: np.ndarray
    global_sfa_directions_full: np.ndarray
    global_sfa_gammas_full: np.ndarray
    global_index: np.ndarray
    subject_id: np.ndarray
    session: np.ndarray
    run: np.ndarray
    phase: np.ndarray
    task: np.ndarray
    family: np.ndarray
    state: np.ndarray
    sample_start: np.ndarray
    sample_end: np.ndarray
    center_sample: np.ndarray
    train_mask: np.ndarray
    test_mask: np.ndarray
    source_fingerprint: str


def load_handoff(manifest_path: Path) -> HandoffData:
    manifest_path = Path(manifest_path)
    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    schema = str(manifest.get("schema_version", ""))
    if schema != EXPECTED_B_SCHEMA:
        raise ValueError(f"Unsupported B handoff schema {schema!r}; expected {EXPECTED_B_SCHEMA!r}")
    files = manifest.get("files", {})
    coord_path = manifest_path.parent / str(files.get("coordinates", "B_to_C_coordinates.npz"))
    transform_path = manifest_path.parent / str(files.get("transform", "B_to_C_transform.npz"))
    for path in (coord_path, transform_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing B handoff file: {path}")
    expected_hashes = manifest.get("file_sha256", {})
    for key, path in (("coordinates", coord_path), ("transform", transform_path)):
        expected = str(expected_hashes.get(key, ""))
        if expected and sha256_file(path) != expected:
            raise RuntimeError(f"B handoff hash mismatch for {path}")

    with np.load(coord_path, allow_pickle=False) as z:
        required = [
            "global_slow_coordinates", "svd_coordinates", "global_index", "subject_id",
            "session_id", "run_id", "phase_id", "task_id", "task_family_id",
            "state_label", "sample_start", "sample_end", "center_sample",
            "is_training_session", "is_heldout_session",
        ]
        missing = [k for k in required if k not in z]
        if missing:
            raise KeyError(f"B coordinate handoff missing arrays: {missing}")
        arrays = {k: np.asarray(z[k]) for k in required}
    with np.load(transform_path, allow_pickle=False) as z:
        required_t = [
            "global_sfa_mean_svd", "global_sfa_directions_svd_full",
            "global_sfa_gammas_full",
        ]
        missing = [k for k in required_t if k not in z]
        if missing:
            raise KeyError(f"B transform handoff missing arrays: {missing}")
        tarrays = {k: np.asarray(z[k]) for k in required_t}

    slow = np.asarray(arrays["global_slow_coordinates"], dtype=np.float64)
    ambient = np.asarray(arrays["svd_coordinates"], dtype=np.float64)
    if slow.ndim != 2 or ambient.ndim != 2 or len(slow) != len(ambient):
        raise ValueError(f"Invalid B coordinate shapes slow={slow.shape}, ambient={ambient.shape}")
    n = len(slow)
    for key, arr in arrays.items():
        if key not in {"global_slow_coordinates", "svd_coordinates"} and len(arr) != n:
            raise ValueError(f"B handoff length mismatch: {key}={len(arr)} vs n={n}")
    if not np.all(np.isfinite(slow)) or not np.all(np.isfinite(ambient)):
        raise FloatingPointError("Non-finite B coordinates")

    model = str(manifest["model"]); subject = int(manifest["subject_id"])
    heldout = int(manifest["heldout_session"])
    training_sessions = sorted(int(x) for x in manifest["training_sessions"])
    candidate = int(manifest["candidate_rank_fixed_topk_block_complete"])
    fixed_topk = int(manifest["handoff_rank_fixed_topk_requested"])
    selection_policy = str(manifest.get(
        "handoff_selection_policy",
        "pre_registered_fixed_topk_then_slow_prefix_nonchaining_block_completion",
    ))
    retained = int(manifest["retained_rank_r99"])
    effective = int(manifest["svd_rank_effective"])
    if slow.shape[1] != candidate or ambient.shape[1] != effective:
        raise ValueError(
            f"B dimension mismatch slow={slow.shape[1]}/{candidate}, "
            f"ambient={ambient.shape[1]}/{effective}"
        )
    mean = np.asarray(tarrays["global_sfa_mean_svd"], dtype=np.float64).reshape(-1)
    directions = np.asarray(tarrays["global_sfa_directions_svd_full"], dtype=np.float64)
    gammas = np.asarray(tarrays["global_sfa_gammas_full"], dtype=np.float64).reshape(-1)
    if len(mean) != effective or directions.shape[0] != effective or directions.shape[1] != len(gammas):
        raise ValueError("B full Global-SFA transform dimensions are inconsistent")

    subject_id = np.asarray(arrays["subject_id"], dtype=np.int64)
    session = np.asarray(arrays["session_id"], dtype=np.int64)
    train_mask = np.asarray(arrays["is_training_session"], dtype=bool)
    test_mask = np.asarray(arrays["is_heldout_session"], dtype=bool)
    if np.any(subject_id != subject):
        raise ValueError("B handoff contains more than one subject")
    if np.any(train_mask & test_mask) or not np.all(train_mask | test_mask):
        raise ValueError("B train/test mask ledger is not a partition")
    if sorted(np.unique(session[train_mask]).astype(int).tolist()) != training_sessions:
        raise ValueError("B training-session mask disagrees with manifest")
    if np.unique(session[test_mask]).astype(int).tolist() != [heldout]:
        raise ValueError("B held-out mask disagrees with manifest")
    state = np.asarray(arrays["state_label"], dtype=np.int64)
    train_counts = np.bincount(state[train_mask], minlength=5)
    test_counts = np.bincount(state[test_mask], minlength=5)
    if np.any(train_counts == 0) or np.any(test_counts == 0):
        raise ValueError(
            f"B handoff must contain all five states in train/test; "
            f"train={train_counts.tolist()} test={test_counts.tolist()}"
        )
    fingerprint = hashlib.sha256(
        (sha256_file(manifest_path) + sha256_file(coord_path) + sha256_file(transform_path)).encode()
    ).hexdigest()
    return HandoffData(
        manifest_path, coord_path, transform_path, manifest, model, subject,
        str(manifest["grid"]), heldout, training_sessions,
        int(manifest["svd_rank_requested"]), effective, candidate,
        fixed_topk, selection_policy, retained,
        slow, ambient, mean, directions, gammas,
        np.asarray(arrays["global_index"], dtype=np.int64), subject_id, session,
        np.asarray(arrays["run_id"], dtype=np.int64),
        np.asarray(arrays["phase_id"], dtype=np.int64),
        np.asarray(arrays["task_id"], dtype=np.int64),
        np.asarray(arrays["task_family_id"], dtype=np.int64), state,
        np.asarray(arrays["sample_start"], dtype=np.int64),
        np.asarray(arrays["sample_end"], dtype=np.int64),
        np.asarray(arrays["center_sample"], dtype=np.int64),
        train_mask, test_mask, fingerprint,
    )


def complete_block_ends(manifest: Mapping[str, Any], candidate_rank: int) -> List[int]:
    ends: List[int] = []
    for block in manifest.get("all_degenerate_blocks_zero_based", []):
        axes = sorted(int(x) for x in block)
        if axes and max(axes) + 1 <= int(candidate_rank):
            ends.append(int(max(axes) + 1))
    if candidate_rank > 0 and candidate_rank not in ends:
        ends.append(int(candidate_rank))
    return sorted(set(ends))


def complete_block_suffix_ranks(manifest: Mapping[str, Any], support_rank: int) -> List[int]:
    ranks: List[int] = []
    for block in manifest.get("all_degenerate_blocks_zero_based", []):
        axes = sorted(int(x) for x in block)
        if axes and min(axes) < int(support_rank):
            ranks.append(int(support_rank - min(axes)))
    ranks.append(int(support_rank))
    return sorted(set(r for r in ranks if 0 < r <= int(support_rank)))


def axis_coordinates(
    data: HandoffData, source: str, rank: int, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    rank = int(rank)
    if rank <= 0:
        raise ValueError("axis rank must be positive")
    if source == "slow_prefix":
        return np.asarray(data.slow[:, :rank], dtype=np.float64)
    if source == "pca_prefix":
        return np.asarray(data.ambient[:, :rank], dtype=np.float64)
    centered = data.ambient - data.global_sfa_mean[None, :]
    if source == "fast_suffix":
        if rank > data.global_sfa_directions_full.shape[1]:
            raise ValueError("fast suffix exceeds Global-SFA support")
        return np.asarray(centered @ data.global_sfa_directions_full[:, -rank:], dtype=np.float64)
    if source == "ambient_svd":
        return np.asarray(data.ambient[:, :rank], dtype=np.float64)
    if source == "random":
        if rng is None:
            raise ValueError("random source requires rng")
        q, _ = np.linalg.qr(rng.standard_normal((data.svd_rank_effective, rank)), mode="reduced")
        return np.asarray(centered @ q[:, :rank], dtype=np.float64)
    raise KeyError(f"Unknown axis source {source!r}")


def qda_feasibility_error(X: np.ndarray, data: HandoffData) -> Optional[str]:
    masks = [data.session == int(s) for s in data.training_sessions] + [data.train_mask]
    labels = [f"session_{s}" for s in data.training_sessions] + ["outer_train"]
    for label, mask in zip(labels, masks):
        counts = np.bincount(data.state[mask], minlength=5)
        if np.any(counts == 0):
            return f"{label}: missing class counts={counts.tolist()}"
        try:
            fit_class_stats(X[mask], data.state[mask], STATE_NAMES)
        except Exception as exc:  # noqa: BLE001
            return f"{label}: {type(exc).__name__}: {exc}"
    return None


def choose_analysis_rank(
    data: HandoffData, max_candidate_rank: int, requested_sources: Sequence[str]
) -> Dict[str, Any]:
    """Choose a primary slow/PCA rank and an independent fast-control rank.

    The B handoff rank is defined by the fixed-k slow prefix only.  Slow and PCA
    remain exactly rank matched.  The fast suffix is completed at its nearest
    non-chaining spectral-block boundary and may therefore differ in dimension;
    that mismatch is reported instead of invalidating the primary slow-space
    fold.  No rank is selected by held-out labels or performance.
    """
    candidate = int(data.candidate_rank_b)
    base = {
        "candidate_rank_b": candidate,
        "handoff_rank_fixed_topk_requested": data.handoff_rank_fixed_topk_requested,
        "handoff_selection_policy": data.handoff_selection_policy,
        "retained_rank_r99": int(data.retained_rank_r99),
        "requested_sources": list(requested_sources),
    }
    if candidate <= 0:
        return {
            **base,
            "analysis_rank": 0,
            "source_ranks": {},
            "status": "unavailable_no_B_fixed_topk_candidate",
        }
    if candidate >= int(data.svd_rank_effective):
        return {
            **base,
            "analysis_rank": 0,
            "source_ranks": {},
            "status": "unavailable_B_fixed_topk_saturates_ambient",
            "rank_ratio": float(candidate / max(data.svd_rank_effective, 1)),
        }

    cap = candidate
    if int(max_candidate_rank) > 0:
        cap = min(cap, int(max_candidate_rank))
    slow_prefix_ends = [
        x for x in complete_block_ends(data.manifest, candidate) if x <= cap
    ]
    diagnostics: List[Dict[str, Any]] = []
    primary_sources = ["slow_prefix"]
    if "pca_prefix" in requested_sources:
        primary_sources.append("pca_prefix")

    slow_rank = 0
    for rank in sorted(slow_prefix_ends, reverse=True):
        errors: Dict[str, str] = {}
        for source in primary_sources:
            try:
                err = qda_feasibility_error(axis_coordinates(data, source, rank), data)
            except Exception as exc:  # noqa: BLE001
                err = f"{type(exc).__name__}: {exc}"
            if err is not None:
                errors[source] = err
        diagnostics.append({"rank": int(rank), "errors": errors})
        if not errors:
            slow_rank = int(rank)
            break
    if slow_rank <= 0:
        return {
            **base,
            "analysis_rank": 0,
            "source_ranks": {},
            "slow_prefix_complete_block_ends": slow_prefix_ends,
            "status": "unavailable_no_slow_prefix_block_jointly_QDA_feasible",
            "covers_fixed_topk": False,
            "covers_r99_auxiliary": False,
            "feasibility_diagnostics": diagnostics,
        }

    source_ranks: Dict[str, int] = {
        "slow_prefix": slow_rank,
        "pca_prefix": slow_rank,
    }
    source_status: Dict[str, str] = {
        "slow_prefix": "available",
        "pca_prefix": "available",
    }

    fast_suffix_ranks = [
        r for r in complete_block_suffix_ranks(
            data.manifest, data.global_sfa_directions_full.shape[1]
        )
        if 0 < int(r) < int(data.svd_rank_effective)
    ]
    fast_diagnostics: List[Dict[str, Any]] = []
    fast_rank = 0
    if "fast_suffix" in requested_sources:
        # Prefer the nearest complete rank.  At equal distance prefer the rank
        # not exceeding the slow rank, so the fast control does not receive a
        # dimensional advantage merely because of a tie.
        ordered_fast = sorted(
            set(int(r) for r in fast_suffix_ranks),
            key=lambda r: (abs(r - slow_rank), r > slow_rank, r),
        )
        for rank in ordered_fast:
            try:
                err = qda_feasibility_error(
                    axis_coordinates(data, "fast_suffix", rank), data
                )
            except Exception as exc:  # noqa: BLE001
                err = f"{type(exc).__name__}: {exc}"
            fast_diagnostics.append({"rank": int(rank), "error": err})
            if err is None:
                fast_rank = int(rank)
                break
        if fast_rank > 0:
            source_ranks["fast_suffix"] = fast_rank
            source_status["fast_suffix"] = (
                "available_exact_rank_match"
                if fast_rank == slow_rank
                else "available_nearest_block_complete_rank"
            )
        else:
            source_status["fast_suffix"] = "unavailable_no_block_complete_QDA_feasible_rank"

    status = (
        "full_B_candidate"
        if slow_rank == candidate
        else "block_complete_joint_feasibility_prefix"
    )
    return {
        **base,
        "analysis_rank": slow_rank,
        "source_ranks": source_ranks,
        "source_status": source_status,
        "status": status,
        "covers_fixed_topk": bool(
            slow_rank >= data.handoff_rank_fixed_topk_requested
        ),
        "covers_r99_auxiliary": bool(
            slow_rank >= data.retained_rank_r99 and data.retained_rank_r99 > 0
        ),
        "slow_prefix_complete_block_ends": slow_prefix_ends,
        "fast_suffix_block_complete_ranks": fast_suffix_ranks,
        "fast_suffix_rank": int(fast_rank),
        "fast_minus_slow_rank": int(fast_rank - slow_rank) if fast_rank > 0 else None,
        "fast_rank_match_exact": bool(fast_rank == slow_rank and fast_rank > 0),
        "feasibility_diagnostics": diagnostics,
        "fast_feasibility_diagnostics": fast_diagnostics,
    }


# =============================================================================
# Shared A-compatible readout protocol
# =============================================================================


def scenario_data(
    features: Mapping[str, np.ndarray],
    state: np.ndarray,
    scenario: str,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, int, List[str], np.ndarray]:
    state = np.asarray(state, dtype=np.int64)
    if scenario == "five_state":
        mask = np.ones(len(state), dtype=bool)
        return {a: np.asarray(features[a]) for a in ARMS}, state, 5, list(STATE_NAMES), mask
    if scenario == "context2":
        mask = np.ones(len(state), dtype=bool)
        y = (state > 0).astype(np.int64)
        return {a: np.asarray(features[a]) for a in ARMS}, y, 2, ["Baseline", "Task"], mask
    if scenario == "task4":
        mask = state > 0
        y = state[mask] - 1
        return {a: np.asarray(features[a])[mask] for a in ARMS}, y, 4, list(TASK_STATE_NAMES), mask
    raise KeyError(scenario)


def stable_select_alpha(scores: Mapping[float, Sequence[float]], n_splits: int) -> float:
    valid: List[Tuple[float, float]] = []
    for alpha, values in scores.items():
        arr = np.asarray(values, dtype=np.float64)
        if len(arr) == n_splits and np.all(np.isfinite(arr)):
            valid.append((float(np.mean(arr)), float(alpha)))
    if not valid:
        raise RuntimeError("No alpha candidate was valid in every inner split")
    valid.sort(key=lambda x: (x[0], x[1]))  # tie -> more pooling
    return float(valid[-1][1])


def stable_select_ridge(
    scores: Mapping[Tuple[str, str, float, float], Sequence[float]],
    scenario: str,
    arm: str,
    alpha: float,
    ridge_grid: Sequence[float],
    n_splits: int,
) -> float:
    valid: List[Tuple[float, float]] = []
    for lam in ridge_grid:
        arr = np.asarray(
            scores.get((scenario, arm, float(alpha), float(lam)), []), dtype=np.float64
        )
        if len(arr) == n_splits and np.all(np.isfinite(arr)):
            valid.append((float(np.mean(arr)), float(lam)))
    if not valid:
        raise RuntimeError(f"No ridge candidate for {scenario}/{arm}")
    valid.sort(key=lambda x: (x[0], x[1]))  # tie -> stronger regularization
    return float(valid[-1][1])


@dataclass
class SelectionResult:
    alpha: float
    ridge_by_scenario_arm: Dict[str, Dict[str, float]]
    alpha_rows: List[Dict[str, Any]]
    ridge_rows: List[Dict[str, Any]]


def nested_select(
    X: np.ndarray,
    state: np.ndarray,
    sessions: np.ndarray,
    training_sessions: Sequence[int],
    alpha_grid: Sequence[float],
    ridge_grid: Sequence[float],
    covariance_interpolation: str,
    degenerate_tol: float,
) -> SelectionResult:
    train_sessions = sorted(int(x) for x in training_sessions)
    if len(train_sessions) != 2:
        raise ValueError(f"C expects two B training sessions, got {train_sessions}")
    alpha_scores: Dict[float, List[float]] = {float(a): [] for a in alpha_grid}
    ridge_scores: Dict[Tuple[str, str, float, float], List[float]] = {}
    alpha_rows: List[Dict[str, Any]] = []
    ridge_rows: List[Dict[str, Any]] = []

    for inner_index, val_session in enumerate(train_sessions, start=1):
        tr_session = [x for x in train_sessions if x != val_session][0]
        tr = sessions == int(tr_session)
        va = sessions == int(val_session)
        Xtr, Xva = X[tr], X[va]
        ytr, yva = state[tr], state[va]
        stats = fit_class_stats(Xtr, ytr, STATE_NAMES)
        ell_tr = lda_relative_scores(Xtr, stats)
        ell_va = lda_relative_scores(Xva, stats)

        for alpha in alpha_grid:
            alpha = float(alpha)
            try:
                qda = fit_qda_model(stats, alpha, covariance_interpolation)
                qtr = qda_relative_scores(Xtr, stats, qda)
                qva = qda_relative_scores(Xva, stats, qda)
                pred = native_prediction(qva)
                score = float(evaluate_predictions(yva, pred, 5)["balanced_accuracy"])
                alpha_scores[alpha].append(score)
                valid = 1
                error = ""
            except Exception as exc:  # noqa: BLE001
                qtr = qva = None
                score = float("nan")
                valid = 0
                error = repr(exc)

            alpha_rows.append({
                "inner_split": inner_index,
                "train_session": tr_session,
                "validation_session": val_session,
                "slow_dimension": int(X.shape[1]),
                "alpha": alpha,
                "native_qda_validation_bacc_five_state": score,
                "valid": valid,
                "error": error,
            })
            if not valid:
                continue

            builder = FeatureBuilder.fit(ell_tr, qtr, degenerate_tol)
            ftr = builder.transform(ell_tr, qtr)
            fva = builder.transform(ell_va, qva)
            for scenario in SCENARIOS:
                Xtr_map, ytr_s, K, names, _ = scenario_data(ftr, ytr, scenario)
                Xva_map, yva_s, _, _, _ = scenario_data(fva, yva, scenario)
                for arm in ARMS:
                    for lam in ridge_grid:
                        readout = fit_ridge_readout(Xtr_map[arm], ytr_s, K, float(lam))
                        pred = readout.predict(Xva_map[arm])
                        metric = evaluate_predictions(yva_s, pred, K)
                        value = float(metric["balanced_accuracy"])
                        ridge_scores.setdefault(
                            (scenario, arm, alpha, float(lam)), []
                        ).append(value)
                        ridge_rows.append({
                            "inner_split": inner_index,
                            "train_session": tr_session,
                            "validation_session": val_session,
                            "slow_dimension": int(X.shape[1]),
                            "scenario": scenario,
                            "class_names": names,
                            "arm": arm,
                            "feature_dim": int(Xtr_map[arm].shape[1]),
                            "alpha": alpha,
                            "ridge_lambda": float(lam),
                            "validation_balanced_accuracy": value,
                            "weight_norm": readout.weight_norm,
                            "degenerate_features": builder.degenerate_report(),
                        })

    selected_alpha = stable_select_alpha(alpha_scores, len(train_sessions))
    selected_ridge: Dict[str, Dict[str, float]] = {s: {} for s in SCENARIOS}
    for scenario in SCENARIOS:
        for arm in ARMS:
            selected_ridge[scenario][arm] = stable_select_ridge(
                ridge_scores, scenario, arm, selected_alpha, ridge_grid, len(train_sessions)
            )
    return SelectionResult(selected_alpha, selected_ridge, alpha_rows, ridge_rows)


# =============================================================================
# Structured held-out nulls
# =============================================================================


def heldout_run_positions(data: HandoffData) -> Dict[int, np.ndarray]:
    test_positions = np.flatnonzero(data.test_mask)
    out: Dict[int, np.ndarray] = {}
    for run_id in sorted(np.unique(data.run[data.test_mask]).astype(int).tolist()):
        pos = test_positions[data.run[test_positions] == int(run_id)]
        out[int(run_id)] = pos[np.argsort(data.sample_start[pos], kind="mergesort")]
    return out


def permuted_heldout_labels(
    data: HandoffData,
    scenario: str,
    rng: np.random.Generator,
) -> np.ndarray:
    runs = heldout_run_positions(data)
    if len(runs) != 4:
        raise ValueError(f"Held-out ledger must contain four task runs, got {sorted(runs)}")
    run_task_class: Dict[int, int] = {}
    for run_id, pos in runs.items():
        states = np.unique(data.state[pos][data.phase[pos] == 1])
        if len(states) != 1 or not (1 <= int(states[0]) <= 4):
            raise ValueError(f"Run {run_id} has invalid task-state ledger: {states.tolist()}")
        run_task_class[run_id] = int(states[0])
    run_ids = sorted(runs)
    original = np.asarray([run_task_class[r] for r in run_ids], dtype=np.int64)
    permuted = rng.permutation(original)
    task_map = {r: int(c) for r, c in zip(run_ids, permuted)}

    test_pos = np.flatnonzero(data.test_mask)
    if scenario == "task4":
        task_pos = test_pos[data.state[test_pos] > 0]
        return np.asarray([task_map[int(data.run[p])] - 1 for p in task_pos], dtype=np.int64)

    local = {int(p): i for i, p in enumerate(test_pos.tolist())}
    out_state = np.empty(len(test_pos), dtype=np.int64)
    for run_id, pos in runs.items():
        phase = data.phase[pos].astype(np.int64)
        if len(phase) < 2:
            raise ValueError(f"Run {run_id} too short for phase circular shift")
        shift = int(rng.integers(1, len(phase)))
        shifted = np.roll(phase, shift)
        values = np.where(shifted == 0, 0, task_map[run_id]).astype(np.int64)
        for p, value in zip(pos, values):
            out_state[local[int(p)]] = int(value)
    if scenario == "five_state":
        return out_state
    if scenario == "context2":
        return (out_state > 0).astype(np.int64)
    raise KeyError(scenario)


def exact_task4_permuted_labels(data: HandoffData) -> List[np.ndarray]:
    """Enumerate the complete 4! run-to-task randomization group."""
    runs = heldout_run_positions(data)
    run_ids = sorted(runs)
    if len(run_ids) != 4:
        raise ValueError(f"Held-out ledger must contain four task runs, got {run_ids}")
    run_task_class: Dict[int, int] = {}
    for run_id, pos in runs.items():
        states = np.unique(data.state[pos][data.phase[pos] == 1])
        if len(states) != 1 or not (1 <= int(states[0]) <= 4):
            raise ValueError(f"Run {run_id} has invalid task-state ledger: {states.tolist()}")
        run_task_class[run_id] = int(states[0])
    original = [run_task_class[r] for r in run_ids]
    task_pos = np.flatnonzero(data.test_mask)[data.state[data.test_mask] > 0]
    outputs: List[np.ndarray] = []
    for assignment in itertools.permutations(original):
        task_map = {r: int(c) for r, c in zip(run_ids, assignment)}
        outputs.append(np.asarray(
            [task_map[int(data.run[p])] - 1 for p in task_pos], dtype=np.int64
        ))
    return outputs


def structured_null_bacc(
    data: HandoffData,
    scenario: str,
    predictions: np.ndarray,
    n_permutations: int,
    seed: int,
) -> np.ndarray:
    if int(n_permutations) <= 0:
        return np.empty(0, dtype=np.float64)
    K = 5 if scenario == "five_state" else (2 if scenario == "context2" else 4)
    if scenario == "task4":
        labels = exact_task4_permuted_labels(data)
        return np.asarray([
            evaluate_predictions(y, predictions, K)["balanced_accuracy"] for y in labels
        ], dtype=np.float64)
    rng = np.random.default_rng(seed)
    out = np.empty(int(n_permutations), dtype=np.float64)
    for b in range(int(n_permutations)):
        y = permuted_heldout_labels(data, scenario, rng)
        out[b] = evaluate_predictions(y, predictions, K)["balanced_accuracy"]
    return out


# =============================================================================
# A alignment and semantic retention
# =============================================================================


def a_fold_path(a_root: Path, data: HandoffData) -> Path:
    return (
        Path(a_root) / "folds" / data.model / f"sub-{data.subject:03d}"
        / f"heldout-session-{data.heldout_session}"
        / f"svd-rank-{data.svd_rank_requested}"
    )


def compare_a_ledger(a_dir: Path, data: HandoffData) -> Dict[str, Any]:
    ledger_path = a_dir / "heldout_ledger.csv"
    if not ledger_path.exists():
        return {"status": "missing_A_ledger", "a_fold_dir": str(a_dir)}
    rows = csv_read(ledger_path)
    rows = sorted(rows, key=lambda r: int(r["global_index"]))
    bpos = np.flatnonzero(data.test_mask)
    bpos = bpos[np.argsort(data.global_index[bpos])]
    if len(rows) != len(bpos):
        return {
            "status": "row_count_mismatch",
            "a_rows": len(rows), "b_rows": len(bpos), "a_fold_dir": str(a_dir),
        }
    fields = {
        "global_index": data.global_index,
        "session_id": data.session,
        "run_id": data.run,
        "sample_start": data.sample_start,
        "sample_end": data.sample_end,
        "phase_id": data.phase,
        "task_id": data.task,
        "state_label": data.state,
    }
    mismatches: Dict[str, int] = {}
    for field, arr in fields.items():
        a = np.asarray([int(r[field]) for r in rows], dtype=np.int64)
        b = np.asarray(arr[bpos], dtype=np.int64)
        mismatches[field] = int(np.sum(a != b))
    status = "pass" if max(mismatches.values(), default=0) == 0 else "value_mismatch"
    return {
        "status": status,
        "a_fold_dir": str(a_dir),
        "n_rows": len(rows),
        "mismatch_counts": mismatches,
    }


def load_a_metrics(a_dir: Path) -> Dict[Tuple[str, str], Dict[str, str]]:
    rows = csv_read(a_dir / "outer_metrics.csv")
    return {
        (r.get("scenario", ""), r.get("arm", "")): r
        for r in rows if r.get("arm", "") in ARMS
    }


def external_a_retention_rows(
    c_rows: Sequence[Mapping[str, Any]],
    a_root: Optional[Path],
    data: HandoffData,
    alignment: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    if a_root is None:
        return []
    if alignment.get("status") != "pass":
        return [{
            "model": data.model, "subject_id": data.subject,
            "grid": data.grid, "heldout_session": data.heldout_session,
            "M_requested": data.svd_rank_requested,
            "status": f"unavailable_alignment_{alignment.get('status')}",
        }]
    a_map = load_a_metrics(a_fold_path(a_root, data))
    rows: List[Dict[str, Any]] = []
    for c in c_rows:
        if c.get("axis_source") != "slow_prefix" or c.get("arm") not in ARMS:
            continue
        scenario, arm = str(c["scenario"]), str(c["arm"])
        a = a_map.get((scenario, arm))
        if a is None:
            rows.append({
                "model": data.model, "subject_id": data.subject, "grid": data.grid,
                "heldout_session": data.heldout_session, "M_requested": data.svd_rank_requested,
                "scenario": scenario, "arm": arm, "status": "missing_A_metric",
            })
            continue
        av = as_float(a.get("test_balanced_accuracy")); cv = float(c["test_balanced_accuracy"])
        chance = CHANCE_BY_SCENARIO[scenario]; denom = av - chance
        rows.append({
            "model": data.model, "subject_id": data.subject, "grid": data.grid,
            "heldout_session": data.heldout_session, "M_requested": data.svd_rank_requested,
            "axis_source": "slow_prefix", "axis_rank": c.get("axis_rank"),
            "rank_ratio": c.get("rank_ratio"), "scenario": scenario, "arm": arm,
            "chance": chance, "A_test_balanced_accuracy": av,
            "C_test_balanced_accuracy": cv, "A_minus_C_gap": av - cv,
            "C_minus_A_delta": cv - av,
            "chance_normalized_retention": (cv - chance) / denom
            if np.isfinite(denom) and abs(denom) > 1e-12 else float("nan"),
            "A_heldout_permutation_p": as_float(a.get("heldout_permutation_p")),
            "C_heldout_permutation_p": float(c.get("heldout_permutation_p", float("nan"))),
            "status": "ok",
        })
    return rows


def internal_retention_rows(
    c_rows: Sequence[Mapping[str, Any]], data: HandoffData
) -> List[Dict[str, Any]]:
    metric = {
        (str(r.get("axis_source")), int(r.get("random_rep", -1)), str(r.get("scenario")), str(r.get("arm"))): r
        for r in c_rows if r.get("arm") in ARMS
    }
    rows: List[Dict[str, Any]] = []
    for key, r in metric.items():
        source, rep, scenario, arm = key
        if source == "ambient_svd":
            continue
        ambient = metric.get(("ambient_svd", -1, scenario, arm))
        fast = metric.get(("fast_suffix", -1, scenario, arm))
        pca = metric.get(("pca_prefix", -1, scenario, arm))
        slow = metric.get(("slow_prefix", -1, scenario, arm))
        chance = CHANCE_BY_SCENARIO[scenario]
        source_bacc = float(r["test_balanced_accuracy"])
        ambient_bacc = as_float(ambient.get("test_balanced_accuracy")) if ambient else float("nan")
        denom = ambient_bacc - chance
        rows.append({
            "model": data.model, "subject_id": data.subject, "grid": data.grid,
            "heldout_session": data.heldout_session, "M_requested": data.svd_rank_requested,
            "axis_source": source, "random_rep": rep, "axis_rank": r.get("axis_rank"),
            "rank_ratio": r.get("rank_ratio"), "scenario": scenario, "arm": arm,
            "chance": chance, "ambient_test_balanced_accuracy": ambient_bacc,
            "source_test_balanced_accuracy": source_bacc,
            "source_minus_ambient_delta": source_bacc - ambient_bacc,
            "chance_normalized_retention_vs_ambient": (source_bacc - chance) / denom
            if np.isfinite(denom) and abs(denom) > 1e-12 else float("nan"),
            "slow_minus_fast_delta": (as_float(slow.get("test_balanced_accuracy")) - as_float(fast.get("test_balanced_accuracy")))
            if slow and fast else float("nan"),
            "slow_minus_pca_delta": (as_float(slow.get("test_balanced_accuracy")) - as_float(pca.get("test_balanced_accuracy")))
            if slow and pca else float("nan"),
            "status": "ok" if ambient is not None else "ambient_unavailable",
        })
    return rows


# =============================================================================
# Fold execution
# =============================================================================


def covariance_diagnostic_rows(
    stats: ClassStats,
    base: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    cov_rows: List[Dict[str, Any]] = []
    spectra: List[Dict[str, Any]] = []
    rank, cond, logdet, eig = rank_condition(stats.pooled)
    cov_rows.append({
        **base, "class_id": "pooled", "class_name": "pooled_equal_class",
        "n_class": int(np.sum(stats.class_counts)), "rank": rank,
        "dimension": int(stats.pooled.shape[0]), "condition_number": cond,
        "logdet": logdet, "min_eigenvalue": float(np.min(eig)),
        "max_eigenvalue": float(np.max(eig)),
    })
    for c, name in enumerate(stats.class_names):
        rank, cond, logdet, eig = rank_condition(stats.covs[c])
        cov_rows.append({
            **base, "class_id": c, "class_name": name,
            "n_class": int(stats.class_counts[c]), "rank": rank,
            "dimension": int(stats.covs[c].shape[0]), "condition_number": cond,
            "logdet": logdet, "min_eigenvalue": float(np.min(eig)),
            "max_eigenvalue": float(np.max(eig)),
        })
        for j, value in enumerate(stats.lambda_spectra[c]):
            spectra.append({
                **base, "class_id": c, "class_name": name,
                "eigen_index": j + 1, "generalized_lambda": float(value),
            })
    return cov_rows, spectra


def alpha_selection_diagnostics(selection: SelectionResult) -> Dict[str, Any]:
    by_alpha: Dict[float, List[float]] = {}
    split_winners: Dict[int, Tuple[float, float]] = {}
    for row in selection.alpha_rows:
        if int(row.get("valid", 0)) != 1:
            continue
        alpha = float(row["alpha"]); score = float(row["native_qda_validation_bacc_five_state"])
        by_alpha.setdefault(alpha, []).append(score)
        split = int(row["inner_split"]); current = split_winners.get(split)
        if current is None or (score, alpha) > current:
            split_winners[split] = (score, alpha)
    means = sorted((float(np.mean(v)), a) for a, v in by_alpha.items() if len(v) == 2)
    margin = means[-1][0] - means[-2][0] if len(means) > 1 else float("nan")
    winners = [x[1] for x in split_winners.values()]
    return {
        "selected_alpha": selection.alpha,
        "selected_alpha_mean_margin_to_runner_up": margin,
        "inner_split_alpha_winners": winners,
        "inner_split_winner_agreement": bool(len(set(winners)) == 1) if winners else False,
    }


def evaluate_axis_space(
    data: HandoffData, args: argparse.Namespace, source: str, X: np.ndarray,
    axis_rank: int, rank_info: Mapping[str, Any], random_rep: int = -1,
) -> Dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    selection = nested_select(
        X=X, state=data.state, sessions=data.session,
        training_sessions=data.training_sessions, alpha_grid=args.alpha_grid_values,
        ridge_grid=args.ridge_grid_values, covariance_interpolation=args.cov_interpolation,
        degenerate_tol=args.degenerate_tol,
    )
    Xtr, Xte = X[data.train_mask], X[data.test_mask]
    ytr, yte = data.state[data.train_mask], data.state[data.test_mask]
    train_counts = np.bincount(ytr, minlength=5); test_counts = np.bincount(yte, minlength=5)
    if np.any(train_counts == 0) or np.any(test_counts == 0):
        raise ValueError(f"C train/test misses a state: train={train_counts.tolist()} test={test_counts.tolist()}")
    stats = fit_class_stats(Xtr, ytr, STATE_NAMES)
    ell_tr = lda_relative_scores(Xtr, stats); ell_te = lda_relative_scores(Xte, stats)
    qda = fit_qda_model(stats, selection.alpha, args.cov_interpolation)
    qtr = qda_relative_scores(Xtr, stats, qda); qte = qda_relative_scores(Xte, stats, qda)
    builder = FeatureBuilder.fit(ell_tr, qtr, args.degenerate_tol)
    ftr = builder.transform(ell_tr, qtr); fte = builder.transform(ell_te, qte)
    rank_ratio = float(axis_rank / data.svd_rank_effective)
    base = {
        "script_version": SCRIPT_VERSION, "model": data.model,
        "subject_id": data.subject, "heldout_session": data.heldout_session,
        "training_sessions": data.training_sessions, "grid": data.grid,
        "M_requested": data.svd_rank_requested, "M_effective": data.svd_rank_effective,
        "B_retained_rank_r99_auxiliary": data.retained_rank_r99,
        "B_handoff_rank_fixed_topk_requested": data.handoff_rank_fixed_topk_requested,
        "B_candidate_rank_fixed_topk_block_complete": data.candidate_rank_b,
        "B_handoff_selection_policy": data.handoff_selection_policy,
        "C_analysis_rank": int(rank_info.get("analysis_rank", axis_rank)),
        "C_rank_policy_status": rank_info.get("status", "ambient_or_random"),
        "C_rank_covers_B_fixed_topk": rank_info.get("covers_fixed_topk", False),
        "C_rank_covers_B_r99_auxiliary": rank_info.get("covers_r99_auxiliary", False),
        "C_fast_suffix_rank": rank_info.get("fast_suffix_rank", ""),
        "C_fast_minus_slow_rank": rank_info.get("fast_minus_slow_rank", ""),
        "C_fast_rank_match_exact": rank_info.get("fast_rank_match_exact", False),
        "axis_source": source, "axis_rank": int(axis_rank), "rank_ratio": rank_ratio,
        "random_rep": int(random_rep), "selected_alpha": selection.alpha,
        "covariance_interpolation": args.cov_interpolation,
        "source_B_manifest": str(data.manifest_path),
        "source_B_fingerprint": data.source_fingerprint,
    }
    outer_rows: List[Dict[str, Any]] = []; perm_rows: List[Dict[str, Any]] = []
    selected_rows: List[Dict[str, Any]] = [{
        **base, "parameter_type": "alpha", "scenario": "five_state_native_QDA",
        "arm": "native_QDA", "selected_value": selection.alpha,
    }]
    prediction_payload: Dict[str, np.ndarray] = {}
    null_payload: Dict[str, np.ndarray] = {}
    for arm_name, pred in (("native_LDA", native_prediction(ell_te)), ("native_QDA", native_prediction(qte))):
        metric = evaluate_predictions(yte, pred, 5)
        outer_rows.append({
            **base, "scenario": "five_state", "arm": arm_name, "feature_dim": 4,
            "ridge_lambda": "", "train_balanced_accuracy": "",
            "test_balanced_accuracy": metric["balanced_accuracy"],
            "test_accuracy": metric["accuracy"], "train_test_gap": "", "weight_norm": "",
            "class_names": STATE_NAMES, "class_recalls": metric["recalls"],
            "confusion_matrix": metric["confusion_matrix"].tolist(),
            "degenerate_features": builder.degenerate_report(),
        })
    for scenario_index, scenario in enumerate(SCENARIOS):
        Xtr_map, ytr_s, K, names, _ = scenario_data(ftr, ytr, scenario)
        Xte_map, yte_s, _, _, test_scenario_mask = scenario_data(fte, yte, scenario)
        for arm_index, arm in enumerate(ARMS):
            lam = selection.ridge_by_scenario_arm[scenario][arm]
            readout = fit_ridge_readout(Xtr_map[arm], ytr_s, K, lam)
            ptr = readout.predict(Xtr_map[arm]); pte = readout.predict(Xte_map[arm])
            mtr = evaluate_predictions(ytr_s, ptr, K); mte = evaluate_predictions(yte_s, pte, K)
            null = structured_null_bacc(
                data, scenario, pte, args.n_permutations,
                args.seed + 700000 * data.subject + 10000 * data.heldout_session
                + 1000 * {"slow_prefix": 11, "fast_suffix": 23, "pca_prefix": 37, "ambient_svd": 41, "random": 53}[source] + 100 * scenario_index + arm_index
                + max(0, random_rep) * 1000000,
            )
            exact_task4 = scenario == "task4"
            pvalue = (
                float(np.mean(null >= mte["balanced_accuracy"] - 1e-15))
                if exact_task4 and len(null)
                else empirical_upper_p(mte["balanced_accuracy"], null)
            )
            key = f"{source}__rep{random_rep}__{scenario}__{arm}"
            null_payload[key] = null
            outer_rows.append({
                **base, "scenario": scenario, "arm": arm,
                "feature_dim": int(Xtr_map[arm].shape[1]), "ridge_lambda": float(lam),
                "train_balanced_accuracy": mtr["balanced_accuracy"],
                "test_balanced_accuracy": mte["balanced_accuracy"],
                "test_accuracy": mte["accuracy"],
                "train_test_gap": mtr["balanced_accuracy"] - mte["balanced_accuracy"],
                "weight_norm": readout.weight_norm, "class_names": names,
                "class_recalls": mte["recalls"],
                "confusion_matrix": mte["confusion_matrix"].tolist(),
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
                "heldout_permutation_p": pvalue,
                "heldout_null_mean": float(np.mean(null)) if len(null) else float("nan"),
                "heldout_null_q95": float(np.quantile(null, 0.95)) if len(null) else float("nan"),
            })
            perm_rows.append({
                **base, "scenario": scenario, "arm": arm,
                "real_test_balanced_accuracy": mte["balanced_accuracy"],
                "n_permutations": int(len(null)),
                "permutation_support_size": 24 if exact_task4 else "",
                "pvalue_method": "exact_full_4_factorial" if exact_task4 else "monte_carlo_plus_one",
                "minimum_attainable_p": (1.0 / 24.0) if exact_task4 else (1.0 / (len(null) + 1.0) if len(null) else float("nan")),
                "upper_tail_p": pvalue,
                "null_mean": float(np.mean(null)) if len(null) else float("nan"),
                "null_std": float(np.std(null)) if len(null) else float("nan"),
                "null_q05": float(np.quantile(null, 0.05)) if len(null) else float("nan"),
                "null_q50": float(np.quantile(null, 0.50)) if len(null) else float("nan"),
                "null_q95": float(np.quantile(null, 0.95)) if len(null) else float("nan"),
            })
            selected_rows.append({
                **base, "parameter_type": "ridge_lambda", "scenario": scenario,
                "arm": arm, "selected_value": lam,
            })
            full_pred = np.full(len(yte), -1, dtype=np.int64); full_pred[test_scenario_mask] = pte
            prediction_payload[f"pred__{source}__rep{random_rep}__{scenario}__{arm}"] = full_pred
    cov_rows, spectrum_rows = covariance_diagnostic_rows(stats, base)
    alpha1_q = qda_relative_scores(Xtr, stats, fit_qda_model(stats, 1.0, args.cov_interpolation))
    return {
        "base": base, "outer_rows": outer_rows, "perm_rows": perm_rows,
        "selected_rows": selected_rows,
        "alpha_rows": [{**base, **r} for r in selection.alpha_rows],
        "ridge_rows": [{**base, **r} for r in selection.ridge_rows],
        "cov_rows": cov_rows, "spectrum_rows": spectrum_rows,
        "predictions": prediction_payload, "nulls": null_payload,
        "summary": {
            **base, "status": "ok", "train_class_counts": train_counts.tolist(),
            "test_class_counts": test_counts.tolist(), "heldout_has_all_five_states": True,
            "feature_dimensions": feature_dimensions(4),
            "selected_ridge": selection.ridge_by_scenario_arm,
            "degenerate_features": builder.degenerate_report(),
            "alpha1_max_abs_q_minus_ell_train": float(np.max(np.abs(alpha1_q - ell_tr))),
            "alpha_selection_diagnostics": alpha_selection_diagnostics(selection),
        },
    }


def c_fold_fingerprint(data: HandoffData, args: argparse.Namespace) -> str:
    payload = {
        "script_version": SCRIPT_VERSION,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_B_fingerprint": str(data.source_fingerprint),
        "alpha_grid": [float(x) for x in args.alpha_grid_values],
        "ridge_grid": [float(x) for x in args.ridge_grid_values],
        "cov_interpolation": str(args.cov_interpolation),
        "axis_sources": [str(x) for x in args.axis_source_values],
        "n_random_subspaces": int(args.n_random_subspaces),
        "degenerate_tol": float(args.degenerate_tol),
        "max_candidate_rank": int(args.max_candidate_rank),
        "n_permutations": int(args.n_permutations),
        "require_a_alignment": bool(args.require_a_alignment),
        "a_root": str(args.a_root.resolve()) if args.a_root is not None else None,
        "seed": int(args.seed),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def run_fold(
    data: HandoffData,
    args: argparse.Namespace,
    fold_dir: Path,
    run_fingerprint: str,
) -> str:
    requested_sources = [
        x for x in args.axis_source_values
        if x in {"slow_prefix", "fast_suffix", "pca_prefix"}
    ]
    if "slow_prefix" not in requested_sources:
        requested_sources = ["slow_prefix"] + requested_sources
    rank_info = choose_analysis_rank(data, args.max_candidate_rank, requested_sources)
    rank = int(rank_info["analysis_rank"])
    source_ranks = {
        str(k): int(v) for k, v in rank_info.get("source_ranks", {}).items()
    }
    if rank <= 0:
        json_dump(fold_dir / "SKIPPED.json", {
            "script_version": SCRIPT_VERSION, "model": data.model,
            "subject_id": data.subject, "grid": data.grid,
            "heldout_session": data.heldout_session, "M_requested": data.svd_rank_requested,
            "status": "skipped_no_feasible_block_complete_slow_prefix",
            "rank_policy": rank_info, "source_B_manifest": str(data.manifest_path),
            "source_B_fingerprint": data.source_fingerprint,
            "c_run_fingerprint": run_fingerprint,
        })
        log(f"[SKIP] {data.model} sub-{data.subject:03d} {data.grid} heldout={data.heldout_session} M={data.svd_rank_requested}: {rank_info['status']}")
        return "skipped"

    results: List[Dict[str, Any]] = []
    for source in args.axis_source_values:
        if source == "random":
            for rep in range(args.n_random_subspaces):
                rng = np.random.default_rng(
                    args.seed + 9100000 * data.subject + 100000 * data.heldout_session
                    + 1000 * data.svd_rank_requested + rep
                )
                X = axis_coordinates(data, "random", rank, rng)
                err = qda_feasibility_error(X, data)
                if err is not None:
                    log(f"[RANDOM SKIP] rep={rep}: {err}"); continue
                results.append(evaluate_axis_space(data, args, "random", X, rank, rank_info, rep))
        elif source == "ambient_svd":
            X = axis_coordinates(data, source, data.svd_rank_effective)
            err = qda_feasibility_error(X, data)
            if err is not None:
                log(f"[AMBIENT UNAVAILABLE] {data.model} M={data.svd_rank_requested}: {err}")
                continue
            results.append(evaluate_axis_space(
                data, args, source, X, data.svd_rank_effective,
                {"analysis_rank": data.svd_rank_effective, "status": "full_ambient",
                 "covers_fixed_topk": True, "covers_r99_auxiliary": True}, -1
            ))
        else:
            source_rank = int(source_ranks.get(source, 0))
            if source_rank <= 0:
                log(
                    f"[SOURCE UNAVAILABLE] {data.model} sub-{data.subject:03d} "
                    f"heldout={data.heldout_session} M={data.svd_rank_requested} "
                    f"source={source}: {rank_info.get('source_status', {}).get(source, 'no_rank')}"
                )
                continue
            X = axis_coordinates(data, source, source_rank)
            results.append(
                evaluate_axis_space(
                    data, args, source, X, source_rank, rank_info, -1
                )
            )
    if not any(r["base"]["axis_source"] == "slow_prefix" for r in results):
        raise RuntimeError("slow_prefix result was not produced")

    outer_rows = [x for r in results for x in r["outer_rows"]]
    perm_rows = [x for r in results for x in r["perm_rows"]]
    selected_rows = [x for r in results for x in r["selected_rows"]]
    alpha_rows = [x for r in results for x in r["alpha_rows"]]
    ridge_rows = [x for r in results for x in r["ridge_rows"]]
    cov_rows = [x for r in results for x in r["cov_rows"]]
    spectrum_rows = [x for r in results for x in r["spectrum_rows"]]
    prediction_payload: Dict[str, np.ndarray] = {
        "global_index": data.global_index[data.test_mask],
        "session_id": data.session[data.test_mask], "run_id": data.run[data.test_mask],
        "sample_start": data.sample_start[data.test_mask], "sample_end": data.sample_end[data.test_mask],
        "phase_id": data.phase[data.test_mask], "task_id": data.task[data.test_mask],
        "state_label": data.state[data.test_mask],
    }
    null_payload: Dict[str, np.ndarray] = {}
    for r in results:
        prediction_payload.update(r["predictions"]); null_payload.update(r["nulls"])

    alignment: Dict[str, Any] = {"status": "not_requested"}
    if args.a_root is not None:
        adir = a_fold_path(args.a_root, data)
        if (adir / "SKIPPED.json").exists():
            alignment = {"status": "A_skipped", "a_fold_dir": str(adir)}
        else:
            alignment = compare_a_ledger(adir, data)
        if args.require_a_alignment and alignment.get("status") != "pass":
            raise RuntimeError(f"A/B held-out ledger alignment failed: {alignment}")
    external_retention = external_a_retention_rows(outer_rows, args.a_root, data, alignment)
    internal_retention = internal_retention_rows(outer_rows, data)

    heldout_pos = np.flatnonzero(data.test_mask)
    ledger_rows = [{
        "global_index": int(data.global_index[p]), "model": data.model,
        "subject_id": data.subject, "grid": data.grid,
        "heldout_session": data.heldout_session, "session_id": int(data.session[p]),
        "run_id": int(data.run[p]), "sample_start": int(data.sample_start[p]),
        "sample_end": int(data.sample_end[p]), "center_sample": int(data.center_sample[p]),
        "phase_id": int(data.phase[p]), "task_id": int(data.task[p]),
        "task_family_id": int(data.family[p]), "state_label": int(data.state[p]),
        "state_name": STATE_NAMES[int(data.state[p])], "heldout_row_index": i,
    } for i, p in enumerate(heldout_pos)]
    csv_write(fold_dir / "outer_metrics.csv", outer_rows)
    csv_write(fold_dir / "heldout_permutation_summary.csv", perm_rows)
    csv_write(fold_dir / "selected_hyperparameters.csv", selected_rows)
    csv_write(fold_dir / "alpha_selection_candidates.csv", alpha_rows)
    csv_write(fold_dir / "ridge_selection_candidates.csv", ridge_rows)
    csv_write(fold_dir / "covariance_diagnostics.csv", cov_rows)
    csv_write(fold_dir / "generalized_lambda_spectra.csv", spectrum_rows)
    csv_write(fold_dir / "heldout_ledger.csv", ledger_rows)
    csv_write(fold_dir / "A_to_C_semantic_retention.csv", external_retention)
    csv_write(fold_dir / "internal_semantic_retention.csv", internal_retention)
    atomic_npz(fold_dir / "heldout_predictions.npz", **prediction_payload)
    atomic_npz(fold_dir / "heldout_null_distributions.npz", **null_payload)
    json_dump(fold_dir / "ABC_fold_alignment.json", alignment)
    summary = {
        "script_version": SCRIPT_VERSION, "status": "ok", "model": data.model,
        "subject_id": data.subject, "grid": data.grid,
        "heldout_session": data.heldout_session, "training_sessions": data.training_sessions,
        "M_requested": data.svd_rank_requested, "M_effective": data.svd_rank_effective,
        "B_retained_rank_r99_auxiliary": data.retained_rank_r99,
        "B_handoff_rank_fixed_topk_requested": data.handoff_rank_fixed_topk_requested,
        "B_candidate_rank_fixed_topk_block_complete": data.candidate_rank_b,
        "B_handoff_selection_policy": data.handoff_selection_policy,
        "C_analysis_rank": rank, "rank_ratio": rank / data.svd_rank_effective,
        "C_rank_policy_status": rank_info["status"],
        "C_rank_covers_B_fixed_topk": rank_info.get("covers_fixed_topk", False),
        "C_rank_covers_B_r99_auxiliary": rank_info.get("covers_r99_auxiliary", False),
        "rank_policy": rank_info, "axis_sources_completed": [r["base"]["axis_source"] for r in results],
        "n_random_subspaces_completed": sum(r["base"]["axis_source"] == "random" for r in results),
        "A_B_ledger_alignment": alignment, "source_B_manifest": str(data.manifest_path),
        "source_B_fingerprint": data.source_fingerprint,
        "space_summaries": [r["summary"] for r in results],
        "notes": [
            "B is frozen before C and is never refitted or selected with C labels.",
            "Slow and PCA are exactly rank-matched. The fast suffix is completed independently at the nearest valid spectral-block boundary, and any rank mismatch is reported explicitly.",
            "The slow-prefix rank is pre-registered by fixed k and never selected by r99 or C labels; inability to match the fast boundary no longer invalidates the primary slow-space fold.",
            "The slow-prefix result is coordinate-free within its selected subspace; slowness specificity is tested by comparisons with fast and PCA subspaces.",
            "A is an external endpoint audit only. Internal ambient/slow/control comparisons are constructed from the same B ledger and reducer.",
        ],
    }
    json_dump(fold_dir / "fold_summary.json", summary)
    json_dump(fold_dir / "DONE.json", {
        "script_version": SCRIPT_VERSION, "model": data.model, "subject_id": data.subject,
        "grid": data.grid, "heldout_session": data.heldout_session,
        "M_requested": data.svd_rank_requested, "source_B_fingerprint": data.source_fingerprint,
        "c_run_fingerprint": run_fingerprint, "finished_at": now(),
    })
    log(f"C complete {data.model} sub-{data.subject:03d} {data.grid} heldout={data.heldout_session} M={data.svd_rank_requested}; r={rank}")
    return "completed"


# =============================================================================
# Discovery and aggregation
# =============================================================================


def discover_manifests(
    b_root: Path, models: Sequence[str], subjects: Sequence[int], grids: Sequence[str],
    heldout_sessions: Sequence[int], svd_ranks: Sequence[int],
) -> List[Path]:
    paths = sorted(Path(b_root).glob("*/sub-*/grid-*/heldout-session-*/svd-rank-*/B_to_C_manifest.json"))
    out: List[Path] = []
    model_set, subject_set = set(models), set(int(x) for x in subjects)
    grid_set, heldout_set, rank_set = set(grids), set(heldout_sessions), set(svd_ranks)
    for path in paths:
        try:
            with path.open("r", encoding="utf-8") as fh:
                manifest = json.load(fh)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Unreadable B manifest {path}: {exc}") from exc
        if str(manifest.get("schema_version", "")) != EXPECTED_B_SCHEMA:
            raise ValueError(f"Unsupported B schema in {path}: {manifest.get('schema_version')}")
        model = str(manifest.get("model", "")); subject = int(manifest.get("subject_id", -1))
        grid = str(manifest.get("grid", "")); heldout = int(manifest.get("heldout_session", -1))
        rank = int(manifest.get("svd_rank_requested", -1))
        if model_set and model not in model_set: continue
        if subject_set and subject not in subject_set: continue
        if grid_set and grid not in grid_set: continue
        if heldout_set and heldout not in heldout_set: continue
        if rank_set and rank not in rank_set: continue
        out.append(path)
    return out


def aggregate_outputs(root: Path) -> None:
    table_names = [
        "outer_metrics.csv", "heldout_permutation_summary.csv",
        "selected_hyperparameters.csv", "alpha_selection_candidates.csv",
        "ridge_selection_candidates.csv", "covariance_diagnostics.csv",
        "generalized_lambda_spectra.csv", "A_to_C_semantic_retention.csv",
        "internal_semantic_retention.csv",
    ]
    tables = ensure_dir(root / "tables")
    aggregated: Dict[str, List[Dict[str, str]]] = {}
    for name in table_names:
        rows: List[Dict[str, str]] = []
        for path in sorted((root / "folds").glob(f"*/*/*/*/*/{name}")):
            rows.extend(csv_read(path))
        aggregated[name] = rows; csv_write(tables / name, rows)
    metrics = [r for r in aggregated["outer_metrics.csv"] if r.get("arm") in ARMS]
    groups: Dict[Tuple[str, ...], List[Dict[str, str]]] = {}
    for r in metrics:
        key = (
            r.get("model", ""), r.get("M_requested", ""), r.get("axis_source", ""),
            r.get("axis_rank", ""), r.get("rank_ratio", ""), r.get("scenario", ""),
            r.get("arm", ""), r.get("C_rank_policy_status", ""),
        )
        groups.setdefault(key, []).append(r)
    summary: List[Dict[str, Any]] = []
    for key, rows in groups.items():
        pvals = [as_float(r.get("heldout_permutation_p")) for r in rows]
        summary.append({
            "model": key[0], "M_requested": key[1], "axis_source": key[2],
            "axis_rank": key[3], "rank_ratio": key[4], "scenario": key[5],
            "arm": key[6], "C_rank_policy_status": key[7], "n_folds": len(rows),
            "test_balanced_accuracy_mean": finite_mean(as_float(r.get("test_balanced_accuracy")) for r in rows),
            "test_balanced_accuracy_sem": finite_sem(as_float(r.get("test_balanced_accuracy")) for r in rows),
            "heldout_permutation_p_median": float(np.nanmedian(pvals)) if np.any(np.isfinite(pvals)) else float("nan"),
        })
    csv_write(tables / "summary_by_model_rank_source_scenario_arm.csv", summary)
    headline = [
        r for r in summary
        if r.get("axis_source") == "slow_prefix"
        and as_float(r.get("rank_ratio")) < 1.0 - 1e-12
    ]
    csv_write(tables / "headline_nontrivial_slow_prefix.csv", headline)

    retention = aggregated["internal_semantic_retention.csv"]
    ret_groups: Dict[Tuple[str, ...], List[Dict[str, str]]] = {}
    for r in retention:
        if r.get("status") != "ok":
            continue
        key = (r.get("model", ""), r.get("M_requested", ""), r.get("axis_source", ""),
               r.get("axis_rank", ""), r.get("scenario", ""), r.get("arm", ""))
        ret_groups.setdefault(key, []).append(r)
    ret_summary: List[Dict[str, Any]] = []
    for key, rows in ret_groups.items():
        source_mean = finite_mean(as_float(r.get("source_test_balanced_accuracy")) for r in rows)
        ambient_mean = finite_mean(as_float(r.get("ambient_test_balanced_accuracy")) for r in rows)
        chance = as_float(rows[0].get("chance"))
        denom = ambient_mean - chance
        ret_summary.append({
            "model": key[0], "M_requested": key[1], "axis_source": key[2],
            "axis_rank": key[3], "scenario": key[4], "arm": key[5], "n_folds": len(rows),
            "source_test_bacc_mean": source_mean, "ambient_test_bacc_mean": ambient_mean,
            "source_minus_ambient_delta_mean": finite_mean(
                as_float(r.get("source_minus_ambient_delta")) for r in rows
            ),
            "aggregate_chance_normalized_retention": (source_mean - chance) / denom
            if np.isfinite(denom) and abs(denom) > 1e-12 else float("nan"),
        })
    csv_write(tables / "internal_retention_summary.csv", ret_summary)


def self_test(seed: int = 0) -> Dict[str, Any]:
    core = core_self_test(seed); rng = np.random.default_rng(seed + 11)
    sessions=[]; runs=[]; phases=[]; tasks=[]; states=[]; starts=[]; cursor=0
    for session in (1,2,3):
        for j, task in enumerate((0,1,5,6)):
            for t in range(70):
                states.append(0 if t < 25 else j+1); sessions.append(session); runs.append(10*session+j)
                phases.append(0 if t < 25 else 1); tasks.append(task); starts.append(cursor); cursor += 1
    y=np.asarray(states,dtype=np.int64); n=len(y); M=8
    ambient=rng.standard_normal((n,M))
    for c in range(5):
        mask=y==c; ambient[mask,c%M]+=0.4*c; ambient[mask,(c+1)%M]*=0.8+0.08*c
    q,_=np.linalg.qr(rng.standard_normal((M,M))); gammas=np.arange(M,dtype=float)
    mean=np.zeros(M); slow=(ambient-mean)@q[:,:6]
    train=np.asarray(sessions)!=3; test=~train
    manifest={
        "schema_version":EXPECTED_B_SCHEMA,"model":"synthetic","subject_id":1,"grid":"raw",
        "heldout_session":3,"training_sessions":[1,2],"svd_rank_requested":M,
        "svd_rank_effective":M,
        "candidate_rank_fixed_topk_block_complete":6,
        "candidate_rank_block_complete":6,
        "handoff_rank_fixed_topk_requested":5,
        "handoff_selection_policy":"pre_registered_fixed_topk_then_bidirectional_nonchaining_block_completion",
        "retained_rank_r99":4,
        "all_degenerate_blocks_zero_based":[[0],[1],[2,3],[4],[5]],
    }
    data=HandoffData(
        Path("manifest"),Path("coords"),Path("transform"),manifest,"synthetic",1,"raw",3,[1,2],
        M,M,6,5,"pre_registered_fixed_topk_then_bidirectional_nonchaining_block_completion",4,
        slow,ambient,mean,q,gammas,np.arange(n),np.ones(n,dtype=int),
        np.asarray(sessions),np.asarray(runs),np.asarray(phases),np.asarray(tasks),np.zeros(n,dtype=int),y,
        np.asarray(starts),np.asarray(starts)+1,np.asarray(starts),train,test,"synthetic-fingerprint",
    )
    rank_info=choose_analysis_rank(data,0,["slow_prefix","fast_suffix","pca_prefix"])
    if rank_info["analysis_rank"] <= 0: raise AssertionError(rank_info)
    for source in ("slow_prefix","fast_suffix","pca_prefix"):
        X=axis_coordinates(data,source,rank_info["analysis_rank"]);
        if qda_feasibility_error(X,data) is not None: raise AssertionError(source)
    selection=nested_select(axis_coordinates(data,"slow_prefix",rank_info["analysis_rank"]),y,np.asarray(sessions),[1,2],[0.5,1.0],[0.01,0.1],"arithmetic",1e-10)
    null=structured_null_bacc(data,"five_state",np.zeros(int(np.sum(test)),dtype=int),3,seed)
    if len(null)!=3 or not np.all(np.isfinite(null)): raise AssertionError("structured null")
    task4_exact = structured_null_bacc(
        data, "task4", np.zeros(int(np.sum(test & (y > 0))), dtype=int), 3, seed
    )
    if len(task4_exact) != 24:
        raise AssertionError("task4 exact null did not enumerate 4!=24 assignments")
    return {"core":core,"rank_policy":rank_info,"selected_alpha":selection.alpha,"structured_null":"passed","task4_exact_support":24}


def build_parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(
        description="Experiment C v5.0: semantic readout with rank-matched slow/fast/PCA controls",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--b-root",type=Path,required=False); p.add_argument("--a-root",type=Path,default=None)
    p.add_argument("--models",default=""); p.add_argument("--subjects",default="")
    p.add_argument("--grids",default="raw"); p.add_argument("--heldout-sessions",default="")
    p.add_argument("--svd-ranks",default="")
    p.add_argument("--axis-sources",default="slow_prefix,fast_suffix,pca_prefix,ambient_svd")
    p.add_argument("--n-random-subspaces",type=int,default=0)
    p.add_argument("--alpha-grid",default="0.001,0.01,0.05,0.1,0.25,0.5,0.9,1")
    p.add_argument("--ridge-grid",default="0.0001,0.001,0.01,0.1,1,10,100")
    p.add_argument("--cov-interpolation",choices=["geometric","arithmetic"],default="arithmetic")
    p.add_argument("--degenerate-tol",type=float,default=1e-10)
    p.add_argument("--max-candidate-rank",type=int,default=0)
    p.add_argument("--n-permutations",type=int,default=200)
    p.add_argument("--require-a-alignment",action="store_true")
    p.add_argument("--seed",type=int,default=0); p.add_argument("--resume",action="store_true")
    p.add_argument("--require-at-least-one-completed",action="store_true")
    p.add_argument("--fail-fast",action="store_true"); p.add_argument("--self-test",action="store_true")
    p.add_argument("--outdir",type=Path,default=Path("/mnt/dataset4/yinuo/FM_flow/dataset/expC_slowspace_ld_qd_tensor_v5_1"))
    return p


def resume_matches(path: Path, source_fingerprint: str, run_fingerprint: str) -> bool:
    try:
        with path.open("r",encoding="utf-8") as fh: payload=json.load(fh)
        return (
            str(payload.get("source_B_fingerprint","")) == str(source_fingerprint)
            and str(payload.get("c_run_fingerprint","")) == str(run_fingerprint)
            and str(payload.get("script_version","")) == SCRIPT_VERSION
        )
    except Exception:
        return False


def main() -> None:
    args=build_parser().parse_args()
    if args.self_test:
        print(json.dumps(self_test(args.seed),indent=2,ensure_ascii=False)); return
    if args.b_root is None: raise ValueError("--b-root is required unless --self-test is used")
    args.alpha_grid_values=sorted(set(parse_float_csv(args.alpha_grid)))
    args.ridge_grid_values=sorted(set(parse_float_csv(args.ridge_grid)))
    args.axis_source_values=parse_str_list(args.axis_sources)
    allowed={"slow_prefix","fast_suffix","pca_prefix","ambient_svd","random"}
    unknown=sorted(set(args.axis_source_values)-allowed)
    if unknown: raise ValueError(f"Unknown --axis-sources: {unknown}")
    if "random" in args.axis_source_values and args.n_random_subspaces <= 0:
        raise ValueError("axis source random requires --n-random-subspaces > 0")
    if args.n_random_subspaces < 0 or args.n_permutations < 0 or args.max_candidate_rank < 0:
        raise ValueError("counts and rank cap must be non-negative")
    if not args.alpha_grid_values or not args.ridge_grid_values:
        raise ValueError("alpha/ridge grid cannot be empty")
    if any((a < 0.0 or a > 1.0) for a in args.alpha_grid_values):
        raise ValueError("--alpha-grid values must lie in [0,1]")
    if args.cov_interpolation == "arithmetic" and any(a <= 0.0 for a in args.alpha_grid_values):
        raise ValueError(
            "Arithmetic pooled-covariance shrinkage requires alpha > 0; "
            "remove alpha=0 from --alpha-grid"
        )
    if args.a_root is not None and not args.a_root.exists():
        raise FileNotFoundError(f"A root does not exist: {args.a_root}")
    ensure_dir(args.outdir); ensure_dir(args.outdir/"folds")
    manifests=discover_manifests(
        args.b_root,parse_str_list(args.models),parse_int_csv(args.subjects),
        parse_str_list(args.grids),parse_int_csv(args.heldout_sessions),parse_int_csv(args.svd_ranks),
    )
    if not manifests: raise RuntimeError(f"No matching B-to-C manifests under {args.b_root}")
    log(f"C discovered {len(manifests)} B handoff fold(s)")
    json_dump(args.outdir/"arguments.json",{
        "script_version":SCRIPT_VERSION,"arguments":vars(args),"python":sys.version,
        "platform":platform.platform(),"numpy":np.__version__,"started_at":now(),
        "n_discovered_handoffs":len(manifests),
    })
    failures=[]; completed_this_run=skipped_this_run=resumed_done=resumed_skipped=0
    for manifest_path in manifests:
        try:
            data=load_handoff(manifest_path)
            fold_dir=ensure_dir(
                args.outdir/"folds"/data.model/f"sub-{data.subject:03d}"/f"grid-{data.grid}"
                /f"heldout-session-{data.heldout_session}"/f"svd-rank-{data.svd_rank_requested}"
            )
            done_marker=fold_dir/"DONE.json"; skipped_marker=fold_dir/"SKIPPED.json"
            run_fingerprint = c_fold_fingerprint(data, args)
            if args.resume and done_marker.exists() and resume_matches(done_marker,data.source_fingerprint,run_fingerprint):
                resumed_done+=1; log(f"[RESUME DONE] {data.model} sub-{data.subject:03d} {data.grid} heldout={data.heldout_session} M={data.svd_rank_requested}"); continue
            if args.resume and skipped_marker.exists() and resume_matches(skipped_marker,data.source_fingerprint,run_fingerprint):
                resumed_skipped+=1; log(f"[RESUME SKIPPED] {data.model} sub-{data.subject:03d} {data.grid} heldout={data.heldout_session} M={data.svd_rank_requested}"); continue
            # Recompute after a scientific-version change and prevent stale
            # DONE/SKIPPED markers from coexisting with the new result.
            for marker in (done_marker, skipped_marker):
                if marker.exists(): marker.unlink()
            status=run_fold(data,args,fold_dir,run_fingerprint)
            if status=="completed": completed_this_run+=1
            elif status=="skipped": skipped_this_run+=1
            else: raise RuntimeError(f"Unknown C fold status {status!r}")
            gc.collect()
        except Exception as exc:  # noqa: BLE001
            failure={"manifest":str(manifest_path),"error":repr(exc),"traceback":traceback.format_exc()}
            failures.append(failure); log(f"C FAILED {manifest_path}: {exc!r}")
            if args.fail_fast: raise
    aggregate_outputs(args.outdir)
    done_total=len(list((args.outdir/"folds").glob("*/*/*/*/*/DONE.json")))
    skipped_total=len(list((args.outdir/"folds").glob("*/*/*/*/*/SKIPPED.json")))
    json_dump(args.outdir/"summary.json",{
        "script_version":SCRIPT_VERSION,"finished_at":now(),"n_discovered":len(manifests),
        "n_completed_this_run":completed_this_run,"n_skipped_this_run":skipped_this_run,
        "n_resumed_done":resumed_done,"n_resumed_skipped":resumed_skipped,
        "n_done_total":done_total,"n_skipped_total":skipped_total,"failures":failures,
    })
    log(f"C done; completed={completed_this_run}, skipped={skipped_this_run}, done_total={done_total}, skipped_total={skipped_total}, failures={len(failures)}")
    if failures: raise SystemExit(1)
    if args.require_at_least_one_completed and done_total==0: raise SystemExit(2)


if __name__=="__main__":
    main()
