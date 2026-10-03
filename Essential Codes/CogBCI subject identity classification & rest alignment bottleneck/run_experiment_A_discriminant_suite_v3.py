#!/usr/bin/env python3
"""
Experiment A discriminant suite for EEG FM embeddings.

This is a refactored analysis suite for global task-state separability in frozen EEG
foundation-model embeddings.

Main state probe
----------------
Default CogBCI setting:
  class label = task_id
  classes     = 0,1,2 = Resting, N-Back, MATB

For each subject-heldout fold:
  1) fit train-only mean and train-only randomized SVD;
  2) fit train-only multiclass LDA/LD in the SVD score space;
  3) classify train samples in the train-fitted LD space for in-sample diagnosis;
  4) project held-out subjects into the same train-fitted space;
  5) classify by nearest train class centroid;
  6) save fold-level, subject-level, class-level, centroid, point-level, and plot outputs.

Subject identity control
------------------------
Closed-set subject probe:
  target = subject_id
  split  = within each subject and task class, so every subject appears in train/test
  goal   = test whether the embedding carries a strong subject axis.

Backbone parameters are never updated. Everything is a linear readout fitted on
train rows only.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def parse_int_list(s: str | None) -> List[int]:
    if s is None or str(s).strip() == "":
        return []
    return [int(x.strip()) for x in str(s).split(",") if x.strip() != ""]


def parse_subject_groups(s: str | None) -> List[List[int]]:
    """Parse manual groups like '1,2,3;4,5,6;7,8'."""
    if s is None or str(s).strip() == "":
        return []
    out: List[List[int]] = []
    for part in str(s).split(";"):
        vals = parse_int_list(part)
        if vals:
            out.append(sorted(vals))
    return out


def decode_attr(x):
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    if isinstance(x, np.generic):
        return x.item()
    return x


def read_h5_vector(f: h5py.File, key: str) -> np.ndarray:
    if key not in f:
        raise KeyError(f"Key not found in H5: {key}")
    arr = np.asarray(f[key][()])
    if arr.dtype.kind == "S":
        arr = np.array([v.decode("utf-8", errors="replace") for v in arr])
    return arr


def get_flat_dim(ds: h5py.Dataset) -> int:
    if len(ds.shape) < 2:
        raise ValueError(f"Embedding dataset must have shape (N, ...), got {ds.shape}")
    return int(np.prod(ds.shape[1:]))


def read_embedding_rows(
    ds: h5py.Dataset,
    indices: np.ndarray,
    batch_size: int = 4096,
    dtype=np.float32,
) -> np.ndarray:
    """Read selected H5 rows and flatten to (n, d).

    h5py fancy indexing is happiest with sorted integer indices. We rely on
    np.flatnonzero / original-order selected indices, which are sorted.
    """
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError("indices must be 1D")
    if len(indices) == 0:
        raise ValueError("cannot read zero rows")

    n = len(indices)
    d = get_flat_dim(ds)
    out = np.empty((n, d), dtype=dtype)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        idx = indices[start:end]
        chunk = ds[idx]
        out[start:end] = np.asarray(chunk).reshape(len(idx), d).astype(dtype, copy=False)
    return out


def safe_json(x) -> str:
    return json.dumps(x, ensure_ascii=False)


def write_csv(path: str | Path, rows: List[Dict]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        print(f"[{timestamp()}] [WARN] no rows for {path}", flush=True)
        return
    fields: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                fields.append(k)
                seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[{timestamp()}] Saved: {path}", flush=True)


def nanmean(x: Iterable[float]) -> float:
    arr = np.asarray(list(x), dtype=np.float64)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def nanstd(x: Iterable[float], ddof: int = 1) -> float:
    arr = np.asarray(list(x), dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size <= ddof:
        return 0.0
    return float(np.std(arr, ddof=ddof))


def sem(x: Iterable[float]) -> float:
    arr = np.asarray(list(x), dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / math.sqrt(arr.size))


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, classes: Sequence[int]) -> Tuple[float, Dict[int, float]]:
    recalls: Dict[int, float] = {}
    for c in classes:
        c = int(c)
        mask = (y_true == c)
        if int(np.sum(mask)) == 0:
            recalls[c] = float("nan")
        else:
            recalls[c] = float(np.mean(y_pred[mask] == c))
    vals = [v for v in recalls.values() if not math.isnan(v)]
    return (float(np.mean(vals)) if vals else float("nan")), recalls


def confusion_matrix_fixed(y_true: np.ndarray, y_pred: np.ndarray, classes: Sequence[int]) -> np.ndarray:
    c2i = {int(c): i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for yt, yp in zip(y_true, y_pred):
        yt_i = c2i.get(int(yt))
        yp_i = c2i.get(int(yp))
        if yt_i is not None and yp_i is not None:
            cm[yt_i, yp_i] += 1
    return cm


def binary_auc_rank(y_binary: np.ndarray, scores: np.ndarray) -> float:
    """Compute binary ROC-AUC from ranks.

    y_binary: bool / 0-1 array, True means positive.
    scores: higher score means more positive.
    Returns NaN if either class is absent.
    """
    y_binary = np.asarray(y_binary).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int(np.sum(y_binary))
    n_neg = int(len(y_binary) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)

    # Average ranks for ties. Ranks are 1-based for the Mann-Whitney formula.
    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = avg_rank
        i = j

    sum_pos_ranks = float(np.sum(ranks[y_binary]))
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def macro_auc_ovr_from_distances(y_true: np.ndarray, dist: np.ndarray, classes: Sequence[int]) -> Tuple[float, Dict[int, float]]:
    """One-vs-rest macro AUC using nearest-centroid distances.

    For class c, score_c = -distance_to_centroid_c. AUC only depends on ranking,
    so a softmax over distances would give the same ordering in most cases and is
    not necessary.
    """
    y_true = np.asarray(y_true)
    dist = np.asarray(dist, dtype=np.float64)
    aucs: Dict[int, float] = {}
    for j, c in enumerate(classes):
        c = int(c)
        aucs[c] = binary_auc_rank(y_true == c, -dist[:, j])
    vals = [v for v in aucs.values() if not math.isnan(v)]
    return (float(np.mean(vals)) if vals else float("nan")), aucs


def subject_center_ld(
    Z: np.ndarray,
    subject: np.ndarray,
    train_reference_mean: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[int, List[float]]]:
    """Unsupervised per-subject translation alignment in LD space.

    For each held-out subject, subtract its global LD mean and add the train
    reference mean. This uses no task labels. In the state probe, the train LD
    global mean is usually close to zero because train embeddings were centered
    before SVD, but we keep it explicit.
    """
    Z = np.asarray(Z)
    subject = np.asarray(subject)
    if train_reference_mean is None:
        train_reference_mean = np.zeros(Z.shape[1], dtype=Z.dtype)
    train_reference_mean = np.asarray(train_reference_mean, dtype=Z.dtype)

    Zc = np.array(Z, copy=True)
    offsets: Dict[int, List[float]] = {}
    for s in sorted(np.unique(subject).astype(int)):
        mask = (subject == s)
        if not np.any(mask):
            continue
        subj_mean = Z[mask].mean(axis=0)
        offset = subj_mean - train_reference_mean
        Zc[mask] = Z[mask] - offset
        offsets[int(s)] = [float(x) for x in offset]
    return Zc, offsets


def per_class_counts(y: np.ndarray, classes: Sequence[int]) -> Dict[int, int]:
    return {int(c): int(np.sum(y == int(c))) for c in classes}


# -----------------------------------------------------------------------------
# Mechanism diagnostics: margins, boundary projections, Procrustes, self-readout
# -----------------------------------------------------------------------------


def signed_margins_from_distances(y_true: np.ndarray, dist: np.ndarray, classes: Sequence[int]) -> np.ndarray:
    """Nearest-centroid signed margin.

    margin = nearest_wrong_distance - true_class_distance.
    Positive margin means the true class centroid is closer than every wrong centroid.
    """
    y_true = np.asarray(y_true)
    dist = np.asarray(dist, dtype=np.float64)
    c2i = {int(c): i for i, c in enumerate(classes)}
    margins = np.full(len(y_true), np.nan, dtype=np.float64)
    for i, yt in enumerate(y_true):
        j = c2i.get(int(yt))
        if j is None:
            continue
        true_d = dist[i, j]
        wrong = np.delete(dist[i], j)
        margins[i] = float(np.min(wrong) - true_d)
    return margins


def summarize_margins(margins: np.ndarray, prefix: str) -> Dict[str, float]:
    margins = np.asarray(margins, dtype=np.float64)
    ok = margins[np.isfinite(margins)]
    if ok.size == 0:
        return {
            f"{prefix}_mean_margin": float("nan"),
            f"{prefix}_median_margin": float("nan"),
            f"{prefix}_q25_margin": float("nan"),
            f"{prefix}_q75_margin": float("nan"),
            f"{prefix}_frac_positive_margin": float("nan"),
        }
    return {
        f"{prefix}_mean_margin": float(np.mean(ok)),
        f"{prefix}_median_margin": float(np.median(ok)),
        f"{prefix}_q25_margin": float(np.percentile(ok, 25)),
        f"{prefix}_q75_margin": float(np.percentile(ok, 75)),
        f"{prefix}_frac_positive_margin": float(np.mean(ok > 0)),
    }


def boundary_projection_diagnostic(
    offset: np.ndarray,
    centroids: Dict[int, np.ndarray],
    classes: Sequence[int],
    class_names: Dict[int, str],
) -> Dict[str, float | str]:
    """Project a subject offset onto train class-centroid difference directions.

    This tests whether a large subject offset lies along decision-relevant class
    separation directions, or mostly in their weak/orthogonal complement.
    """
    offset = np.asarray(offset, dtype=np.float64).reshape(-1)
    offset_norm = float(np.linalg.norm(offset))
    row: Dict[str, float | str] = {"offset_norm": offset_norm}
    max_abs_proj = 0.0
    max_pair = ""
    max_proj_over_sep = 0.0
    for ia, ca in enumerate(classes):
        for cb in classes[ia + 1:]:
            ca = int(ca); cb = int(cb)
            va = np.asarray(centroids[ca], dtype=np.float64).reshape(-1)
            vb = np.asarray(centroids[cb], dtype=np.float64).reshape(-1)
            diff = vb - va
            sep = float(np.linalg.norm(diff))
            pair_name = f"{class_names.get(ca, str(ca))}_vs_{class_names.get(cb, str(cb))}".replace(" ", "_")
            if sep <= 1e-12 or offset_norm <= 1e-12:
                signed_proj = 0.0
                abs_proj = 0.0
                ratio = 0.0
                proj_over_sep = 0.0
            else:
                unit = diff / sep
                signed_proj = float(np.dot(offset, unit))
                abs_proj = float(abs(signed_proj))
                ratio = float(abs_proj / offset_norm)
                proj_over_sep = float(abs_proj / sep)
            row[f"signed_projection_{pair_name}"] = signed_proj
            row[f"abs_projection_{pair_name}"] = abs_proj
            row[f"projection_ratio_{pair_name}"] = ratio
            row[f"projection_over_class_distance_{pair_name}"] = proj_over_sep
            row[f"class_distance_{pair_name}"] = sep
            if abs_proj > max_abs_proj:
                max_abs_proj = abs_proj
                max_pair = pair_name
                max_proj_over_sep = proj_over_sep
    row["max_boundary_projection"] = float(max_abs_proj)
    row["max_boundary_projection_pair"] = max_pair
    row["boundary_projection_ratio"] = float(max_abs_proj / offset_norm) if offset_norm > 1e-12 else 0.0
    row["max_projection_over_class_distance"] = float(max_proj_over_sep)
    return row


def fit_similarity_procrustes(
    source: np.ndarray,
    target: np.ndarray,
    weights: Optional[np.ndarray] = None,
    allow_reflection: bool = False,
) -> Tuple[np.ndarray, float, np.ndarray, float]:
    """Fit source @ (scale * R) + b ≈ target for row-vector points.

    Returns R, scale, translation, residual_rms.
    This is an oracle diagnostic when source points are true held-out class centroids.
    """
    X = np.asarray(source, dtype=np.float64)
    Y = np.asarray(target, dtype=np.float64)
    if X.shape != Y.shape:
        raise ValueError(f"source and target must have same shape, got {X.shape} and {Y.shape}")
    n, q = X.shape
    if n < 2:
        R = np.eye(q)
        scale = 1.0
        b = Y.mean(axis=0) - X.mean(axis=0)
        resid = float(np.sqrt(np.mean(np.sum((X + b - Y) ** 2, axis=1))))
        return R, scale, b, resid
    if weights is None:
        w = np.ones(n, dtype=np.float64) / n
    else:
        w = np.asarray(weights, dtype=np.float64)
        if np.sum(w) <= 0 or not np.all(np.isfinite(w)):
            w = np.ones(n, dtype=np.float64)
        w = w / np.sum(w)
    muX = np.sum(X * w[:, None], axis=0)
    muY = np.sum(Y * w[:, None], axis=0)
    X0 = X - muX
    Y0 = Y - muY
    H = X0.T @ (Y0 * w[:, None])
    U, S, Vt = np.linalg.svd(H, full_matrices=False)
    R = U @ Vt
    if not allow_reflection and np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    denom = float(np.sum(w * np.sum(X0 ** 2, axis=1)))
    scale = float(np.sum(S) / denom) if denom > 1e-12 else 1.0
    b = muY - scale * (muX @ R)
    Xhat = scale * (X @ R) + b
    resid = float(np.sqrt(np.sum(w * np.sum((Xhat - Y) ** 2, axis=1))))
    return R, scale, b, resid


def apply_similarity_transform(Z: np.ndarray, R: np.ndarray, scale: float, b: np.ndarray) -> np.ndarray:
    return float(scale) * (np.asarray(Z, dtype=np.float64) @ np.asarray(R, dtype=np.float64)) + np.asarray(b, dtype=np.float64)


def rotation_angle_degrees(R: np.ndarray) -> float:
    R = np.asarray(R, dtype=np.float64)
    if R.shape[0] < 2 or R.shape[1] < 2:
        return float("nan")
    return float(np.degrees(np.arctan2(R[0, 1], R[0, 0])))


def stratified_label_split(y: np.ndarray, test_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return local train/test indices stratified by class label."""
    rng = np.random.default_rng(seed)
    train_parts = []
    test_parts = []
    y = np.asarray(y)
    for c in sorted(np.unique(y).astype(int)):
        idx = np.flatnonzero(y == c)
        if len(idx) <= 2:
            train_parts.append(idx)
            continue
        perm = np.array(idx, copy=True)
        rng.shuffle(perm)
        n_test = int(round(len(perm) * test_fraction))
        n_test = max(1, min(n_test, len(perm) - 1))
        test_parts.append(np.sort(perm[:n_test]))
        train_parts.append(np.sort(perm[n_test:]))
    train_idx = np.sort(np.concatenate(train_parts)) if train_parts else np.array([], dtype=np.int64)
    test_idx = np.sort(np.concatenate(test_parts)) if test_parts else np.array([], dtype=np.int64)
    return train_idx, test_idx


def within_subject_self_readout(
    R_subject: np.ndarray,
    y_subject: np.ndarray,
    classes: Sequence[int],
    ridge: float,
    test_fraction: float,
    seed: int,
) -> Dict[str, float | str | int]:
    """Oracle within-subject upper-bound readout in the fold's train-SVD space."""
    R_subject = np.asarray(R_subject, dtype=np.float64)
    y_subject = np.asarray(y_subject, dtype=np.int64)
    counts = per_class_counts(y_subject, classes)
    if any(v < 3 for v in counts.values()):
        return {
            "self_readout_available": 0,
            "self_readout_reason": "too_few_samples_for_at_least_one_class",
            "self_readout_acc": float("nan"),
            "self_readout_bacc": float("nan"),
            "self_readout_macro_auc": float("nan"),
        }
    tr, te = stratified_label_split(y_subject, test_fraction=test_fraction, seed=seed)
    if len(tr) == 0 or len(te) == 0:
        return {
            "self_readout_available": 0,
            "self_readout_reason": "empty_split",
            "self_readout_acc": float("nan"),
            "self_readout_bacc": float("nan"),
            "self_readout_macro_auc": float("nan"),
        }
    try:
        lda_s = fit_multiclass_lda(R_subject[tr], y_subject[tr], classes=classes, ridge=ridge, max_ld_dims=None)
        Z_te = R_subject[te] @ lda_s.W
        pred, dist = predict_nearest_centroid(Z_te, lda_s.class_centroids_ld, classes)
        acc = float(np.mean(pred == y_subject[te]))
        bacc, recalls = balanced_accuracy(y_subject[te], pred, classes)
        auc, auc_by_class = macro_auc_ovr_from_distances(y_subject[te], dist, classes)
        return {
            "self_readout_available": 1,
            "self_readout_reason": "ok",
            "self_readout_n_train": int(len(tr)),
            "self_readout_n_test": int(len(te)),
            "self_readout_acc": acc,
            "self_readout_bacc": bacc,
            "self_readout_macro_auc": auc,
            "self_readout_class_recalls": safe_json(recalls),
            "self_readout_auc_by_class": safe_json(auc_by_class),
        }
    except Exception as e:
        return {
            "self_readout_available": 0,
            "self_readout_reason": f"error:{type(e).__name__}:{e}",
            "self_readout_acc": float("nan"),
            "self_readout_bacc": float("nan"),
            "self_readout_macro_auc": float("nan"),
        }


# -----------------------------------------------------------------------------
# Randomized SVD and LDA
# -----------------------------------------------------------------------------


def randomized_svd_dense(
    X: np.ndarray,
    n_components: int,
    n_oversamples: int = 20,
    n_iter: int = 2,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Randomized truncated SVD for a dense centered matrix X."""
    n, d = X.shape
    k = int(min(n_components, n, d))
    if k <= 0:
        raise ValueError(f"Invalid n_components={n_components} for X shape={X.shape}")

    if k >= min(n, d) - 1 and min(n, d) <= 512:
        U, S, Vt = np.linalg.svd(X.astype(np.float32, copy=False), full_matrices=False)
        return U[:, :k], S[:k], Vt[:k, :]

    rng = np.random.default_rng(seed)
    l = int(min(k + n_oversamples, d))
    Omega = rng.standard_normal(size=(d, l)).astype(np.float32)
    Y = X @ Omega
    for _ in range(max(0, n_iter)):
        Y = X @ (X.T @ Y)
    Q, _ = np.linalg.qr(Y, mode="reduced")
    Q = Q.astype(np.float32, copy=False)
    B = Q.T @ X
    Ub, S, Vt = np.linalg.svd(B.astype(np.float32, copy=False), full_matrices=False)
    U = Q @ Ub[:, :k]
    return U[:, :k], S[:k], Vt[:k, :]


@dataclass
class LDAResult:
    eigvals: np.ndarray
    W: np.ndarray
    class_centroids_ld: Dict[int, np.ndarray]
    train_ld: np.ndarray


def fit_multiclass_lda(
    R: np.ndarray,
    y: np.ndarray,
    classes: Sequence[int],
    ridge: float = 1e-4,
    max_ld_dims: Optional[int] = None,
) -> LDAResult:
    """Fit multiclass LDA in the SVD score space.

    For C classes, at most C-1 discriminant directions exist. max_ld_dims can
    optionally restrict the returned number of directions.
    """
    R = np.asarray(R, dtype=np.float64)
    y = np.asarray(y)
    n, m = R.shape
    C = len(classes)
    q = min(C - 1, m)
    if max_ld_dims is not None:
        q = min(q, int(max_ld_dims))
    if q <= 0:
        raise ValueError("Need at least two classes and one feature dimension for LDA")

    mu = R.mean(axis=0)
    Sw = np.zeros((m, m), dtype=np.float64)
    Sb = np.zeros((m, m), dtype=np.float64)

    for c in classes:
        Rc = R[y == int(c)]
        if len(Rc) == 0:
            raise ValueError(f"Class {c} has no samples in training fold")
        muc = Rc.mean(axis=0)
        Xc = Rc - muc
        Sw += Xc.T @ Xc
        dm = (muc - mu).reshape(-1, 1)
        Sb += len(Rc) * (dm @ dm.T)

    Sw /= max(n - C, 1)
    Sb /= max(C - 1, 1)
    scale = float(np.trace(Sw) / max(m, 1))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    Sw_reg = Sw + (ridge * scale) * np.eye(m, dtype=np.float64)

    try:
        from scipy.linalg import eigh
        eigvals, eigvecs = eigh(Sb, Sw_reg)
    except Exception:
        A = np.linalg.solve(Sw_reg, Sb)
        eigvals, eigvecs = np.linalg.eig(A)
        eigvals = np.real(eigvals)
        eigvecs = np.real(eigvecs)

    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(np.real(eigvals[order]), 0.0)
    W = np.real(eigvecs[:, order[:q]])

    # Normalize columns for stable plots and distances.
    for j in range(W.shape[1]):
        norm = np.linalg.norm(W[:, j])
        if norm > 0:
            W[:, j] /= norm

    train_ld = R @ W
    centroids = {int(c): train_ld[y == int(c)].mean(axis=0) for c in classes}
    return LDAResult(eigvals=eigvals, W=W, class_centroids_ld=centroids, train_ld=train_ld)


def distances_to_centroids(Z: np.ndarray, centroids: Dict[int, np.ndarray], classes: Sequence[int]) -> np.ndarray:
    Cmat = np.stack([centroids[int(c)] for c in classes], axis=0)
    return np.sum((Z[:, None, :] - Cmat[None, :, :]) ** 2, axis=2)


def predict_nearest_centroid(Z: np.ndarray, centroids: Dict[int, np.ndarray], classes: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    dist = distances_to_centroids(Z, centroids, classes)
    pred_idx = np.argmin(dist, axis=1)
    preds = np.array([int(classes[i]) for i in pred_idx], dtype=np.int64)
    return preds, dist


# -----------------------------------------------------------------------------
# Fold construction
# -----------------------------------------------------------------------------


def make_subject_kfold_groups(subjects: Sequence[int], subj_sel: np.ndarray, n_folds: int, seed: int = 0) -> List[List[int]]:
    """Greedy subject-level folds with roughly balanced sample counts."""
    subjects = [int(s) for s in subjects]
    if n_folds < 2:
        raise ValueError("--n-subject-folds must be at least 2")
    if n_folds > len(subjects):
        raise ValueError(f"--n-subject-folds={n_folds} exceeds n_subjects={len(subjects)}")

    counts = {int(s): int(np.sum(subj_sel == int(s))) for s in subjects}
    rng = np.random.default_rng(seed)
    shuffled = list(subjects)
    rng.shuffle(shuffled)
    ordered = sorted(shuffled, key=lambda x: counts[x], reverse=True)

    groups: List[List[int]] = [[] for _ in range(n_folds)]
    totals = [0 for _ in range(n_folds)]
    for s in ordered:
        j = int(np.argmin(totals))
        groups[j].append(int(s))
        totals[j] += counts[int(s)]
    groups = [sorted(g) for g in groups]
    groups = sorted(groups, key=lambda g: (min(g), len(g)))
    return groups


def build_subject_folds(args, subjects: List[int], subj_sel: np.ndarray) -> List[Dict]:
    manual_groups = parse_subject_groups(args.heldout_subject_groups)
    if manual_groups:
        allowed = set(subjects)
        groups = []
        for g in manual_groups:
            gg = [int(s) for s in g if int(s) in allowed]
            if gg:
                groups.append(sorted(gg))
        cv_name = "manual_subject_groups"
    elif args.cv == "loso":
        groups = [[int(s)] for s in subjects]
        cv_name = "loso"
    elif args.cv == "subject_kfold":
        groups = make_subject_kfold_groups(subjects, subj_sel, n_folds=args.n_subject_folds, seed=args.seed)
        cv_name = f"subject_{args.n_subject_folds}fold"
    else:
        raise ValueError(f"Unknown --cv: {args.cv}")

    if args.folds:
        keep = set(parse_int_list(args.folds))
        groups = [g for i, g in enumerate(groups, start=1) if i in keep]

    total_n = int(len(subj_sel))
    specs = []
    for i, g in enumerate(groups, start=1):
        test_n = int(np.sum(np.isin(subj_sel, np.asarray(g, dtype=np.int64))))
        specs.append({
            "fold_id": int(i),
            "cv": cv_name,
            "heldout_subjects": [int(x) for x in g],
            "n_heldout_subjects": int(len(g)),
            "estimated_n_test": test_n,
            "estimated_test_fraction": float(test_n / max(total_n, 1)),
        })
    return specs


def stratified_subject_control_split(
    subj: np.ndarray,
    stratify_labels: np.ndarray,
    test_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Closed-set subject-control split.

    For every subject and every stratify label, randomly allocate a fraction to
    test. This preserves the task-state composition inside each subject.
    """
    rng = np.random.default_rng(seed)
    train_loc: List[np.ndarray] = []
    test_loc: List[np.ndarray] = []
    for s in sorted(np.unique(subj).astype(int)):
        for lab in sorted(np.unique(stratify_labels[subj == s]).astype(int)):
            idx = np.flatnonzero((subj == s) & (stratify_labels == lab))
            if len(idx) <= 1:
                train_loc.append(idx)
                continue
            perm = np.array(idx, copy=True)
            rng.shuffle(perm)
            n_test = int(round(len(perm) * test_fraction))
            n_test = max(1, min(n_test, len(perm) - 1))
            test_loc.append(np.sort(perm[:n_test]))
            train_loc.append(np.sort(perm[n_test:]))
    train_idx = np.sort(np.concatenate(train_loc)) if train_loc else np.array([], dtype=np.int64)
    test_idx = np.sort(np.concatenate(test_loc)) if test_loc else np.array([], dtype=np.int64)
    return train_idx, test_idx


# -----------------------------------------------------------------------------
# Names and stats
# -----------------------------------------------------------------------------


def infer_class_names(label_key: str, classes: Sequence[int]) -> Dict[int, str]:
    if label_key == "task_id" and set([int(c) for c in classes]) == {0, 1, 2}:
        return {0: "Resting", 1: "N-Back", 2: "MATB"}
    return {int(c): str(c) for c in classes}


def class_coord_stats(Z: np.ndarray, y: np.ndarray, classes: Sequence[int], prefix: str) -> Dict[int, Dict]:
    rows: Dict[int, Dict] = {}
    for c in classes:
        c = int(c)
        mask = (y == c)
        Zc = Z[mask]
        if len(Zc) == 0:
            rows[c] = {
                f"{prefix}_n": 0,
                f"{prefix}_mean_ld1": np.nan,
                f"{prefix}_mean_ld2": np.nan,
                f"{prefix}_std_ld1": np.nan,
                f"{prefix}_std_ld2": np.nan,
                f"{prefix}_median_ld1": np.nan,
                f"{prefix}_median_ld2": np.nan,
                f"{prefix}_q25_ld1": np.nan,
                f"{prefix}_q25_ld2": np.nan,
                f"{prefix}_q75_ld1": np.nan,
                f"{prefix}_q75_ld2": np.nan,
            }
            continue
        if Zc.shape[1] == 1:
            Zc2 = np.c_[Zc[:, 0], np.zeros(len(Zc))]
        else:
            Zc2 = Zc[:, :2]
        rows[c] = {
            f"{prefix}_n": int(len(Zc2)),
            f"{prefix}_mean_ld1": float(np.mean(Zc2[:, 0])),
            f"{prefix}_mean_ld2": float(np.mean(Zc2[:, 1])),
            f"{prefix}_std_ld1": float(np.std(Zc2[:, 0], ddof=1)) if len(Zc2) > 1 else 0.0,
            f"{prefix}_std_ld2": float(np.std(Zc2[:, 1], ddof=1)) if len(Zc2) > 1 else 0.0,
            f"{prefix}_median_ld1": float(np.median(Zc2[:, 0])),
            f"{prefix}_median_ld2": float(np.median(Zc2[:, 1])),
            f"{prefix}_q25_ld1": float(np.percentile(Zc2[:, 0], 25)),
            f"{prefix}_q25_ld2": float(np.percentile(Zc2[:, 1], 25)),
            f"{prefix}_q75_ld1": float(np.percentile(Zc2[:, 0], 75)),
            f"{prefix}_q75_ld2": float(np.percentile(Zc2[:, 1], 75)),
        }
    return rows


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------


def _two_dim(Z: np.ndarray) -> np.ndarray:
    if Z.shape[1] == 1:
        return np.c_[Z[:, 0], np.zeros(len(Z))]
    return Z[:, :2]


def _robust_extent(Z_list: List[np.ndarray], centroids: Dict[int, np.ndarray]) -> Tuple[float, float, float, float]:
    pts = []
    for Z in Z_list:
        if Z is not None and len(Z):
            pts.append(_two_dim(Z))
    if centroids:
        pts.append(np.stack([_two_dim(np.asarray(v).reshape(1, -1))[0] for v in centroids.values()], axis=0))
    if not pts:
        return -1, 1, -1, 1
    Z = np.concatenate(pts, axis=0)
    xlo, xhi = np.percentile(Z[:, 0], [1, 99])
    ylo, yhi = np.percentile(Z[:, 1], [1, 99])
    padx = 0.12 * max(1e-6, xhi - xlo)
    pady = 0.12 * max(1e-6, yhi - ylo)
    return float(xlo - padx), float(xhi + padx), float(ylo - pady), float(yhi + pady)


def plot_ld_regions(
    Z: np.ndarray,
    y_true: np.ndarray,
    centroids: Dict[int, np.ndarray],
    classes: Sequence[int],
    class_names: Dict[int, str],
    title: str,
    out_png: str | Path,
    y_pred: Optional[np.ndarray] = None,
    max_points: int = 6000,
    seed: int = 0,
    extra_Z_for_extent: Optional[List[np.ndarray]] = None,
) -> None:
    ensure_dir(Path(out_png).parent)
    Z2 = _two_dim(Z)
    rng = np.random.default_rng(seed)
    if len(Z2) > max_points:
        idx = rng.choice(len(Z2), size=max_points, replace=False)
    else:
        idx = np.arange(len(Z2))
    Zp, yp = Z2[idx], y_true[idx]

    extent_list = [Z2]
    if extra_Z_for_extent:
        extent_list.extend([_two_dim(z) for z in extra_Z_for_extent if z is not None and len(z)])
    xlo, xhi, ylo, yhi = _robust_extent(extent_list, centroids)

    # Decision regions only use LD1/LD2. For >2-D LDA this is a qualitative view.
    xx, yy = np.meshgrid(np.linspace(xlo, xhi, 260), np.linspace(ylo, yhi, 260))
    grid2 = np.c_[xx.ravel(), yy.ravel()]
    C2 = np.stack([_two_dim(centroids[int(c)].reshape(1, -1))[0] for c in classes], axis=0)
    dist2 = ((grid2[:, None, :] - C2[None, :, :]) ** 2).sum(axis=2)
    pred_grid = np.argmin(dist2, axis=1).reshape(xx.shape)

    plt.figure(figsize=(7.2, 6.0))
    plt.contourf(xx, yy, pred_grid, levels=np.arange(len(classes) + 1) - 0.5, alpha=0.10)
    if len(classes) <= 8:
        plt.contour(xx, yy, pred_grid, levels=np.arange(0.5, len(classes) - 0.5 + 1e-9, 1.0), linewidths=0.8)

    for c in classes:
        c = int(c)
        mask = (yp == c)
        if np.any(mask):
            plt.scatter(Zp[mask, 0], Zp[mask, 1], s=8, alpha=0.35, label=f"true {class_names.get(c, str(c))}")

    for c in classes:
        c = int(c)
        cen = _two_dim(centroids[c].reshape(1, -1))[0]
        plt.scatter([cen[0]], [cen[1]], marker="*", s=250, edgecolors="black", linewidths=1.0,
                    label=f"centroid {class_names.get(c, str(c))}")
        plt.text(cen[0], cen[1], f"  C{c}", fontsize=10, weight="bold")

    plt.title(title)
    plt.xlabel("LD1")
    plt.ylabel("LD2")
    plt.xlim(xlo, xhi)
    plt.ylim(ylo, yhi)
    plt.legend(fontsize=8, ncol=2, loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def plot_confusion(cm: np.ndarray, classes: Sequence[int], class_names: Dict[int, str], title: str, out_png: str | Path) -> None:
    ensure_dir(Path(out_png).parent)
    cm = cm.astype(np.float64)
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, np.maximum(row_sum, 1.0))
    plt.figure(figsize=(5.2, 4.3))
    plt.imshow(cm_norm, aspect="auto")
    plt.colorbar(label="Row-normalized recall")
    ticks = np.arange(len(classes))
    labels = [class_names.get(int(c), str(c)) for c in classes]
    plt.xticks(ticks, labels, rotation=30, ha="right")
    plt.yticks(ticks, labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, f"{cm_norm[i, j]:.2f}\n({int(cm[i, j])})", ha="center", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def plot_summary_bacc(summary_rows: List[Dict], out_png: str | Path, title: str, chance: float) -> None:
    ensure_dir(Path(out_png).parent)
    xs = [int(r["svd_dim"]) for r in summary_rows]
    ys = [float(r["balanced_acc_mean"]) for r in summary_rows]
    es = [float(r["balanced_acc_sem"]) for r in summary_rows]
    plt.figure(figsize=(6.2, 4.4))
    plt.errorbar(xs, ys, yerr=es, marker="o", capsize=3)
    plt.axhline(chance, linestyle="--", linewidth=1, label=f"chance={chance:.3f}")
    plt.xlabel("SVD dimension M")
    plt.ylabel("Mean balanced accuracy")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def plot_subject_bacc_heatmap(subject_metric_rows: List[Dict], out_png: str | Path, plot_svd_dims: Sequence[int]) -> None:
    ensure_dir(Path(out_png).parent)
    # Use first requested plot dim, usually 500.
    m = int(plot_svd_dims[-1]) if plot_svd_dims else None
    rows = [r for r in subject_metric_rows if m is None or int(r["svd_dim"]) == m]
    if not rows:
        return
    subjects = sorted({int(r["heldout_subject"]) for r in rows})
    # One column per fold, value per subject.
    folds = sorted({int(r["heldout_fold"]) for r in rows})
    mat = np.full((len(subjects), len(folds)), np.nan)
    s2i = {s: i for i, s in enumerate(subjects)}
    f2j = {f: j for j, f in enumerate(folds)}
    for r in rows:
        mat[s2i[int(r["heldout_subject"])], f2j[int(r["heldout_fold"])]] = float(r["balanced_acc"])
    plt.figure(figsize=(max(5, len(folds) * 1.2), max(5, len(subjects) * 0.28)))
    plt.imshow(mat, aspect="auto", vmin=0.0, vmax=1.0)
    plt.colorbar(label="Balanced accuracy")
    plt.xticks(np.arange(len(folds)), [str(f) for f in folds])
    plt.yticks(np.arange(len(subjects)), [str(s) for s in subjects], fontsize=7)
    plt.xlabel("Held-out fold")
    plt.ylabel("Held-out subject")
    plt.title(f"Subject-level balanced accuracy, M={m}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


# -----------------------------------------------------------------------------
# State probe
# -----------------------------------------------------------------------------


def run_state_probe(args, f: h5py.File, emb_ds: h5py.Dataset, y_all: np.ndarray, subj_all: np.ndarray, selected_idx: np.ndarray, y_sel: np.ndarray, subj_sel: np.ndarray, classes: List[int], class_names: Dict[int, str], model_dir: Path) -> None:
    print(f"[{timestamp()}] === State probe: z -> {args.label_key} ===", flush=True)
    rng = np.random.default_rng(args.seed + 2026)
    base = model_dir / "state_probe"
    tables_dir = base / "tables"
    train_plot_dir = base / "plots_train"
    fold_plot_dir = base / "plots_fold_test"
    subj_plot_dir = base / "plots_subject_test"
    summary_plot_dir = base / "plots_summary"
    point_dir = base / "points"
    for p in [tables_dir, train_plot_dir, fold_plot_dir, subj_plot_dir, summary_plot_dir, point_dir]:
        ensure_dir(p)

    subjects = sorted([int(s) for s in np.unique(subj_sel)])
    if args.subjects:
        keep_subjects = set(parse_int_list(args.subjects))
        candidate_mask = np.isin(subj_sel, np.asarray(sorted(keep_subjects), dtype=np.int64))
        selected_idx = selected_idx[candidate_mask]
        y_sel = y_sel[candidate_mask]
        subj_sel = subj_sel[candidate_mask]
        subjects = sorted([int(s) for s in np.unique(subj_sel)])

    fold_specs = build_subject_folds(args, subjects, subj_sel)
    d = get_flat_dim(emb_ds)
    svd_dims = sorted(set([int(min(m, d)) for m in parse_int_list(args.svd_dims)]))
    plot_svd_dims = sorted(set([int(min(m, d)) for m in parse_int_list(args.plot_svd_dims)]))
    if not plot_svd_dims:
        plot_svd_dims = [max(svd_dims)]
    m_max = max(svd_dims)

    metadata = {
        "model_name": args.model_name,
        "analysis": "state_probe",
        "h5": args.h5,
        "embedding_key": args.embedding_key,
        "label_key": args.label_key,
        "subject_key": args.subject_key,
        "classes": classes,
        "class_names": class_names,
        "selected_samples": int(len(selected_idx)),
        "subjects": subjects,
        "fold_specs": fold_specs,
        "svd_dims": svd_dims,
        "plot_svd_dims": plot_svd_dims,
        "n_shuffle": int(args.n_shuffle),
        "subject_centering_ablation": "LD-space per-heldout-subject global mean alignment, label-free",
        "created_at": timestamp(),
    }
    with (tables_dir / "state_probe_metadata.json").open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2, ensure_ascii=False)

    print(f"[{timestamp()}] State probe selected samples={len(selected_idx)}, subjects={len(subjects)}, folds={len(fold_specs)}", flush=True)
    for fs in fold_specs:
        print(f"  fold {fs['fold_id']}: heldout_subjects={fs['heldout_subjects']} n_test≈{fs['estimated_n_test']} frac≈{fs['estimated_test_fraction']:.3f}", flush=True)

    fold_rows: List[Dict] = []
    train_class_rows: List[Dict] = []
    fold_class_rows: List[Dict] = []
    subject_rows: List[Dict] = []
    subject_class_rows: List[Dict] = []
    centroid_rows: List[Dict] = []
    subject_centroid_rows: List[Dict] = []
    subject_boundary_rows: List[Dict] = []
    subject_margin_rows: List[Dict] = []
    oracle_procrustes_rows: List[Dict] = []
    within_subject_rows: List[Dict] = []
    cm_by_m: Dict[int, np.ndarray] = {m: np.zeros((len(classes), len(classes)), dtype=np.int64) for m in svd_dims}

    for fold_counter, fs in enumerate(fold_specs, start=1):
        fold_id = int(fs["fold_id"])
        t0 = time.time()
        heldout_subjects = np.asarray(fs["heldout_subjects"], dtype=np.int64)
        is_test = np.isin(subj_sel, heldout_subjects)
        train_local = np.flatnonzero(~is_test)
        test_local = np.flatnonzero(is_test)
        train_idx = selected_idx[train_local]
        test_idx = selected_idx[test_local]
        y_train = y_sel[train_local]
        y_test = y_sel[test_local]
        subj_train = subj_sel[train_local]
        subj_test = subj_sel[test_local]

        train_counts = per_class_counts(y_train, classes)
        test_counts = per_class_counts(y_test, classes)
        if any(v == 0 for v in train_counts.values()) or any(v == 0 for v in test_counts.values()):
            print(f"[{timestamp()}] [WARN] skip fold={fold_id} due missing class train={train_counts}, test={test_counts}", flush=True)
            continue

        print(f"[{timestamp()}] State fold {fold_id}/{len(fold_specs)} heldout={heldout_subjects.tolist()} train={len(train_idx)} test={len(test_idx)}", flush=True)
        X_train = read_embedding_rows(emb_ds, train_idx, batch_size=args.read_batch_size, dtype=np.float32)
        X_test = read_embedding_rows(emb_ds, test_idx, batch_size=args.read_batch_size, dtype=np.float32)
        mu = X_train.mean(axis=0, dtype=np.float64).astype(np.float32)
        X_train -= mu
        X_test -= mu

        m_eff_max = int(min(m_max, X_train.shape[0] - 1, X_train.shape[1]))
        U, S, Vt = randomized_svd_dense(
            X_train,
            n_components=m_eff_max,
            n_oversamples=args.svd_oversamples,
            n_iter=args.svd_power_iter,
            seed=args.seed + fold_id,
        )
        R_train_full = U * S[None, :]
        R_test_full = X_test @ Vt.T
        del X_train, X_test, U
        gc.collect()

        for m in svd_dims:
            m_eff = int(min(m, R_train_full.shape[1]))
            R_train = R_train_full[:, :m_eff]
            R_test = R_test_full[:, :m_eff]
            lda = fit_multiclass_lda(R_train, y_train, classes=classes, ridge=args.lda_ridge, max_ld_dims=None)
            Z_train = lda.train_ld
            Z_test = R_test @ lda.W
            y_train_pred, train_dist = predict_nearest_centroid(Z_train, lda.class_centroids_ld, classes)
            y_test_pred, test_dist = predict_nearest_centroid(Z_test, lda.class_centroids_ld, classes)

            # Label-free per-subject centering ablation. This estimates a pure
            # translation offset for each held-out subject in the train-fitted LD
            # space, without using task labels.
            train_ld_global_mean = Z_train.mean(axis=0)
            Z_test_centered, subject_center_offsets = subject_center_ld(
                Z_test, subj_test, train_reference_mean=train_ld_global_mean
            )
            y_test_pred_centered, test_dist_centered = predict_nearest_centroid(
                Z_test_centered, lda.class_centroids_ld, classes
            )

            train_acc = float(np.mean(y_train_pred == y_train))
            train_bacc, train_recalls = balanced_accuracy(y_train, y_train_pred, classes)
            train_macro_auc, train_auc_by_class = macro_auc_ovr_from_distances(y_train, train_dist, classes)

            test_acc = float(np.mean(y_test_pred == y_test))
            test_bacc, test_recalls = balanced_accuracy(y_test, y_test_pred, classes)
            test_macro_auc, test_auc_by_class = macro_auc_ovr_from_distances(y_test, test_dist, classes)

            test_acc_centered = float(np.mean(y_test_pred_centered == y_test))
            test_bacc_centered, test_recalls_centered = balanced_accuracy(y_test, y_test_pred_centered, classes)
            test_macro_auc_centered, test_auc_by_class_centered = macro_auc_ovr_from_distances(y_test, test_dist_centered, classes)

            cm = confusion_matrix_fixed(y_test, y_test_pred, classes)
            cm_centered = confusion_matrix_fixed(y_test, y_test_pred_centered, classes)
            cm_by_m[m] += cm

            # Optional label-shuffle null. It answers whether the same pipeline can
            # obtain comparable held-out performance when train labels carry no
            # state information. This is distinct from subject-heldout validation.
            shuffle_bacc = []
            shuffle_auc = []
            shuffle_bacc_centered = []
            shuffle_auc_centered = []
            shuffle_lambdas = []
            if args.n_shuffle > 0:
                for _ in range(args.n_shuffle):
                    y_shuf = np.array(y_train, copy=True)
                    rng.shuffle(y_shuf)
                    lda_shuf = fit_multiclass_lda(R_train, y_shuf, classes=classes, ridge=args.lda_ridge, max_ld_dims=None)
                    Z_test_shuf = R_test @ lda_shuf.W
                    pred_shuf, dist_shuf = predict_nearest_centroid(Z_test_shuf, lda_shuf.class_centroids_ld, classes)
                    bacc_shuf, _ = balanced_accuracy(y_test, pred_shuf, classes)
                    auc_shuf, _ = macro_auc_ovr_from_distances(y_test, dist_shuf, classes)
                    shuffle_bacc.append(bacc_shuf)
                    shuffle_auc.append(auc_shuf)

                    Z_test_shuf_centered, _ = subject_center_ld(
                        Z_test_shuf, subj_test, train_reference_mean=lda_shuf.train_ld.mean(axis=0)
                    )
                    pred_shuf_centered, dist_shuf_centered = predict_nearest_centroid(
                        Z_test_shuf_centered, lda_shuf.class_centroids_ld, classes
                    )
                    bacc_shuf_centered, _ = balanced_accuracy(y_test, pred_shuf_centered, classes)
                    auc_shuf_centered, _ = macro_auc_ovr_from_distances(y_test, dist_shuf_centered, classes)
                    shuffle_bacc_centered.append(bacc_shuf_centered)
                    shuffle_auc_centered.append(auc_shuf_centered)

                    vals = lda_shuf.eigvals[: len(classes) - 1]
                    if len(vals) < len(classes) - 1:
                        vals = np.pad(vals, (0, len(classes) - 1 - len(vals)), constant_values=np.nan)
                    shuffle_lambdas.append(vals)

            shuffle_bacc = np.asarray(shuffle_bacc, dtype=np.float64)
            shuffle_auc = np.asarray(shuffle_auc, dtype=np.float64)
            shuffle_bacc_centered = np.asarray(shuffle_bacc_centered, dtype=np.float64)
            shuffle_auc_centered = np.asarray(shuffle_auc_centered, dtype=np.float64)
            shuffle_lambdas = np.asarray(shuffle_lambdas, dtype=np.float64) if len(shuffle_lambdas) else np.empty((0, len(classes) - 1))

            def _p_ge(null_arr: np.ndarray, real_val: float) -> float:
                if null_arr.size == 0 or not np.isfinite(real_val):
                    return float("nan")
                return float((1.0 + np.sum(null_arr >= real_val)) / (1.0 + len(null_arr)))

            shuffle_bacc95 = float(np.nanpercentile(shuffle_bacc, 95)) if shuffle_bacc.size else float("nan")
            shuffle_auc95 = float(np.nanpercentile(shuffle_auc, 95)) if shuffle_auc.size else float("nan")
            shuffle_bacc_centered95 = float(np.nanpercentile(shuffle_bacc_centered, 95)) if shuffle_bacc_centered.size else float("nan")
            shuffle_auc_centered95 = float(np.nanpercentile(shuffle_auc_centered, 95)) if shuffle_auc_centered.size else float("nan")

            eig = lda.eigvals[: len(classes) - 1]
            if len(eig) < len(classes) - 1:
                eig = np.pad(eig, (0, len(classes) - 1 - len(eig)), constant_values=np.nan)

            fold_rows.append({
                "model": args.model_name,
                "analysis": "state_probe",
                "heldout_fold": fold_id,
                "heldout_subjects": safe_json([int(x) for x in heldout_subjects.tolist()]),
                "n_heldout_subjects": int(len(heldout_subjects)),
                "heldout_fraction": float(len(y_test) / max(len(selected_idx), 1)),
                "svd_dim": int(m),
                "svd_dim_effective": int(m_eff),
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
                "train_class_counts": safe_json(train_counts),
                "test_class_counts": safe_json(test_counts),

                "train_acc": train_acc,
                "train_balanced_acc": train_bacc,
                "train_macro_auc_ovr": train_macro_auc,
                "train_class_recalls": safe_json(train_recalls),
                "train_auc_by_class": safe_json(train_auc_by_class),

                "test_acc": test_acc,
                "test_balanced_acc": test_bacc,
                "heldout_macro_auc": test_macro_auc,
                "test_macro_auc_ovr": test_macro_auc,
                "test_class_recalls": safe_json(test_recalls),
                "test_auc_by_class": safe_json(test_auc_by_class),

                "test_acc_subject_centered": test_acc_centered,
                "test_balanced_acc_subject_centered": test_bacc_centered,
                "heldout_macro_auc_subject_centered": test_macro_auc_centered,
                "test_macro_auc_ovr_subject_centered": test_macro_auc_centered,
                "test_class_recalls_subject_centered": safe_json(test_recalls_centered),
                "test_auc_by_class_subject_centered": safe_json(test_auc_by_class_centered),
                "delta_bacc_subject_centered_minus_raw": float(test_bacc_centered - test_bacc),
                "delta_auc_subject_centered_minus_raw": float(test_macro_auc_centered - test_macro_auc),
                "subject_center_offsets": safe_json(subject_center_offsets),

                "lambda1": float(eig[0]) if len(eig) > 0 else np.nan,
                "lambda2": float(eig[1]) if len(eig) > 1 else np.nan,
                "shuffle_bacc95": shuffle_bacc95,
                "shuffle_macro_auc95": shuffle_auc95,
                "shuffle_bacc_subject_centered95": shuffle_bacc_centered95,
                "shuffle_macro_auc_subject_centered95": shuffle_auc_centered95,
                "p_bacc": _p_ge(shuffle_bacc, test_bacc),
                "p_macro_auc": _p_ge(shuffle_auc, test_macro_auc),
                "p_bacc_subject_centered": _p_ge(shuffle_bacc_centered, test_bacc_centered),
                "p_macro_auc_subject_centered": _p_ge(shuffle_auc_centered, test_macro_auc_centered),
                "shuffle_lambda1_95": float(np.nanpercentile(shuffle_lambdas[:, 0], 95)) if shuffle_lambdas.shape[0] and shuffle_lambdas.shape[1] > 0 else np.nan,
                "shuffle_lambda2_95": float(np.nanpercentile(shuffle_lambdas[:, 1], 95)) if shuffle_lambdas.shape[0] and shuffle_lambdas.shape[1] > 1 else np.nan,
                "p_lambda1": _p_ge(shuffle_lambdas[:, 0], float(eig[0])) if shuffle_lambdas.shape[0] and shuffle_lambdas.shape[1] > 0 else np.nan,
                "p_lambda2": _p_ge(shuffle_lambdas[:, 1], float(eig[1])) if shuffle_lambdas.shape[0] and shuffle_lambdas.shape[1] > 1 and len(eig) > 1 else np.nan,

                "confusion_matrix": safe_json(cm.tolist()),
                "confusion_matrix_subject_centered": safe_json(cm_centered.tolist()),
            })

            # Train class stats and centroids.
            train_stats = class_coord_stats(Z_train, y_train, classes, prefix="train")
            for c in classes:
                c = int(c)
                cen = lda.class_centroids_ld[c]
                row = {
                    "model": args.model_name,
                    "heldout_fold": fold_id,
                    "svd_dim": int(m),
                    "class_id": c,
                    "class_name": class_names.get(c, str(c)),
                    "train_centroid_ld1": float(_two_dim(cen.reshape(1, -1))[0, 0]),
                    "train_centroid_ld2": float(_two_dim(cen.reshape(1, -1))[0, 1]),
                    "train_in_sample_recall": train_recalls.get(c, np.nan),
                }
                row.update(train_stats[c])
                train_class_rows.append(row)
                centroid_rows.append({
                    "model": args.model_name,
                    "heldout_fold": fold_id,
                    "svd_dim": int(m),
                    "space": "state_LD",
                    "centroid_source": "train_class",
                    "centroid_label": c,
                    "centroid_name": class_names.get(c, str(c)),
                    "ld1": float(_two_dim(cen.reshape(1, -1))[0, 0]),
                    "ld2": float(_two_dim(cen.reshape(1, -1))[0, 1]),
                    "n": int(np.sum(y_train == c)),
                })

            # Fold-level heldout class stats.
            test_stats = class_coord_stats(Z_test, y_test, classes, prefix="test_true")
            test_centered_stats = class_coord_stats(Z_test_centered, y_test, classes, prefix="test_true_subject_centered")
            for c in classes:
                c = int(c)
                mask = (y_test == c)
                recall_c = float(np.mean(y_test_pred[mask] == c)) if np.any(mask) else np.nan
                recall_c_centered = float(np.mean(y_test_pred_centered[mask] == c)) if np.any(mask) else np.nan
                row = {
                    "model": args.model_name,
                    "heldout_fold": fold_id,
                    "heldout_subjects": safe_json([int(x) for x in heldout_subjects.tolist()]),
                    "svd_dim": int(m),
                    "true_class": c,
                    "class_name": class_names.get(c, str(c)),
                    "recall_this_class": recall_c,
                    "recall_this_class_subject_centered": recall_c_centered,
                    "delta_recall_subject_centered_minus_raw": float(recall_c_centered - recall_c) if np.isfinite(recall_c) and np.isfinite(recall_c_centered) else np.nan,
                }
                row.update(test_stats[c])
                row.update(test_centered_stats[c])
                if np.any(mask):
                    for j, cc in enumerate(classes):
                        cname = class_names.get(int(cc), str(cc))
                        row[f"mean_dist_to_{cname}"] = float(np.mean(test_dist[mask, j]))
                        row[f"mean_dist_subject_centered_to_{cname}"] = float(np.mean(test_dist_centered[mask, j]))
                    pred_counts = {int(cc): int(np.sum(y_test_pred[mask] == int(cc))) for cc in classes}
                    pred_counts_centered = {int(cc): int(np.sum(y_test_pred_centered[mask] == int(cc))) for cc in classes}
                    row["pred_counts"] = safe_json(pred_counts)
                    row["pred_counts_subject_centered"] = safe_json(pred_counts_centered)
                fold_class_rows.append(row)

            # Subject-level metrics and class stats in state LD space.
            for s in heldout_subjects.tolist():
                s = int(s)
                smask = (subj_test == s)
                if not np.any(smask):
                    continue
                ys = y_test[smask]
                ps = y_test_pred[smask]
                Zs = Z_test[smask]
                ds = test_dist[smask]
                ps_centered = y_test_pred_centered[smask]
                Zs_centered = Z_test_centered[smask]
                ds_centered = test_dist_centered[smask]

                s_acc = float(np.mean(ps == ys))
                s_bacc, s_recalls = balanced_accuracy(ys, ps, classes)
                s_macro_auc, s_auc_by_class = macro_auc_ovr_from_distances(ys, ds, classes)
                s_cm = confusion_matrix_fixed(ys, ps, classes)

                s_acc_centered = float(np.mean(ps_centered == ys))
                s_bacc_centered, s_recalls_centered = balanced_accuracy(ys, ps_centered, classes)
                s_macro_auc_centered, s_auc_by_class_centered = macro_auc_ovr_from_distances(ys, ds_centered, classes)
                s_cm_centered = confusion_matrix_fixed(ys, ps_centered, classes)

                # Mechanism diagnostic 1: boundary-relevant projection of subject offset.
                offset = np.asarray(subject_center_offsets.get(s, [np.nan] * Z_test.shape[1]), dtype=np.float64)
                boundary_row = {
                    "model": args.model_name,
                    "heldout_fold": fold_id,
                    "heldout_subject": s,
                    "svd_dim": int(m),
                }
                boundary_row.update(boundary_projection_diagnostic(offset, lda.class_centroids_ld, classes, class_names))
                boundary_row["delta_bacc_subject_centered_minus_raw"] = float(s_bacc_centered - s_bacc)
                boundary_row["delta_auc_subject_centered_minus_raw"] = float(s_macro_auc_centered - s_macro_auc)
                subject_boundary_rows.append(boundary_row)

                # Mechanism diagnostic 2: signed margins before and after centering.
                margins_raw = signed_margins_from_distances(ys, ds, classes)
                margins_centered = signed_margins_from_distances(ys, ds_centered, classes)

                # Mechanism diagnostic 3: oracle similarity Procrustes using true held-out labels.
                proc_available = 0
                proc_reason = "missing_class"
                proc_acc = proc_bacc = proc_auc = float("nan")
                proc_recalls = {}
                proc_auc_by_class = {}
                proc_cm = np.zeros((len(classes), len(classes)), dtype=np.int64)
                proc_scale = proc_resid = proc_angle = float("nan")
                proc_margins = np.full(len(ys), np.nan, dtype=np.float64)
                if all(np.any(ys == int(c)) for c in classes):
                    try:
                        source_centroids = np.stack([Zs[ys == int(c)].mean(axis=0) for c in classes], axis=0)
                        target_centroids = np.stack([lda.class_centroids_ld[int(c)] for c in classes], axis=0)
                        proc_weights = np.asarray([np.sum(ys == int(c)) for c in classes], dtype=np.float64)
                        R_proc, proc_scale, b_proc, proc_resid = fit_similarity_procrustes(
                            source_centroids, target_centroids, weights=proc_weights, allow_reflection=False
                        )
                        Zs_proc = apply_similarity_transform(Zs, R_proc, proc_scale, b_proc)
                        ps_proc, ds_proc = predict_nearest_centroid(Zs_proc, lda.class_centroids_ld, classes)
                        proc_acc = float(np.mean(ps_proc == ys))
                        proc_bacc, proc_recalls = balanced_accuracy(ys, ps_proc, classes)
                        proc_auc, proc_auc_by_class = macro_auc_ovr_from_distances(ys, ds_proc, classes)
                        proc_cm = confusion_matrix_fixed(ys, ps_proc, classes)
                        proc_angle = rotation_angle_degrees(R_proc)
                        proc_margins = signed_margins_from_distances(ys, ds_proc, classes)
                        proc_available = 1
                        proc_reason = "ok"
                    except Exception as e:
                        proc_reason = f"error:{type(e).__name__}:{e}"

                oracle_procrustes_rows.append({
                    "model": args.model_name,
                    "heldout_fold": fold_id,
                    "heldout_subject": s,
                    "svd_dim": int(m),
                    "oracle_procrustes_available": int(proc_available),
                    "oracle_procrustes_reason": proc_reason,
                    "bacc_raw": s_bacc,
                    "bacc_subject_centered": s_bacc_centered,
                    "bacc_oracle_procrustes": proc_bacc,
                    "auc_raw": s_macro_auc,
                    "auc_subject_centered": s_macro_auc_centered,
                    "auc_oracle_procrustes": proc_auc,
                    "delta_bacc_procrustes_minus_raw": float(proc_bacc - s_bacc) if np.isfinite(proc_bacc) else np.nan,
                    "delta_bacc_procrustes_minus_centered": float(proc_bacc - s_bacc_centered) if np.isfinite(proc_bacc) else np.nan,
                    "delta_auc_procrustes_minus_raw": float(proc_auc - s_macro_auc) if np.isfinite(proc_auc) else np.nan,
                    "delta_auc_procrustes_minus_centered": float(proc_auc - s_macro_auc_centered) if np.isfinite(proc_auc) else np.nan,
                    "procrustes_scale": proc_scale,
                    "procrustes_rotation_angle_deg": proc_angle,
                    "procrustes_residual_rms": proc_resid,
                    "confusion_matrix_oracle_procrustes": safe_json(proc_cm.tolist()),
                    "class_recalls_oracle_procrustes": safe_json(proc_recalls),
                    "auc_by_class_oracle_procrustes": safe_json(proc_auc_by_class),
                })

                # Mechanism diagnostic 4: within-subject self-readout oracle upper bound.
                R_s = R_test[smask]
                self_row = within_subject_self_readout(
                    R_s, ys, classes=classes, ridge=args.lda_ridge,
                    test_fraction=args.self_readout_test_fraction, seed=args.seed + 100000 + fold_id * 1000 + s + m,
                )
                within_subject_rows.append({
                    "model": args.model_name,
                    "heldout_fold": fold_id,
                    "heldout_subject": s,
                    "svd_dim": int(m),
                    "cross_subject_bacc": s_bacc,
                    "cross_subject_auc": s_macro_auc,
                    "subject_centered_bacc": s_bacc_centered,
                    "oracle_procrustes_bacc": proc_bacc,
                    **self_row,
                })

                margin_row = {
                    "model": args.model_name,
                    "heldout_fold": fold_id,
                    "heldout_subject": s,
                    "svd_dim": int(m),
                    "n_test": int(len(ys)),
                    "balanced_acc_raw": s_bacc,
                    "balanced_acc_subject_centered": s_bacc_centered,
                    "balanced_acc_oracle_procrustes": proc_bacc,
                    "n_prediction_changed_by_centering": int(np.sum(ps_centered != ps)),
                }
                margin_row.update(summarize_margins(margins_raw, "raw"))
                margin_row.update(summarize_margins(margins_centered, "subject_centered"))
                margin_row.update(summarize_margins(proc_margins, "oracle_procrustes"))
                margin_row["delta_mean_margin_centered_minus_raw"] = float(np.nanmean(margins_centered) - np.nanmean(margins_raw))
                margin_row["delta_mean_margin_procrustes_minus_raw"] = float(np.nanmean(proc_margins) - np.nanmean(margins_raw)) if np.any(np.isfinite(proc_margins)) else np.nan
                subject_margin_rows.append(margin_row)

                subject_rows.append({
                    "model": args.model_name,
                    "heldout_fold": fold_id,
                    "heldout_subject": s,
                    "svd_dim": int(m),
                    "n_test": int(len(ys)),
                    "acc": s_acc,
                    "balanced_acc": s_bacc,
                    "macro_auc_ovr": s_macro_auc,
                    "class_recalls": safe_json(s_recalls),
                    "auc_by_class": safe_json(s_auc_by_class),
                    "confusion_matrix": safe_json(s_cm.tolist()),
                    "acc_subject_centered": s_acc_centered,
                    "balanced_acc_subject_centered": s_bacc_centered,
                    "macro_auc_ovr_subject_centered": s_macro_auc_centered,
                    "class_recalls_subject_centered": safe_json(s_recalls_centered),
                    "auc_by_class_subject_centered": safe_json(s_auc_by_class_centered),
                    "confusion_matrix_subject_centered": safe_json(s_cm_centered.tolist()),
                    "delta_bacc_subject_centered_minus_raw": float(s_bacc_centered - s_bacc),
                    "delta_auc_subject_centered_minus_raw": float(s_macro_auc_centered - s_macro_auc),
                    "acc_oracle_procrustes": proc_acc,
                    "balanced_acc_oracle_procrustes": proc_bacc,
                    "macro_auc_ovr_oracle_procrustes": proc_auc,
                    "delta_bacc_procrustes_minus_raw": float(proc_bacc - s_bacc) if np.isfinite(proc_bacc) else np.nan,
                    "delta_auc_procrustes_minus_raw": float(proc_auc - s_macro_auc) if np.isfinite(proc_auc) else np.nan,
                    "self_readout_bacc": self_row.get("self_readout_bacc", np.nan),
                    "self_readout_macro_auc": self_row.get("self_readout_macro_auc", np.nan),
                    "self_minus_cross_bacc": float(self_row.get("self_readout_bacc", np.nan) - s_bacc) if np.isfinite(self_row.get("self_readout_bacc", np.nan)) else np.nan,
                })

                # Overall subject centroid in state LD space. This is the estimated
                # subject translation offset that the centered ablation removes.
                Zs2 = _two_dim(Zs)
                Zs_centered2 = _two_dim(Zs_centered)
                offset = np.asarray(subject_center_offsets.get(s, [np.nan] * Z_test.shape[1]), dtype=np.float64)
                offset2 = _two_dim(offset.reshape(1, -1))[0]
                subject_centroid_rows.append({
                    "model": args.model_name,
                    "heldout_fold": fold_id,
                    "heldout_subject": s,
                    "svd_dim": int(m),
                    "space": "state_LD",
                    "n": int(len(Zs2)),
                    "subject_mean_ld1": float(np.mean(Zs2[:, 0])),
                    "subject_mean_ld2": float(np.mean(Zs2[:, 1])),
                    "subject_std_ld1": float(np.std(Zs2[:, 0], ddof=1)) if len(Zs2) > 1 else 0.0,
                    "subject_std_ld2": float(np.std(Zs2[:, 1], ddof=1)) if len(Zs2) > 1 else 0.0,
                    "subject_center_offset_ld1": float(offset2[0]),
                    "subject_center_offset_ld2": float(offset2[1]),
                    "subject_centered_mean_ld1": float(np.mean(Zs_centered2[:, 0])),
                    "subject_centered_mean_ld2": float(np.mean(Zs_centered2[:, 1])),
                })

                for c in classes:
                    c = int(c)
                    cmask = smask & (y_test == c)
                    if not np.any(cmask):
                        continue
                    local = np.flatnonzero(smask)
                    class_local_mask = (ys == c)
                    Zsc = Zs[class_local_mask]
                    dsc = ds[class_local_mask]
                    psc = ps[class_local_mask]
                    psc_centered = ps_centered[class_local_mask]
                    dsc_centered = ds_centered[class_local_mask]
                    stats = class_coord_stats(Zs, ys, [c], prefix="test_true")[c]
                    stats_centered = class_coord_stats(Zs_centered, ys, [c], prefix="test_true_subject_centered")[c]
                    recall_raw = float(np.mean(psc == c))
                    recall_centered = float(np.mean(psc_centered == c))
                    row = {
                        "model": args.model_name,
                        "heldout_fold": fold_id,
                        "heldout_subject": s,
                        "svd_dim": int(m),
                        "true_class": c,
                        "class_name": class_names.get(c, str(c)),
                        "recall_this_class": recall_raw,
                        "recall_this_class_subject_centered": recall_centered,
                        "delta_recall_subject_centered_minus_raw": float(recall_centered - recall_raw),
                    }
                    row.update(stats)
                    row.update(stats_centered)
                    for j, cc in enumerate(classes):
                        cname = class_names.get(int(cc), str(cc))
                        row[f"mean_dist_to_{cname}"] = float(np.mean(dsc[:, j]))
                        row[f"mean_dist_subject_centered_to_{cname}"] = float(np.mean(dsc_centered[:, j]))
                    row["pred_counts"] = safe_json({int(cc): int(np.sum(psc == int(cc))) for cc in classes})
                    row["pred_counts_subject_centered"] = safe_json({int(cc): int(np.sum(psc_centered == int(cc))) for cc in classes})
                    subject_class_rows.append(row)

                if m in plot_svd_dims and args.plot_each_subject:
                    title = f"{args.model_name} state probe: fold {fold_id}, subject {s}, M={m}\nbACC={s_bacc:.3f}, ACC={s_acc:.3f}"
                    out_png = subj_plot_dir / f"fold{fold_id:02d}_subject{s:02d}_M{m}_test_regions.png"
                    plot_ld_regions(Zs, ys, lda.class_centroids_ld, classes, class_names, title, out_png,
                                    y_pred=ps, max_points=args.max_plot_points_per_subject, seed=args.seed + s,
                                    extra_Z_for_extent=[Z_train])

            if args.save_test_points and m in plot_svd_dims:
                # Save held-out point-level evidence for plotted dims only.
                point_rows = []
                Z_test2 = _two_dim(Z_test)
                Z_test_centered2 = _two_dim(Z_test_centered)
                for i in range(len(y_test)):
                    r = {
                        "model": args.model_name,
                        "heldout_fold": fold_id,
                        "sample_index": int(test_idx[i]),
                        "subject_id": int(subj_test[i]),
                        "true_label": int(y_test[i]),
                        "pred_label": int(y_test_pred[i]),
                        "pred_label_subject_centered": int(y_test_pred_centered[i]),
                        "ld1": float(Z_test2[i, 0]),
                        "ld2": float(Z_test2[i, 1]),
                        "ld1_subject_centered": float(Z_test_centered2[i, 0]),
                        "ld2_subject_centered": float(Z_test_centered2[i, 1]),
                    }
                    for j, cc in enumerate(classes):
                        cname = class_names.get(int(cc), str(cc)).replace(' ', '_')
                        r[f"dist_to_{cname}"] = float(test_dist[i, j])
                        r[f"dist_subject_centered_to_{cname}"] = float(test_dist_centered[i, j])
                    point_rows.append(r)
                write_csv(point_dir / f"state_test_points_fold{fold_id:02d}_M{m}.csv", point_rows)

            if args.save_train_points and m in plot_svd_dims:
                point_rows = []
                Ztr2 = _two_dim(Z_train)
                for i in range(len(y_train)):
                    r = {
                        "model": args.model_name,
                        "heldout_fold": fold_id,
                        "sample_index": int(train_idx[i]),
                        "subject_id": int(subj_train[i]),
                        "true_label": int(y_train[i]),
                        "pred_label": int(y_train_pred[i]),
                        "ld1": float(Ztr2[i, 0]),
                        "ld2": float(Ztr2[i, 1]),
                    }
                    for j, cc in enumerate(classes):
                        r[f"dist_to_{class_names.get(int(cc), str(cc)).replace(' ', '_')}"] = float(train_dist[i, j])
                    point_rows.append(r)
                write_csv(point_dir / f"state_train_points_fold{fold_id:02d}_M{m}.csv", point_rows)

            if m in plot_svd_dims:
                train_title = f"{args.model_name} train LD map: fold {fold_id}, M={m}\nin-sample bACC={train_bacc:.3f}, ACC={train_acc:.3f}"
                plot_ld_regions(Z_train, y_train, lda.class_centroids_ld, classes, class_names, train_title,
                                train_plot_dir / f"fold{fold_id:02d}_M{m}_train_regions.png",
                                y_pred=y_train_pred, max_points=args.max_plot_points, seed=args.seed + fold_id)
                fold_title = f"{args.model_name} held-out LD map: fold {fold_id}, M={m}\nheldout subjects={heldout_subjects.tolist()}; bACC={test_bacc:.3f}, ACC={test_acc:.3f}"
                plot_ld_regions(Z_test, y_test, lda.class_centroids_ld, classes, class_names, fold_title,
                                fold_plot_dir / f"fold{fold_id:02d}_M{m}_heldout_regions.png",
                                y_pred=y_test_pred, max_points=args.max_plot_points, seed=args.seed + fold_id,
                                extra_Z_for_extent=[Z_train])

            print(
                f"    M={m:<4d} train_bacc={train_bacc:.4f} "
                f"test_bacc={test_bacc:.4f} test_auc={test_macro_auc:.4f} "
                f"centered_bacc={test_bacc_centered:.4f} centered_auc={test_macro_auc_centered:.4f} "
                f"test_acc={test_acc:.4f}",
                flush=True,
            )

        del R_train_full, R_test_full, Vt, S
        gc.collect()
        print(f"[{timestamp()}] State fold {fold_id} finished in {(time.time() - t0) / 60:.2f} min", flush=True)

    # Tables.
    write_csv(tables_dir / "fold_metrics.csv", fold_rows)
    write_csv(tables_dir / "train_class_ld_stats.csv", train_class_rows)
    write_csv(tables_dir / "fold_class_ld_stats.csv", fold_class_rows)
    write_csv(tables_dir / "subject_metrics.csv", subject_rows)
    write_csv(tables_dir / "subject_class_ld_stats.csv", subject_class_rows)
    write_csv(tables_dir / "centroids_by_fold.csv", centroid_rows)
    write_csv(tables_dir / "subject_centroids_in_state_ld_space.csv", subject_centroid_rows)
    write_csv(tables_dir / "subject_boundary_projection.csv", subject_boundary_rows)
    write_csv(tables_dir / "subject_margin_stats.csv", subject_margin_rows)
    write_csv(tables_dir / "oracle_procrustes_metrics.csv", oracle_procrustes_rows)
    write_csv(tables_dir / "within_subject_self_readout_oracle.csv", within_subject_rows)

    # Summary by M.
    summary_rows: List[Dict] = []
    for m in svd_dims:
        rows_m = [r for r in fold_rows if int(r["svd_dim"]) == int(m)]
        if not rows_m:
            continue
        summary_rows.append({
            "model": args.model_name,
            "analysis": "state_probe",
            "svd_dim": int(m),
            "n_folds": int(len(rows_m)),

            "train_balanced_acc_mean": nanmean([float(r["train_balanced_acc"]) for r in rows_m]),
            "train_balanced_acc_sem": sem([float(r["train_balanced_acc"]) for r in rows_m]),
            "train_macro_auc_mean": nanmean([float(r["train_macro_auc_ovr"]) for r in rows_m]),
            "train_macro_auc_sem": sem([float(r["train_macro_auc_ovr"]) for r in rows_m]),

            "test_balanced_acc_mean": nanmean([float(r["test_balanced_acc"]) for r in rows_m]),
            "test_balanced_acc_sem": sem([float(r["test_balanced_acc"]) for r in rows_m]),
            "balanced_acc_mean": nanmean([float(r["test_balanced_acc"]) for r in rows_m]),
            "balanced_acc_sem": sem([float(r["test_balanced_acc"]) for r in rows_m]),
            "heldout_macro_auc_mean": nanmean([float(r["heldout_macro_auc"]) for r in rows_m]),
            "heldout_macro_auc_sem": sem([float(r["heldout_macro_auc"]) for r in rows_m]),
            "test_macro_auc_mean": nanmean([float(r["test_macro_auc_ovr"]) for r in rows_m]),
            "test_macro_auc_sem": sem([float(r["test_macro_auc_ovr"]) for r in rows_m]),
            "test_acc_mean": nanmean([float(r["test_acc"]) for r in rows_m]),
            "test_acc_sem": sem([float(r["test_acc"]) for r in rows_m]),

            "test_balanced_acc_subject_centered_mean": nanmean([float(r["test_balanced_acc_subject_centered"]) for r in rows_m]),
            "test_balanced_acc_subject_centered_sem": sem([float(r["test_balanced_acc_subject_centered"]) for r in rows_m]),
            "heldout_macro_auc_subject_centered_mean": nanmean([float(r["heldout_macro_auc_subject_centered"]) for r in rows_m]),
            "heldout_macro_auc_subject_centered_sem": sem([float(r["heldout_macro_auc_subject_centered"]) for r in rows_m]),
            "delta_bacc_subject_centered_minus_raw_mean": nanmean([float(r["delta_bacc_subject_centered_minus_raw"]) for r in rows_m]),
            "delta_auc_subject_centered_minus_raw_mean": nanmean([float(r["delta_auc_subject_centered_minus_raw"]) for r in rows_m]),

            "lambda1_mean": nanmean([float(r["lambda1"]) for r in rows_m]),
            "lambda2_mean": nanmean([float(r["lambda2"]) for r in rows_m]),
            "shuffle_bacc95_mean": nanmean([float(r["shuffle_bacc95"]) for r in rows_m]),
            "shuffle_macro_auc95_mean": nanmean([float(r["shuffle_macro_auc95"]) for r in rows_m]),
            "p_bacc_median": float(np.nanmedian([float(r["p_bacc"]) for r in rows_m])) if not np.all(np.isnan([float(r["p_bacc"]) for r in rows_m])) else np.nan,
            "p_macro_auc_median": float(np.nanmedian([float(r["p_macro_auc"]) for r in rows_m])) if not np.all(np.isnan([float(r["p_macro_auc"]) for r in rows_m])) else np.nan,
            "p_bacc_subject_centered_median": float(np.nanmedian([float(r["p_bacc_subject_centered"]) for r in rows_m])) if not np.all(np.isnan([float(r["p_bacc_subject_centered"]) for r in rows_m])) else np.nan,
            "p_macro_auc_subject_centered_median": float(np.nanmedian([float(r["p_macro_auc_subject_centered"]) for r in rows_m])) if not np.all(np.isnan([float(r["p_macro_auc_subject_centered"]) for r in rows_m])) else np.nan,
        })
    write_csv(tables_dir / "summary_by_svd_dim.csv", summary_rows)

    # Compact mechanism summary by SVD dimension.
    mechanism_summary_rows: List[Dict] = []
    for m in svd_dims:
        br = [r for r in subject_boundary_rows if int(r["svd_dim"]) == int(m)]
        mr = [r for r in subject_margin_rows if int(r["svd_dim"]) == int(m)]
        pr = [r for r in oracle_procrustes_rows if int(r["svd_dim"]) == int(m)]
        sr = [r for r in within_subject_rows if int(r["svd_dim"]) == int(m)]
        if not (br or mr or pr or sr):
            continue
        mechanism_summary_rows.append({
            "model": args.model_name,
            "analysis": "state_probe_mechanism_summary",
            "svd_dim": int(m),
            "n_subject_rows": int(max(len(br), len(mr), len(pr), len(sr))),
            "offset_norm_mean": nanmean([float(r.get("offset_norm", np.nan)) for r in br]),
            "boundary_projection_ratio_mean": nanmean([float(r.get("boundary_projection_ratio", np.nan)) for r in br]),
            "max_projection_over_class_distance_mean": nanmean([float(r.get("max_projection_over_class_distance", np.nan)) for r in br]),
            "raw_mean_margin_mean": nanmean([float(r.get("raw_mean_margin", np.nan)) for r in mr]),
            "subject_centered_mean_margin_mean": nanmean([float(r.get("subject_centered_mean_margin", np.nan)) for r in mr]),
            "oracle_procrustes_mean_margin_mean": nanmean([float(r.get("oracle_procrustes_mean_margin", np.nan)) for r in mr]),
            "delta_mean_margin_centered_minus_raw_mean": nanmean([float(r.get("delta_mean_margin_centered_minus_raw", np.nan)) for r in mr]),
            "delta_mean_margin_procrustes_minus_raw_mean": nanmean([float(r.get("delta_mean_margin_procrustes_minus_raw", np.nan)) for r in mr]),
            "oracle_procrustes_bacc_mean": nanmean([float(r.get("bacc_oracle_procrustes", np.nan)) for r in pr]),
            "delta_bacc_procrustes_minus_raw_mean": nanmean([float(r.get("delta_bacc_procrustes_minus_raw", np.nan)) for r in pr]),
            "delta_bacc_procrustes_minus_centered_mean": nanmean([float(r.get("delta_bacc_procrustes_minus_centered", np.nan)) for r in pr]),
            "procrustes_scale_mean": nanmean([float(r.get("procrustes_scale", np.nan)) for r in pr]),
            "procrustes_residual_rms_mean": nanmean([float(r.get("procrustes_residual_rms", np.nan)) for r in pr]),
            "within_subject_self_readout_bacc_mean": nanmean([float(r.get("self_readout_bacc", np.nan)) for r in sr]),
            "within_subject_self_readout_auc_mean": nanmean([float(r.get("self_readout_macro_auc", np.nan)) for r in sr]),
            "self_minus_cross_bacc_mean": nanmean([float(r.get("self_readout_bacc", np.nan)) - float(r.get("cross_subject_bacc", np.nan)) for r in sr]),
        })
    write_csv(tables_dir / "mechanism_summary_by_svd_dim.csv", mechanism_summary_rows)

    # Aggregate confusion plots.
    for m in svd_dims:
        plot_confusion(cm_by_m[m], classes, class_names,
                       title=f"{args.model_name} state probe confusion, M={m}",
                       out_png=summary_plot_dir / f"confusion_matrix_M{m}.png")
    if summary_rows:
        plot_summary_bacc(summary_rows, summary_plot_dir / "summary_balanced_acc_vs_svd_dim.png",
                          title=f"{args.model_name} state probe: subject-heldout bACC", chance=1.0 / len(classes))
        centered_plot_rows = []
        for r in summary_rows:
            rr = dict(r)
            rr["balanced_acc_mean"] = rr.get("test_balanced_acc_subject_centered_mean", np.nan)
            rr["balanced_acc_sem"] = rr.get("test_balanced_acc_subject_centered_sem", 0.0)
            centered_plot_rows.append(rr)
        plot_summary_bacc(centered_plot_rows, summary_plot_dir / "summary_balanced_acc_subject_centered_vs_svd_dim.png",
                          title=f"{args.model_name} state probe: per-subject centered bACC", chance=1.0 / len(classes))
    if subject_rows:
        plot_subject_bacc_heatmap(subject_rows, summary_plot_dir / "subject_bacc_heatmap.png", plot_svd_dims)


# -----------------------------------------------------------------------------
# Subject identity control
# -----------------------------------------------------------------------------


def run_subject_identity_control(args, f: h5py.File, emb_ds: h5py.Dataset, y_all: np.ndarray, subj_all: np.ndarray, selected_idx: np.ndarray, y_sel: np.ndarray, subj_sel: np.ndarray, model_dir: Path) -> None:
    print(f"[{timestamp()}] === Subject identity control: z -> subject_id ===", flush=True)
    base = model_dir / "subject_identity_control"
    tables_dir = base / "tables"
    plots_dir = base / "plots"
    for p in [tables_dir, plots_dir]:
        ensure_dir(p)

    subjects = sorted([int(s) for s in np.unique(subj_sel)])
    subject_classes = subjects
    subject_names = {int(s): f"S{s}" for s in subjects}
    d = get_flat_dim(emb_ds)
    svd_dims = sorted(set([int(min(m, d)) for m in parse_int_list(args.control_svd_dims or args.svd_dims)]))
    plot_svd_dims = sorted(set([int(min(m, d)) for m in parse_int_list(args.control_plot_svd_dims or args.plot_svd_dims)]))
    if not plot_svd_dims:
        plot_svd_dims = [max(svd_dims)]
    m_max = max(svd_dims)

    train_local, test_local = stratified_subject_control_split(
        subj_sel, stratify_labels=y_sel, test_fraction=args.subject_control_test_fraction, seed=args.seed
    )
    train_idx = selected_idx[train_local]
    test_idx = selected_idx[test_local]
    y_train_subj = subj_sel[train_local].astype(np.int64)
    y_test_subj = subj_sel[test_local].astype(np.int64)
    state_train = y_sel[train_local].astype(np.int64)
    state_test = y_sel[test_local].astype(np.int64)

    print(f"[{timestamp()}] Subject control train={len(train_idx)} test={len(test_idx)} subjects={len(subjects)} chance={1/len(subjects):.4f}", flush=True)
    with (tables_dir / "subject_identity_control_metadata.json").open("w", encoding="utf-8") as fp:
        json.dump({
            "model_name": args.model_name,
            "analysis": "subject_identity_control",
            "h5": args.h5,
            "selected_samples": int(len(selected_idx)),
            "subjects": subjects,
            "n_subjects": len(subjects),
            "test_fraction": args.subject_control_test_fraction,
            "split": "within-subject-and-state stratified closed-set",
            "svd_dims": svd_dims,
            "plot_svd_dims": plot_svd_dims,
            "created_at": timestamp(),
        }, fp, indent=2, ensure_ascii=False)

    X_train = read_embedding_rows(emb_ds, train_idx, batch_size=args.read_batch_size, dtype=np.float32)
    X_test = read_embedding_rows(emb_ds, test_idx, batch_size=args.read_batch_size, dtype=np.float32)
    mu = X_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    X_train -= mu
    X_test -= mu
    U, S, Vt = randomized_svd_dense(
        X_train,
        n_components=int(min(m_max, X_train.shape[0] - 1, X_train.shape[1])),
        n_oversamples=args.svd_oversamples,
        n_iter=args.svd_power_iter,
        seed=args.seed + 991,
    )
    R_train_full = U * S[None, :]
    R_test_full = X_test @ Vt.T
    del X_train, X_test, U
    gc.collect()

    summary_rows: List[Dict] = []
    subject_rows: List[Dict] = []
    for m in svd_dims:
        m_eff = int(min(m, R_train_full.shape[1]))
        R_train = R_train_full[:, :m_eff]
        R_test = R_test_full[:, :m_eff]
        lda = fit_multiclass_lda(R_train, y_train_subj, classes=subject_classes, ridge=args.lda_ridge, max_ld_dims=None)
        Z_train = lda.train_ld
        Z_test = R_test @ lda.W
        pred_train, _ = predict_nearest_centroid(Z_train, lda.class_centroids_ld, subject_classes)
        pred_test, dist_test = predict_nearest_centroid(Z_test, lda.class_centroids_ld, subject_classes)
        train_acc = float(np.mean(pred_train == y_train_subj))
        train_bacc, train_recalls = balanced_accuracy(y_train_subj, pred_train, subject_classes)
        test_acc = float(np.mean(pred_test == y_test_subj))
        test_bacc, test_recalls = balanced_accuracy(y_test_subj, pred_test, subject_classes)
        summary_rows.append({
            "model": args.model_name,
            "analysis": "subject_identity_control",
            "svd_dim": int(m),
            "svd_dim_effective": int(m_eff),
            "n_train": int(len(y_train_subj)),
            "n_test": int(len(y_test_subj)),
            "n_subjects": int(len(subject_classes)),
            "chance": float(1.0 / len(subject_classes)),
            "train_acc": train_acc,
            "train_balanced_acc": train_bacc,
            "test_acc": test_acc,
            "test_balanced_acc": test_bacc,
            "lambda1": float(lda.eigvals[0]) if len(lda.eigvals) > 0 else np.nan,
            "lambda2": float(lda.eigvals[1]) if len(lda.eigvals) > 1 else np.nan,
        })
        for s in subject_classes:
            s = int(s)
            mask = (y_test_subj == s)
            subject_rows.append({
                "model": args.model_name,
                "analysis": "subject_identity_control",
                "svd_dim": int(m),
                "subject_id": s,
                "n_test": int(np.sum(mask)),
                "recall": float(np.mean(pred_test[mask] == s)) if np.any(mask) else np.nan,
                "mean_dist_to_own_centroid": float(np.mean(dist_test[mask, subject_classes.index(s)])) if np.any(mask) else np.nan,
            })
        print(f"    subject-control M={m:<4d} train_bacc={train_bacc:.4f} test_bacc={test_bacc:.4f} test_acc={test_acc:.4f}", flush=True)

        if m in plot_svd_dims:
            # Qualitative LD1/LD2 scatter for subject identity. Too many classes for regions.
            Zt2 = _two_dim(Z_test)
            rng = np.random.default_rng(args.seed + m)
            if len(Zt2) > args.max_plot_points:
                idx = rng.choice(len(Zt2), size=args.max_plot_points, replace=False)
            else:
                idx = np.arange(len(Zt2))
            plt.figure(figsize=(7.2, 6.0))
            sc = plt.scatter(Zt2[idx, 0], Zt2[idx, 1], c=y_test_subj[idx], s=8, alpha=0.55)
            plt.colorbar(sc, label="Subject ID")
            plt.xlabel("LD1")
            plt.ylabel("LD2")
            plt.title(f"{args.model_name} subject identity control, M={m}\ntest bACC={test_bacc:.3f}, chance={1/len(subject_classes):.3f}")
            plt.tight_layout()
            plt.savefig(plots_dir / f"subject_identity_test_LD12_M{m}.png", dpi=220)
            plt.close()

    write_csv(tables_dir / "subject_identity_summary_by_svd_dim.csv", summary_rows)
    write_csv(tables_dir / "subject_identity_per_subject.csv", subject_rows)
    if summary_rows:
        # Reuse expected fields by mapping test_bacc to balanced_acc_mean.
        rows_for_plot = []
        for r in summary_rows:
            rr = dict(r)
            rr["balanced_acc_mean"] = rr["test_balanced_acc"]
            rr["balanced_acc_sem"] = 0.0
            rows_for_plot.append(rr)
        plot_summary_bacc(rows_for_plot, plots_dir / "subject_identity_bacc_vs_svd_dim.png",
                          title=f"{args.model_name} subject identity control", chance=1.0 / len(subject_classes))

    del R_train_full, R_test_full, Vt, S
    gc.collect()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def run(args) -> None:
    outdir = Path(args.outdir)
    model_dir = outdir / args.model_name
    ensure_dir(model_dir)

    include_labels = parse_int_list(args.include_labels)
    if not include_labels:
        raise ValueError("--include-labels must be non-empty for this suite, e.g. 0,1,2")

    print(f"[{timestamp()}] Opening H5: {args.h5}", flush=True)
    with h5py.File(args.h5, "r") as f:
        if args.embedding_key not in f:
            raise KeyError(f"Embedding key not found: {args.embedding_key}")
        emb_ds = f[args.embedding_key]
        N_total = emb_ds.shape[0]
        y_all = read_h5_vector(f, args.label_key)
        subj_all = read_h5_vector(f, args.subject_key)
        if len(y_all) != N_total or len(subj_all) != N_total:
            raise ValueError(f"Length mismatch: embedding N={N_total}, label len={len(y_all)}, subject len={len(subj_all)}")
        mask = np.isin(y_all, include_labels)
        selected_idx = np.flatnonzero(mask)
        y_sel = y_all[selected_idx].astype(np.int64)
        subj_sel = subj_all[selected_idx].astype(np.int64)
        classes = sorted([int(c) for c in include_labels])
        class_names = infer_class_names(args.label_key, classes)

        global_meta = {
            "script": os.path.basename(__file__),
            "created_at": timestamp(),
            "model_name": args.model_name,
            "h5": args.h5,
            "embedding_key": args.embedding_key,
            "label_key": args.label_key,
            "subject_key": args.subject_key,
            "include_labels": classes,
            "class_names": class_names,
            "n_total_windows": int(N_total),
            "n_selected_windows": int(len(selected_idx)),
            "embedding_shape": tuple(int(x) for x in emb_ds.shape),
            "flat_dim": int(get_flat_dim(emb_ds)),
            "args": vars(args),
        }
        for k, v in f.attrs.items():
            try:
                global_meta[f"h5_attr/{k}"] = decode_attr(v)
            except Exception:
                pass
        with (model_dir / "run_metadata.json").open("w", encoding="utf-8") as fp:
            json.dump(global_meta, fp, indent=2, ensure_ascii=False)

        print("=" * 100, flush=True)
        print(f"Model: {args.model_name}", flush=True)
        print(f"Embedding shape: {emb_ds.shape}, flat_dim={get_flat_dim(emb_ds)}", flush=True)
        print(f"State classes: {classes} ({class_names})", flush=True)
        print(f"Selected samples: {len(selected_idx)} / {N_total}", flush=True)
        print(f"Subjects: {len(np.unique(subj_sel))} -> {sorted(np.unique(subj_sel).astype(int).tolist())}", flush=True)
        print("=" * 100, flush=True)

        if not args.skip_state_probe:
            run_state_probe(args, f, emb_ds, y_all, subj_all, selected_idx, y_sel, subj_sel, classes, class_names, model_dir)
        if not args.skip_subject_control:
            run_subject_identity_control(args, f, emb_ds, y_all, subj_all, selected_idx, y_sel, subj_sel, model_dir)

    print(f"[{timestamp()}] Done. Output dir: {model_dir}", flush=True)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Experiment A discriminant analysis suite")
    p.add_argument("--h5", required=True, help="Path to H5 embedding file")
    p.add_argument("--embedding-key", default="embedding")
    p.add_argument("--label-key", default="task_id", help="State label key; CogBCI global 3-class uses task_id")
    p.add_argument("--subject-key", default="subject_id")
    p.add_argument("--include-labels", default="0,1,2")
    p.add_argument("--model-name", required=True)
    p.add_argument("--outdir", required=True, help="Root output directory. A model subdirectory is created inside.")

    # State probe CV.
    p.add_argument("--cv", default="subject_kfold", choices=["subject_kfold", "loso"])
    p.add_argument("--n-subject-folds", type=int, default=4)
    p.add_argument("--folds", default="", help="Optional comma-separated state-probe fold IDs, e.g. 1")
    p.add_argument("--heldout-subject-groups", default="", help="Manual heldout subject groups, e.g. '1,2;3,4'")
    p.add_argument("--subjects", default="", help="Optional candidate subjects")

    # Dimensions and numerics.
    p.add_argument("--svd-dims", default="100,200,300,500")
    p.add_argument("--plot-svd-dims", default="500", help="SVD dims for detailed plots/points")
    p.add_argument("--svd-oversamples", type=int, default=20)
    p.add_argument("--svd-power-iter", type=int, default=2)
    p.add_argument("--lda-ridge", type=float, default=1e-4)
    p.add_argument("--n-shuffle", type=int, default=0, help="Train-label permutation nulls for state probe. 0 disables.")
    p.add_argument("--read-batch-size", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)

    # Plot and point saving.
    p.add_argument("--max-plot-points", type=int, default=8000)
    p.add_argument("--max-plot-points-per-subject", type=int, default=4000)
    p.add_argument("--plot-each-subject", action="store_true", help="Save one held-out LD map per subject for plot dims")
    p.add_argument("--save-test-points", action="store_true", help="Save held-out point-level LD coords/distances for plot dims")
    p.add_argument("--save-train-points", action="store_true", help="Save train point-level LD coords/distances for plot dims")

    # Mechanism diagnostics.
    p.add_argument("--self-readout-test-fraction", type=float, default=0.25, help="Within-subject oracle self-readout test fraction")

    # Subject identity control.
    p.add_argument("--skip-state-probe", action="store_true")
    p.add_argument("--skip-subject-control", action="store_true")
    p.add_argument("--subject-control-test-fraction", type=float, default=0.25)
    p.add_argument("--control-svd-dims", default="", help="Defaults to --svd-dims")
    p.add_argument("--control-plot-svd-dims", default="", help="Defaults to --plot-svd-dims")
    return p


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    run(args)
