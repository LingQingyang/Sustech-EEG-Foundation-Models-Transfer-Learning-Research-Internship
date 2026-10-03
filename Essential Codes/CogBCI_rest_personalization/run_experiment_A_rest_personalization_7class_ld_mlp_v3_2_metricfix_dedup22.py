#!/usr/bin/env python3
"""
Experiment A upgrade v3: seven-state subject-heldout diagnosis and rest-driven
personalization in the six-dimensional Fisher/LDA space.

Scientific question
-------------------
CogBCI contains four resting-state conditions, three N-Back conditions, and
three MATB conditions.  This script merges the four Rest conditions into one
class and keeps the six cognitive conditions separate:

    class 0: Rest (raw labels 0,1,2,3 merged)
    class 1: N-Back zeroBACK   (raw label 10)
    class 2: N-Back oneBACK    (raw label 11)
    class 3: N-Back twoBACK    (raw label 12)
    class 4: MATB easy         (raw label 20)
    class 5: MATB medium       (raw label 21)
    class 6: MATB difficult    (raw label 22)

Seven classes imply at most six Fisher discriminant dimensions.  For every
subject-heldout fold, the train subjects alone fit:

    raw embedding -> train mean -> randomized SVD -> seven-class LDA (6-D)

Each held-out subject is then evaluated with the following readouts:

    cross                 No adaptation.
    all_state_centered     Label-free transductive centering using all held-out
                           windows.  Diagnostic only.
    rest_centered          Mean alignment estimated only from held-out Rest.
    rest_coral             Regularized mean/covariance alignment estimated only
                           from held-out Rest.
    rest_mlp               One small residual 6->H1->H2->6 MLP per held-out
                           subject, fitted only from that subject's Rest.  The two
                           nonlinear layers deliberately use different activation
                           functions (default: Tanh then GELU).
    oracle_procrustes      Label-using similarity transform based on seven class
                           centroids.  Oracle diagnostic.
    self_readout           Within-subject seven-class LDA fitted in the shared
                           six-dimensional global Fisher space. Label-using diagnostic,
                           evaluated on a held-out within-subject split.

The Rest-MLP is trained without pointwise cross-subject pairing.  It aligns the
held-out Rest distribution to a balanced train-subject Rest reference using a
sliced-Wasserstein loss, while residual-displacement and canonical-anchor losses
prevent arbitrary extrapolation away from Rest.  Once fitted, the same frozen
mapping is applied to all six cognitive task conditions.

Primary outcome
---------------
The primary personalization metric is task6 balanced accuracy: balanced
accuracy on the six cognitive classes only, while predictions are still allowed
to be any of the seven classes.  This prevents an easy gain in Rest recall from
masquerading as successful transfer to cognitive tasks.  Complete seven-class
balanced accuracy and Rest recall are reported separately.

Identity controls
-----------------
Optional closed-set subject-ID probes are run for Rest-only, Task-only, and all
states.  Chance is 1 / number_of_subjects.  Within each subject and raw condition,
windows are split between train and test so state composition is preserved.

Dependencies
------------
Python 3.9+, numpy, scipy, h5py, matplotlib. PyTorch is optional.
If PyTorch cannot be imported, the tiny residual MLP automatically uses a
pure-NumPy AdamW backend. No scikit-learn dependency is required.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import math
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
except Exception as exc:  # pragma: no cover - handled explicitly at runtime
    torch = None
    nn = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None


SCRIPT_VERSION = "3.2-lda-metricfix-dedup22-selfreadout-ld6"


COGBCI_5S_MODEL_PATHS: Dict[str, str] = {
    "BIOT": "/mnt/dataset4/fuxy/FMS/FM/BIOT/output_emb_5s/cogbci_sesS1_embeddings.h5",
    "LaBraM": "/mnt/dataset4/fuxy/FMS/FM/LaBraM/output_emb_5s/cogbci_sesS1_embeddings.h5",
    "CBraMod": "/mnt/dataset4/fuxy/FMS/FM/CBraMod/output_emb_5s/cogbci_sesS1_embeddings.h5",
    "EEGPT": "/mnt/dataset4/fuxy/FMS/FM/EEGPT/output_emb_5s/cogbci_sesS1_embeddings.h5",
    "EEGMamba": "/mnt/dataset4/fuxy/FMS/FM/EEGMamba/output_emb_5s/cogbci_sesS1_embeddings.h5",
    "s-JEPA": "/mnt/dataset4/fuxy/FMS/FM/s-JEPA/output_5s/cogbci_sesS1_embeddings.h5",
}

METHODS = [
    "cross",
    "all_state_centered",
    "rest_centered",
    "rest_coral",
    "rest_mlp",
    "oracle_procrustes",
]

METHOD_TITLES = {
    "cross": "Cross-subject",
    "all_state_centered": "All-state centered",
    "rest_centered": "Rest-centered",
    "rest_coral": "Rest-CORAL",
    "rest_mlp": "Rest-MLP",
    "oracle_procrustes": "Oracle Procrustes",
    "self_readout": "Self-readout",
}

DEFAULT_CLASS_NAMES = {
    0: "Rest",
    1: "NBack-0",
    2: "NBack-1",
    3: "NBack-2",
    4: "MATB-easy",
    5: "MATB-medium",
    6: "MATB-difficult",
}


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def parse_int_list(s: str | None) -> List[int]:
    if s is None or not str(s).strip():
        return []
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def parse_str_list(s: str | None) -> List[str]:
    if s is None or not str(s).strip():
        return []
    return [x.strip() for x in str(s).split(",") if x.strip()]


def parse_subject_groups(s: str | None) -> List[List[int]]:
    if s is None or not str(s).strip():
        return []
    groups: List[List[int]] = []
    for part in str(s).split(";"):
        vals = parse_int_list(part)
        if vals:
            groups.append(sorted(vals))
    return groups


def groups_to_cli(groups: Sequence[Sequence[int]]) -> str:
    return ";".join(",".join(str(int(x)) for x in group) for group in groups)


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
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError("indices must be one-dimensional")
    if len(indices) == 0:
        raise ValueError("cannot read zero embedding rows")
    if np.any(np.diff(indices) < 0):
        raise ValueError("indices must be sorted for reliable h5py fancy indexing")

    n = len(indices)
    d = get_flat_dim(ds)
    out = np.empty((n, d), dtype=dtype)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        idx = indices[start:end]
        chunk = np.asarray(ds[idx])
        out[start:end] = chunk.reshape(len(idx), d).astype(dtype, copy=False)
    return out


def write_csv(path: str | Path, rows: List[Dict]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        print(f"[{timestamp()}] [WARN] no rows for {path}", flush=True)
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[{timestamp()}] Saved: {path}", flush=True)


def safe_json(x) -> str:
    return json.dumps(x, ensure_ascii=False)


def nanmean(x: Iterable[float]) -> float:
    arr = np.asarray(list(x), dtype=np.float64)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def nanstd(x: Iterable[float], ddof: int = 1) -> float:
    arr = np.asarray(list(x), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= ddof:
        return 0.0
    return float(np.std(arr, ddof=ddof))


def sem(x: Iterable[float]) -> float:
    arr = np.asarray(list(x), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / math.sqrt(arr.size))


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------


def balanced_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: Sequence[int],
) -> Tuple[float, Dict[int, float]]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    recalls: Dict[int, float] = {}
    for c in classes:
        c = int(c)
        mask = y_true == c
        recalls[c] = float(np.mean(y_pred[mask] == c)) if np.any(mask) else float("nan")
    vals = [v for v in recalls.values() if np.isfinite(v)]
    return (float(np.mean(vals)) if vals else float("nan")), recalls


def confusion_matrix_fixed(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: Sequence[int],
) -> np.ndarray:
    class_to_idx = {int(c): i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for yt, yp in zip(y_true, y_pred):
        i = class_to_idx.get(int(yt))
        j = class_to_idx.get(int(yp))
        if i is not None and j is not None:
            cm[i, j] += 1
    return cm


def binary_auc_rank(y_binary: np.ndarray, scores: np.ndarray) -> float:
    y_binary = np.asarray(y_binary).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int(np.sum(y_binary))
    n_neg = int(len(y_binary) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * ((i + 1) + j)
        i = j
    rank_sum = float(np.sum(ranks[y_binary]))
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def macro_auc_from_distances(
    y_true: np.ndarray,
    dist: np.ndarray,
    classes: Sequence[int],
    distance_class_order: Sequence[int],
) -> Tuple[float, Dict[int, float]]:
    order = {int(c): j for j, c in enumerate(distance_class_order)}
    aucs: Dict[int, float] = {}
    for c in classes:
        c = int(c)
        if c not in order:
            aucs[c] = float("nan")
        else:
            aucs[c] = binary_auc_rank(np.asarray(y_true) == c, -dist[:, order[c]])
    vals = [v for v in aucs.values() if np.isfinite(v)]
    return (float(np.mean(vals)) if vals else float("nan")), aucs


def signed_margins_from_distances(
    y_true: np.ndarray,
    dist: np.ndarray,
    classes: Sequence[int],
) -> np.ndarray:
    class_to_idx = {int(c): i for i, c in enumerate(classes)}
    margins = np.full(len(y_true), np.nan, dtype=np.float64)
    for i, yt in enumerate(y_true):
        j = class_to_idx.get(int(yt))
        if j is None:
            continue
        true_d = dist[i, j]
        other = np.delete(dist[i], j)
        if other.size:
            margins[i] = float(np.min(other) - true_d)
    return margins


def evaluate_seven_class(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dist: np.ndarray,
    classes: Sequence[int],
    rest_class: int,
) -> Dict:
    classes = [int(c) for c in classes]
    task_classes = [c for c in classes if c != int(rest_class)]
    acc = float(np.mean(y_true == y_pred))
    bacc7, recalls7 = balanced_accuracy(y_true, y_pred, classes)
    auc7, auc_by_class = macro_auc_from_distances(y_true, dist, classes, classes)

    task_mask = np.isin(y_true, task_classes)
    task_acc = float(np.mean(y_pred[task_mask] == y_true[task_mask])) if np.any(task_mask) else float("nan")
    task_bacc, task_recalls = balanced_accuracy(y_true[task_mask], y_pred[task_mask], task_classes)
    task_auc, task_auc_by_class = macro_auc_from_distances(
        y_true[task_mask], dist[task_mask], task_classes, classes
    ) if np.any(task_mask) else (float("nan"), {})

    rest_mask = y_true == int(rest_class)
    rest_recall = float(np.mean(y_pred[rest_mask] == int(rest_class))) if np.any(rest_mask) else float("nan")
    cm = confusion_matrix_fixed(y_true, y_pred, classes)
    margins = signed_margins_from_distances(y_true, dist, classes)
    return {
        "acc_all7": acc,
        "bacc_all7": bacc7,
        "macro_auc_all7": auc7,
        "class_recalls_all7": recalls7,
        "auc_by_class_all7": auc_by_class,
        "acc_task6": task_acc,
        "bacc_task6": task_bacc,
        "macro_auc_task6": task_auc,
        "class_recalls_task6": task_recalls,
        "auc_by_class_task6": task_auc_by_class,
        "rest_recall": rest_recall,
        "confusion_matrix": cm,
        "mean_margin": float(np.nanmean(margins)) if np.any(np.isfinite(margins)) else float("nan"),
        "median_margin": float(np.nanmedian(margins)) if np.any(np.isfinite(margins)) else float("nan"),
        "positive_margin_fraction": float(np.mean(margins > 0)) if np.any(np.isfinite(margins)) else float("nan"),
    }


# -----------------------------------------------------------------------------
# Randomized SVD and Fisher/LDA
# -----------------------------------------------------------------------------


def randomized_svd_dense(
    X: np.ndarray,
    n_components: int,
    n_oversamples: int = 20,
    n_iter: int = 2,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Randomized truncated SVD for an already-centered dense matrix."""
    n, d = X.shape
    k = int(min(n_components, n, d))
    if k <= 0:
        raise ValueError(f"Invalid n_components={n_components} for X shape={X.shape}")

    if k >= min(n, d) - 1 and min(n, d) <= 512:
        U, S, Vt = np.linalg.svd(X.astype(np.float32, copy=False), full_matrices=False)
        return U[:, :k], S[:k], Vt[:k]

    rng = np.random.default_rng(seed)
    l = int(min(k + n_oversamples, d))
    omega = rng.standard_normal((d, l)).astype(np.float32)
    Y = X @ omega
    for _ in range(max(0, int(n_iter))):
        Y = X @ (X.T @ Y)
    Q, _ = np.linalg.qr(Y, mode="reduced")
    Q = Q.astype(np.float32, copy=False)
    B = Q.T @ X
    Ub, S, Vt = np.linalg.svd(B.astype(np.float32, copy=False), full_matrices=False)
    U = Q @ Ub[:, :k]
    return U[:, :k], S[:k], Vt[:k]


@dataclass
class LDAResult:
    eigvals: np.ndarray
    W: np.ndarray
    class_centroids_ld: Dict[int, np.ndarray]
    train_ld: np.ndarray
    within_metric_diag_min: float
    within_metric_diag_max: float
    within_metric_diag_ratio: float
    within_metric_max_offdiag: float


def fit_multiclass_lda(
    R: np.ndarray,
    y: np.ndarray,
    classes: Sequence[int],
    ridge: float = 1e-4,
) -> LDAResult:
    R = np.asarray(R, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    classes = [int(c) for c in classes]
    n, m = R.shape
    q = min(len(classes) - 1, m)
    if q <= 0:
        raise ValueError("Need at least two classes and one feature dimension")

    # Equal-class Fisher geometry.  Rest merges four raw conditions and therefore
    # contains roughly four times as many windows as each cognitive class.  Using
    # empirical sample weights would let the merged Rest class dominate both Sw
    # and Sb.  We instead average class covariances and class-mean scatter with
    # equal class weights, matching the balanced-accuracy target.
    class_means: Dict[int, np.ndarray] = {}
    class_covariances: Dict[int, np.ndarray] = {}
    for c in classes:
        Rc = R[y == c]
        if len(Rc) == 0:
            raise ValueError(f"Class {c} has no training samples")
        muc = Rc.mean(axis=0)
        centered = Rc - muc
        class_means[c] = muc
        class_covariances[c] = (centered.T @ centered) / max(len(Rc) - 1, 1)

    mu = np.mean(np.stack([class_means[c] for c in classes], axis=0), axis=0)
    Sw = np.mean(np.stack([class_covariances[c] for c in classes], axis=0), axis=0)
    Sb = np.zeros((m, m), dtype=np.float64)
    for c in classes:
        dm = (class_means[c] - mu).reshape(-1, 1)
        Sb += dm @ dm.T
    Sb /= max(len(classes) - 1, 1)
    scale = float(np.trace(Sw) / max(m, 1))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    Sw_reg = Sw + float(ridge) * scale * np.eye(m, dtype=np.float64)

    try:
        # scipy.linalg.eigh solves Sb w = lambda Sw_reg w and returns a
        # Sw_reg-orthonormal basis: W.T @ Sw_reg @ W = I.
        from scipy.linalg import eigh
        eigvals, eigvecs = eigh(Sb, Sw_reg, check_finite=False)
    except Exception:
        # Dependency-free symmetric fallback.  Do NOT use eig(inv(Sw_reg) @ Sb):
        # although it has the same eigenvalues, the matrix is not Euclidean
        # symmetric and its returned eigenvectors need not be numerically
        # Sw_reg-orthogonal.  Cholesky whitening preserves the symmetric problem.
        L = np.linalg.cholesky(Sw_reg)
        tmp = np.linalg.solve(L, Sb)
        whitened = np.linalg.solve(L, tmp.T).T  # L^{-1} Sb L^{-T}
        whitened = 0.5 * (whitened + whitened.T)
        eigvals, U = np.linalg.eigh(whitened)
        eigvecs = np.linalg.solve(L.T, U)

    order = np.argsort(np.asarray(eigvals))[::-1]
    eigvals = np.maximum(np.asarray(eigvals, dtype=np.float64)[order], 0.0)
    W = np.asarray(eigvecs, dtype=np.float64)[:, order[:q]]

    # Preserve the within-class whitening metric.  scipy.linalg.eigh already
    # supplies this normalization, while the explicit step also removes small
    # numerical drift and covers the Cholesky fallback.  Unit Euclidean-norm
    # normalization is incorrect here because it makes the LD axes anisotropic
    # under Euclidean nearest-centroid, SWD, CORAL and MLP losses.
    for j in range(W.shape[1]):
        s2 = float(W[:, j] @ Sw_reg @ W[:, j])
        if not np.isfinite(s2) or s2 <= 0:
            raise FloatingPointError(f"Invalid generalized-eigenvector Sw norm: {s2}")
        W[:, j] /= math.sqrt(s2)

    metric_gram = 0.5 * (W.T @ Sw_reg @ W + (W.T @ Sw_reg @ W).T)
    metric_diag = np.diag(metric_gram)
    offdiag = metric_gram - np.diag(metric_diag)
    diag_min = float(np.min(metric_diag))
    diag_max = float(np.max(metric_diag))
    diag_ratio = float(diag_max / max(diag_min, 1e-15))
    max_offdiag = float(np.max(np.abs(offdiag))) if offdiag.size else 0.0

    train_ld = R @ W
    centroids = {c: train_ld[y == c].mean(axis=0) for c in classes}
    return LDAResult(
        eigvals=eigvals,
        W=W,
        class_centroids_ld=centroids,
        train_ld=train_ld,
        within_metric_diag_min=diag_min,
        within_metric_diag_max=diag_max,
        within_metric_diag_ratio=diag_ratio,
        within_metric_max_offdiag=max_offdiag,
    )


def distances_to_centroids(
    Z: np.ndarray,
    centroids: Dict[int, np.ndarray],
    classes: Sequence[int],
) -> np.ndarray:
    C = np.stack([np.asarray(centroids[int(c)], dtype=np.float64) for c in classes], axis=0)
    Z = np.asarray(Z, dtype=np.float64)
    return np.sum((Z[:, None, :] - C[None, :, :]) ** 2, axis=2)


def predict_nearest_centroid(
    Z: np.ndarray,
    centroids: Dict[int, np.ndarray],
    classes: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    classes = [int(c) for c in classes]
    dist = distances_to_centroids(Z, centroids, classes)
    pred = np.asarray([classes[i] for i in np.argmin(dist, axis=1)], dtype=np.int64)
    return pred, dist


# -----------------------------------------------------------------------------
# Label construction and folds
# -----------------------------------------------------------------------------


def construct_seven_class_labels(
    raw_label: np.ndarray,
    task_id: np.ndarray,
    rest_task_value: int,
    nback_task_value: int,
    matb_task_value: int,
    expected_rest_raw: Sequence[int],
    expected_nback_raw: Sequence[int],
    expected_matb_raw: Sequence[int],
) -> Tuple[np.ndarray, Dict[int, str], Dict]:
    raw_label = np.asarray(raw_label, dtype=np.int64)
    task_id = np.asarray(task_id, dtype=np.int64)
    y7 = np.full(len(raw_label), -1, dtype=np.int64)

    rest_raw_observed = sorted(int(x) for x in np.unique(raw_label[task_id == int(rest_task_value)]))
    nback_raw_observed = sorted(int(x) for x in np.unique(raw_label[task_id == int(nback_task_value)]))
    matb_raw_observed = sorted(int(x) for x in np.unique(raw_label[task_id == int(matb_task_value)]))

    expected_rest_raw = sorted(int(x) for x in expected_rest_raw)
    expected_nback_raw = sorted(int(x) for x in expected_nback_raw)
    expected_matb_raw = sorted(int(x) for x in expected_matb_raw)

    if rest_raw_observed != expected_rest_raw:
        raise ValueError(
            f"Rest raw labels mismatch: observed={rest_raw_observed}, expected={expected_rest_raw}. "
            "Check --label-key/--task-key or override --rest-raw-labels."
        )
    if nback_raw_observed != expected_nback_raw:
        raise ValueError(
            f"N-Back raw labels mismatch: observed={nback_raw_observed}, expected={expected_nback_raw}. "
            "Check --nback-task-value or --nback-raw-labels."
        )
    if matb_raw_observed != expected_matb_raw:
        raise ValueError(
            f"MATB raw labels mismatch: observed={matb_raw_observed}, expected={expected_matb_raw}. "
            "Check --matb-task-value or --matb-raw-labels."
        )

    y7[task_id == int(rest_task_value)] = 0
    for j, raw in enumerate(expected_nback_raw, start=1):
        y7[(task_id == int(nback_task_value)) & (raw_label == raw)] = j
    for j, raw in enumerate(expected_matb_raw, start=4):
        y7[(task_id == int(matb_task_value)) & (raw_label == raw)] = j

    selected = y7 >= 0
    if sorted(np.unique(y7[selected]).astype(int).tolist()) != list(range(7)):
        raise ValueError(f"Seven-class construction failed; observed class IDs={sorted(np.unique(y7[selected]).tolist())}")

    meta = {
        "rest_raw_observed": rest_raw_observed,
        "nback_raw_observed": nback_raw_observed,
        "matb_raw_observed": matb_raw_observed,
        "seven_class_mapping": {
            "rest_raw_to_class0": expected_rest_raw,
            **{str(raw): 1 + i for i, raw in enumerate(expected_nback_raw)},
            **{str(raw): 4 + i for i, raw in enumerate(expected_matb_raw)},
        },
    }
    return y7, dict(DEFAULT_CLASS_NAMES), meta


def make_subject_kfold_groups(
    subjects: Sequence[int],
    subj: np.ndarray,
    n_folds: int,
    seed: int,
) -> List[List[int]]:
    subjects = [int(s) for s in subjects]
    if n_folds < 2 or n_folds > len(subjects):
        raise ValueError(f"Invalid n_folds={n_folds} for {len(subjects)} subjects")
    counts = {s: int(np.sum(subj == s)) for s in subjects}
    rng = np.random.default_rng(seed)
    shuffled = list(subjects)
    rng.shuffle(shuffled)
    ordered = sorted(shuffled, key=lambda s: counts[s], reverse=True)
    groups: List[List[int]] = [[] for _ in range(n_folds)]
    totals = [0] * n_folds
    for s in ordered:
        j = int(np.argmin(totals))
        groups[j].append(s)
        totals[j] += counts[s]
    groups = [sorted(g) for g in groups]
    return sorted(groups, key=lambda g: (min(g), len(g)))


def build_subject_folds(
    args: argparse.Namespace,
    subjects: List[int],
    subj_sel: np.ndarray,
) -> List[Dict]:
    manual = parse_subject_groups(args.heldout_subject_groups)
    if manual:
        allowed = set(subjects)
        groups = [sorted([s for s in group if s in allowed]) for group in manual]
        groups = [group for group in groups if group]
        cv_name = "manual_subject_groups"
    elif args.cv == "loso":
        groups = [[s] for s in subjects]
        cv_name = "loso"
    else:
        groups = make_subject_kfold_groups(subjects, subj_sel, args.n_subject_folds, args.seed)
        cv_name = f"subject_{args.n_subject_folds}fold"

    selected_fold_ids = set(parse_int_list(args.folds)) if args.folds else None
    specs: List[Dict] = []
    for fold_id, group in enumerate(groups, start=1):
        if selected_fold_ids is not None and fold_id not in selected_fold_ids:
            continue
        mask = np.isin(subj_sel, np.asarray(group, dtype=np.int64))
        specs.append({
            "fold_id": int(fold_id),
            "cv": cv_name,
            "heldout_subjects": [int(x) for x in group],
            "n_heldout_subjects": int(len(group)),
            "estimated_n_test": int(np.sum(mask)),
            "estimated_test_fraction": float(np.mean(mask)),
        })
    if not specs:
        raise ValueError("No folds selected")
    return specs


def stratified_within_subject_split(
    subject: np.ndarray,
    strata: np.ndarray,
    test_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts: List[np.ndarray] = []
    test_parts: List[np.ndarray] = []
    for s in sorted(np.unique(subject).astype(int)):
        for g in sorted(np.unique(strata[subject == s]).astype(int)):
            idx = np.flatnonzero((subject == s) & (strata == g))
            if len(idx) <= 1:
                train_parts.append(idx)
                continue
            perm = np.array(idx, copy=True)
            rng.shuffle(perm)
            n_test = max(1, min(int(round(len(perm) * test_fraction)), len(perm) - 1))
            test_parts.append(np.sort(perm[:n_test]))
            train_parts.append(np.sort(perm[n_test:]))
    train_idx = np.sort(np.concatenate(train_parts)) if train_parts else np.array([], dtype=np.int64)
    test_idx = np.sort(np.concatenate(test_parts)) if test_parts else np.array([], dtype=np.int64)
    return train_idx, test_idx


def stratified_label_split(
    y: np.ndarray,
    test_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts: List[np.ndarray] = []
    test_parts: List[np.ndarray] = []
    for c in sorted(np.unique(y).astype(int)):
        idx = np.flatnonzero(y == c)
        if len(idx) <= 1:
            train_parts.append(idx)
            continue
        perm = np.array(idx, copy=True)
        rng.shuffle(perm)
        n_test = max(1, min(int(round(len(perm) * test_fraction)), len(perm) - 1))
        test_parts.append(np.sort(perm[:n_test]))
        train_parts.append(np.sort(perm[n_test:]))
    return (
        np.sort(np.concatenate(train_parts)) if train_parts else np.array([], dtype=np.int64),
        np.sort(np.concatenate(test_parts)) if test_parts else np.array([], dtype=np.int64),
    )


# -----------------------------------------------------------------------------
# Balanced pools
# -----------------------------------------------------------------------------


def balanced_indices_by_cells(
    cell_arrays: Sequence[np.ndarray],
    max_per_cell: int,
    seed: int,
) -> np.ndarray:
    nonempty = [np.asarray(x, dtype=np.int64) for x in cell_arrays if len(x) > 0]
    if not nonempty:
        return np.array([], dtype=np.int64)
    n_per = min(min(len(x) for x in nonempty), int(max_per_cell) if max_per_cell > 0 else 10**12)
    rng = np.random.default_rng(seed)
    chosen: List[np.ndarray] = []
    for idx in nonempty:
        if len(idx) > n_per:
            chosen.append(np.sort(rng.choice(idx, size=n_per, replace=False)))
        else:
            chosen.append(np.sort(idx))
    return np.sort(np.concatenate(chosen))


def build_balanced_rest_indices(
    subject: np.ndarray,
    raw_label: np.ndarray,
    y7: np.ndarray,
    rest_class: int,
    rest_raw_labels: Sequence[int],
    max_per_cell: int,
    seed: int,
) -> np.ndarray:
    cells: List[np.ndarray] = []
    for s in sorted(np.unique(subject[y7 == int(rest_class)]).astype(int)):
        for raw in rest_raw_labels:
            cells.append(np.flatnonzero((subject == s) & (raw_label == int(raw)) & (y7 == int(rest_class))))
    return balanced_indices_by_cells(cells, max_per_cell=max_per_cell, seed=seed)


def build_balanced_class_indices(
    y: np.ndarray,
    classes: Sequence[int],
    max_per_class: int,
    seed: int,
) -> np.ndarray:
    cells = [np.flatnonzero(y == int(c)) for c in classes]
    return balanced_indices_by_cells(cells, max_per_cell=max_per_class, seed=seed)


# -----------------------------------------------------------------------------
# Geometric transforms
# -----------------------------------------------------------------------------


def center_to_reference(Z: np.ndarray, source_mask: np.ndarray, target_mean: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if not np.any(source_mask):
        raise ValueError("Centering source mask is empty")
    source_mean = np.mean(Z[source_mask], axis=0)
    offset = np.asarray(target_mean) - source_mean
    return Z + offset, offset


def symmetric_matrix_power(C: np.ndarray, power: float, floor: float) -> np.ndarray:
    C = 0.5 * (np.asarray(C, dtype=np.float64) + np.asarray(C, dtype=np.float64).T)
    vals, vecs = np.linalg.eigh(C)
    vals = np.maximum(vals, floor)
    return (vecs * (vals ** power)[None, :]) @ vecs.T


def fit_coral(
    source_rest: np.ndarray,
    target_rest: np.ndarray,
    ridge: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_rest = np.asarray(source_rest, dtype=np.float64)
    target_rest = np.asarray(target_rest, dtype=np.float64)
    if len(source_rest) < 2 or len(target_rest) < 2:
        raise ValueError("CORAL needs at least two source and target Rest samples")
    mu_s = source_rest.mean(axis=0)
    mu_t = target_rest.mean(axis=0)
    Cs = np.cov(source_rest, rowvar=False, ddof=1)
    Ct = np.cov(target_rest, rowvar=False, ddof=1)
    q = source_rest.shape[1]
    scale_s = float(np.trace(Cs) / max(q, 1))
    scale_t = float(np.trace(Ct) / max(q, 1))
    scale_s = scale_s if np.isfinite(scale_s) and scale_s > 0 else 1.0
    scale_t = scale_t if np.isfinite(scale_t) and scale_t > 0 else 1.0
    Cs = Cs + float(ridge) * scale_s * np.eye(q)
    Ct = Ct + float(ridge) * scale_t * np.eye(q)
    floor = max(1e-12, float(ridge) * min(scale_s, scale_t))
    A = symmetric_matrix_power(Cs, -0.5, floor) @ symmetric_matrix_power(Ct, 0.5, floor)
    return A, mu_s, mu_t


def apply_coral(Z: np.ndarray, A: np.ndarray, mu_s: np.ndarray, mu_t: np.ndarray) -> np.ndarray:
    return (np.asarray(Z) - mu_s) @ A + mu_t


def fit_similarity_procrustes(
    source_centroids: np.ndarray,
    target_centroids: np.ndarray,
    allow_scale: bool = True,
) -> Tuple[np.ndarray, float, np.ndarray, Dict[str, float]]:
    X = np.asarray(source_centroids, dtype=np.float64)
    Y = np.asarray(target_centroids, dtype=np.float64)
    if X.shape != Y.shape or X.ndim != 2:
        raise ValueError(f"Procrustes centroid shape mismatch: X={X.shape}, Y={Y.shape}")
    mx = X.mean(axis=0)
    my = Y.mean(axis=0)
    Xc = X - mx
    Yc = Y - my
    U, _, Vt = np.linalg.svd(Xc.T @ Yc, full_matrices=False)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    denom = float(np.sum(Xc ** 2))
    scale = float(np.sum((Xc @ R) * Yc) / denom) if allow_scale and denom > 0 else 1.0
    b = my - scale * (mx @ R)
    before = float(np.mean(np.sum((X - Y) ** 2, axis=1)))
    after = float(np.mean(np.sum((scale * (X @ R) + b - Y) ** 2, axis=1)))
    return R, scale, b, {
        "centroid_mse_before": before,
        "centroid_mse_after": after,
        "scale": scale,
        "det_rotation": float(np.linalg.det(R)),
    }


def apply_similarity(Z: np.ndarray, R: np.ndarray, scale: float, b: np.ndarray) -> np.ndarray:
    return float(scale) * (np.asarray(Z) @ np.asarray(R)) + np.asarray(b)


# -----------------------------------------------------------------------------
# Rest residual MLP
# -----------------------------------------------------------------------------


def activation_factory(name: str):
    """Create a PyTorch activation when the torch backend is available."""
    if nn is None:
        raise RuntimeError(f"PyTorch is unavailable: {TORCH_IMPORT_ERROR}")
    key = name.strip().lower()
    options = {
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
        "elu": nn.ELU,
        "leaky_relu": lambda: nn.LeakyReLU(negative_slope=0.1),
    }
    if key not in options:
        raise ValueError(f"Unsupported activation {name!r}; choose from {sorted(options)}")
    factory = options[key]
    return factory() if isinstance(factory, type) else factory()


if nn is not None:
    class ResidualAdapter(nn.Module):
        def __init__(
            self,
            dim: int,
            hidden1: int,
            hidden2: int,
            activation1: str,
            activation2: str,
        ) -> None:
            super().__init__()
            if activation1.strip().lower() == activation2.strip().lower():
                raise ValueError(
                    "The two nonlinear layers must use different activation functions. "
                    f"Got activation1={activation1}, activation2={activation2}."
                )
            self.fc1 = nn.Linear(dim, hidden1)
            self.act1 = activation_factory(activation1)
            self.fc2 = nn.Linear(hidden1, hidden2)
            self.act2 = activation_factory(activation2)
            self.fc3 = nn.Linear(hidden2, dim)
            # Start exactly at the identity map through the residual connection.
            nn.init.zeros_(self.fc3.weight)
            nn.init.zeros_(self.fc3.bias)

        def forward(self, x):
            residual = self.fc3(self.act2(self.fc2(self.act1(self.fc1(x)))))
            return x + residual
else:
    # A placeholder keeps module import valid when PyTorch is absent.  The NumPy
    # backend below is selected automatically and this class is never instantiated.
    class ResidualAdapter:  # type: ignore[no-redef]
        pass


def _numpy_activation(x: np.ndarray, name: str) -> np.ndarray:
    key = name.strip().lower()
    if key == "tanh":
        return np.tanh(x)
    if key == "gelu":
        # Hendrycks-Gimpel tanh approximation.  Smooth, stable, dependency-free.
        c = math.sqrt(2.0 / math.pi)
        return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x ** 3)))
    if key == "relu":
        return np.maximum(x, 0.0)
    if key == "silu":
        s = 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))
        return x * s
    if key == "elu":
        return np.where(x > 0.0, x, np.expm1(np.clip(x, -60.0, 60.0)))
    if key == "leaky_relu":
        return np.where(x >= 0.0, x, 0.1 * x)
    raise ValueError(
        f"Unsupported activation {name!r}; choose from "
        "['elu', 'gelu', 'leaky_relu', 'relu', 'silu', 'tanh']"
    )


def _numpy_activation_grad(x: np.ndarray, name: str) -> np.ndarray:
    key = name.strip().lower()
    if key == "tanh":
        t = np.tanh(x)
        return 1.0 - t * t
    if key == "gelu":
        c = math.sqrt(2.0 / math.pi)
        u = c * (x + 0.044715 * x ** 3)
        t = np.tanh(u)
        du = c * (1.0 + 3.0 * 0.044715 * x ** 2)
        return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * du
    if key == "relu":
        return (x > 0.0).astype(np.float64)
    if key == "silu":
        s = 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))
        return s + x * s * (1.0 - s)
    if key == "elu":
        return np.where(x > 0.0, 1.0, np.exp(np.clip(x, -60.0, 60.0)))
    if key == "leaky_relu":
        return np.where(x >= 0.0, 1.0, 0.1)
    raise ValueError(f"Unsupported activation {name!r}")


class NumpyResidualAdapter:
    """Tiny residual 6->H1->H2->6 MLP with manual NumPy backpropagation."""

    backend = "numpy"

    def __init__(
        self,
        dim: int,
        hidden1: int,
        hidden2: int,
        activation1: str,
        activation2: str,
        seed: int,
    ) -> None:
        if activation1.strip().lower() == activation2.strip().lower():
            raise ValueError(
                "The two nonlinear layers must use different activation functions. "
                f"Got activation1={activation1}, activation2={activation2}."
            )
        self.dim = int(dim)
        self.hidden1 = int(hidden1)
        self.hidden2 = int(hidden2)
        self.activation1 = activation1.strip().lower()
        self.activation2 = activation2.strip().lower()
        rng = np.random.default_rng(seed)
        lim1 = math.sqrt(6.0 / (self.dim + self.hidden1))
        lim2 = math.sqrt(6.0 / (self.hidden1 + self.hidden2))
        self.params: Dict[str, np.ndarray] = {
            "W1": rng.uniform(-lim1, lim1, size=(self.dim, self.hidden1)).astype(np.float64),
            "b1": np.zeros(self.hidden1, dtype=np.float64),
            "W2": rng.uniform(-lim2, lim2, size=(self.hidden1, self.hidden2)).astype(np.float64),
            "b2": np.zeros(self.hidden2, dtype=np.float64),
            # Zero final layer makes the initial mapping exactly identity.
            "W3": np.zeros((self.hidden2, self.dim), dtype=np.float64),
            "b3": np.zeros(self.dim, dtype=np.float64),
        }

    def forward(self, x: np.ndarray, return_cache: bool = False):
        x = np.asarray(x, dtype=np.float64)
        z1 = x @ self.params["W1"] + self.params["b1"]
        a1 = _numpy_activation(z1, self.activation1)
        z2 = a1 @ self.params["W2"] + self.params["b2"]
        a2 = _numpy_activation(z2, self.activation2)
        residual = a2 @ self.params["W3"] + self.params["b3"]
        out = x + residual
        if return_cache:
            return out, (x, z1, a1, z2, a2)
        return out

    def backward(self, cache, grad_out: np.ndarray) -> Dict[str, np.ndarray]:
        x, z1, a1, z2, a2 = cache
        g = np.asarray(grad_out, dtype=np.float64)
        grads: Dict[str, np.ndarray] = {}
        grads["W3"] = a2.T @ g
        grads["b3"] = np.sum(g, axis=0)
        ga2 = g @ self.params["W3"].T
        gz2 = ga2 * _numpy_activation_grad(z2, self.activation2)
        grads["W2"] = a1.T @ gz2
        grads["b2"] = np.sum(gz2, axis=0)
        ga1 = gz2 @ self.params["W2"].T
        gz1 = ga1 * _numpy_activation_grad(z1, self.activation1)
        grads["W1"] = x.T @ gz1
        grads["b1"] = np.sum(gz1, axis=0)
        return grads

    def predict(self, x: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        out = np.empty_like(x, dtype=np.float64)
        for start in range(0, len(x), int(batch_size)):
            end = min(start + int(batch_size), len(x))
            out[start:end] = self.forward(x[start:end], return_cache=False)
        return out

    def get_state(self) -> Dict[str, np.ndarray]:
        return {k: np.array(v, copy=True) for k, v in self.params.items()}

    def set_state(self, state: Dict[str, np.ndarray]) -> None:
        for k in self.params:
            self.params[k][...] = state[k]


def sample_torch_rows(x: "torch.Tensor", n: int, generator: "torch.Generator") -> "torch.Tensor":
    if len(x) == 0:
        raise ValueError("Cannot sample from an empty tensor")
    idx = torch.randint(0, len(x), (int(n),), generator=generator, device=x.device)
    return x[idx]


def sliced_wasserstein_squared(
    x: "torch.Tensor",
    y: "torch.Tensor",
    n_projections: int,
    generator: "torch.Generator",
) -> "torch.Tensor":
    if x.shape != y.shape:
        raise ValueError(f"SWD batches must have equal shape; got {x.shape} and {y.shape}")
    q = x.shape[1]
    directions = torch.randn(
        q, int(n_projections), device=x.device, dtype=x.dtype, generator=generator
    )
    directions = directions / torch.clamp(torch.linalg.norm(directions, dim=0, keepdim=True), min=1e-12)
    px = torch.sort(x @ directions, dim=0).values
    py = torch.sort(y @ directions, dim=0).values
    return torch.mean((px - py) ** 2)


def numpy_swd_loss_and_grad(
    x: np.ndarray,
    y: np.ndarray,
    n_projections: int,
    rng: np.random.Generator,
) -> Tuple[float, np.ndarray]:
    """Squared sliced-Wasserstein loss and exact subgradient wrt x.

    Sorting is piecewise linear.  Away from ties, the gradient is obtained by
    differentiating sorted projected coordinates and undoing each permutation.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"SWD batches must have equal shape; got {x.shape} and {y.shape}")
    n, q = x.shape
    p = int(n_projections)
    directions = rng.standard_normal((q, p))
    directions /= np.maximum(np.linalg.norm(directions, axis=0, keepdims=True), 1e-12)
    proj_x = x @ directions
    proj_y = y @ directions
    order_x = np.argsort(proj_x, axis=0)
    order_y = np.argsort(proj_y, axis=0)
    sorted_x = np.take_along_axis(proj_x, order_x, axis=0)
    sorted_y = np.take_along_axis(proj_y, order_y, axis=0)
    diff = sorted_x - sorted_y
    loss = float(np.mean(diff ** 2))
    grad_sorted = 2.0 * diff / float(n * p)
    grad_proj = np.zeros_like(proj_x)
    for j in range(p):
        grad_proj[order_x[:, j], j] = grad_sorted[:, j]
    grad_x = grad_proj @ directions.T
    return loss, grad_x


def split_rest_train_val(
    raw_labels: np.ndarray,
    val_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if val_fraction <= 0:
        idx = np.arange(len(raw_labels), dtype=np.int64)
        return idx, idx
    rng = np.random.default_rng(seed)
    train_parts: List[np.ndarray] = []
    val_parts: List[np.ndarray] = []
    for raw in sorted(np.unique(raw_labels).astype(int)):
        idx = np.flatnonzero(raw_labels == raw)
        if len(idx) <= 2:
            train_parts.append(idx)
            val_parts.append(idx)
            continue
        perm = np.array(idx, copy=True)
        rng.shuffle(perm)
        n_val = max(1, min(int(round(len(perm) * val_fraction)), len(perm) - 1))
        val_parts.append(np.sort(perm[:n_val]))
        train_parts.append(np.sort(perm[n_val:]))
    return np.sort(np.concatenate(train_parts)), np.sort(np.concatenate(val_parts))


def resolve_torch_device(requested: str) -> str:
    if torch is None:
        raise RuntimeError(f"PyTorch import failed: {TORCH_IMPORT_ERROR}")
    requested = requested.lower().strip()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        warnings.warn("CUDA requested but unavailable; falling back to CPU")
        return "cpu"
    return requested


def resolve_mlp_backend(requested: str) -> str:
    key = requested.strip().lower()
    if key == "auto":
        return "torch" if torch is not None else "numpy"
    if key == "torch" and torch is None:
        raise RuntimeError(
            "--mlp-backend torch was requested, but PyTorch could not be imported. "
            f"Original import error: {type(TORCH_IMPORT_ERROR).__name__}: {TORCH_IMPORT_ERROR}"
        )
    if key not in {"torch", "numpy"}:
        raise ValueError(f"Unknown MLP backend: {requested}")
    return key


def numpy_swd_squared(
    x: np.ndarray,
    y: np.ndarray,
    n_projections: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    n = min(len(x), len(y))
    if n == 0:
        return float("nan")
    ix = rng.choice(len(x), size=n, replace=len(x) < n)
    iy = rng.choice(len(y), size=n, replace=len(y) < n)
    xx = np.asarray(x[ix], dtype=np.float64)
    yy = np.asarray(y[iy], dtype=np.float64)
    directions = rng.standard_normal((xx.shape[1], int(n_projections)))
    directions /= np.maximum(np.linalg.norm(directions, axis=0, keepdims=True), 1e-12)
    px = np.sort(xx @ directions, axis=0)
    py = np.sort(yy @ directions, axis=0)
    return float(np.mean((px - py) ** 2))


def train_rest_adapter_torch(
    source_rest: np.ndarray,
    source_rest_raw: np.ndarray,
    target_rest: np.ndarray,
    anchor: np.ndarray,
    args: argparse.Namespace,
    seed: int,
):
    if torch is None:
        raise RuntimeError(f"PyTorch is unavailable: {TORCH_IMPORT_ERROR}")
    device = resolve_torch_device(args.mlp_device)
    set_global_seed(seed)
    model = ResidualAdapter(
        dim=6,
        hidden1=args.mlp_hidden1,
        hidden2=args.mlp_hidden2,
        activation1=args.mlp_activation1,
        activation2=args.mlp_activation2,
    ).to(device)

    src = torch.as_tensor(source_rest, dtype=torch.float32, device=device)
    tgt = torch.as_tensor(target_rest, dtype=torch.float32, device=device)
    anc = torch.as_tensor(anchor, dtype=torch.float32, device=device)
    train_idx, val_idx = split_rest_train_val(source_rest_raw, args.mlp_val_fraction, seed + 11)
    src_train = src[torch.as_tensor(train_idx, dtype=torch.long, device=device)]
    src_val = src[torch.as_tensor(val_idx, dtype=torch.long, device=device)]

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.mlp_lr,
        weight_decay=args.mlp_weight_decay,
    )
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed + 101))
    val_gen = torch.Generator(device=device)
    val_gen.manual_seed(int(seed + 202))

    batch_size = int(max(2, args.mlp_batch_size))
    history: List[Dict] = []
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    best_epoch = 0
    bad_epochs = 0

    for epoch in range(1, int(args.mlp_epochs) + 1):
        model.train()
        xb = sample_torch_rows(src_train, batch_size, gen)
        yb = sample_torch_rows(tgt, batch_size, gen)
        ab = sample_torch_rows(anc, batch_size, gen)
        mapped = model(xb)
        align = sliced_wasserstein_squared(mapped, yb, args.mlp_swd_projections, gen)
        move = torch.mean((mapped - xb) ** 2)
        anchor_loss = torch.mean((model(ab) - ab) ** 2)
        loss = align + args.mlp_lambda_move * move + args.mlp_lambda_anchor * anchor_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.mlp_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.mlp_grad_clip)
        optimizer.step()

        do_eval = epoch == 1 or epoch % args.mlp_eval_every == 0 or epoch == args.mlp_epochs
        if do_eval:
            model.eval()
            with torch.no_grad():
                n_val = max(2, min(batch_size, max(len(src_val), 2)))
                xv = sample_torch_rows(src_val, n_val, val_gen)
                yv = sample_torch_rows(tgt, n_val, val_gen)
                av = sample_torch_rows(anc, n_val, val_gen)
                mapped_v = model(xv)
                val_align = sliced_wasserstein_squared(mapped_v, yv, args.mlp_swd_projections, val_gen)
                val_move = torch.mean((mapped_v - xv) ** 2)
                val_anchor = torch.mean((model(av) - av) ** 2)
                val_loss = val_align + args.mlp_lambda_move * val_move + args.mlp_lambda_anchor * val_anchor
                val_value = float(val_loss.detach().cpu())

            history.append({
                "epoch": int(epoch),
                "train_total": float(loss.detach().cpu()),
                "train_align": float(align.detach().cpu()),
                "train_move": float(move.detach().cpu()),
                "train_anchor": float(anchor_loss.detach().cpu()),
                "val_total": val_value,
                "val_align": float(val_align.detach().cpu()),
                "val_move": float(val_move.detach().cpu()),
                "val_anchor": float(val_anchor.detach().cpu()),
            })

            if val_value < best_val - args.mlp_min_delta:
                best_val = val_value
                best_epoch = int(epoch)
                best_state = copy.deepcopy(model.state_dict())
                bad_epochs = 0
            else:
                bad_epochs += 1
            if args.mlp_patience > 0 and bad_epochs >= args.mlp_patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        mapped_all = model(torch.as_tensor(source_rest, dtype=torch.float32, device=device)).cpu().numpy()
    rest_mean_displacement, rest_mean_input_norm, rest_relative_displacement = mean_relative_displacement(
        mapped_all, source_rest
    )
    rest_swd_before = numpy_swd_squared(
        source_rest, target_rest, args.mlp_swd_projections, seed + 303
    )
    rest_swd_after = numpy_swd_squared(
        mapped_all, target_rest, args.mlp_swd_projections, seed + 303
    )
    diagnostics = {
        "backend": "torch",
        "device": device,
        "activation1": args.mlp_activation1,
        "activation2": args.mlp_activation2,
        "hidden1": int(args.mlp_hidden1),
        "hidden2": int(args.mlp_hidden2),
        "n_source_rest": int(len(source_rest)),
        "n_target_rest": int(len(target_rest)),
        "n_anchor": int(len(anchor)),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "n_eval_records": int(len(history)),
        "rest_swd_before": rest_swd_before,
        "rest_swd_after": rest_swd_after,
        "rest_mean_displacement": rest_mean_displacement,
        "rest_mean_input_norm": rest_mean_input_norm,
        "rest_relative_displacement": rest_relative_displacement,
        "rest_rms_displacement": float(np.sqrt(np.mean((mapped_all - source_rest) ** 2))),
        "rest_swd_reduction_fraction": float(
            (rest_swd_before - rest_swd_after) / max(rest_swd_before, 1e-15)
        ),
    }
    return model, diagnostics, history


def mean_relative_displacement(mapped: np.ndarray, original: np.ndarray) -> Tuple[float, float, float]:
    mapped = np.asarray(mapped, dtype=np.float64)
    original = np.asarray(original, dtype=np.float64)
    if mapped.shape != original.shape or mapped.ndim != 2:
        raise ValueError(f"Displacement shape mismatch: mapped={mapped.shape}, original={original.shape}")
    delta_mean = float(np.mean(np.linalg.norm(mapped - original, axis=1)))
    input_mean = float(np.mean(np.linalg.norm(original, axis=1)))
    relative = float(delta_mean / input_mean) if input_mean > 0 else float("nan")
    return delta_mean, input_mean, relative


def _add_grads(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray], scale: float = 1.0) -> Dict[str, np.ndarray]:
    return {k: a[k] + float(scale) * b[k] for k in a}


def _clip_numpy_grads(grads: Dict[str, np.ndarray], max_norm: float) -> float:
    norm = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
    if max_norm > 0 and norm > max_norm:
        scale = max_norm / max(norm, 1e-12)
        for k in grads:
            grads[k] *= scale
    return norm


def _adamw_numpy_step(
    model: NumpyResidualAdapter,
    grads: Dict[str, np.ndarray],
    state: Dict[str, Dict[str, np.ndarray]],
    step: int,
    lr: float,
    weight_decay: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> None:
    for k, p in model.params.items():
        g = grads[k]
        state["m"][k] = beta1 * state["m"][k] + (1.0 - beta1) * g
        state["v"][k] = beta2 * state["v"][k] + (1.0 - beta2) * (g * g)
        m_hat = state["m"][k] / (1.0 - beta1 ** step)
        v_hat = state["v"][k] / (1.0 - beta2 ** step)
        if weight_decay > 0:
            p *= (1.0 - lr * weight_decay)
        p -= lr * m_hat / (np.sqrt(v_hat) + eps)


def train_rest_adapter_numpy(
    source_rest: np.ndarray,
    source_rest_raw: np.ndarray,
    target_rest: np.ndarray,
    anchor: np.ndarray,
    args: argparse.Namespace,
    seed: int,
):
    model = NumpyResidualAdapter(
        dim=6,
        hidden1=args.mlp_hidden1,
        hidden2=args.mlp_hidden2,
        activation1=args.mlp_activation1,
        activation2=args.mlp_activation2,
        seed=seed,
    )
    src = np.asarray(source_rest, dtype=np.float64)
    tgt = np.asarray(target_rest, dtype=np.float64)
    anc = np.asarray(anchor, dtype=np.float64)
    train_idx, val_idx = split_rest_train_val(source_rest_raw, args.mlp_val_fraction, seed + 11)
    src_train = src[train_idx]
    src_val = src[val_idx]
    batch_size = int(max(2, args.mlp_batch_size))
    train_rng = np.random.default_rng(seed + 101)
    val_rng = np.random.default_rng(seed + 202)

    opt_state = {
        "m": {k: np.zeros_like(v) for k, v in model.params.items()},
        "v": {k: np.zeros_like(v) for k, v in model.params.items()},
    }
    history: List[Dict] = []
    best_state = model.get_state()
    best_val = float("inf")
    best_epoch = 0
    bad_epochs = 0

    def sample_rows(x: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
        if len(x) == 0:
            raise ValueError("Cannot sample from an empty array")
        return x[rng.integers(0, len(x), size=int(n))]

    for epoch in range(1, int(args.mlp_epochs) + 1):
        xb = sample_rows(src_train, batch_size, train_rng)
        yb = sample_rows(tgt, batch_size, train_rng)
        ab = sample_rows(anc, batch_size, train_rng)

        mapped, cache_src = model.forward(xb, return_cache=True)
        align, grad_align = numpy_swd_loss_and_grad(
            mapped, yb, args.mlp_swd_projections, train_rng
        )
        move_diff = mapped - xb
        move = float(np.mean(move_diff ** 2))
        grad_move = 2.0 * move_diff / float(move_diff.size)
        grad_mapped = grad_align + float(args.mlp_lambda_move) * grad_move
        grads_src = model.backward(cache_src, grad_mapped)

        mapped_anchor, cache_anchor = model.forward(ab, return_cache=True)
        anchor_diff = mapped_anchor - ab
        anchor_loss = float(np.mean(anchor_diff ** 2))
        grad_anchor_out = 2.0 * anchor_diff / float(anchor_diff.size)
        grads_anchor = model.backward(cache_anchor, grad_anchor_out)
        grads = _add_grads(grads_src, grads_anchor, scale=args.mlp_lambda_anchor)
        _clip_numpy_grads(grads, args.mlp_grad_clip)
        _adamw_numpy_step(
            model, grads, opt_state, epoch,
            lr=float(args.mlp_lr),
            weight_decay=float(args.mlp_weight_decay),
        )
        train_total = align + args.mlp_lambda_move * move + args.mlp_lambda_anchor * anchor_loss

        do_eval = epoch == 1 or epoch % args.mlp_eval_every == 0 or epoch == args.mlp_epochs
        if do_eval:
            n_val = max(2, min(batch_size, max(len(src_val), 2)))
            xv = sample_rows(src_val, n_val, val_rng)
            yv = sample_rows(tgt, n_val, val_rng)
            av = sample_rows(anc, n_val, val_rng)
            mapped_v = model.forward(xv)
            val_align, _ = numpy_swd_loss_and_grad(
                mapped_v, yv, args.mlp_swd_projections, val_rng
            )
            val_move = float(np.mean((mapped_v - xv) ** 2))
            mapped_av = model.forward(av)
            val_anchor = float(np.mean((mapped_av - av) ** 2))
            val_value = val_align + args.mlp_lambda_move * val_move + args.mlp_lambda_anchor * val_anchor
            history.append({
                "epoch": int(epoch),
                "train_total": float(train_total),
                "train_align": float(align),
                "train_move": float(move),
                "train_anchor": float(anchor_loss),
                "val_total": float(val_value),
                "val_align": float(val_align),
                "val_move": float(val_move),
                "val_anchor": float(val_anchor),
            })
            if val_value < best_val - args.mlp_min_delta:
                best_val = float(val_value)
                best_epoch = int(epoch)
                best_state = model.get_state()
                bad_epochs = 0
            else:
                bad_epochs += 1
            if args.mlp_patience > 0 and bad_epochs >= args.mlp_patience:
                break

    model.set_state(best_state)
    mapped_all = model.predict(source_rest, batch_size=args.mlp_apply_batch_size)
    rest_mean_displacement, rest_mean_input_norm, rest_relative_displacement = mean_relative_displacement(
        mapped_all, source_rest
    )
    rest_swd_before = numpy_swd_squared(
        source_rest, target_rest, args.mlp_swd_projections, seed + 303
    )
    rest_swd_after = numpy_swd_squared(
        mapped_all, target_rest, args.mlp_swd_projections, seed + 303
    )
    diagnostics = {
        "backend": "numpy",
        "device": "numpy-cpu",
        "activation1": args.mlp_activation1,
        "activation2": args.mlp_activation2,
        "hidden1": int(args.mlp_hidden1),
        "hidden2": int(args.mlp_hidden2),
        "n_source_rest": int(len(source_rest)),
        "n_target_rest": int(len(target_rest)),
        "n_anchor": int(len(anchor)),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "n_eval_records": int(len(history)),
        "rest_swd_before": rest_swd_before,
        "rest_swd_after": rest_swd_after,
        "rest_mean_displacement": rest_mean_displacement,
        "rest_mean_input_norm": rest_mean_input_norm,
        "rest_relative_displacement": rest_relative_displacement,
        "rest_rms_displacement": float(np.sqrt(np.mean((mapped_all - source_rest) ** 2))),
        "rest_swd_reduction_fraction": float(
            (rest_swd_before - rest_swd_after) / max(rest_swd_before, 1e-15)
        ),
    }
    return model, diagnostics, history


def train_rest_adapter(
    source_rest: np.ndarray,
    source_rest_raw: np.ndarray,
    target_rest: np.ndarray,
    anchor: np.ndarray,
    args: argparse.Namespace,
    seed: int,
):
    if source_rest.shape[1] != 6 or target_rest.shape[1] != 6 or anchor.shape[1] != 6:
        raise ValueError(
            f"Rest-MLP is defined in the six-dimensional seven-class LD space; got "
            f"source={source_rest.shape}, target={target_rest.shape}, anchor={anchor.shape}"
        )
    if args.mlp_activation1.lower() == args.mlp_activation2.lower():
        raise ValueError("--mlp-activation1 and --mlp-activation2 must be different")
    backend = resolve_mlp_backend(args.mlp_backend)
    if backend == "torch":
        return train_rest_adapter_torch(
            source_rest, source_rest_raw, target_rest, anchor, args, seed
        )
    return train_rest_adapter_numpy(
        source_rest, source_rest_raw, target_rest, anchor, args, seed
    )


def apply_adapter_numpy(model, Z: np.ndarray, device: str, batch_size: int = 4096) -> np.ndarray:
    if isinstance(model, NumpyResidualAdapter):
        return model.predict(Z, batch_size=batch_size).astype(np.float64)
    if torch is None:
        raise RuntimeError("A PyTorch adapter was supplied but PyTorch is unavailable")
    model.eval()
    out = np.empty_like(np.asarray(Z, dtype=np.float32))
    with torch.no_grad():
        for start in range(0, len(Z), batch_size):
            end = min(start + batch_size, len(Z))
            x = torch.as_tensor(Z[start:end], dtype=torch.float32, device=device)
            out[start:end] = model(x).cpu().numpy()
    return out.astype(np.float64)


# -----------------------------------------------------------------------------
# Self-readout and identity diagnostics
# -----------------------------------------------------------------------------


def within_subject_self_readout(
    R_subject: np.ndarray,
    y_subject: np.ndarray,
    classes: Sequence[int],
    test_fraction: float,
    ridge: float,
    seed: int,
    rest_class: int,
) -> Dict:
    counts = {int(c): int(np.sum(y_subject == int(c))) for c in classes}
    if any(v < 2 for v in counts.values()):
        return {
            "available": 0,
            "reason": "too_few_samples_for_at_least_one_class",
            "bacc_all7": float("nan"),
            "bacc_task6": float("nan"),
            "rest_recall": float("nan"),
        }
    train_idx, test_idx = stratified_label_split(y_subject, test_fraction, seed)
    try:
        lda = fit_multiclass_lda(R_subject[train_idx], y_subject[train_idx], classes, ridge)
        Z_test = R_subject[test_idx] @ lda.W
        pred, dist = predict_nearest_centroid(Z_test, lda.class_centroids_ld, classes)
        metrics = evaluate_seven_class(y_subject[test_idx], pred, dist, classes, rest_class)
        return {
            "available": 1,
            "reason": "ok",
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            **{k: v for k, v in metrics.items() if k != "confusion_matrix"},
            "confusion_matrix": safe_json(metrics["confusion_matrix"].tolist()),
        }
    except Exception as exc:
        return {
            "available": 0,
            "reason": f"error:{type(exc).__name__}:{exc}",
            "bacc_all7": float("nan"),
            "bacc_task6": float("nan"),
            "rest_recall": float("nan"),
        }


def subject_identity_in_lowdim(
    Z: np.ndarray,
    subject: np.ndarray,
    strata: np.ndarray,
    test_fraction: float,
    ridge: float,
    seed: int,
) -> Dict:
    subjects = sorted(np.unique(subject).astype(int).tolist())
    if len(subjects) < 2:
        return {"available": 0, "reason": "fewer_than_two_subjects", "bacc": float("nan")}
    train_idx, test_idx = stratified_within_subject_split(subject, strata, test_fraction, seed)
    if len(train_idx) == 0 or len(test_idx) == 0:
        return {"available": 0, "reason": "empty_split", "bacc": float("nan")}
    try:
        lda = fit_multiclass_lda(Z[train_idx], subject[train_idx], subjects, ridge)
        Z_test = Z[test_idx] @ lda.W
        pred, _ = predict_nearest_centroid(Z_test, lda.class_centroids_ld, subjects)
        bacc, recalls = balanced_accuracy(subject[test_idx], pred, subjects)
        return {
            "available": 1,
            "reason": "ok",
            "n_subjects": int(len(subjects)),
            "chance": float(1.0 / len(subjects)),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "acc": float(np.mean(pred == subject[test_idx])),
            "bacc": bacc,
            "lda_within_metric_diag_ratio": lda.within_metric_diag_ratio,
            "lda_within_metric_max_offdiag": lda.within_metric_max_offdiag,
            "recalls": recalls,
        }
    except Exception as exc:
        return {"available": 0, "reason": f"error:{type(exc).__name__}:{exc}", "bacc": float("nan")}


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def plot_confusion(
    cm: np.ndarray,
    classes: Sequence[int],
    class_names: Dict[int, str],
    title: str,
    out_png: str | Path,
) -> None:
    cm = np.asarray(cm, dtype=np.float64)
    norm = np.divide(cm, np.maximum(cm.sum(axis=1, keepdims=True), 1.0))
    plt.figure(figsize=(7.2, 6.2))
    plt.imshow(norm, aspect="auto", vmin=0, vmax=1)
    plt.colorbar(label="Row-normalized recall")
    ticks = np.arange(len(classes))
    labels = [class_names.get(int(c), str(c)) for c in classes]
    plt.xticks(ticks, labels, rotation=35, ha="right")
    plt.yticks(ticks, labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, f"{norm[i, j]:.2f}\n({int(cm[i, j])})", ha="center", va="center", fontsize=6.5)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def plot_summary_metric(
    summary_rows: List[Dict],
    metric_mean: str,
    metric_sem: str,
    ylabel: str,
    out_png: str | Path,
    chance: Optional[float] = None,
) -> None:
    plt.figure(figsize=(9, 5.8))
    for method in METHODS + ["self_readout"]:
        rows = sorted([r for r in summary_rows if r["method"] == method], key=lambda r: int(r["svd_dim"]))
        if not rows:
            continue
        xs = [int(r["svd_dim"]) for r in rows]
        ys = [float(r.get(metric_mean, np.nan)) for r in rows]
        es = [float(r.get(metric_sem, 0.0)) for r in rows]
        plt.errorbar(xs, ys, yerr=es, marker="o", capsize=2.5, label=METHOD_TITLES.get(method, method))
    if chance is not None:
        plt.axhline(chance, linestyle="--", linewidth=1, label=f"chance={chance:.3f}")
    plt.xlabel("SVD dimension M")
    plt.ylabel(ylabel)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def plot_mlp_delta_by_subject(subject_rows: List[Dict], svd_dim: int, out_png: str | Path) -> None:
    rows = [r for r in subject_rows if int(r["svd_dim"]) == int(svd_dim)]
    if not rows:
        return
    rows = sorted(rows, key=lambda r: int(r["subject"]))
    xs = np.arange(len(rows))
    ys = [float(r.get("delta_rest_mlp_task6_minus_cross", np.nan)) for r in rows]
    labels = [str(int(r["subject"])) for r in rows]
    plt.figure(figsize=(12, 4.8))
    plt.bar(xs, ys)
    plt.axhline(0.0, linewidth=1)
    plt.xticks(xs, labels, rotation=0)
    plt.xlabel("Held-out subject")
    plt.ylabel("Delta task6 bACC: Rest-MLP minus cross")
    plt.title(f"Per-subject Rest-MLP transfer, M={svd_dim}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


# -----------------------------------------------------------------------------
# Identity controls in embedding/SVD space
# -----------------------------------------------------------------------------


def run_subject_identity_controls(
    args: argparse.Namespace,
    emb_ds: h5py.Dataset,
    selected_idx: np.ndarray,
    y7: np.ndarray,
    raw_label: np.ndarray,
    subject: np.ndarray,
    model_dir: Path,
) -> None:
    scopes = parse_str_list(args.identity_scopes)
    if not scopes:
        return
    valid_scopes = {"rest", "task", "all"}
    unknown = set(scopes) - valid_scopes
    if unknown:
        raise ValueError(f"Unknown identity scopes: {sorted(unknown)}")

    out_dir = model_dir / "identity_controls"
    ensure_dir(out_dir)
    summary_rows: List[Dict] = []
    per_subject_rows: List[Dict] = []
    dims_requested = parse_int_list(args.identity_svd_dims) or parse_int_list(args.svd_dims)

    for scope_i, scope in enumerate(scopes):
        if scope == "rest":
            scope_mask = y7 == int(args.rest_class)
        elif scope == "task":
            scope_mask = y7 != int(args.rest_class)
        else:
            scope_mask = np.ones(len(y7), dtype=bool)

        idx_global = selected_idx[scope_mask]
        y_scope = y7[scope_mask]
        raw_scope = raw_label[scope_mask]
        subj_scope = subject[scope_mask]
        subjects = sorted(np.unique(subj_scope).astype(int).tolist())
        train_loc, test_loc = stratified_within_subject_split(
            subj_scope, raw_scope, args.identity_test_fraction, args.seed + 10000 + scope_i
        )
        if len(train_loc) == 0 or len(test_loc) == 0:
            print(f"[{timestamp()}] [WARN] Identity scope {scope}: empty split", flush=True)
            continue

        print(f"[{timestamp()}] Identity control scope={scope}: reading {len(idx_global)} rows", flush=True)
        X_train = read_embedding_rows(emb_ds, idx_global[train_loc], args.read_batch_size)
        X_test = read_embedding_rows(emb_ds, idx_global[test_loc], args.read_batch_size)
        mu = X_train.mean(axis=0, dtype=np.float64).astype(np.float32)
        X_train -= mu
        X_test -= mu
        max_dim = min(max(dims_requested), X_train.shape[0], X_train.shape[1])
        U, S, Vt = randomized_svd_dense(
            X_train, max_dim, args.svd_oversamples, args.svd_power_iter,
            args.seed + 20000 + scope_i,
        )
        R_train_max = U * S[None, :]
        R_test_max = X_test @ Vt.T
        del X_train, X_test, U, S, Vt
        gc.collect()

        for m_requested in dims_requested:
            m = min(int(m_requested), R_train_max.shape[1])
            R_train = R_train_max[:, :m]
            R_test = R_test_max[:, :m]
            lda = fit_multiclass_lda(R_train, subj_scope[train_loc], subjects, args.lda_ridge)
            Z_test = R_test @ lda.W
            pred, _ = predict_nearest_centroid(Z_test, lda.class_centroids_ld, subjects)
            bacc, recalls = balanced_accuracy(subj_scope[test_loc], pred, subjects)
            summary_rows.append({
                "model": args.model_name,
                "scope": scope,
                "svd_dim_requested": int(m_requested),
                "svd_dim": int(m),
                "n_subjects": int(len(subjects)),
                "chance": float(1.0 / len(subjects)),
                "n_train": int(len(train_loc)),
                "n_test": int(len(test_loc)),
                "acc": float(np.mean(pred == subj_scope[test_loc])),
                "balanced_acc": bacc,
                "lda_within_metric_diag_min": lda.within_metric_diag_min,
                "lda_within_metric_diag_max": lda.within_metric_diag_max,
                "lda_within_metric_diag_ratio": lda.within_metric_diag_ratio,
                "lda_within_metric_max_offdiag": lda.within_metric_max_offdiag,
                "subject_recalls": safe_json(recalls),
            })
            for s in subjects:
                mask = subj_scope[test_loc] == s
                per_subject_rows.append({
                    "model": args.model_name,
                    "scope": scope,
                    "svd_dim_requested": int(m_requested),
                    "svd_dim": int(m),
                    "subject": int(s),
                    "n_test": int(np.sum(mask)),
                    "recall": float(np.mean(pred[mask] == s)) if np.any(mask) else float("nan"),
                })

        del R_train_max, R_test_max
        gc.collect()

    write_csv(out_dir / "identity_summary.csv", summary_rows)
    write_csv(out_dir / "identity_per_subject.csv", per_subject_rows)
    if summary_rows:
        plt.figure(figsize=(8, 5))
        for scope in scopes:
            rows = sorted([r for r in summary_rows if r["scope"] == scope], key=lambda r: int(r["svd_dim"]))
            if rows:
                plt.plot([r["svd_dim"] for r in rows], [r["balanced_acc"] for r in rows], marker="o", label=scope)
        plt.axhline(1.0 / max(1, len(np.unique(subject))), linestyle="--", linewidth=1, label="chance")
        plt.xlabel("SVD dimension M")
        plt.ylabel("Subject-ID balanced accuracy")
        plt.title(f"{args.model_name}: closed-set subject identity")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "identity_bacc_vs_svd_dim.png", dpi=220)
        plt.close()


# -----------------------------------------------------------------------------
# Main seven-class experiment
# -----------------------------------------------------------------------------


def record_subject_method_metrics(
    base: Dict,
    method: str,
    y_true: np.ndarray,
    Z: np.ndarray,
    centroids: Dict[int, np.ndarray],
    classes: Sequence[int],
    rest_class: int,
) -> Tuple[Dict, np.ndarray, np.ndarray, Dict]:
    pred, dist = predict_nearest_centroid(Z, centroids, classes)
    metrics = evaluate_seven_class(y_true, pred, dist, classes, rest_class)
    row = {
        **base,
        "method": method,
        "acc_all7": metrics["acc_all7"],
        "bacc_all7": metrics["bacc_all7"],
        "macro_auc_all7": metrics["macro_auc_all7"],
        "acc_task6": metrics["acc_task6"],
        "bacc_task6": metrics["bacc_task6"],
        "macro_auc_task6": metrics["macro_auc_task6"],
        "rest_recall": metrics["rest_recall"],
        "mean_margin": metrics["mean_margin"],
        "median_margin": metrics["median_margin"],
        "positive_margin_fraction": metrics["positive_margin_fraction"],
        "class_recalls_all7": safe_json(metrics["class_recalls_all7"]),
        "class_recalls_task6": safe_json(metrics["class_recalls_task6"]),
        "confusion_matrix": safe_json(metrics["confusion_matrix"].tolist()),
    }
    return row, pred, dist, metrics


def run_seven_class_probe(
    args: argparse.Namespace,
    emb_ds: h5py.Dataset,
    selected_idx: np.ndarray,
    y7: np.ndarray,
    raw_label: np.ndarray,
    task_id: np.ndarray,
    subject: np.ndarray,
    class_names: Dict[int, str],
    fold_specs: List[Dict],
    model_dir: Path,
) -> None:
    classes = list(range(7))
    rest_class = int(args.rest_class)
    task_classes = [c for c in classes if c != rest_class]
    rest_raw_labels = parse_int_list(args.rest_raw_labels)
    requested_dims = parse_int_list(args.svd_dims)
    if not requested_dims:
        raise ValueError("--svd-dims must contain at least one integer")
    plot_dims = set(parse_int_list(args.plot_svd_dims))

    tables_dir = model_dir / "tables"
    plots_dir = model_dir / "plots"
    states_dir = model_dir / "mlp_states"
    ensure_dir(tables_dir)
    ensure_dir(plots_dir)
    if args.save_mlp_state:
        ensure_dir(states_dir)

    fold_method_rows: List[Dict] = []
    subject_method_rows_long: List[Dict] = []
    subject_comparison_rows: List[Dict] = []
    class_rows: List[Dict] = []
    mlp_rows: List[Dict] = []
    self_rows: List[Dict] = []
    fold_identity_rows: List[Dict] = []
    point_rows: List[Dict] = []
    confusion_accum: Dict[Tuple[int, str], np.ndarray] = {
        (int(m), method): np.zeros((7, 7), dtype=np.int64)
        for m in requested_dims for method in METHODS
    }

    for fold_spec in fold_specs:
        fold_id = int(fold_spec["fold_id"])
        heldout = [int(x) for x in fold_spec["heldout_subjects"]]
        test_mask = np.isin(subject, np.asarray(heldout, dtype=np.int64))
        train_mask = ~test_mask
        train_loc = np.flatnonzero(train_mask)
        test_loc = np.flatnonzero(test_mask)
        if len(train_loc) == 0 or len(test_loc) == 0:
            raise ValueError(f"Fold {fold_id}: empty train or test set")

        print("\n" + "=" * 110, flush=True)
        print(f"[{timestamp()}] Fold {fold_id}: held out subjects {heldout}", flush=True)
        print(f"Train N={len(train_loc)}, test N={len(test_loc)}", flush=True)
        print("=" * 110, flush=True)

        X_train = read_embedding_rows(emb_ds, selected_idx[train_loc], args.read_batch_size)
        X_test = read_embedding_rows(emb_ds, selected_idx[test_loc], args.read_batch_size)
        mu_train = X_train.mean(axis=0, dtype=np.float64).astype(np.float32)
        X_train -= mu_train
        X_test -= mu_train

        m_max_requested = max(requested_dims)
        m_max = min(m_max_requested, X_train.shape[0], X_train.shape[1])
        U, S, Vt = randomized_svd_dense(
            X_train,
            m_max,
            args.svd_oversamples,
            args.svd_power_iter,
            args.seed + fold_id * 1000,
        )
        R_train_max = U * S[None, :]
        R_test_max = X_test @ Vt.T
        del X_train, X_test, U, S, Vt
        gc.collect()

        y_train = y7[train_loc]
        y_test = y7[test_loc]
        raw_train = raw_label[train_loc]
        raw_test = raw_label[test_loc]
        subj_train = subject[train_loc]
        subj_test = subject[test_loc]

        for m_requested in requested_dims:
            m = min(int(m_requested), R_train_max.shape[1])
            print(f"[{timestamp()}] Fold {fold_id}, M={m_requested} (effective {m})", flush=True)
            R_train = np.asarray(R_train_max[:, :m], dtype=np.float64)
            R_test = np.asarray(R_test_max[:, :m], dtype=np.float64)
            lda = fit_multiclass_lda(R_train, y_train, classes, args.lda_ridge)
            if lda.W.shape[1] != 6:
                raise ValueError(
                    f"Seven-class LDA must produce six LD axes, got {lda.W.shape[1]}. "
                    "Choose SVD dimension M >= 6 and ensure all seven classes are present."
                )
            Z_train = lda.train_ld
            Z_test = R_test @ lda.W
            centroids = lda.class_centroids_ld
            train_global_mean = Z_train.mean(axis=0)

            # Balanced canonical pools.  Local indices refer to train arrays.
            rest_ref_idx = build_balanced_rest_indices(
                subj_train, raw_train, y_train, rest_class, rest_raw_labels,
                args.canonical_rest_per_subject_subtype,
                args.seed + fold_id * 10000 + m,
            )
            if len(rest_ref_idx) == 0:
                raise ValueError(f"Fold {fold_id}, M={m}: canonical Rest pool is empty")
            canonical_rest = Z_train[rest_ref_idx]
            canonical_rest_mean = canonical_rest.mean(axis=0)
            anchor_idx = build_balanced_class_indices(
                y_train, classes, args.mlp_anchor_per_class,
                args.seed + fold_id * 20000 + m,
            )
            anchor_pool = Z_train[anchor_idx]

            method_arrays = {method: np.empty_like(Z_test, dtype=np.float64) for method in METHODS}
            method_arrays["cross"][:] = Z_test
            subject_metric_map: Dict[int, Dict[str, Dict]] = {}

            for s in sorted(np.unique(subj_test).astype(int)):
                smask = subj_test == s
                Zs = Z_test[smask]
                ys = y_test[smask]
                raw_s = raw_test[smask]
                rest_s_mask = ys == rest_class
                if not np.any(rest_s_mask):
                    raise ValueError(f"Held-out subject {s} has no Rest samples")

                # All-state centering: diagnostic, uses the unlabeled task point cloud.
                Z_all_centered, all_offset = center_to_reference(
                    Zs, np.ones(len(Zs), dtype=bool), train_global_mean
                )
                method_arrays["all_state_centered"][smask] = Z_all_centered

                # Rest-centered: deployment-compatible mean shift.
                Z_rest_centered, rest_offset = center_to_reference(Zs, rest_s_mask, canonical_rest_mean)
                method_arrays["rest_centered"][smask] = Z_rest_centered

                # Rest-CORAL: deployment-compatible first/second-order alignment.
                A, mu_s, mu_t = fit_coral(Zs[rest_s_mask], canonical_rest, args.coral_ridge)
                Z_coral = apply_coral(Zs, A, mu_s, mu_t)
                method_arrays["rest_coral"][smask] = Z_coral

                # Rest-MLP: balance the four Rest subtypes for the source subject.
                local_rest_cells = [
                    np.flatnonzero(rest_s_mask & (raw_s == int(raw))) for raw in rest_raw_labels
                ]
                source_rest_idx = balanced_indices_by_cells(
                    local_rest_cells,
                    max_per_cell=args.subject_rest_per_subtype,
                    seed=args.seed + fold_id * 100000 + m * 100 + s,
                )
                if len(source_rest_idx) == 0:
                    raise ValueError(f"Subject {s}: balanced Rest calibration pool is empty")
                source_rest = Zs[source_rest_idx]
                source_rest_raw = raw_s[source_rest_idx]
                adapter, adapter_diag, history = train_rest_adapter(
                    source_rest=source_rest,
                    source_rest_raw=source_rest_raw,
                    target_rest=canonical_rest,
                    anchor=anchor_pool,
                    args=args,
                    seed=args.seed + fold_id * 1000000 + m * 1000 + s,
                )
                Z_mlp = apply_adapter_numpy(adapter, Zs, adapter_diag["device"], args.mlp_apply_batch_size)
                method_arrays["rest_mlp"][smask] = Z_mlp
                all_disp, all_input_norm, all_rel_disp = mean_relative_displacement(Z_mlp, Zs)
                if np.any(~rest_s_mask):
                    task_disp, task_input_norm, task_rel_disp = mean_relative_displacement(
                        Z_mlp[~rest_s_mask], Zs[~rest_s_mask]
                    )
                else:
                    task_disp = task_input_norm = task_rel_disp = float("nan")
                adapter_diag.update({
                    "model": args.model_name,
                    "fold": fold_id,
                    "svd_dim": int(m),
                    "svd_dim_requested": int(m_requested),
                    "subject": int(s),
                    "all_mean_displacement": all_disp,
                    "all_mean_input_norm": all_input_norm,
                    "all_relative_displacement": all_rel_disp,
                    "task_mean_displacement": task_disp,
                    "task_mean_input_norm": task_input_norm,
                    "task_relative_displacement": task_rel_disp,
                    "all_state_center_offset_norm": float(np.linalg.norm(all_offset)),
                    "rest_center_offset_norm": float(np.linalg.norm(rest_offset)),
                    "coral_matrix_frobenius_from_identity": float(np.linalg.norm(A - np.eye(6), ord="fro")),
                })
                mlp_rows.append(adapter_diag)
                for hrow in history:
                    mlp_rows.append({
                        "record_type": "history",
                        "model": args.model_name,
                        "fold": fold_id,
                        "svd_dim": int(m),
                        "svd_dim_requested": int(m_requested),
                        "subject": int(s),
                        **hrow,
                    })
                if args.save_mlp_state:
                    state_dir = states_dir / f"M{m}" / f"fold{fold_id:02d}"
                    ensure_dir(state_dir)
                    if isinstance(adapter, NumpyResidualAdapter):
                        np.savez_compressed(
                            state_dir / f"subject{s:02d}.npz",
                            **adapter.get_state(),
                            metadata_json=np.array(safe_json(adapter_diag)),
                        )
                    else:
                        if torch is None:
                            raise RuntimeError("Cannot save a PyTorch adapter because PyTorch is unavailable")
                        torch.save({
                            "state_dict": adapter.cpu().state_dict(),
                            "metadata": adapter_diag,
                        }, state_dir / f"subject{s:02d}.pt")
                        adapter.to(adapter_diag["device"])

                # Oracle similarity Procrustes from true seven-class subject centroids.
                source_centroids = []
                target_centroids = []
                for c in classes:
                    if not np.any(ys == c):
                        raise ValueError(f"Subject {s} is missing class {c}; oracle Procrustes unavailable")
                    source_centroids.append(Zs[ys == c].mean(axis=0))
                    target_centroids.append(centroids[c])
                R_proc, scale_proc, b_proc, proc_diag = fit_similarity_procrustes(
                    np.stack(source_centroids), np.stack(target_centroids), allow_scale=True
                )
                Z_proc = apply_similarity(Zs, R_proc, scale_proc, b_proc)
                method_arrays["oracle_procrustes"][smask] = Z_proc

                subject_metric_map[s] = {
                    "procrustes": proc_diag,
                    "adapter": adapter_diag,
                }

                # Label-using within-subject readout in the shared 6-D global Fisher space.
                # This avoids the unstable high-dimensional small-sample LDA previously fitted
                # directly in the M-dimensional SVD score space.
                self_result = within_subject_self_readout(
                    Zs, ys, classes,
                    args.self_readout_test_fraction,
                    args.lda_ridge,
                    args.seed + fold_id * 100000 + int(m_requested) * 100 + s,
                    rest_class,
                )
                self_rows.append({
                    "model": args.model_name,
                    "fold": fold_id,
                    "svd_dim": int(m),
                    "svd_dim_requested": int(m_requested),
                    "subject": int(s),
                    **self_result,
                })

            # Fold-level metrics for transforms using the same held-out windows.
            fold_method_metrics: Dict[str, Dict] = {}
            for method in METHODS:
                pred, dist = predict_nearest_centroid(method_arrays[method], centroids, classes)
                metrics = evaluate_seven_class(y_test, pred, dist, classes, rest_class)
                fold_method_metrics[method] = metrics
                confusion_accum[(int(m_requested), method)] += metrics["confusion_matrix"]
                fold_method_rows.append({
                    "model": args.model_name,
                    "fold": fold_id,
                    "heldout_subjects": safe_json(heldout),
                    "svd_dim_requested": int(m_requested),
                    "svd_dim": int(m),
                    "ld_dim": 6,
                    "method": method,
                    "n_train": int(len(train_loc)),
                    "n_test": int(len(test_loc)),
                    "acc_all7": metrics["acc_all7"],
                    "bacc_all7": metrics["bacc_all7"],
                    "macro_auc_all7": metrics["macro_auc_all7"],
                    "acc_task6": metrics["acc_task6"],
                    "bacc_task6": metrics["bacc_task6"],
                    "macro_auc_task6": metrics["macro_auc_task6"],
                    "rest_recall": metrics["rest_recall"],
                    "mean_margin": metrics["mean_margin"],
                    "positive_margin_fraction": metrics["positive_margin_fraction"],
                    "class_recalls_all7": safe_json(metrics["class_recalls_all7"]),
                    "confusion_matrix": safe_json(metrics["confusion_matrix"].tolist()),
                    "lda_within_metric_diag_min": lda.within_metric_diag_min,
                    "lda_within_metric_diag_max": lda.within_metric_diag_max,
                    "lda_within_metric_diag_ratio": lda.within_metric_diag_ratio,
                    "lda_within_metric_max_offdiag": lda.within_metric_max_offdiag,
                    **{f"lambda{j+1}": float(lda.eigvals[j]) if j < len(lda.eigvals) else np.nan for j in range(6)},
                })
                for c in classes:
                    class_rows.append({
                        "model": args.model_name,
                        "fold": fold_id,
                        "svd_dim": int(m),
                        "method": method,
                        "class_id": int(c),
                        "class_name": class_names[c],
                        "n_test": int(np.sum(y_test == c)),
                        "recall": float(metrics["class_recalls_all7"].get(c, np.nan)),
                    })

            # Per-subject method metrics and compact comparison table.
            for s in sorted(np.unique(subj_test).astype(int)):
                smask = subj_test == s
                base = {
                    "model": args.model_name,
                    "fold": fold_id,
                    "svd_dim_requested": int(m_requested),
                    "svd_dim": int(m),
                    "subject": int(s),
                    "n_test": int(np.sum(smask)),
                    "n_rest": int(np.sum(smask & (y_test == rest_class))),
                    "n_task": int(np.sum(smask & (y_test != rest_class))),
                }
                compact = dict(base)
                for method in METHODS:
                    row, pred_s, dist_s, metrics_s = record_subject_method_metrics(
                        base, method, y_test[smask], method_arrays[method][smask],
                        centroids, classes, rest_class,
                    )
                    subject_method_rows_long.append(row)
                    compact[f"{method}_bacc_all7"] = metrics_s["bacc_all7"]
                    compact[f"{method}_bacc_task6"] = metrics_s["bacc_task6"]
                    compact[f"{method}_rest_recall"] = metrics_s["rest_recall"]
                    compact[f"{method}_mean_margin"] = metrics_s["mean_margin"]

                    if args.save_test_points and int(m_requested) in plot_dims:
                        idx_subject_local = np.flatnonzero(smask)
                        for local_j, global_test_j in enumerate(idx_subject_local):
                            point_rows.append({
                                "model": args.model_name,
                                "fold": fold_id,
                                "svd_dim": int(m),
                                "subject": int(s),
                                "test_local_index": int(global_test_j),
                                "raw_label": int(raw_test[global_test_j]),
                                "class_id": int(y_test[global_test_j]),
                                "method": method,
                                "predicted_class": int(pred_s[local_j]),
                                **{f"ld{k+1}": float(method_arrays[method][global_test_j, k]) for k in range(6)},
                            })

                self_match = [r for r in self_rows if r["fold"] == fold_id and r["svd_dim"] == m and r["subject"] == s]
                self_result = self_match[-1] if self_match else {}
                compact["self_readout_bacc_all7"] = self_result.get("bacc_all7", np.nan)
                compact["self_readout_bacc_task6"] = self_result.get("bacc_task6", np.nan)
                compact["self_readout_rest_recall"] = self_result.get("rest_recall", np.nan)
                compact["delta_rest_mlp_all7_minus_cross"] = compact["rest_mlp_bacc_all7"] - compact["cross_bacc_all7"]
                compact["delta_rest_mlp_task6_minus_cross"] = compact["rest_mlp_bacc_task6"] - compact["cross_bacc_task6"]
                compact["delta_rest_coral_task6_minus_cross"] = compact["rest_coral_bacc_task6"] - compact["cross_bacc_task6"]
                compact["delta_rest_centered_task6_minus_cross"] = compact["rest_centered_bacc_task6"] - compact["cross_bacc_task6"]
                compact["delta_oracle_task6_minus_cross"] = compact["oracle_procrustes_bacc_task6"] - compact["cross_bacc_task6"]
                compact["delta_self_task6_minus_cross"] = compact["self_readout_bacc_task6"] - compact["cross_bacc_task6"] if np.isfinite(compact["self_readout_bacc_task6"]) else np.nan
                compact.update({
                    "mlp_rest_swd_before": subject_metric_map[s]["adapter"].get("rest_swd_before", np.nan),
                    "mlp_rest_swd_after": subject_metric_map[s]["adapter"].get("rest_swd_after", np.nan),
                    "mlp_rest_swd_reduction_fraction": subject_metric_map[s]["adapter"].get("rest_swd_reduction_fraction", np.nan),
                    "mlp_rest_relative_displacement": subject_metric_map[s]["adapter"].get("rest_relative_displacement", np.nan),
                    "mlp_task_relative_displacement": subject_metric_map[s]["adapter"].get("task_relative_displacement", np.nan),
                    "mlp_all_relative_displacement": subject_metric_map[s]["adapter"].get("all_relative_displacement", np.nan),
                    "mlp_best_epoch": subject_metric_map[s]["adapter"].get("best_epoch", np.nan),
                    "procrustes_centroid_mse_before": subject_metric_map[s]["procrustes"].get("centroid_mse_before", np.nan),
                    "procrustes_centroid_mse_after": subject_metric_map[s]["procrustes"].get("centroid_mse_after", np.nan),
                    "procrustes_scale": subject_metric_map[s]["procrustes"].get("scale", np.nan),
                })
                subject_comparison_rows.append(compact)

            # Diagnostic: how much held-out subject identity remains in shared 6-D LD.
            for scope in ("rest", "all"):
                scope_mask = (y_test == rest_class) if scope == "rest" else np.ones(len(y_test), dtype=bool)
                raw_identity = subject_identity_in_lowdim(
                    method_arrays["cross"][scope_mask], subj_test[scope_mask], raw_test[scope_mask],
                    args.identity_test_fraction, args.lda_ridge,
                    args.seed + fold_id * 100000 + m * 10 + (0 if scope == "rest" else 1),
                )
                mlp_identity = subject_identity_in_lowdim(
                    method_arrays["rest_mlp"][scope_mask], subj_test[scope_mask], raw_test[scope_mask],
                    args.identity_test_fraction, args.lda_ridge,
                    args.seed + fold_id * 100000 + m * 10 + (0 if scope == "rest" else 1),
                )
                fold_identity_rows.append({
                    "model": args.model_name,
                    "fold": fold_id,
                    "svd_dim": int(m),
                    "scope": scope,
                    "n_subjects": raw_identity.get("n_subjects", len(np.unique(subj_test[scope_mask]))),
                    "chance": raw_identity.get("chance", np.nan),
                    "raw_bacc": raw_identity.get("bacc", np.nan),
                    "rest_mlp_bacc": mlp_identity.get("bacc", np.nan),
                    "delta_rest_mlp_minus_raw": mlp_identity.get("bacc", np.nan) - raw_identity.get("bacc", np.nan),
                    "raw_reason": raw_identity.get("reason", ""),
                    "rest_mlp_reason": mlp_identity.get("reason", ""),
                })

        del R_train_max, R_test_max
        gc.collect()

    # Aggregate subject-level self-readout into fold-like rows for plotting.
    for m_requested in requested_dims:
        m_eff_values = [r["svd_dim"] for r in self_rows if r["svd_dim_requested"] == m_requested]
        if not m_eff_values:
            continue
        fold_method_rows.append({
            "model": args.model_name,
            "fold": -1,
            "heldout_subjects": "all_subjects",
            "svd_dim_requested": int(m_requested),
            "svd_dim": int(m_eff_values[0]),
            "ld_dim": 6,
            "method": "self_readout",
            "n_train": np.nan,
            "n_test": np.nan,
            "acc_all7": nanmean([r.get("acc_all7", np.nan) for r in self_rows if r["svd_dim_requested"] == m_requested]),
            "bacc_all7": nanmean([r.get("bacc_all7", np.nan) for r in self_rows if r["svd_dim_requested"] == m_requested]),
            "macro_auc_all7": nanmean([r.get("macro_auc_all7", np.nan) for r in self_rows if r["svd_dim_requested"] == m_requested]),
            "acc_task6": nanmean([r.get("acc_task6", np.nan) for r in self_rows if r["svd_dim_requested"] == m_requested]),
            "bacc_task6": nanmean([r.get("bacc_task6", np.nan) for r in self_rows if r["svd_dim_requested"] == m_requested]),
            "macro_auc_task6": nanmean([r.get("macro_auc_task6", np.nan) for r in self_rows if r["svd_dim_requested"] == m_requested]),
            "rest_recall": nanmean([r.get("rest_recall", np.nan) for r in self_rows if r["svd_dim_requested"] == m_requested]),
            "mean_margin": np.nan,
            "positive_margin_fraction": np.nan,
            "class_recalls_all7": "subject_mean",
            "confusion_matrix": "not_aggregated",
        })

    # Summaries use true fold rows for shared methods, and subject rows for self-readout.
    summary_rows: List[Dict] = []
    for m_requested in requested_dims:
        for method in METHODS:
            rows = [r for r in fold_method_rows if r["svd_dim_requested"] == m_requested and r["method"] == method and r["fold"] >= 0]
            if not rows:
                continue
            summary_rows.append({
                "model": args.model_name,
                "svd_dim_requested": int(m_requested),
                "svd_dim": int(rows[0]["svd_dim"]),
                "method": method,
                "n_folds": int(len(rows)),
                "bacc_all7_mean": nanmean([r["bacc_all7"] for r in rows]),
                "bacc_all7_sem": sem([r["bacc_all7"] for r in rows]),
                "bacc_task6_mean": nanmean([r["bacc_task6"] for r in rows]),
                "bacc_task6_sem": sem([r["bacc_task6"] for r in rows]),
                "rest_recall_mean": nanmean([r["rest_recall"] for r in rows]),
                "rest_recall_sem": sem([r["rest_recall"] for r in rows]),
                "macro_auc_all7_mean": nanmean([r["macro_auc_all7"] for r in rows]),
                "macro_auc_task6_mean": nanmean([r["macro_auc_task6"] for r in rows]),
                "mean_margin_mean": nanmean([r["mean_margin"] for r in rows]),
            })
        sr = [r for r in self_rows if r["svd_dim_requested"] == m_requested and int(r.get("available", 0)) == 1]
        if sr:
            summary_rows.append({
                "model": args.model_name,
                "svd_dim_requested": int(m_requested),
                "svd_dim": int(sr[0]["svd_dim"]),
                "method": "self_readout",
                "n_folds": int(len(sr)),
                "bacc_all7_mean": nanmean([r.get("bacc_all7", np.nan) for r in sr]),
                "bacc_all7_sem": sem([r.get("bacc_all7", np.nan) for r in sr]),
                "bacc_task6_mean": nanmean([r.get("bacc_task6", np.nan) for r in sr]),
                "bacc_task6_sem": sem([r.get("bacc_task6", np.nan) for r in sr]),
                "rest_recall_mean": nanmean([r.get("rest_recall", np.nan) for r in sr]),
                "rest_recall_sem": sem([r.get("rest_recall", np.nan) for r in sr]),
                "macro_auc_all7_mean": nanmean([r.get("macro_auc_all7", np.nan) for r in sr]),
                "macro_auc_task6_mean": nanmean([r.get("macro_auc_task6", np.nan) for r in sr]),
                "mean_margin_mean": nanmean([r.get("mean_margin", np.nan) for r in sr]),
            })

    write_csv(tables_dir / "fold_method_metrics.csv", fold_method_rows)
    write_csv(tables_dir / "subject_method_metrics_long.csv", subject_method_rows_long)
    write_csv(tables_dir / "subject_comparison_wide.csv", subject_comparison_rows)
    write_csv(tables_dir / "class_metrics.csv", class_rows)
    write_csv(tables_dir / "mlp_training_and_diagnostics.csv", mlp_rows)
    write_csv(tables_dir / "self_readout.csv", self_rows)
    write_csv(tables_dir / "fold_subject_identity_before_after_mlp.csv", fold_identity_rows)
    write_csv(tables_dir / "summary_by_svd_dim.csv", summary_rows)
    if point_rows:
        write_csv(tables_dir / "heldout_points_selected_dims.csv", point_rows)

    for m_requested in requested_dims:
        if int(m_requested) not in plot_dims:
            continue
        for method in METHODS:
            cm = confusion_accum[(int(m_requested), method)]
            plot_confusion(
                cm, classes, class_names,
                f"{args.model_name} | {METHOD_TITLES[method]} | M={m_requested}",
                plots_dir / f"confusion_{method}_M{m_requested}.png",
            )
        matching_rows = [r for r in subject_comparison_rows if int(r["svd_dim_requested"]) == int(m_requested)]
        effective_for_plot = int(matching_rows[0]["svd_dim"]) if matching_rows else int(m_requested)
        plot_mlp_delta_by_subject(
            subject_comparison_rows, effective_for_plot,
            plots_dir / f"rest_mlp_task6_delta_by_subject_M{m_requested}.png",
        )

    plot_summary_metric(
        summary_rows,
        "bacc_all7_mean", "bacc_all7_sem",
        "Seven-class balanced accuracy",
        plots_dir / "summary_bacc_all7_vs_svd_dim.png",
        chance=1.0 / 7.0,
    )
    plot_summary_metric(
        summary_rows,
        "bacc_task6_mean", "bacc_task6_sem",
        "Task-six balanced accuracy (primary)",
        plots_dir / "summary_bacc_task6_vs_svd_dim.png",
        chance=1.0 / 7.0,
    )
    plot_summary_metric(
        summary_rows,
        "rest_recall_mean", "rest_recall_sem",
        "Rest recall",
        plots_dir / "summary_rest_recall_vs_svd_dim.png",
        chance=1.0 / 7.0,
    )

    compact_json = {
        "model": args.model_name,
        "created_at": timestamp(),
        "summary_by_svd_dim": summary_rows,
        "primary_metric": "bacc_task6_mean",
        "interpretation_note": (
            "Rest-MLP transfer is supported only when task6 balanced accuracy improves; "
            "an isolated Rest-recall gain is insufficient."
        ),
    }
    with (model_dir / "result_summary.json").open("w", encoding="utf-8") as fp:
        json.dump(compact_json, fp, indent=2, ensure_ascii=False)


# -----------------------------------------------------------------------------
# Single-model and batch runners
# -----------------------------------------------------------------------------


def subject_selection_mask(
    subject_vector: np.ndarray,
    include_subjects_text: str,
    exclude_subjects_text: str,
) -> np.ndarray:
    """Return the explicit subject-selection mask used by both batch inspection and runs."""
    mask = np.ones(len(subject_vector), dtype=bool)
    include_subjects = parse_int_list(include_subjects_text)
    exclude_subjects = parse_int_list(exclude_subjects_text)
    if include_subjects:
        mask &= np.isin(subject_vector, np.asarray(include_subjects, dtype=np.int64))
    if exclude_subjects:
        mask &= ~np.isin(subject_vector, np.asarray(exclude_subjects, dtype=np.int64))
    return mask


def validate_requested_svd_sweep(args: argparse.Namespace) -> None:
    """
    Project constraint: every formal run must request the full sweep through M=500.
    Effective rank may reduce the realised dimension, but the requested value remains recorded.
    """
    svd_dims = parse_int_list(args.svd_dims)
    identity_dims = parse_int_list(args.identity_svd_dims) or svd_dims
    if 500 not in svd_dims:
        raise ValueError(
            f"--svd-dims must include 500 for this experiment; received {svd_dims}"
        )
    if not args.skip_identity_controls and 500 not in identity_dims:
        raise ValueError(
            f"Identity-control dimensions must include 500; received {identity_dims}"
        )


def inspect_h5_for_batch(path: str, args: argparse.Namespace) -> Dict:
    with h5py.File(path, "r") as f:
        raw = read_h5_vector(f, args.label_key).astype(np.int64)
        task = read_h5_vector(f, args.task_key).astype(np.int64)
        subj = read_h5_vector(f, args.subject_key).astype(np.int64)
        y7, _, _ = construct_seven_class_labels(
            raw, task,
            args.rest_task_value, args.nback_task_value, args.matb_task_value,
            parse_int_list(args.rest_raw_labels),
            parse_int_list(args.nback_raw_labels),
            parse_int_list(args.matb_raw_labels),
        )
        mask = y7 >= 0
        mask &= subject_selection_mask(subj, args.subjects, args.exclude_subjects)
        selected_subjects = sorted(np.unique(subj[mask]).astype(int).tolist())
        return {
            "subjects": selected_subjects,
            "subject_vector": subj[mask],
            "n_selected": int(np.sum(mask)),
            "class_counts": {int(c): int(np.sum(y7[mask] == c)) for c in range(7)},
            "excluded_subjects": parse_int_list(args.exclude_subjects),
            "included_subjects": parse_int_list(args.subjects),
        }


def prepare_shared_batch_folds(model_paths: Dict[str, str], args: argparse.Namespace) -> Tuple[str, Dict]:
    inspections: Dict[str, Dict] = {}
    reference_subjects: Optional[List[int]] = None
    reference_vector: Optional[np.ndarray] = None
    for model, path in model_paths.items():
        info = inspect_h5_for_batch(path, args)
        inspections[model] = {
            "subjects": info["subjects"],
            "n_selected": info["n_selected"],
            "class_counts": info["class_counts"],
            "included_subjects": info["included_subjects"],
            "excluded_subjects": info["excluded_subjects"],
        }
        if reference_subjects is None:
            reference_subjects = info["subjects"]
            reference_vector = info["subject_vector"]
        elif info["subjects"] != reference_subjects:
            raise ValueError(f"Subject mismatch in batch: {model} has {info['subjects']}, reference={reference_subjects}")
    assert reference_subjects is not None and reference_vector is not None
    if args.heldout_subject_groups:
        shared = args.heldout_subject_groups
    elif args.cv == "loso":
        shared = groups_to_cli([[s] for s in reference_subjects])
    else:
        groups = make_subject_kfold_groups(reference_subjects, reference_vector, args.n_subject_folds, args.seed)
        shared = groups_to_cli(groups)
    return shared, inspections


def run_single(args: argparse.Namespace) -> None:
    validate_requested_svd_sweep(args)
    model_dir = Path(args.outdir) / args.model_name
    ensure_dir(model_dir)
    if args.mlp_activation1.lower() == args.mlp_activation2.lower():
        raise ValueError("The two MLP nonlinear layers must use different activation functions")
    if int(args.rest_class) != 0:
        raise ValueError("The seven-class mapping fixes merged Rest as class 0; keep --rest-class 0")
    backend = resolve_mlp_backend(args.mlp_backend)
    if torch is not None and backend == "torch" and args.mlp_torch_threads > 0:
        torch.set_num_threads(int(args.mlp_torch_threads))
    if args.mlp_backend == "auto" and backend == "numpy" and TORCH_IMPORT_ERROR is not None:
        print(
            f"[{timestamp()}] [WARN] PyTorch import failed; using NumPy MLP backend. "
            f"Original error: {type(TORCH_IMPORT_ERROR).__name__}: {TORCH_IMPORT_ERROR}",
            flush=True,
        )
    print(f"[{timestamp()}] MLP backend: {backend}", flush=True)

    print(f"[{timestamp()}] Opening H5: {args.h5}", flush=True)
    with h5py.File(args.h5, "r") as f:
        if args.embedding_key not in f:
            raise KeyError(f"Embedding key not found: {args.embedding_key}")
        emb_ds = f[args.embedding_key]
        n_total = int(emb_ds.shape[0])
        raw_all = read_h5_vector(f, args.label_key).astype(np.int64)
        task_all = read_h5_vector(f, args.task_key).astype(np.int64)
        subj_all = read_h5_vector(f, args.subject_key).astype(np.int64)
        for key, arr in ((args.label_key, raw_all), (args.task_key, task_all), (args.subject_key, subj_all)):
            if len(arr) != n_total:
                raise ValueError(f"Length mismatch: embedding N={n_total}, {key} len={len(arr)}")

        y7_all, class_names, label_meta = construct_seven_class_labels(
            raw_all, task_all,
            args.rest_task_value, args.nback_task_value, args.matb_task_value,
            parse_int_list(args.rest_raw_labels),
            parse_int_list(args.nback_raw_labels),
            parse_int_list(args.matb_raw_labels),
        )
        selected_mask = y7_all >= 0
        selected_mask &= subject_selection_mask(
            subj_all, args.subjects, args.exclude_subjects
        )
        selected_idx = np.flatnonzero(selected_mask)
        y7 = y7_all[selected_idx]
        raw = raw_all[selected_idx]
        task = task_all[selected_idx]
        subj = subj_all[selected_idx]
        subjects = sorted(np.unique(subj).astype(int).tolist())
        folds = build_subject_folds(args, subjects, subj)

        metadata = {
            "script": os.path.basename(__file__),
            "script_version": SCRIPT_VERSION,
            "created_at": timestamp(),
            "model_name": args.model_name,
            "h5": args.h5,
            "embedding_key": args.embedding_key,
            "label_key": args.label_key,
            "task_key": args.task_key,
            "subject_key": args.subject_key,
            "embedding_shape": [int(x) for x in emb_ds.shape],
            "flat_dim": int(get_flat_dim(emb_ds)),
            "n_total": n_total,
            "n_selected": int(len(selected_idx)),
            "subjects": subjects,
            "n_effective_subjects": int(len(subjects)),
            "subject_selection": {
                "included_subjects": parse_int_list(args.subjects),
                "excluded_subjects": parse_int_list(args.exclude_subjects),
                "duplicate_block_retained_representative": int(args.duplicate_block_representative),
                "selection_rationale": (
                    "Subjects 18-24 and 27 have identical 419-window embeddings in the current "
                    "five H5 files. This run excludes 18-24 and retains subject 27 as the single "
                    "representative, restoring 22 independent subject blocks."
                ),
            },
            "class_names": class_names,
            "class_counts": {int(c): int(np.sum(y7 == c)) for c in range(7)},
            "label_construction": label_meta,
            "folds": folds,
            "rest_mlp_architecture": {
                "input_dim": 6,
                "hidden1": args.mlp_hidden1,
                "activation1": args.mlp_activation1,
                "hidden2": args.mlp_hidden2,
                "activation2": args.mlp_activation2,
                "output_dim": 6,
                "residual": True,
                "last_layer_zero_initialized": True,
            },
            "primary_metric": "task6 balanced accuracy",
            "lda_class_weighting": "equal class covariance and equal class-mean scatter",
            "lda_axis_metric": "Sw_reg-orthonormal; W.T @ Sw_reg @ W = I",
            "args": vars(args),
        }
        for k, v in f.attrs.items():
            try:
                metadata[f"h5_attr/{k}"] = decode_attr(v)
            except Exception:
                pass
        with (model_dir / "run_metadata.json").open("w", encoding="utf-8") as fp:
            json.dump(metadata, fp, indent=2, ensure_ascii=False)

        print("=" * 110, flush=True)
        print(f"Model: {args.model_name}", flush=True)
        print(f"Embedding shape: {emb_ds.shape}, flat_dim={get_flat_dim(emb_ds)}", flush=True)
        print(f"Seven classes: {class_names}", flush=True)
        print(f"Class counts: {metadata['class_counts']}", flush=True)
        print(f"Subjects: {subjects}", flush=True)
        print(f"Held-out groups: {[x['heldout_subjects'] for x in folds]}", flush=True)
        print(
            f"Rest-MLP: 6->{args.mlp_hidden1}({args.mlp_activation1})->"
            f"{args.mlp_hidden2}({args.mlp_activation2})->6 residual",
            flush=True,
        )
        print("=" * 110, flush=True)

        if not args.skip_identity_controls:
            run_subject_identity_controls(
                args, emb_ds, selected_idx, y7, raw, subj, model_dir
            )
        if not args.skip_seven_class_probe:
            run_seven_class_probe(
                args, emb_ds, selected_idx, y7, raw, task, subj,
                class_names, folds, model_dir,
            )

    print(f"[{timestamp()}] Done. Output: {model_dir}", flush=True)


def save_batch_summary(outroot: str, models: Sequence[str]) -> None:
    rows: List[Dict] = []
    for model in models:
        path = Path(outroot) / model / "tables" / "summary_by_svd_dim.csv"
        if not path.exists():
            print(f"[{timestamp()}] [WARN] Missing summary: {path}", flush=True)
            continue
        with path.open("r", encoding="utf-8", newline="") as fp:
            for row in csv.DictReader(fp):
                row = dict(row)
                row["model"] = model
                rows.append(row)
    if not rows:
        return
    outroot_path = Path(outroot)
    write_csv(outroot_path / "all_models_summary_by_svd_dim.csv", rows)

    best_rows: List[Dict] = []
    for model in models:
        model_rows = [r for r in rows if r["model"] == model and r["method"] == "rest_mlp"]
        if model_rows:
            best_rows.append(max(model_rows, key=lambda r: float(r.get("bacc_task6_mean", "nan"))))
    write_csv(outroot_path / "all_models_best_rest_mlp_task6.csv", best_rows)

    plt.figure(figsize=(9, 5.8))
    for model in models:
        mr = sorted(
            [r for r in rows if r["model"] == model and r["method"] == "rest_mlp"],
            key=lambda r: int(r["svd_dim"]),
        )
        if mr:
            plt.plot(
                [int(r["svd_dim"]) for r in mr],
                [float(r["bacc_task6_mean"]) for r in mr],
                marker="o", label=model,
            )
    plt.xlabel("SVD dimension M")
    plt.ylabel("Rest-MLP task6 balanced accuracy")
    plt.title("CogBCI 5-second: rest-driven personalization across models")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outroot_path / "all_models_rest_mlp_task6_vs_svd_dim.png", dpi=220)
    plt.close()


def run_batch(args: argparse.Namespace) -> None:
    validate_requested_svd_sweep(args)
    models = parse_str_list(args.models)
    unknown = [m for m in models if m not in COGBCI_5S_MODEL_PATHS]
    if unknown:
        raise ValueError(f"Unknown models: {unknown}; choices={list(COGBCI_5S_MODEL_PATHS)}")
    paths = {m: COGBCI_5S_MODEL_PATHS[m] for m in models}
    ensure_dir(args.outroot)
    shared_groups, inspections = prepare_shared_batch_folds(paths, args)
    selected_subject_sets = {tuple(v["subjects"]) for v in inspections.values()}
    if len(selected_subject_sets) != 1:
        raise ValueError(f"Selected subject mismatch across models: {selected_subject_sets}")
    selected_subjects = list(next(iter(selected_subject_sets)))
    expected_subjects = list(range(1, 18)) + [25, 26, 27, 28, 29]
    if selected_subjects != expected_subjects:
        raise ValueError(
            "Deduplicated cohort mismatch. Expected subjects "
            f"{expected_subjects}, got {selected_subjects}. "
            "Do not continue until the selection is explicit and correct."
        )
    with open(Path(args.outroot) / "batch_metadata.json", "w", encoding="utf-8") as fp:
        json.dump({
            "script": os.path.basename(__file__),
            "script_version": SCRIPT_VERSION,
            "created_at": timestamp(),
            "models": models,
            "paths": paths,
            "shared_heldout_subject_groups": shared_groups,
            "effective_subjects": selected_subjects,
            "n_effective_subjects": len(selected_subjects),
            "deduplication": {
                "excluded_subjects": parse_int_list(args.exclude_subjects),
                "retained_representative": int(args.duplicate_block_representative),
                "reason": (
                    "Subjects 18-24 and 27 contain elementwise-identical 419-window embeddings "
                    "in all five current CogBCI 5s H5 files; retain 27 once."
                ),
            },
            "inspections": inspections,
            "args": vars(args),
        }, fp, indent=2, ensure_ascii=False)

    print("=" * 110, flush=True)
    print(f"CogBCI 5-second seven-class Rest-personalization batch [{SCRIPT_VERSION}]", flush=True)
    print(f"Models: {models}", flush=True)
    print(f"Shared folds: {shared_groups}", flush=True)
    print(f"Output root: {args.outroot}", flush=True)
    print("=" * 110, flush=True)

    for i, model in enumerate(models, start=1):
        model_args = argparse.Namespace(**vars(args).copy())
        model_args.batch_5s = False
        model_args.h5 = paths[model]
        model_args.model_name = model
        model_args.outdir = args.outroot
        model_args.heldout_subject_groups = shared_groups
        print("\n" + "#" * 110, flush=True)
        print(f"[{timestamp()}] Batch model {i}/{len(models)}: {model}", flush=True)
        print("#" * 110, flush=True)
        run_single(model_args)

    save_batch_summary(args.outroot, models)
    print(f"[{timestamp()}] Batch complete", flush=True)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CogBCI seven-class subject-heldout Rest personalization in six-dimensional LD space"
    )

    # Input and batch mode.
    p.add_argument("--batch-5s", action="store_true", help="Run registered 5-second models with shared subject folds")
    p.add_argument("--models", default="BIOT,LaBraM,CBraMod,EEGPT,EEGMamba")
    p.add_argument(
        "--outroot",
        default="/mnt/dataset4/yinuo/FM_flow/dataset/expA_rest_personalization_7class_ld_mlp_5s",
        help="Batch output root",
    )
    p.add_argument("--h5", default="", help="Single-file H5 path")
    p.add_argument("--model-name", default="", help="Model name in single-file mode")
    p.add_argument("--outdir", default="", help="Output root in single-file mode; model subdirectory is created")
    p.add_argument("--embedding-key", default="embedding")
    p.add_argument("--label-key", default="label", help="Raw/global condition label key")
    p.add_argument("--task-key", default="task_id")
    p.add_argument("--subject-key", default="subject_id")
    p.add_argument("--window-seconds", type=float, default=5.0)

    # Seven-class semantics.
    p.add_argument("--rest-task-value", type=int, default=0)
    p.add_argument("--nback-task-value", type=int, default=1)
    p.add_argument("--matb-task-value", type=int, default=2)
    p.add_argument("--rest-raw-labels", default="0,1,2,3")
    p.add_argument("--nback-raw-labels", default="10,11,12")
    p.add_argument("--matb-raw-labels", default="20,21,22")
    p.add_argument("--rest-class", type=int, default=0)

    # Subject-heldout folds.
    p.add_argument("--cv", default="subject_kfold", choices=["subject_kfold", "loso"])
    p.add_argument("--n-subject-folds", type=int, default=4)
    p.add_argument("--heldout-subject-groups", default="")
    p.add_argument("--folds", default="", help="Optional fold IDs for smoke test, e.g. 1")
    p.add_argument("--subjects", default="", help="Optional explicit inclusion list")
    p.add_argument(
        "--exclude-subjects",
        default="18,19,20,21,22,23,24",
        help=(
            "Subjects excluded before fold construction. The current CogBCI 5s H5 files "
            "contain one identical 419-window block under subjects 18-24 and 27; keep 27 "
            "as the representative and exclude 18-24."
        ),
    )
    p.add_argument(
        "--duplicate-block-representative",
        type=int,
        default=27,
        help="Representative retained from the duplicated subject block; metadata only.",
    )

    # SVD/LDA.
    p.add_argument("--svd-dims", default="100,200,300,500")
    p.add_argument("--plot-svd-dims", default="500")
    p.add_argument("--svd-oversamples", type=int, default=20)
    p.add_argument("--svd-power-iter", type=int, default=2)
    p.add_argument("--lda-ridge", type=float, default=1e-4)
    p.add_argument("--read-batch-size", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)

    # Rest pools and linear baselines.
    p.add_argument("--canonical-rest-per-subject-subtype", type=int, default=128)
    p.add_argument("--subject-rest-per-subtype", type=int, default=128)
    p.add_argument("--coral-ridge", type=float, default=1e-3)

    # Rest residual MLP.  Activations are intentionally different by default.
    p.add_argument("--mlp-hidden1", type=int, default=16)
    p.add_argument("--mlp-hidden2", type=int, default=16)
    p.add_argument("--mlp-activation1", default="tanh")
    p.add_argument("--mlp-activation2", default="gelu")
    p.add_argument("--mlp-backend", default="auto", choices=["auto", "torch", "numpy"], help="auto uses PyTorch when importable, otherwise the NumPy fallback")
    p.add_argument("--mlp-device", default="auto", help="PyTorch backend only: auto, cpu, cuda, cuda:0, ...")
    p.add_argument("--mlp-torch-threads", type=int, default=1, help="CPU threads for tiny per-subject MLPs")
    p.add_argument("--mlp-epochs", type=int, default=400)
    p.add_argument("--mlp-batch-size", type=int, default=128)
    p.add_argument("--mlp-apply-batch-size", type=int, default=4096)
    p.add_argument("--mlp-lr", type=float, default=1e-3)
    p.add_argument("--mlp-weight-decay", type=float, default=1e-4)
    p.add_argument("--mlp-swd-projections", type=int, default=64)
    p.add_argument("--mlp-lambda-move", type=float, default=0.05)
    p.add_argument("--mlp-lambda-anchor", type=float, default=0.10)
    p.add_argument("--mlp-anchor-per-class", type=int, default=256)
    p.add_argument("--mlp-val-fraction", type=float, default=0.25)
    p.add_argument("--mlp-eval-every", type=int, default=5)
    p.add_argument("--mlp-patience", type=int, default=20, help="Patience counted in evaluation records; 0 disables")
    p.add_argument("--mlp-min-delta", type=float, default=1e-6)
    p.add_argument("--mlp-grad-clip", type=float, default=5.0)
    p.add_argument("--save-mlp-state", action="store_true")

    # Oracles and diagnostics.
    p.add_argument("--self-readout-test-fraction", type=float, default=0.25)
    p.add_argument("--identity-test-fraction", type=float, default=0.25)
    p.add_argument("--identity-scopes", default="rest,task,all")
    p.add_argument(
        "--identity-svd-dims",
        default="",
        help=(
            "Identity-control SVD dimensions. Empty means follow --svd-dims exactly; "
            "the formal sweep must include requested M=500 even when effective rank is lower."
        ),
    )
    p.add_argument("--skip-identity-controls", action="store_true")
    p.add_argument("--skip-seven-class-probe", action="store_true")
    p.add_argument("--save-test-points", action="store_true")

    return p


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    if args.batch_5s:
        run_batch(args)
        return
    missing = [name for name, value in (("--h5", args.h5), ("--model-name", args.model_name), ("--outdir", args.outdir)) if not value]
    if missing:
        parser.error("Single-file mode requires " + ", ".join(missing) + "; or use --batch-5s")
    run_single(args)


if __name__ == "__main__":
    main()
