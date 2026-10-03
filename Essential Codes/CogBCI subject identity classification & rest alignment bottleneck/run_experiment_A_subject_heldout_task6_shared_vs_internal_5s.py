#!/usr/bin/env python3
"""
Experiment A on 5-second CogBCI task windows: six-class shared LDA axes versus
N-Back/MATB task-specific three-class LDA axes.

Scientific question
-------------------
Rest windows are removed.  The six remaining task labels are analyzed in one
common train-only SVD space.  For every subject-heldout fold and every SVD
truncation M, the script compares five readouts:

1. task6_shared5
   Fit one six-class LDA on all non-rest training samples.  Six classes imply at
   most five LD axes.  Classify all six held-out task classes in that 5-D LD space.

2. nback_on_shared5
   Keep the same five LD axes learned by the six-class problem.  Restrict to the
   three N-Back classes, estimate their train centroids in the shared 5-D LD
   space, and classify held-out N-Back samples.

3. matb_on_shared5
   Same as (2), but for the three MATB classes.

4. nback_internal2
   In the same common SVD space, fit a separate N-Back-only three-class LDA.
   Three classes imply at most two LD axes.

5. matb_internal2
   Same as (4), but fit on MATB only.

This directly answers whether the five axes learned from the joint six-class
problem are already useful within each task family, or whether N-Back and MATB
need their own supervised two-axis readouts.

Important geometry
------------------
- The embedding is first flattened per window.
- Train-only centering and train-only randomized SVD create the common
  unsupervised M-dimensional space.
- The six-class shared LDA then maps M -> 5.
- Each task-specific LDA maps M -> 2.
- Held-out subjects never contribute to centering, SVD, LDA directions, or train
  centroids.

Default CogBCI semantics
------------------------
- task_id == 0: Rest, removed.
- label among non-rest samples: six raw task labels, sorted and remapped to
  contiguous class IDs 0..5.
- task_id == 1: N-Back family by default.
- task_id == 2: MATB family by default.
- The three class IDs in each family are inferred from task_id, so the script
  does not silently depend on raw-label ordering.

Recommended batch command
-------------------------
python3 run_experiment_A_subject_heldout_task6_shared_vs_internal_5s.py \
  --batch-5s \
  --cv subject_kfold \
  --n-subject-folds 4 \
  --svd-dims 40,500,600 \
  --n-shuffle 50 \
  --outroot /mnt/dataset4/yinuo/FM_flow/dataset/expA_task6_shared_vs_internal_5s_M40_500_600

Smoke test
----------
python3 run_experiment_A_subject_heldout_task6_shared_vs_internal_5s.py \
  --h5 /mnt/dataset4/fuxy/FMS/FM/LaBraM/output_emb_5s/cogbci_sesS1_embeddings.h5 \
  --model-name LaBraM \
  --outdir /mnt/dataset4/yinuo/FM_flow/dataset/expA_task6_smoke/LaBraM \
  --folds 1 \
  --svd-dims 40 \
  --n-shuffle 0
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
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


COGBCI_5S_MODEL_PATHS: Dict[str, str] = {
    "BIOT": "/mnt/dataset4/fuxy/FMS/FM/BIOT/output_emb_5s/cogbci_sesS1_embeddings.h5",
    "LaBraM": "/mnt/dataset4/fuxy/FMS/FM/LaBraM/output_emb_5s/cogbci_sesS1_embeddings.h5",
    "CBraMod": "/mnt/dataset4/fuxy/FMS/FM/CBraMod/output_emb_5s/cogbci_sesS1_embeddings.h5",
    "EEGPT": "/mnt/dataset4/fuxy/FMS/FM/EEGPT/output_emb_5s/cogbci_sesS1_embeddings.h5",
    "EEGMamba": "/mnt/dataset4/fuxy/FMS/FM/EEGMamba/output_emb_5s/cogbci_sesS1_embeddings.h5",
    "s-JEPA": "/mnt/dataset4/fuxy/FMS/FM/s-JEPA/output_5s/cogbci_sesS1_embeddings.h5",
}

ANALYSIS_ORDER = [
    "task6_shared5",
    "nback_on_shared5",
    "matb_on_shared5",
    "nback_internal2",
    "matb_internal2",
]

ANALYSIS_TITLES = {
    "task6_shared5": "Six-task classification on joint 5 LD axes",
    "nback_on_shared5": "N-Back 3-class on joint six-task LD axes",
    "matb_on_shared5": "MATB 3-class on joint six-task LD axes",
    "nback_internal2": "N-Back 3-class on task-specific 2 LD axes",
    "matb_internal2": "MATB 3-class on task-specific 2 LD axes",
}


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_int_list(s: str) -> List[int]:
    if s is None or str(s).strip() == "":
        return []
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


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
    """Read selected H5 rows and flatten every sample to one vector."""
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError("indices must be 1D")
    if len(indices) == 0:
        raise ValueError("cannot read zero rows")
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


def balanced_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: Sequence[int],
) -> Tuple[float, Dict[int, float]]:
    recalls: Dict[int, float] = {}
    for c in classes:
        mask = y_true == c
        recalls[int(c)] = float(np.mean(y_pred[mask] == c)) if np.any(mask) else float("nan")
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
        if int(yt) in class_to_idx and int(yp) in class_to_idx:
            cm[class_to_idx[int(yt)], class_to_idx[int(yp)]] += 1
    return cm


def nanmean_or_nan(x: Sequence[float]) -> float:
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def nanmedian_or_nan(x: Sequence[float]) -> float:
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmedian(arr))


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
    """Randomized truncated SVD for an already centered dense matrix."""
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
    for _ in range(max(0, n_iter)):
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


def fit_multiclass_lda(
    R: np.ndarray,
    y: np.ndarray,
    classes: Sequence[int],
    ridge: float = 1e-4,
) -> LDAResult:
    """Fit multiclass LDA and return at most C-1 discriminant axes."""
    R = np.asarray(R, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    n, m = R.shape
    classes = [int(c) for c in classes]
    q = min(len(classes) - 1, m)
    if q <= 0:
        raise ValueError("Need at least two classes and one feature dimension")

    mu = R.mean(axis=0)
    Sw = np.zeros((m, m), dtype=np.float64)
    Sb = np.zeros((m, m), dtype=np.float64)
    for c in classes:
        Rc = R[y == c]
        if len(Rc) == 0:
            raise ValueError(f"Class {c} has no training samples")
        muc = Rc.mean(axis=0)
        Xc = Rc - muc
        Sw += Xc.T @ Xc
        dm = (muc - mu).reshape(-1, 1)
        Sb += len(Rc) * (dm @ dm.T)

    Sw /= max(n - len(classes), 1)
    Sb /= max(len(classes) - 1, 1)
    scale = float(np.trace(Sw) / max(m, 1))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    Sw_reg = Sw + ridge * scale * np.eye(m, dtype=np.float64)

    try:
        from scipy.linalg import eigh
        eigvals, eigvecs = eigh(Sb, Sw_reg)
    except Exception:
        A = np.linalg.solve(Sw_reg, Sb)
        eigvals, eigvecs = np.linalg.eig(A)
        eigvals = np.real(eigvals)
        eigvecs = np.real(eigvecs)

    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(np.asarray(eigvals)[order], 0.0)
    W = np.asarray(eigvecs)[:, order[:q]]
    for j in range(W.shape[1]):
        norm = np.linalg.norm(W[:, j])
        if norm > 0:
            W[:, j] /= norm

    train_ld = R @ W
    centroids = {int(c): train_ld[y == c].mean(axis=0) for c in classes}
    return LDAResult(eigvals=eigvals, W=W, class_centroids_ld=centroids, train_ld=train_ld)


def predict_nearest_centroid(Z: np.ndarray, centroids: Dict[int, np.ndarray]) -> np.ndarray:
    classes = list(centroids.keys())
    C = np.stack([centroids[c] for c in classes], axis=0)
    dist = np.sum((Z[:, None, :] - C[None, :, :]) ** 2, axis=2)
    idx = np.argmin(dist, axis=1)
    return np.asarray([classes[i] for i in idx], dtype=np.int64)


def subgroup_centroids(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    classes: Sequence[int],
) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    for c in classes:
        mask = y_train == c
        if not np.any(mask):
            raise ValueError(f"Class {c} missing while computing subgroup centroids")
        out[int(c)] = Z_train[mask].mean(axis=0)
    return out


def compute_eigen_null(
    R: np.ndarray,
    y: np.ndarray,
    classes: Sequence[int],
    real_eigvals: np.ndarray,
    n_shuffle: int,
    ridge: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Train-label permutation null for the fitted discriminant eigenvalues."""
    q = len(classes) - 1
    if n_shuffle <= 0:
        return np.full(q, np.nan), np.full(q, np.nan)
    null = np.empty((n_shuffle, q), dtype=np.float64)
    for sh in range(n_shuffle):
        y_shuf = np.array(y, copy=True)
        rng.shuffle(y_shuf)
        lda_shuf = fit_multiclass_lda(R, y_shuf, classes=classes, ridge=ridge)
        vals = lda_shuf.eigvals[:q]
        if len(vals) < q:
            vals = np.pad(vals, (0, q - len(vals)), constant_values=np.nan)
        null[sh] = vals
    shuffle95 = np.nanpercentile(null, 95, axis=0)
    pvals = np.empty(q, dtype=np.float64)
    for j in range(q):
        real = real_eigvals[j] if j < len(real_eigvals) else np.nan
        pvals[j] = (1.0 + np.sum(null[:, j] >= real)) / (1.0 + n_shuffle)
    return shuffle95, pvals


# -----------------------------------------------------------------------------
# Fold construction
# -----------------------------------------------------------------------------


def parse_subject_groups(s: str) -> List[List[int]]:
    if s is None or not str(s).strip():
        return []
    groups: List[List[int]] = []
    for part in str(s).split(";"):
        vals = parse_int_list(part)
        if vals:
            groups.append(vals)
    return groups


def make_subject_kfold_groups(
    subjects: Sequence[int],
    subj_sel: np.ndarray,
    n_folds: int,
    seed: int,
) -> List[List[int]]:
    subjects = [int(s) for s in subjects]
    if n_folds < 2 or n_folds > len(subjects):
        raise ValueError(f"Invalid n_folds={n_folds} for {len(subjects)} subjects")
    counts = {s: int(np.sum(subj_sel == s)) for s in subjects}
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
        groups = [sorted([s for s in g if s in allowed]) for g in manual]
        groups = [g for g in groups if g]
        cv_name = "manual_subject_groups"
    elif args.cv == "loso":
        groups = [[s] for s in subjects]
        cv_name = "loso"
    else:
        groups = make_subject_kfold_groups(subjects, subj_sel, args.n_subject_folds, args.seed)
        cv_name = f"subject_{args.n_subject_folds}fold"

    if args.folds:
        keep = set(parse_int_list(args.folds))
        groups = [g for i, g in enumerate(groups, start=1) if i in keep]

    specs = []
    total = len(subj_sel)
    for i, g in enumerate(groups, start=1):
        n_test = int(np.sum(np.isin(subj_sel, np.asarray(g, dtype=np.int64))))
        specs.append({
            "fold_id": i,
            "cv": cv_name,
            "heldout_subjects": [int(x) for x in g],
            "n_heldout_subjects": len(g),
            "estimated_n_test": n_test,
            "estimated_test_fraction": float(n_test / max(total, 1)),
        })
    return specs


# -----------------------------------------------------------------------------
# Evaluation and plotting
# -----------------------------------------------------------------------------


def make_class_names(raw_classes: Sequence[int], family: str) -> Dict[int, str]:
    names: Dict[int, str] = {}
    for cid, raw in enumerate(raw_classes):
        if cid <= 2:
            names[cid] = f"NBack_{raw}"
        else:
            names[cid] = f"MATB_{raw}"
    return names


def evaluate_projection(
    analysis: str,
    Z_train: np.ndarray,
    Z_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    classes: Sequence[int],
) -> Dict:
    classes = [int(c) for c in classes]
    train_mask = np.isin(y_train, classes)
    test_mask = np.isin(y_test, classes)
    if not np.any(train_mask) or not np.any(test_mask):
        raise ValueError(f"No samples for analysis={analysis}, classes={classes}")
    centroids = subgroup_centroids(Z_train[train_mask], y_train[train_mask], classes)
    pred = predict_nearest_centroid(Z_test[test_mask], centroids)
    yt = y_test[test_mask]
    acc = float(np.mean(pred == yt))
    bacc, recalls = balanced_accuracy(yt, pred, classes)
    cm = confusion_matrix_fixed(yt, pred, classes)
    return {
        "analysis": analysis,
        "classes": classes,
        "n_train_analysis": int(np.sum(train_mask)),
        "n_test_analysis": int(np.sum(test_mask)),
        "acc": acc,
        "balanced_acc": bacc,
        "class_recalls": recalls,
        "confusion_matrix": cm,
        "Z_test": Z_test[test_mask],
        "y_test": yt,
    }


def evaluate_internal_lda(
    analysis: str,
    R_train: np.ndarray,
    R_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    classes: Sequence[int],
    ridge: float,
    n_shuffle: int,
    rng: np.random.Generator,
) -> Dict:
    classes = [int(c) for c in classes]
    train_mask = np.isin(y_train, classes)
    test_mask = np.isin(y_test, classes)
    lda = fit_multiclass_lda(R_train[train_mask], y_train[train_mask], classes, ridge)
    Z_train = lda.train_ld
    Z_test = R_test[test_mask] @ lda.W
    pred = predict_nearest_centroid(Z_test, lda.class_centroids_ld)
    yt = y_test[test_mask]
    acc = float(np.mean(pred == yt))
    bacc, recalls = balanced_accuracy(yt, pred, classes)
    cm = confusion_matrix_fixed(yt, pred, classes)
    shuffle95, pvals = compute_eigen_null(
        R_train[train_mask], y_train[train_mask], classes,
        lda.eigvals, n_shuffle, ridge, rng,
    )
    return {
        "analysis": analysis,
        "classes": classes,
        "n_train_analysis": int(np.sum(train_mask)),
        "n_test_analysis": int(np.sum(test_mask)),
        "acc": acc,
        "balanced_acc": bacc,
        "class_recalls": recalls,
        "confusion_matrix": cm,
        "Z_test": Z_test,
        "y_test": yt,
        "eigvals": lda.eigvals[: len(classes) - 1],
        "shuffle95": shuffle95,
        "pvals": pvals,
    }


def result_to_row(
    result: Dict,
    args: argparse.Namespace,
    fold_id: int,
    heldout_subjects: Sequence[int],
    svd_dim: int,
    m_eff: int,
    n_train_total: int,
    n_test_total: int,
    shared_eigvals: Optional[np.ndarray] = None,
    shared_shuffle95: Optional[np.ndarray] = None,
    shared_pvals: Optional[np.ndarray] = None,
) -> Dict:
    eigvals = result.get("eigvals")
    shuffle95 = result.get("shuffle95")
    pvals = result.get("pvals")
    eig_source = "task_specific_lda"
    if eigvals is None and shared_eigvals is not None:
        eigvals = shared_eigvals
        shuffle95 = shared_shuffle95
        pvals = shared_pvals
        eig_source = "shared_six_class_lda"
    if eigvals is None:
        eigvals = np.array([], dtype=np.float64)
    if shuffle95 is None:
        shuffle95 = np.full(len(eigvals), np.nan)
    if pvals is None:
        pvals = np.full(len(eigvals), np.nan)

    row = {
        "model": args.model_name,
        "analysis": result["analysis"],
        "task_name": args.task_name,
        "h5": args.h5,
        "heldout_fold": int(fold_id),
        "heldout_subjects": json.dumps([int(x) for x in heldout_subjects]),
        "svd_dim": int(svd_dim),
        "svd_dim_effective": int(m_eff),
        "ld_dim": int(result["Z_test"].shape[1]),
        "eigenvalue_source": eig_source,
        "classes": json.dumps([int(x) for x in result["classes"]]),
        "n_train_total": int(n_train_total),
        "n_test_total": int(n_test_total),
        "n_train_analysis": int(result["n_train_analysis"]),
        "n_test_analysis": int(result["n_test_analysis"]),
        "acc": float(result["acc"]),
        "balanced_acc": float(result["balanced_acc"]),
        "class_recalls": json.dumps(result["class_recalls"], ensure_ascii=False),
        "confusion_matrix": json.dumps(result["confusion_matrix"].tolist()),
    }
    for j in range(5):
        row[f"lambda{j + 1}"] = float(eigvals[j]) if j < len(eigvals) else np.nan
        row[f"shuffle95_{j + 1}"] = float(shuffle95[j]) if j < len(shuffle95) else np.nan
        row[f"p{j + 1}"] = float(pvals[j]) if j < len(pvals) else np.nan
    return row


def save_confusion_plot(
    cm: np.ndarray,
    classes: Sequence[int],
    class_names: Dict[int, str],
    title: str,
    path: str,
) -> None:
    cm = np.asarray(cm, dtype=np.float64)
    row_sum = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, np.maximum(row_sum, 1.0))
    plt.figure(figsize=(6, 5))
    plt.imshow(norm, aspect="auto")
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
            plt.text(j, i, f"{norm[i, j]:.2f}\n({int(cm[i, j])})", ha="center", va="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_ld_scatter(
    Z: np.ndarray,
    y: np.ndarray,
    classes: Sequence[int],
    class_names: Dict[int, str],
    title: str,
    path: str,
    max_points: int,
    seed: int,
) -> None:
    if Z.shape[1] == 1:
        Z = np.column_stack([Z[:, 0], np.zeros(len(Z))])
    n = len(y)
    if n > max_points:
        idx = np.random.default_rng(seed).choice(n, max_points, replace=False)
        Z, y = Z[idx], y[idx]
    plt.figure(figsize=(6, 5))
    for c in classes:
        mask = y == c
        if np.any(mask):
            plt.scatter(Z[mask, 0], Z[mask, 1], s=6, alpha=0.4, label=class_names.get(int(c), str(c)))
    plt.xlabel("LD1")
    plt.ylabel("LD2")
    plt.title(title)
    plt.legend(fontsize=7, markerscale=2)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_model_summary_plot(summary_rows: List[Dict], outdir: str) -> None:
    plt.figure(figsize=(9, 6))
    for analysis in ANALYSIS_ORDER:
        rows = [r for r in summary_rows if r["analysis"] == analysis]
        rows = sorted(rows, key=lambda r: int(r["svd_dim"]))
        if not rows:
            continue
        xs = [int(r["svd_dim"]) for r in rows]
        ys = [float(r["balanced_acc_mean"]) for r in rows]
        es = [float(r["balanced_acc_sem"]) for r in rows]
        plt.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=analysis)
    plt.axhline(1 / 6, linestyle="--", linewidth=1, label="six-class chance")
    plt.axhline(1 / 3, linestyle=":", linewidth=1, label="three-class chance")
    plt.xlabel("Effective SVD dimension M")
    plt.ylabel("Mean subject-heldout balanced accuracy")
    plt.title("Shared six-class axes versus task-specific axes")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "summary_all_analyses_vs_svd_dim.png"), dpi=200)
    plt.close()


# -----------------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    ensure_dir(args.outdir)
    rng = np.random.default_rng(args.seed)
    requested_dims = parse_int_list(args.svd_dims)
    if not requested_dims:
        raise ValueError("--svd-dims must contain at least one integer")
    nback_override = parse_int_list(args.nback_class_ids)
    matb_override = parse_int_list(args.matb_class_ids)

    print(f"[{timestamp()}] Opening H5: {args.h5}", flush=True)
    with h5py.File(args.h5, "r") as f:
        if args.embedding_key not in f:
            raise KeyError(f"Embedding key not found: {args.embedding_key}")
        emb_ds = f[args.embedding_key]
        n_total = int(emb_ds.shape[0])
        flat_dim = get_flat_dim(emb_ds)
        raw_label_all = read_h5_vector(f, args.label_key).astype(np.int64)
        task_all = read_h5_vector(f, args.task_key).astype(np.int64)
        subj_all = read_h5_vector(f, args.subject_key).astype(np.int64)
        for name, arr in ((args.label_key, raw_label_all), (args.task_key, task_all), (args.subject_key, subj_all)):
            if len(arr) != n_total:
                raise ValueError(f"Length mismatch: embedding N={n_total}, {name} len={len(arr)}")

        nonrest = task_all != int(args.rest_task_value)
        raw_classes = sorted(int(x) for x in np.unique(raw_label_all[nonrest]))
        if len(raw_classes) != int(args.expected_task_classes):
            raise ValueError(
                f"Expected {args.expected_task_classes} non-rest raw labels, got {raw_classes}. "
                f"Check --label-key, --task-key, and --rest-task-value."
            )
        raw_to_id = {raw: i for i, raw in enumerate(raw_classes)}
        class_id_all = np.full(n_total, -1, dtype=np.int64)
        for raw, cid in raw_to_id.items():
            class_id_all[nonrest & (raw_label_all == raw)] = cid

        selected_idx = np.flatnonzero(nonrest & (class_id_all >= 0))
        y_sel = class_id_all[selected_idx]
        raw_y_sel = raw_label_all[selected_idx]
        task_sel = task_all[selected_idx]
        subj_sel = subj_all[selected_idx]

        if args.subjects:
            keep = np.asarray(parse_int_list(args.subjects), dtype=np.int64)
            mask = np.isin(subj_sel, keep)
            selected_idx = selected_idx[mask]
            y_sel = y_sel[mask]
            raw_y_sel = raw_y_sel[mask]
            task_sel = task_sel[mask]
            subj_sel = subj_sel[mask]

        all_classes = list(range(len(raw_classes)))
        inferred_nback = sorted(int(x) for x in np.unique(y_sel[task_sel == int(args.nback_task_value)]))
        inferred_matb = sorted(int(x) for x in np.unique(y_sel[task_sel == int(args.matb_task_value)]))
        nback_classes = nback_override if nback_override else inferred_nback
        matb_classes = matb_override if matb_override else inferred_matb
        if len(nback_classes) != 3 or len(matb_classes) != 3:
            raise ValueError(
                "N-Back and MATB must each contain exactly three classes. "
                f"Inferred N-Back={inferred_nback} from task_id={args.nback_task_value}; "
                f"MATB={inferred_matb} from task_id={args.matb_task_value}."
            )
        if set(nback_classes) & set(matb_classes):
            raise ValueError("N-Back and MATB class IDs must be disjoint")
        expected_union = sorted(set(nback_classes) | set(matb_classes))
        if expected_union != all_classes:
            raise ValueError(
                f"N-Back + MATB class IDs must cover all six classes. "
                f"Got union={expected_union}, expected={all_classes}."
            )
        # Unless the user explicitly overrides the groups, require exact agreement
        # with task_id so a raw-label ordering accident cannot swap the families.
        if not nback_override and nback_classes != inferred_nback:
            raise AssertionError("Internal N-Back inference mismatch")
        if not matb_override and matb_classes != inferred_matb:
            raise AssertionError("Internal MATB inference mismatch")

        subjects = sorted(int(x) for x in np.unique(subj_sel))
        class_names = make_class_names(raw_classes, "task6")
        fold_specs = build_subject_folds(args, subjects, subj_sel)

        # Requested dimensions are clipped to the raw feature dimension.  Duplicate
        # effective dimensions are removed.  BIOT therefore cannot reach 500/600
        # because its flattened embedding dimension is only 256.
        effective_dims = sorted(set(int(min(m, flat_dim)) for m in requested_dims))
        m_max = max(effective_dims)
        dim_map = {int(m): int(min(m, flat_dim)) for m in requested_dims}

        # Sanity-check how the remapped labels align with task_id.
        class_task_table: Dict[str, Dict[str, int]] = {}
        for cid, raw in enumerate(raw_classes):
            mask = y_sel == cid
            values, counts = np.unique(task_sel[mask], return_counts=True)
            class_task_table[str(cid)] = {
                "raw_label": int(raw),
                **{f"task_id_{int(v)}": int(c) for v, c in zip(values, counts)},
            }

        meta = {
            "script": os.path.basename(__file__),
            "created_at": timestamp(),
            "h5": args.h5,
            "model_name": args.model_name,
            "task_name": args.task_name,
            "window_seconds": float(args.window_seconds),
            "embedding_key": args.embedding_key,
            "label_key": args.label_key,
            "task_key": args.task_key,
            "subject_key": args.subject_key,
            "rest_task_value": int(args.rest_task_value),
            "nback_task_value": int(args.nback_task_value),
            "matb_task_value": int(args.matb_task_value),
            "raw_task_labels_sorted": raw_classes,
            "raw_to_class_id": {str(k): int(v) for k, v in raw_to_id.items()},
            "class_names": {str(k): v for k, v in class_names.items()},
            "nback_class_ids": nback_classes,
            "matb_class_ids": matb_classes,
            "class_task_table": class_task_table,
            "n_total_windows": n_total,
            "n_nonrest_windows": int(len(selected_idx)),
            "embedding_original_shape": [int(x) for x in emb_ds.shape],
            "flat_dim": flat_dim,
            "subjects": subjects,
            "cv": args.cv,
            "fold_specs": fold_specs,
            "svd_dims_requested": requested_dims,
            "requested_to_effective_dim": dim_map,
            "svd_dims_effective": effective_dims,
            "n_shuffle": int(args.n_shuffle),
            "lda_ridge": float(args.lda_ridge),
        }
        for k, v in f.attrs.items():
            try:
                meta[f"h5_attr/{k}"] = decode_attr(v)
            except Exception:
                pass
        with open(os.path.join(args.outdir, "run_metadata.json"), "w", encoding="utf-8") as fp:
            json.dump(meta, fp, indent=2, ensure_ascii=False)

        print("=" * 110, flush=True)
        print(f"Model: {args.model_name}", flush=True)
        print(f"Embedding shape: {emb_ds.shape}; flat_dim={flat_dim}", flush=True)
        print(f"Removed rest task_id={args.rest_task_value}; non-rest N={len(selected_idx)}", flush=True)
        print(f"Raw labels -> class IDs: {raw_to_id}", flush=True)
        print(f"N-Back class IDs: {nback_classes}; MATB class IDs: {matb_classes}", flush=True)
        print(f"Requested M: {requested_dims}; effective M: {effective_dims}", flush=True)
        if any(dim_map[m] != m for m in requested_dims):
            print(f"Dimension clipping map: {dim_map}", flush=True)
        print(f"Subject folds: {[x['heldout_subjects'] for x in fold_specs]}", flush=True)
        print("=" * 110, flush=True)

        fold_rows: List[Dict] = []
        cm_collect: Dict[Tuple[str, int], np.ndarray] = {}
        z_collect: Dict[Tuple[str, int], List[np.ndarray]] = {}
        y_collect: Dict[Tuple[str, int], List[np.ndarray]] = {}
        for analysis in ANALYSIS_ORDER:
            classes = all_classes if analysis == "task6_shared5" else (
                nback_classes if analysis.startswith("nback") else matb_classes
            )
            for m in effective_dims:
                cm_collect[(analysis, m)] = np.zeros((len(classes), len(classes)), dtype=np.int64)
                z_collect[(analysis, m)] = []
                y_collect[(analysis, m)] = []

        for fold_i, fold_spec in enumerate(fold_specs, start=1):
            t0 = time.time()
            heldout = np.asarray(fold_spec["heldout_subjects"], dtype=np.int64)
            test_mask_local = np.isin(subj_sel, heldout)
            train_local = np.flatnonzero(~test_mask_local)
            test_local = np.flatnonzero(test_mask_local)
            train_idx = selected_idx[train_local]
            test_idx = selected_idx[test_local]
            y_train = y_sel[train_local]
            y_test = y_sel[test_local]

            train_counts = {int(c): int(np.sum(y_train == c)) for c in all_classes}
            test_counts = {int(c): int(np.sum(y_test == c)) for c in all_classes}
            if any(v == 0 for v in train_counts.values()) or any(v == 0 for v in test_counts.values()):
                print(f"[{timestamp()}] Fold {fold_i} skipped: train={train_counts}, test={test_counts}", flush=True)
                continue

            print(
                f"[{timestamp()}] Fold {fold_i}/{len(fold_specs)} heldout={heldout.tolist()} "
                f"train={len(train_idx)} test={len(test_idx)}",
                flush=True,
            )
            X_train = read_embedding_rows(emb_ds, train_idx, args.read_batch_size, np.float32)
            X_test = read_embedding_rows(emb_ds, test_idx, args.read_batch_size, np.float32)
            mu = X_train.mean(axis=0, dtype=np.float64).astype(np.float32)
            X_train -= mu
            X_test -= mu

            m_eff_max = int(min(m_max, X_train.shape[0] - 1, X_train.shape[1]))
            U, S, Vt = randomized_svd_dense(
                X_train,
                n_components=m_eff_max,
                n_oversamples=args.svd_oversamples,
                n_iter=args.svd_power_iter,
                seed=args.seed + fold_i,
            )
            R_train_full = U * S[None, :]
            R_test_full = X_test @ Vt.T
            del X_train, X_test, U
            gc.collect()

            for m in effective_dims:
                m_eff = min(m, R_train_full.shape[1])
                R_train = R_train_full[:, :m_eff]
                R_test = R_test_full[:, :m_eff]

                # Shared six-class LDA, fitted once and reused by both task families.
                lda6 = fit_multiclass_lda(R_train, y_train, all_classes, args.lda_ridge)
                Z6_train = lda6.train_ld
                Z6_test = R_test @ lda6.W
                sh95_6, p6 = compute_eigen_null(
                    R_train, y_train, all_classes, lda6.eigvals,
                    args.n_shuffle, args.lda_ridge, rng,
                )

                results = []
                results.append(evaluate_projection(
                    "task6_shared5", Z6_train, Z6_test, y_train, y_test, all_classes
                ))
                results.append(evaluate_projection(
                    "nback_on_shared5", Z6_train, Z6_test, y_train, y_test, nback_classes
                ))
                results.append(evaluate_projection(
                    "matb_on_shared5", Z6_train, Z6_test, y_train, y_test, matb_classes
                ))
                results.append(evaluate_internal_lda(
                    "nback_internal2", R_train, R_test, y_train, y_test,
                    nback_classes, args.lda_ridge, args.n_shuffle, rng,
                ))
                results.append(evaluate_internal_lda(
                    "matb_internal2", R_train, R_test, y_train, y_test,
                    matb_classes, args.lda_ridge, args.n_shuffle, rng,
                ))

                for result in results:
                    use_shared = result["analysis"] in {
                        "task6_shared5", "nback_on_shared5", "matb_on_shared5"
                    }
                    row = result_to_row(
                        result=result,
                        args=args,
                        fold_id=fold_i,
                        heldout_subjects=heldout.tolist(),
                        svd_dim=m,
                        m_eff=m_eff,
                        n_train_total=len(y_train),
                        n_test_total=len(y_test),
                        shared_eigvals=lda6.eigvals[:5] if use_shared else None,
                        shared_shuffle95=sh95_6 if use_shared else None,
                        shared_pvals=p6 if use_shared else None,
                    )
                    row["train_class_counts_all6"] = json.dumps(train_counts)
                    row["test_class_counts_all6"] = json.dumps(test_counts)
                    fold_rows.append(row)
                    key = (result["analysis"], m)
                    cm_collect[key] += result["confusion_matrix"]
                    z_collect[key].append(result["Z_test"].astype(np.float32))
                    y_collect[key].append(result["y_test"].astype(np.int64))
                    print(
                        f"    M={m:<4d} {result['analysis']:<22s} "
                        f"bACC={result['balanced_acc']:.4f} acc={result['acc']:.4f} "
                        f"LDdim={result['Z_test'].shape[1]}",
                        flush=True,
                    )

            del R_train_full, R_test_full, Vt, S
            gc.collect()
            print(f"[{timestamp()}] Fold {fold_i} finished in {(time.time() - t0) / 60:.2f} min", flush=True)

    # Save fold metrics.
    fold_csv = os.path.join(args.outdir, "fold_metrics_all_analyses.csv")
    if fold_rows:
        with open(fold_csv, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(fold_rows[0].keys()))
            writer.writeheader()
            writer.writerows(fold_rows)
    print(f"[{timestamp()}] Saved fold metrics: {fold_csv}", flush=True)

    # Summary by analysis and effective SVD dimension.
    summary_rows: List[Dict] = []
    for analysis in ANALYSIS_ORDER:
        for m in effective_dims:
            rows = [r for r in fold_rows if r["analysis"] == analysis and int(r["svd_dim"]) == m]
            if not rows:
                continue
            b = np.asarray([float(r["balanced_acc"]) for r in rows], dtype=np.float64)
            a = np.asarray([float(r["acc"]) for r in rows], dtype=np.float64)
            n = len(rows)
            out = {
                "model": args.model_name,
                "analysis": analysis,
                "analysis_title": ANALYSIS_TITLES[analysis],
                "svd_dim": m,
                "n_folds": n,
                "ld_dim": int(rows[0]["ld_dim"]),
                "acc_mean": float(np.nanmean(a)),
                "acc_std": float(np.nanstd(a, ddof=1)) if n > 1 else 0.0,
                "acc_sem": float(np.nanstd(a, ddof=1) / math.sqrt(n)) if n > 1 else 0.0,
                "balanced_acc_mean": float(np.nanmean(b)),
                "balanced_acc_std": float(np.nanstd(b, ddof=1)) if n > 1 else 0.0,
                "balanced_acc_sem": float(np.nanstd(b, ddof=1) / math.sqrt(n)) if n > 1 else 0.0,
                "balanced_acc_min": float(np.nanmin(b)),
                "balanced_acc_max": float(np.nanmax(b)),
            }
            for j in range(5):
                out[f"lambda{j + 1}_mean"] = nanmean_or_nan([float(r[f"lambda{j + 1}"]) for r in rows])
                out[f"p{j + 1}_median"] = nanmedian_or_nan([float(r[f"p{j + 1}"]) for r in rows])
            summary_rows.append(out)

    summary_csv = os.path.join(args.outdir, "summary_by_analysis_and_svd_dim.csv")
    if summary_rows:
        with open(summary_csv, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
    print(f"[{timestamp()}] Saved summary: {summary_csv}", flush=True)

    # Direct comparison table: shared five-axis versus task-specific two-axis.
    comparison_rows: List[Dict] = []
    for family in ("nback", "matb"):
        shared_name = f"{family}_on_shared5"
        internal_name = f"{family}_internal2"
        for m in effective_dims:
            shared = next((r for r in summary_rows if r["analysis"] == shared_name and r["svd_dim"] == m), None)
            internal = next((r for r in summary_rows if r["analysis"] == internal_name and r["svd_dim"] == m), None)
            if shared and internal:
                comparison_rows.append({
                    "model": args.model_name,
                    "family": family,
                    "svd_dim": m,
                    "shared5_bacc": shared["balanced_acc_mean"],
                    "internal2_bacc": internal["balanced_acc_mean"],
                    "internal2_minus_shared5": internal["balanced_acc_mean"] - shared["balanced_acc_mean"],
                    "shared5_acc": shared["acc_mean"],
                    "internal2_acc": internal["acc_mean"],
                })
    comparison_csv = os.path.join(args.outdir, "shared5_vs_internal2_comparison.csv")
    if comparison_rows:
        with open(comparison_csv, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(comparison_rows[0].keys()))
            writer.writeheader()
            writer.writerows(comparison_rows)

    # Plots.
    for analysis in ANALYSIS_ORDER:
        classes = all_classes if analysis == "task6_shared5" else (
            nback_classes if analysis.startswith("nback") else matb_classes
        )
        for m in effective_dims:
            key = (analysis, m)
            analysis_dir = os.path.join(args.outdir, analysis)
            ensure_dir(analysis_dir)
            save_confusion_plot(
                cm_collect[key], classes, class_names,
                f"{args.model_name}: {ANALYSIS_TITLES[analysis]}, M={m}",
                os.path.join(analysis_dir, f"confusion_matrix_M{m}.png"),
            )
            if z_collect[key]:
                Z = np.concatenate(z_collect[key], axis=0)
                yy = np.concatenate(y_collect[key], axis=0)
                save_ld_scatter(
                    Z, yy, classes, class_names,
                    f"{args.model_name}: {ANALYSIS_TITLES[analysis]}, M={m}",
                    os.path.join(analysis_dir, f"heldout_ld12_scatter_M{m}.png"),
                    args.max_scatter_points, args.seed,
                )

    if summary_rows:
        save_model_summary_plot(summary_rows, args.outdir)

    result_json = os.path.join(args.outdir, "result_summary.json")
    with open(result_json, "w", encoding="utf-8") as fp:
        json.dump({
            "metadata": meta,
            "summary_by_analysis_and_svd_dim": summary_rows,
            "shared5_vs_internal2_comparison": comparison_rows,
        }, fp, indent=2, ensure_ascii=False)
    print(f"[{timestamp()}] Done. Output dir: {args.outdir}", flush=True)


# -----------------------------------------------------------------------------
# Batch mode
# -----------------------------------------------------------------------------


def _canonical_model_name(token: str) -> str:
    compact = "".join(ch for ch in str(token).lower() if ch.isalnum())
    aliases = {
        "biot": "BIOT",
        "labram": "LaBraM",
        "cbramod": "CBraMod",
        "eegpt": "EEGPT",
        "eegmamba": "EEGMamba",
        "sjepa": "s-JEPA",
    }
    if compact not in aliases:
        raise ValueError(f"Unknown model: {token}")
    return aliases[compact]


def parse_model_list(s: str) -> List[str]:
    raw = [x.strip() for x in str(s).split(",") if x.strip()]
    if not raw or (len(raw) == 1 and raw[0].lower() == "all"):
        return list(COGBCI_5S_MODEL_PATHS)
    out: List[str] = []
    for token in raw:
        name = _canonical_model_name(token)
        if name not in out:
            out.append(name)
    return out


def inspect_selected_subjects(path: str, args: argparse.Namespace) -> Tuple[np.ndarray, List[int], List[int]]:
    with h5py.File(path, "r") as f:
        n = int(f[args.embedding_key].shape[0])
        raw = read_h5_vector(f, args.label_key).astype(np.int64)
        task = read_h5_vector(f, args.task_key).astype(np.int64)
        subj = read_h5_vector(f, args.subject_key).astype(np.int64)
        if len(raw) != n or len(task) != n or len(subj) != n:
            raise ValueError(f"Length mismatch in {path}")
    mask = task != int(args.rest_task_value)
    raw_classes = sorted(int(x) for x in np.unique(raw[mask]))
    subj_sel = subj[mask]
    if args.subjects:
        keep = np.asarray(parse_int_list(args.subjects), dtype=np.int64)
        subj_sel = subj_sel[np.isin(subj_sel, keep)]
    return subj_sel, sorted(int(x) for x in np.unique(subj_sel)), raw_classes


def groups_to_cli(groups: Sequence[Sequence[int]]) -> str:
    return ";".join(",".join(str(int(s)) for s in g) for g in groups)


def prepare_shared_batch_folds(
    model_paths: Dict[str, str],
    args: argparse.Namespace,
) -> Tuple[str, Dict[str, Dict]]:
    inspections: Dict[str, Dict] = {}
    ref_subjects: Optional[List[int]] = None
    ref_raw_classes: Optional[List[int]] = None
    ref_vector: Optional[np.ndarray] = None
    for model, path in model_paths.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"5-second H5 not found for {model}: {path}")
        subj_sel, subjects, raw_classes = inspect_selected_subjects(path, args)
        inspections[model] = {
            "h5": path,
            "n_nonrest_windows": int(len(subj_sel)),
            "subjects": subjects,
            "raw_task_labels": raw_classes,
            "subject_window_counts": {str(s): int(np.sum(subj_sel == s)) for s in subjects},
        }
        if ref_subjects is None:
            ref_subjects = subjects
            ref_raw_classes = raw_classes
            ref_vector = subj_sel
        else:
            if subjects != ref_subjects:
                raise ValueError(f"Subject set mismatch for {model}")
            if raw_classes != ref_raw_classes:
                raise ValueError(f"Raw task label mismatch for {model}: {raw_classes} vs {ref_raw_classes}")
    assert ref_subjects is not None and ref_vector is not None
    if args.heldout_subject_groups:
        shared = args.heldout_subject_groups
    elif args.cv == "loso":
        shared = groups_to_cli([[s] for s in ref_subjects])
    else:
        shared = groups_to_cli(make_subject_kfold_groups(
            ref_subjects, ref_vector, args.n_subject_folds, args.seed
        ))
    return shared, inspections


def save_batch_summary(outroot: str, models: Sequence[str]) -> None:
    combined: List[Dict] = []
    comparisons: List[Dict] = []
    for model in models:
        path = os.path.join(outroot, model, "summary_by_analysis_and_svd_dim.csv")
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8", newline="") as fp:
                for row in csv.DictReader(fp):
                    row = dict(row)
                    row["model"] = model
                    combined.append(row)
        comp_path = os.path.join(outroot, model, "shared5_vs_internal2_comparison.csv")
        if os.path.isfile(comp_path):
            with open(comp_path, "r", encoding="utf-8", newline="") as fp:
                for row in csv.DictReader(fp):
                    row = dict(row)
                    row["model"] = model
                    comparisons.append(row)
    if not combined:
        print(f"[{timestamp()}] No summaries found; batch aggregation skipped", flush=True)
        return

    combined_path = os.path.join(outroot, "all_models_all_analyses_summary.csv")
    with open(combined_path, "w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(combined[0].keys()))
        writer.writeheader()
        writer.writerows(combined)

    if comparisons:
        path = os.path.join(outroot, "all_models_shared5_vs_internal2_comparison.csv")
        with open(path, "w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(comparisons[0].keys()))
            writer.writeheader()
            writer.writerows(comparisons)

    # One cross-model plot per analysis keeps the figure readable.
    for analysis in ANALYSIS_ORDER:
        plt.figure(figsize=(8, 5))
        chance = 1 / 6 if analysis == "task6_shared5" else 1 / 3
        for model in models:
            rows = [r for r in combined if r["model"] == model and r["analysis"] == analysis]
            rows = sorted(rows, key=lambda r: int(r["svd_dim"]))
            if rows:
                xs = [int(r["svd_dim"]) for r in rows]
                ys = [float(r["balanced_acc_mean"]) for r in rows]
                es = [float(r["balanced_acc_sem"]) for r in rows]
                plt.errorbar(xs, ys, yerr=es, marker="o", capsize=2, label=model)
        plt.axhline(chance, linestyle="--", linewidth=1, label=f"chance={chance:.3f}")
        plt.xlabel("Effective SVD dimension M")
        plt.ylabel("Mean subject-heldout balanced accuracy")
        plt.title(ANALYSIS_TITLES[analysis])
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(outroot, f"all_models_{analysis}_vs_svd_dim.png"), dpi=200)
        plt.close()

    print(f"[{timestamp()}] Saved batch summary: {combined_path}", flush=True)


def run_batch_5s(args: argparse.Namespace) -> None:
    models = parse_model_list(args.models)
    model_paths = {model: COGBCI_5S_MODEL_PATHS[model] for model in models}
    ensure_dir(args.outroot)
    shared_groups, inspections = prepare_shared_batch_folds(model_paths, args)
    with open(os.path.join(args.outroot, "batch_metadata.json"), "w", encoding="utf-8") as fp:
        json.dump({
            "script": os.path.basename(__file__),
            "created_at": timestamp(),
            "models": models,
            "model_paths": model_paths,
            "shared_heldout_subject_groups": shared_groups,
            "inspections": inspections,
            "svd_dims_requested": parse_int_list(args.svd_dims),
        }, fp, indent=2, ensure_ascii=False)

    print("=" * 110, flush=True)
    print("CogBCI 5-second task6 shared-vs-internal batch", flush=True)
    print(f"Models: {models}", flush=True)
    print(f"Shared held-out groups: {shared_groups}", flush=True)
    print(f"Output root: {args.outroot}", flush=True)
    print("=" * 110, flush=True)

    for i, model in enumerate(models, start=1):
        model_args = argparse.Namespace(**vars(args).copy())
        model_args.batch_5s = False
        model_args.h5 = model_paths[model]
        model_args.model_name = model
        model_args.outdir = os.path.join(args.outroot, model)
        model_args.heldout_subject_groups = shared_groups
        print("\n" + "#" * 110, flush=True)
        print(f"[{timestamp()}] Model {i}/{len(models)}: {model}", flush=True)
        print("#" * 110, flush=True)
        run(model_args)

    save_batch_summary(args.outroot, models)
    print(f"[{timestamp()}] All requested models finished", flush=True)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CogBCI 5-second six-task shared LDA versus N-Back/MATB internal LDA"
    )
    p.add_argument("--batch-5s", action="store_true")
    p.add_argument("--models", default="BIOT,LaBraM,CBraMod,EEGPT,EEGMamba,s-JEPA")
    p.add_argument(
        "--outroot",
        default="/mnt/dataset4/yinuo/FM_flow/dataset/expA_task6_shared_vs_internal_5s_M40_500_600",
    )
    p.add_argument("--h5", default="")
    p.add_argument("--model-name", default="")
    p.add_argument("--outdir", default="")

    p.add_argument("--embedding-key", default="embedding")
    p.add_argument("--label-key", default="label", help="Raw six-way task label key")
    p.add_argument("--task-key", default="task_id", help="Task-family key; rest is task_id=0")
    p.add_argument("--subject-key", default="subject_id")
    p.add_argument("--rest-task-value", type=int, default=0)
    p.add_argument("--nback-task-value", type=int, default=1)
    p.add_argument("--matb-task-value", type=int, default=2)
    p.add_argument("--expected-task-classes", type=int, default=6)
    p.add_argument(
        "--nback-class-ids", default="",
        help="Optional override. Empty means infer the three class IDs from --nback-task-value.",
    )
    p.add_argument(
        "--matb-class-ids", default="",
        help="Optional override. Empty means infer the three class IDs from --matb-task-value.",
    )

    p.add_argument("--subjects", default="")
    p.add_argument("--cv", default="subject_kfold", choices=["subject_kfold", "loso"])
    p.add_argument("--n-subject-folds", type=int, default=4)
    p.add_argument("--folds", default="")
    p.add_argument("--heldout-subject-groups", default="")

    p.add_argument("--task-name", default="task6_shared5_vs_nback_matb_internal3_5s")
    p.add_argument("--window-seconds", type=float, default=5.0)
    p.add_argument("--svd-dims", default="40,500,600")
    p.add_argument("--svd-oversamples", type=int, default=20)
    p.add_argument("--svd-power-iter", type=int, default=2)
    p.add_argument("--lda-ridge", type=float, default=1e-4)
    p.add_argument("--n-shuffle", type=int, default=0)
    p.add_argument("--read-batch-size", type=int, default=4096)
    p.add_argument("--max-scatter-points", type=int, default=8000)
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    if args.batch_5s:
        run_batch_5s(args)
        return
    missing = [name for name, value in (
        ("--h5", args.h5), ("--model-name", args.model_name), ("--outdir", args.outdir)
    ) if not value]
    if missing:
        parser.error("single-file mode requires " + ", ".join(missing) + "; or use --batch-5s")
    run(args)


if __name__ == "__main__":
    main()
