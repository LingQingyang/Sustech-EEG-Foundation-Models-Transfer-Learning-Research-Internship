#!/usr/bin/env python3
"""
Experiment A subject-heldout on 5-second CogBCI windows: discriminant-spectrum diagnosis.

Goal
----
Cross-subject version of Experiment A for the 5-second-window CogBCI embeddings.
The default split is subject-group 4-fold CV, so each held-out test fold is
approximately 1/4 of all windows while still holding out whole subjects.

The script has two modes:
  - batch mode (recommended): run the six registered 5-second FM embeddings with
    identical subject folds and collect a cross-model summary;
  - single-file mode: run one arbitrary H5 file.
For each held-out subject group:
  1) fit train-only mean and train-only randomized SVD on embeddings;
  2) learn multiclass LD/LDA discriminant directions in the SVD score space;
  3) project the held-out subjects with the train mean + train SVD + train LD;
  4) evaluate whether the global three classes are separable.

Default CogBCI global 3-class setting:
  label-key = task_id
  classes   = 0,1,2 = Resting, N-Back, MATB

Example
-------
# Recommended: run all six 5-second embeddings with shared subject folds
python3 run_experiment_A_subject_heldout_discriminant_spectrum_5s.py \
  --batch-5s \
  --cv subject_kfold \
  --n-subject-folds 4 \
  --svd-dims 100,200,300,500 \
  --n-shuffle 50 \
  --outroot /mnt/dataset4/yinuo/FM_flow/dataset/expA_subject4fold_cogbci_5s_taskid3

# Single-model smoke test
python3 run_experiment_A_subject_heldout_discriminant_spectrum_5s.py \
  --h5 /mnt/dataset4/fuxy/FMS/FM/LaBraM/output_emb_5s/cogbci_sesS1_embeddings.h5 \
  --model-name LaBraM \
  --outdir /mnt/dataset4/yinuo/FM_flow/dataset/expA_subject4fold_cogbci_5s_taskid3/LaBraM_smoke \
  --folds 1 \
  --svd-dims 100 \
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


# Exact 5-second CogBCI embedding paths supplied by Xinyu.  Keeping this registry
# in the script avoids six nearly identical commands and, more importantly, lets
# batch mode enforce identical held-out subject groups across models.
COGBCI_5S_MODEL_PATHS: Dict[str, str] = {
    "BIOT": "/mnt/dataset4/fuxy/FMS/FM/BIOT/output_emb_5s/cogbci_sesS1_embeddings.h5",
    "LaBraM": "/mnt/dataset4/fuxy/FMS/FM/LaBraM/output_emb_5s/cogbci_sesS1_embeddings.h5",
    "CBraMod": "/mnt/dataset4/fuxy/FMS/FM/CBraMod/output_emb_5s/cogbci_sesS1_embeddings.h5",
    "EEGPT": "/mnt/dataset4/fuxy/FMS/FM/EEGPT/output_emb_5s/cogbci_sesS1_embeddings.h5",
    "EEGMamba": "/mnt/dataset4/fuxy/FMS/FM/EEGMamba/output_emb_5s/cogbci_sesS1_embeddings.h5",
    "s-JEPA": "/mnt/dataset4/fuxy/FMS/FM/s-JEPA/output_5s/cogbci_sesS1_embeddings.h5",
}


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_int_list(s: str) -> List[int]:
    if s is None or str(s).strip() == "":
        return []
    return [int(x.strip()) for x in str(s).split(",") if x.strip() != ""]


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def nanmedian_or_nan(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or np.all(np.isnan(x)):
        return float("nan")
    return float(np.nanmedian(x))


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
    """Read selected rows from an H5 embedding dataset and flatten to (n, d).

    h5py fancy indexing is happiest with sorted unique integer indices. This function
    assumes indices are sorted in ascending order, which is true for np.flatnonzero.
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


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, classes: Sequence[int]) -> Tuple[float, Dict[int, float]]:
    recalls = {}
    for c in classes:
        mask = (y_true == c)
        if np.sum(mask) == 0:
            recalls[int(c)] = float("nan")
        else:
            recalls[int(c)] = float(np.mean(y_pred[mask] == c))
    vals = [v for v in recalls.values() if not math.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan"), recalls


def confusion_matrix_fixed(y_true: np.ndarray, y_pred: np.ndarray, classes: Sequence[int]) -> np.ndarray:
    class_to_idx = {int(c): i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for yt, yp in zip(y_true, y_pred):
        if int(yt) in class_to_idx and int(yp) in class_to_idx:
            cm[class_to_idx[int(yt)], class_to_idx[int(yp)]] += 1
    return cm


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
    """Compute a randomized truncated SVD for a dense centered matrix X.

    Returns U, S, Vt such that X ≈ U @ diag(S) @ Vt.
    X is expected to be centered already. To save memory, centering should be done
    in-place before calling this function.
    """
    n, d = X.shape
    k = int(min(n_components, n, d))
    if k <= 0:
        raise ValueError(f"Invalid n_components={n_components} for X shape={X.shape}")

    # For tiny feature spaces, exact SVD can be faster and cleaner.
    if k >= min(n, d) - 1 and min(n, d) <= 512:
        U, S, Vt = np.linalg.svd(X.astype(np.float32, copy=False), full_matrices=False)
        return U[:, :k], S[:k], Vt[:k, :]

    rng = np.random.default_rng(seed)
    l = int(min(k + n_oversamples, d))
    Omega = rng.standard_normal(size=(d, l)).astype(np.float32)

    # Sample column space: Y = X Omega, with optional power iterations.
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
) -> LDAResult:
    """Fit multiclass LDA in the SVD score space.

    R: (n, m) SVD scores.
    y: labels.
    For C classes, returns at most C-1 LD directions.
    """
    R = np.asarray(R, dtype=np.float64)
    y = np.asarray(y)
    n, m = R.shape
    C = len(classes)
    q = min(C - 1, m)
    if q <= 0:
        raise ValueError("Need at least two classes and one feature dimension for LDA")

    mu = R.mean(axis=0)
    Sw = np.zeros((m, m), dtype=np.float64)
    Sb = np.zeros((m, m), dtype=np.float64)

    for c in classes:
        Rc = R[y == c]
        if len(Rc) == 0:
            raise ValueError(f"Class {c} has no samples in training fold")
        muc = Rc.mean(axis=0)
        Xc = Rc - muc
        Sw += Xc.T @ Xc
        dm = (muc - mu).reshape(-1, 1)
        Sb += len(Rc) * (dm @ dm.T)

    # Normalize scatter matrices. The generalized eigenvectors are invariant to
    # common scaling, but normalization keeps ridge numerically interpretable.
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
        # Fallback. Less elegant but avoids hard dependency failure.
        A = np.linalg.solve(Sw_reg, Sb)
        eigvals, eigvecs = np.linalg.eig(A)
        eigvals = np.real(eigvals)
        eigvecs = np.real(eigvecs)

    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    W = eigvecs[:, order[:q]]

    # Normalize columns for stable projection scale.
    for j in range(W.shape[1]):
        norm = np.linalg.norm(W[:, j])
        if norm > 0:
            W[:, j] /= norm

    train_ld = R @ W
    centroids = {int(c): train_ld[y == c].mean(axis=0) for c in classes}
    return LDAResult(eigvals=eigvals, W=W, class_centroids_ld=centroids, train_ld=train_ld)


def predict_nearest_centroid(Z: np.ndarray, centroids: Dict[int, np.ndarray]) -> np.ndarray:
    classes = list(centroids.keys())
    Cmat = np.stack([centroids[c] for c in classes], axis=0)
    # squared Euclidean distances, shape (n, C)
    dist = np.sum((Z[:, None, :] - Cmat[None, :, :]) ** 2, axis=2)
    pred_idx = np.argmin(dist, axis=1)
    return np.array([classes[i] for i in pred_idx], dtype=np.int64)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def save_balanced_acc_plot(rows: List[Dict], svd_dims: List[int], outdir: str) -> None:
    plt.figure(figsize=(8, 5))
    x_key = "heldout_fold" if rows and "heldout_fold" in rows[0] else "heldout_subject"
    for m in svd_dims:
        xs, ys = [], []
        for r in rows:
            if int(r["svd_dim"]) == int(m):
                xs.append(int(r[x_key]))
                ys.append(float(r["balanced_acc"]))
        if xs:
            order = np.argsort(xs)
            xs = np.array(xs)[order]
            ys = np.array(ys)[order]
            plt.plot(xs, ys, marker="o", linewidth=1, label=f"M={m}")
    plt.axhline(1.0 / 3.0, linestyle="--", linewidth=1, label="chance=1/3")
    plt.xlabel("Held-out fold" if x_key == "heldout_fold" else "Held-out subject")
    plt.ylabel("Balanced accuracy")
    plt.title("Subject-heldout balanced accuracy")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "subject_heldout_balanced_acc.png"), dpi=200)
    plt.close()


def save_summary_acc_plot(summary_rows: List[Dict], outdir: str) -> None:
    xs = [int(r["svd_dim"]) for r in summary_rows]
    means = [float(r["balanced_acc_mean"]) for r in summary_rows]
    sems = [float(r["balanced_acc_sem"]) for r in summary_rows]
    plt.figure(figsize=(6, 4))
    plt.errorbar(xs, means, yerr=sems, marker="o", capsize=3)
    plt.axhline(1.0 / 3.0, linestyle="--", linewidth=1, label="chance=1/3")
    plt.xlabel("SVD dimension M")
    plt.ylabel("Mean balanced accuracy")
    plt.title("Subject-heldout mean balanced accuracy vs SVD dimension")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "summary_balanced_acc_vs_svd_dim.png"), dpi=200)
    plt.close()


def save_confusion_plot(cm: np.ndarray, classes: List[int], class_names: Dict[int, str], title: str, path: str) -> None:
    cm = cm.astype(np.float64)
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, np.maximum(row_sum, 1.0))

    plt.figure(figsize=(5, 4))
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
    plt.savefig(path, dpi=200)
    plt.close()


def save_ld_scatter(
    Z: np.ndarray,
    y: np.ndarray,
    subject: np.ndarray,
    classes: List[int],
    class_names: Dict[int, str],
    title: str,
    path: str,
    max_points: int = 8000,
    seed: int = 0,
) -> None:
    if Z.shape[1] == 1:
        Z = np.concatenate([Z, np.zeros((Z.shape[0], 1))], axis=1)
    n = len(y)
    rng = np.random.default_rng(seed)
    if n > max_points:
        idx = rng.choice(n, size=max_points, replace=False)
        Zp, yp, sp = Z[idx], y[idx], subject[idx]
    else:
        Zp, yp, sp = Z, y, subject

    plt.figure(figsize=(6, 5))
    for c in classes:
        mask = (yp == c)
        if np.any(mask):
            plt.scatter(Zp[mask, 0], Zp[mask, 1], s=6, alpha=0.45, label=class_names.get(int(c), str(c)))
    plt.xlabel("LD1")
    plt.ylabel("LD2")
    plt.title(title)
    plt.legend(fontsize=8, markerscale=2)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


# -----------------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------------


def infer_class_names(label_key: str, classes: List[int]) -> Dict[int, str]:
    if label_key == "task_id" and set(classes) == {0, 1, 2}:
        return {0: "Resting", 1: "N-Back", 2: "MATB"}
    return {int(c): str(c) for c in classes}


def parse_subject_groups(s: str) -> List[List[int]]:
    """Parse manual subject groups like '1,2,3;4,5,6;7,8'."""
    if s is None or str(s).strip() == "":
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
    seed: int = 0,
) -> List[List[int]]:
    """Create subject-level folds with roughly balanced sample counts.

    This is a greedy bin-packing split over subjects: whole subjects are assigned
    to folds, and the currently lightest fold receives the next largest subject.
    The goal is heldout/total ≈ 1/n_folds while preventing subject leakage.
    """
    subjects = [int(s) for s in subjects]
    if n_folds < 2:
        raise ValueError("--n-subject-folds must be at least 2 for subject_kfold")
    if n_folds > len(subjects):
        raise ValueError(f"--n-subject-folds={n_folds} exceeds number of subjects={len(subjects)}")

    counts = {int(s): int(np.sum(subj_sel == int(s))) for s in subjects}
    rng = np.random.default_rng(seed)
    # Shuffle first so subjects with identical or near-identical counts do not
    # always fall into the same deterministic order.
    shuffled = list(subjects)
    rng.shuffle(shuffled)
    ordered = sorted(shuffled, key=lambda x: counts[x], reverse=True)

    groups: List[List[int]] = [[] for _ in range(n_folds)]
    totals = [0 for _ in range(n_folds)]
    for s in ordered:
        j = int(np.argmin(totals))
        groups[j].append(int(s))
        totals[j] += counts[int(s)]

    # Sort subjects inside each fold for readability; sort folds by first subject
    # so fold IDs are stable and human-friendly.
    groups = [sorted(g) for g in groups]
    groups = sorted(groups, key=lambda g: (min(g), len(g)))
    return groups


def build_subject_folds(args: argparse.Namespace, subjects: List[int], subj_sel: np.ndarray) -> List[Dict]:
    """Return fold specs with subject-heldout groups."""
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


def run(args: argparse.Namespace) -> None:
    ensure_dir(args.outdir)
    rng = np.random.default_rng(args.seed)

    include_labels = parse_int_list(args.include_labels)
    svd_dims_requested = parse_int_list(args.svd_dims)
    if not svd_dims_requested:
        raise ValueError("--svd-dims must contain at least one integer, e.g. 100,200,300,500")

    print(f"[{timestamp()}] Opening H5: {args.h5}", flush=True)
    with h5py.File(args.h5, "r") as f:
        if args.embedding_key not in f:
            raise KeyError(f"Embedding key not found: {args.embedding_key}")
        emb_ds = f[args.embedding_key]
        N_total = emb_ds.shape[0]
        d = get_flat_dim(emb_ds)

        y_all = read_h5_vector(f, args.label_key)
        subj_all = read_h5_vector(f, args.subject_key)
        if len(y_all) != N_total or len(subj_all) != N_total:
            raise ValueError(
                f"Length mismatch: embedding N={N_total}, label len={len(y_all)}, subject len={len(subj_all)}"
            )

        if include_labels:
            mask = np.isin(y_all, include_labels)
        else:
            include_labels = sorted([int(x) for x in np.unique(y_all)])
            mask = np.ones_like(y_all, dtype=bool)

        selected_idx = np.flatnonzero(mask)
        y_sel = y_all[selected_idx].astype(np.int64)
        subj_sel = subj_all[selected_idx].astype(np.int64)
        classes = sorted([int(c) for c in include_labels])
        subjects = sorted([int(s) for s in np.unique(subj_sel)])
        class_names = infer_class_names(args.label_key, classes)

        if args.subjects:
            # Candidate subject pool. In LOSO, this behaves like the old smoke-test
            # option. In subject_kfold, folds are built only from this subject pool.
            keep_subjects = set(parse_int_list(args.subjects))
            subjects = [s for s in subjects if s in keep_subjects]
            candidate_mask = np.isin(subj_sel, np.asarray(subjects, dtype=np.int64))
            selected_idx = selected_idx[candidate_mask]
            y_sel = y_sel[candidate_mask]
            subj_sel = subj_sel[candidate_mask]
            subjects = sorted([int(s) for s in np.unique(subj_sel)])

        fold_specs = build_subject_folds(args, subjects, subj_sel)

        # SVD dims cannot exceed the raw feature dimension.
        svd_dims = [int(min(m, d)) for m in svd_dims_requested]
        svd_dims = sorted(set(svd_dims))
        m_max = max(svd_dims)

        meta = {
            "script": os.path.basename(__file__),
            "created_at": timestamp(),
            "h5": args.h5,
            "model_name": args.model_name,
            "task_name": args.task_name,
            "window_seconds": float(args.window_seconds),
            "embedding_key": args.embedding_key,
            "label_key": args.label_key,
            "subject_key": args.subject_key,
            "include_labels": classes,
            "class_names": class_names,
            "n_total_windows": int(N_total),
            "n_selected_windows": int(len(selected_idx)),
            "embedding_original_shape": tuple(int(x) for x in emb_ds.shape),
            "flat_dim": int(d),
            "subjects": subjects,
            "cv": args.cv,
            "n_subject_folds": int(args.n_subject_folds),
            "heldout_subject_groups": [fs["heldout_subjects"] for fs in fold_specs],
            "fold_specs": fold_specs,
            "svd_dims_requested": svd_dims_requested,
            "svd_dims_used": svd_dims,
            "n_shuffle": int(args.n_shuffle),
            "lda_ridge": float(args.lda_ridge),
            "svd_oversamples": int(args.svd_oversamples),
            "svd_power_iter": int(args.svd_power_iter),
        }
        for k, v in f.attrs.items():
            try:
                meta[f"h5_attr/{k}"] = decode_attr(v)
            except Exception:
                pass

        with open(os.path.join(args.outdir, "run_metadata.json"), "w", encoding="utf-8") as fp:
            json.dump(meta, fp, indent=2, ensure_ascii=False)

        print("=" * 100, flush=True)
        print(f"Model: {args.model_name}", flush=True)
        print(f"Task:  {args.task_name}", flush=True)
        print(f"Embedding shape: {emb_ds.shape}, flat_dim={d}", flush=True)
        print(f"Selected labels/classes: {classes} ({class_names})", flush=True)
        print(f"Selected samples: {len(selected_idx)} / {N_total}", flush=True)
        print(f"Subjects: {len(subjects)} -> {subjects}", flush=True)
        print(f"CV: {args.cv}; folds={len(fold_specs)}", flush=True)
        for fs in fold_specs:
            print(
                f"  fold {fs['fold_id']}: heldout_subjects={fs['heldout_subjects']} "
                f"n_test≈{fs['estimated_n_test']} frac≈{fs['estimated_test_fraction']:.3f}",
                flush=True,
            )
        print(f"SVD dims: {svd_dims}", flush=True)
        print("=" * 100, flush=True)

        fold_rows: List[Dict] = []
        # For aggregate plots and confusion matrices.
        cm_by_m: Dict[int, np.ndarray] = {m: np.zeros((len(classes), len(classes)), dtype=np.int64) for m in svd_dims}
        ld_collect: Dict[int, List[np.ndarray]] = {m: [] for m in svd_dims}
        y_collect: Dict[int, List[np.ndarray]] = {m: [] for m in svd_dims}
        s_collect: Dict[int, List[np.ndarray]] = {m: [] for m in svd_dims}

        for fold_i, fold_spec in enumerate(fold_specs, start=1):
            t0 = time.time()
            heldout_subjects = np.asarray(fold_spec["heldout_subjects"], dtype=np.int64)
            is_test = np.isin(subj_sel, heldout_subjects)
            train_local = np.flatnonzero(~is_test)
            test_local = np.flatnonzero(is_test)
            train_idx = selected_idx[train_local]
            test_idx = selected_idx[test_local]
            y_train = y_sel[train_local]
            y_test = y_sel[test_local]
            subj_test = subj_sel[test_local]

            # Check class coverage in this fold.
            train_counts = {int(c): int(np.sum(y_train == c)) for c in classes}
            test_counts = {int(c): int(np.sum(y_test == c)) for c in classes}
            if any(v == 0 for v in train_counts.values()) or any(v == 0 for v in test_counts.values()):
                print(
                    f"[{timestamp()}] Fold {fold_i}/{len(fold_specs)} heldout_subjects={heldout_subjects.tolist()}: "
                    f"skipped due to missing class. train={train_counts}, test={test_counts}",
                    flush=True,
                )
                continue

            heldout_frac = float(len(test_idx) / max(len(selected_idx), 1))
            print(
                f"[{timestamp()}] Fold {fold_i}/{len(fold_specs)} heldout_subjects={heldout_subjects.tolist()} | "
                f"reading train={len(train_idx)}, test={len(test_idx)}, test/total={heldout_frac:.4f}",
                flush=True,
            )
            X_train = read_embedding_rows(emb_ds, train_idx, batch_size=args.read_batch_size, dtype=np.float32)
            X_test = read_embedding_rows(emb_ds, test_idx, batch_size=args.read_batch_size, dtype=np.float32)

            # Train-only centering. Test subject never contributes to the mean.
            mu = X_train.mean(axis=0, dtype=np.float64).astype(np.float32)
            X_train -= mu
            X_test -= mu

            # Train-only SVD.
            m_eff_max = int(min(m_max, X_train.shape[0] - 1, X_train.shape[1]))
            if m_eff_max < m_max:
                print(f"[{timestamp()}] Fold {fold_i}: m_max clipped {m_max} -> {m_eff_max}", flush=True)
            U, S, Vt = randomized_svd_dense(
                X_train,
                n_components=m_eff_max,
                n_oversamples=args.svd_oversamples,
                n_iter=args.svd_power_iter,
                seed=args.seed + fold_i,
            )
            # Scores for train are U*S. Test scores use train Vt.
            R_train_full = U * S[None, :]
            R_test_full = X_test @ Vt.T

            # Free the largest raw matrices as early as possible.
            del X_train, X_test, U
            gc.collect()

            for m in svd_dims:
                m_eff = int(min(m, R_train_full.shape[1]))
                R_train = R_train_full[:, :m_eff]
                R_test = R_test_full[:, :m_eff]

                lda = fit_multiclass_lda(R_train, y_train, classes=classes, ridge=args.lda_ridge)
                Z_test = R_test @ lda.W
                y_pred = predict_nearest_centroid(Z_test, lda.class_centroids_ld)
                acc = float(np.mean(y_pred == y_test))
                bacc, recalls = balanced_accuracy(y_test, y_pred, classes)
                cm = confusion_matrix_fixed(y_test, y_pred, classes)
                cm_by_m[m] += cm

                # Shuffle null: keep SVD scores fixed, shuffle training labels only.
                shuffle_top = []
                if args.n_shuffle > 0:
                    for sh in range(args.n_shuffle):
                        y_shuf = np.array(y_train, copy=True)
                        rng.shuffle(y_shuf)
                        lda_shuf = fit_multiclass_lda(R_train, y_shuf, classes=classes, ridge=args.lda_ridge)
                        vals = lda_shuf.eigvals[: len(classes) - 1]
                        if len(vals) < len(classes) - 1:
                            vals = np.pad(vals, (0, len(classes) - 1 - len(vals)), constant_values=np.nan)
                        shuffle_top.append(vals)
                    shuffle_top = np.asarray(shuffle_top, dtype=np.float64)
                    shuffle95 = np.nanpercentile(shuffle_top, 95, axis=0)
                    pvals = []
                    for j in range(len(classes) - 1):
                        real_val = lda.eigvals[j] if j < len(lda.eigvals) else np.nan
                        null_vals = shuffle_top[:, j]
                        pvals.append(float((1.0 + np.sum(null_vals >= real_val)) / (1.0 + len(null_vals))))
                else:
                    shuffle95 = np.full((len(classes) - 1,), np.nan)
                    pvals = [np.nan] * (len(classes) - 1)

                eig = lda.eigvals[: len(classes) - 1]
                if len(eig) < len(classes) - 1:
                    eig = np.pad(eig, (0, len(classes) - 1 - len(eig)), constant_values=np.nan)

                row = {
                    "model": args.model_name,
                    "task_name": args.task_name,
                    "h5": args.h5,
                    "heldout_fold": int(fold_i),
                    "heldout_subjects": json.dumps([int(x) for x in heldout_subjects.tolist()], ensure_ascii=False),
                    "n_heldout_subjects": int(len(heldout_subjects)),
                    "heldout_fraction": float(len(y_test) / max(len(selected_idx), 1)),
                    "svd_dim": int(m),
                    "svd_dim_effective": int(m_eff),
                    "n_train": int(len(y_train)),
                    "n_test": int(len(y_test)),
                    "train_class_counts": json.dumps(train_counts, ensure_ascii=False),
                    "test_class_counts": json.dumps(test_counts, ensure_ascii=False),
                    "acc": acc,
                    "balanced_acc": bacc,
                    "class_recalls": json.dumps(recalls, ensure_ascii=False),
                    "lambda1": float(eig[0]) if len(eig) > 0 else np.nan,
                    "lambda2": float(eig[1]) if len(eig) > 1 else np.nan,
                    "shuffle95_1": float(shuffle95[0]) if len(shuffle95) > 0 else np.nan,
                    "shuffle95_2": float(shuffle95[1]) if len(shuffle95) > 1 else np.nan,
                    "p1": float(pvals[0]) if len(pvals) > 0 else np.nan,
                    "p2": float(pvals[1]) if len(pvals) > 1 else np.nan,
                    "confusion_matrix": json.dumps(cm.tolist()),
                }
                fold_rows.append(row)

                ld_collect[m].append(Z_test.astype(np.float32))
                y_collect[m].append(y_test.astype(np.int64))
                s_collect[m].append(subj_test.astype(np.int64))

                print(
                    f"    M={m:<4d} bacc={bacc:.4f} acc={acc:.4f} "
                    f"lambda=({row['lambda1']:.4g},{row['lambda2']:.4g}) "
                    f"shuffle95=({row['shuffle95_1']:.4g},{row['shuffle95_2']:.4g})",
                    flush=True,
                )

            del R_train_full, R_test_full, Vt, S
            gc.collect()
            print(f"[{timestamp()}] Fold {fold_i} finished in {(time.time() - t0) / 60:.2f} min", flush=True)

    # ---------------------------------------------------------------------
    # Save fold-level metrics.
    # ---------------------------------------------------------------------
    fold_csv = os.path.join(args.outdir, "fold_metrics.csv")
    if fold_rows:
        fieldnames = list(fold_rows[0].keys())
        with open(fold_csv, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(fold_rows)
    print(f"[{timestamp()}] Saved fold metrics: {fold_csv}", flush=True)

    # Summary by SVD dimension.
    summary_rows = []
    for m in svd_dims:
        rows_m = [r for r in fold_rows if int(r["svd_dim"]) == int(m)]
        if not rows_m:
            continue
        baccs = np.array([float(r["balanced_acc"]) for r in rows_m], dtype=np.float64)
        accs = np.array([float(r["acc"]) for r in rows_m], dtype=np.float64)
        lam1 = np.array([float(r["lambda1"]) for r in rows_m], dtype=np.float64)
        lam2 = np.array([float(r["lambda2"]) for r in rows_m], dtype=np.float64)
        p1 = np.array([float(r["p1"]) for r in rows_m], dtype=np.float64)
        p2 = np.array([float(r["p2"]) for r in rows_m], dtype=np.float64)
        n = len(rows_m)
        summary_rows.append({
            "model": args.model_name,
            "task_name": args.task_name,
            "svd_dim": int(m),
            "n_folds": int(n),
            "acc_mean": float(np.nanmean(accs)),
            "acc_std": float(np.nanstd(accs, ddof=1)) if n > 1 else 0.0,
            "acc_sem": float(np.nanstd(accs, ddof=1) / math.sqrt(n)) if n > 1 else 0.0,
            "balanced_acc_mean": float(np.nanmean(baccs)),
            "balanced_acc_std": float(np.nanstd(baccs, ddof=1)) if n > 1 else 0.0,
            "balanced_acc_sem": float(np.nanstd(baccs, ddof=1) / math.sqrt(n)) if n > 1 else 0.0,
            "balanced_acc_min": float(np.nanmin(baccs)),
            "balanced_acc_max": float(np.nanmax(baccs)),
            "lambda1_mean": float(np.nanmean(lam1)),
            "lambda2_mean": float(np.nanmean(lam2)),
            "p1_median": nanmedian_or_nan(p1),
            "p2_median": nanmedian_or_nan(p2),
        })

    summary_csv = os.path.join(args.outdir, "summary_by_svd_dim.csv")
    if summary_rows:
        with open(summary_csv, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
    print(f"[{timestamp()}] Saved summary: {summary_csv}", flush=True)

    # Save aggregate confusion matrices and LD scatter plots.
    for m in svd_dims:
        cm_path = os.path.join(args.outdir, f"confusion_matrix_M{m}.png")
        save_confusion_plot(
            cm_by_m[m],
            classes=classes,
            class_names=class_names,
            title=f"{args.model_name} subject-heldout confusion, M={m}",
            path=cm_path,
        )
        if ld_collect[m]:
            Z_all = np.concatenate(ld_collect[m], axis=0)
            y_all_plot = np.concatenate(y_collect[m], axis=0)
            s_all_plot = np.concatenate(s_collect[m], axis=0)
            scatter_path = os.path.join(args.outdir, f"heldout_ld_scatter_M{m}.png")
            save_ld_scatter(
                Z_all,
                y_all_plot,
                s_all_plot,
                classes=classes,
                class_names=class_names,
                title=f"{args.model_name} held-out LD projection, M={m}",
                path=scatter_path,
                max_points=args.max_scatter_points,
                seed=args.seed,
            )

    if fold_rows:
        save_balanced_acc_plot(fold_rows, svd_dims=svd_dims, outdir=args.outdir)
    if summary_rows:
        save_summary_acc_plot(summary_rows, outdir=args.outdir)

    # Save compact JSON summary.
    result_json = os.path.join(args.outdir, "result_summary.json")
    with open(result_json, "w", encoding="utf-8") as fp:
        json.dump({"metadata": meta, "summary_by_svd_dim": summary_rows}, fp, indent=2, ensure_ascii=False)
    print(f"[{timestamp()}] Saved result JSON: {result_json}", flush=True)
    print(f"[{timestamp()}] Done. Output dir: {args.outdir}", flush=True)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Experiment A: 5-second CogBCI subject-heldout discriminant-spectrum classification"
    )
    p.add_argument(
        "--batch-5s",
        action="store_true",
        help="Run the registered six 5-second CogBCI embedding files sequentially with shared subject folds.",
    )
    p.add_argument(
        "--models",
        default="BIOT,LaBraM,CBraMod,EEGPT,EEGMamba,s-JEPA",
        help="Models to run in --batch-5s mode. Comma-separated; default runs all six.",
    )
    p.add_argument(
        "--outroot",
        default="/mnt/dataset4/yinuo/FM_flow/dataset/expA_subject4fold_cogbci_5s_taskid3",
        help="Batch output root; each model is written to a separate subdirectory.",
    )
    p.add_argument("--h5", default="", help="Path to one H5 embedding file in single-file mode")
    p.add_argument("--embedding-key", default="embedding", help="H5 key for embeddings")
    p.add_argument("--label-key", default="task_id", help="H5 key for class labels. For global CogBCI 3-class use task_id.")
    p.add_argument("--subject-key", default="subject_id", help="H5 key for subject IDs")
    p.add_argument("--include-labels", default="0,1,2", help="Comma-separated class labels to include, e.g. 0,1,2")
    p.add_argument("--subjects", default="", help="Optional comma-separated candidate subjects. LOSO: only these held-out subjects. subject_kfold: build folds from only these subjects.")
    p.add_argument("--cv", default="subject_kfold", choices=["subject_kfold", "loso"], help="Subject-level CV. Default holds out subject groups so test/total is about 1/n-subject-folds.")
    p.add_argument("--n-subject-folds", type=int, default=4, help="Number of subject-level folds for --cv subject_kfold. 4 gives heldout/total ≈ 1/4.")
    p.add_argument("--folds", default="", help="Optional comma-separated fold IDs to run after subject groups are built, useful for smoke tests, e.g. --folds 1")
    p.add_argument("--heldout-subject-groups", default="", help="Optional manual held-out groups, e.g. '1,2,3,4,5,6,7;8,9,10,11,12,13,14'. Overrides --cv.")
    p.add_argument("--model-name", default="", help="Model name for output metadata in single-file mode")
    p.add_argument("--task-name", default="taskid3_rest_task1_task2_5s", help="Task name for output metadata")
    p.add_argument("--window-seconds", type=float, default=5.0, help="Window length recorded in metadata")
    p.add_argument("--outdir", default="", help="Output directory in single-file mode")

    p.add_argument("--svd-dims", default="100,200,300,500", help="Comma-separated SVD dimensions")
    p.add_argument("--svd-oversamples", type=int, default=20, help="Oversampling dimension for randomized SVD")
    p.add_argument("--svd-power-iter", type=int, default=2, help="Power iterations for randomized SVD")
    p.add_argument("--lda-ridge", type=float, default=1e-4, help="Ridge regularization relative to trace(Sw)/m")
    p.add_argument("--n-shuffle", type=int, default=0, help="Number of train-label shuffles per fold and SVD dim. 0 disables null.")

    p.add_argument("--read-batch-size", type=int, default=4096, help="H5 row read batch size")
    p.add_argument("--max-scatter-points", type=int, default=8000, help="Max points in aggregate LD scatter plots")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    return p


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
        raise ValueError(
            f"Unknown model '{token}'. Available: {', '.join(COGBCI_5S_MODEL_PATHS.keys())}"
        )
    return aliases[compact]


def parse_model_list(s: str) -> List[str]:
    raw = [x.strip() for x in str(s).split(",") if x.strip()]
    if not raw or (len(raw) == 1 and raw[0].lower() == "all"):
        return list(COGBCI_5S_MODEL_PATHS.keys())
    names: List[str] = []
    for token in raw:
        name = _canonical_model_name(token)
        if name not in names:
            names.append(name)
    return names


def get_selected_subject_vector(h5_path: str, args: argparse.Namespace) -> Tuple[np.ndarray, List[int], List[int]]:
    """Read only labels/subject IDs for batch validation and shared-fold construction."""
    with h5py.File(h5_path, "r") as f:
        if args.embedding_key not in f:
            raise KeyError(f"Embedding key not found in {h5_path}: {args.embedding_key}")
        n = int(f[args.embedding_key].shape[0])
        y = read_h5_vector(f, args.label_key)
        subj = read_h5_vector(f, args.subject_key)
        if len(y) != n or len(subj) != n:
            raise ValueError(
                f"Length mismatch in {h5_path}: embedding N={n}, label len={len(y)}, subject len={len(subj)}"
            )

    include_labels = parse_int_list(args.include_labels)
    if include_labels:
        mask = np.isin(y, include_labels)
    else:
        include_labels = sorted(int(x) for x in np.unique(y))
        mask = np.ones(len(y), dtype=bool)

    subj_sel = np.asarray(subj[mask], dtype=np.int64)
    if args.subjects:
        keep = np.asarray(parse_int_list(args.subjects), dtype=np.int64)
        subj_sel = subj_sel[np.isin(subj_sel, keep)]
    subjects = sorted(int(x) for x in np.unique(subj_sel))
    return subj_sel, subjects, sorted(int(x) for x in include_labels)


def groups_to_cli(groups: Sequence[Sequence[int]]) -> str:
    return ";".join(",".join(str(int(s)) for s in group) for group in groups)


def prepare_shared_batch_folds(
    model_paths: Dict[str, str],
    args: argparse.Namespace,
) -> Tuple[str, Dict[str, Dict]]:
    """Validate H5 layouts and construct one common subject partition for all models."""
    inspections: Dict[str, Dict] = {}
    reference_subjects: Optional[List[int]] = None
    reference_labels: Optional[List[int]] = None
    reference_vector: Optional[np.ndarray] = None

    for model, path in model_paths.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"5-second H5 not found for {model}: {path}")
        subj_sel, subjects, labels = get_selected_subject_vector(path, args)
        inspections[model] = {
            "h5": path,
            "n_selected_windows": int(len(subj_sel)),
            "subjects": subjects,
            "labels": labels,
            "subject_window_counts": {
                str(s): int(np.sum(subj_sel == s)) for s in subjects
            },
        }
        if reference_subjects is None:
            reference_subjects = subjects
            reference_labels = labels
            reference_vector = subj_sel
        else:
            if subjects != reference_subjects:
                raise ValueError(
                    f"Subject set mismatch: {model} has {subjects}, reference has {reference_subjects}. "
                    "Refusing to compare models with different subject pools."
                )
            if labels != reference_labels:
                raise ValueError(
                    f"Class-label mismatch: {model} has {labels}, reference has {reference_labels}."
                )

    assert reference_subjects is not None and reference_vector is not None
    if args.heldout_subject_groups:
        shared_groups = args.heldout_subject_groups
    elif args.cv == "loso":
        shared_groups = groups_to_cli([[s] for s in reference_subjects])
    else:
        groups = make_subject_kfold_groups(
            reference_subjects,
            reference_vector,
            n_folds=args.n_subject_folds,
            seed=args.seed,
        )
        shared_groups = groups_to_cli(groups)
    return shared_groups, inspections


def save_batch_summary(outroot: str, models: Sequence[str]) -> None:
    """Collect per-model summaries and write a compact comparison table/plot."""
    combined: List[Dict] = []
    for model in models:
        path = os.path.join(outroot, model, "summary_by_svd_dim.csv")
        if not os.path.isfile(path):
            print(f"[{timestamp()}] Warning: missing summary for {model}: {path}", flush=True)
            continue
        with open(path, "r", encoding="utf-8", newline="") as fp:
            for row in csv.DictReader(fp):
                row = dict(row)
                row["model"] = model
                combined.append(row)

    if not combined:
        print(f"[{timestamp()}] No model summaries found; batch aggregation skipped.", flush=True)
        return

    combined_path = os.path.join(outroot, "all_models_summary_by_svd_dim.csv")
    with open(combined_path, "w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(combined[0].keys()))
        writer.writeheader()
        writer.writerows(combined)

    best_rows: List[Dict] = []
    for model in models:
        rows = [r for r in combined if r["model"] == model]
        if rows:
            best_rows.append(max(rows, key=lambda r: float(r["balanced_acc_mean"])))
    best_path = os.path.join(outroot, "all_models_best_svd_dim.csv")
    if best_rows:
        with open(best_path, "w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(best_rows[0].keys()))
            writer.writeheader()
            writer.writerows(best_rows)

    plt.figure(figsize=(8, 5))
    for model in models:
        rows = [r for r in combined if r["model"] == model]
        rows = sorted(rows, key=lambda r: int(r["svd_dim"]))
        if rows:
            xs = [int(r["svd_dim"]) for r in rows]
            ys = [float(r["balanced_acc_mean"]) for r in rows]
            es = [float(r["balanced_acc_sem"]) for r in rows]
            plt.errorbar(xs, ys, yerr=es, marker="o", capsize=2, label=model)
    plt.axhline(1.0 / 3.0, linestyle="--", linewidth=1, label="chance=1/3")
    plt.xlabel("SVD dimension M")
    plt.ylabel("Mean subject-heldout balanced accuracy")
    plt.title("CogBCI 5-second windows: cross-model subject-heldout classification")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(outroot, "all_models_balanced_acc_vs_svd_dim.png"), dpi=200)
    plt.close()

    print(f"[{timestamp()}] Saved batch summary: {combined_path}", flush=True)
    print(f"[{timestamp()}] Saved best-M table: {best_path}", flush=True)


def run_batch_5s(args: argparse.Namespace) -> None:
    models = parse_model_list(args.models)
    model_paths = {model: COGBCI_5S_MODEL_PATHS[model] for model in models}
    ensure_dir(args.outroot)

    shared_groups, inspections = prepare_shared_batch_folds(model_paths, args)
    batch_meta = {
        "script": os.path.basename(__file__),
        "created_at": timestamp(),
        "window_seconds": float(args.window_seconds),
        "models": models,
        "model_paths": model_paths,
        "shared_heldout_subject_groups": shared_groups,
        "inspections": inspections,
    }
    with open(os.path.join(args.outroot, "batch_metadata.json"), "w", encoding="utf-8") as fp:
        json.dump(batch_meta, fp, indent=2, ensure_ascii=False)

    print("=" * 100, flush=True)
    print("CogBCI 5-second batch mode", flush=True)
    print(f"Models: {models}", flush=True)
    print(f"Shared held-out subject groups: {shared_groups}", flush=True)
    print(f"Output root: {args.outroot}", flush=True)
    print("=" * 100, flush=True)

    for i, model in enumerate(models, start=1):
        model_args = argparse.Namespace(**vars(args).copy())
        model_args.batch_5s = False
        model_args.h5 = model_paths[model]
        model_args.model_name = model
        model_args.outdir = os.path.join(args.outroot, model)
        model_args.heldout_subject_groups = shared_groups
        print("\n" + "#" * 100, flush=True)
        print(f"[{timestamp()}] Batch model {i}/{len(models)}: {model}", flush=True)
        print("#" * 100, flush=True)
        run(model_args)

    save_batch_summary(args.outroot, models)
    print(f"[{timestamp()}] All requested 5-second models finished.", flush=True)


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    if args.batch_5s:
        run_batch_5s(args)
        return

    missing = [name for name, value in (("--h5", args.h5), ("--model-name", args.model_name), ("--outdir", args.outdir)) if not value]
    if missing:
        parser.error("single-file mode requires " + ", ".join(missing) + "; or use --batch-5s")
    run(args)


if __name__ == "__main__":
    main()
