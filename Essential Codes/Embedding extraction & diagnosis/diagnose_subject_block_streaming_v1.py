#!/usr/bin/env python3
"""
Streaming integrity audit for subject-block anomalies in CogBCI EEG embeddings.

Goals
-----
1. Verify H5 key lengths and subject/label alignment.
2. Compute per-subject counts, norms, variance, and projected effective rank.
3. Compare suspect-vs-suspect, other-vs-other, and suspect-vs-other centroid distances.
4. Detect exact duplicate rows across subjects using streaming hashes.
5. Detect near-duplicate subject centroids in a shared random projection.
6. Run subject-identity probes in a projected space:
   - stratified train/test split by subject x raw label
   - nearest-centroid classifier
   - kNN classifier
   - confusion matrices and per-subject recall
7. Measure train/test centroid drift per subject.

This is a diagnostic script only. Its random projection does NOT replace the formal
experiment's SVD scan to 500 dimensions.

Example
-------
python3 diagnose_subject_block_streaming_v1.py \
  --h5 /path/to/cogbci_sesS1_embeddings.h5 \
  --model-name LaBraM \
  --outdir /path/to/output
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import h5py
import numpy as np


CORE_SUSPECT_DEFAULT = "18,19,20,21,22,23,24,27"
EXTENDED_SUSPECT_DEFAULT = "17,28"


def parse_int_list(text: str) -> List[int]:
    if not text.strip():
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def stable_softmax_weights(sq_dist: np.ndarray) -> np.ndarray:
    # For diagnostic ranking only.
    score = -sq_dist
    score = score - np.max(score, axis=1, keepdims=True)
    e = np.exp(score)
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-30)


def balanced_accuracy_from_confusion(cm: np.ndarray) -> float:
    denom = cm.sum(axis=1)
    recall = np.divide(
        np.diag(cm),
        denom,
        out=np.zeros_like(denom, dtype=np.float64),
        where=denom > 0,
    )
    return float(recall.mean())


def stratified_split_indices(
    subjects: np.ndarray,
    raw_labels: np.ndarray,
    test_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts: List[np.ndarray] = []
    test_parts: List[np.ndarray] = []
    for s in np.unique(subjects):
        mask_s = subjects == s
        for r in np.unique(raw_labels[mask_s]):
            idx = np.flatnonzero(mask_s & (raw_labels == r))
            rng.shuffle(idx)
            if len(idx) <= 1:
                train_parts.append(idx)
                continue
            n_test = max(1, int(round(len(idx) * test_fraction)))
            n_test = min(n_test, len(idx) - 1)
            test_parts.append(idx[:n_test])
            train_parts.append(idx[n_test:])
    train_idx = np.concatenate(train_parts) if train_parts else np.empty(0, dtype=int)
    test_idx = np.concatenate(test_parts) if test_parts else np.empty(0, dtype=int)
    return np.sort(train_idx), np.sort(test_idx)


def make_projection(
    flat_dim: int,
    proj_dim: int,
    seed: int,
    block_cols: int,
) -> Iterable[Tuple[slice, np.ndarray]]:
    """
    Yield blocks of a Gaussian random projection matrix without storing the full matrix.

    Entries are N(0, 1/proj_dim). The same deterministic RNG sequence is reused for
    every H5 batch by regenerating the column blocks from the same seed.
    """
    rng = np.random.default_rng(seed)
    scale = 1.0 / math.sqrt(proj_dim)
    for start in range(0, flat_dim, block_cols):
        stop = min(flat_dim, start + block_cols)
        block = rng.standard_normal((stop - start, proj_dim), dtype=np.float32)
        block *= scale
        yield slice(start, stop), block


def project_batch(
    batch_flat: np.ndarray,
    flat_dim: int,
    proj_dim: int,
    seed: int,
    block_cols: int,
) -> np.ndarray:
    out = np.zeros((batch_flat.shape[0], proj_dim), dtype=np.float32)
    for sl, R in make_projection(flat_dim, proj_dim, seed, block_cols):
        out += batch_flat[:, sl].astype(np.float32, copy=False) @ R
    return out


def exact_hash_rows(batch_flat: np.ndarray) -> List[str]:
    """
    Exact row hashes after canonical conversion to contiguous float32.
    This detects exact equality up to the source's float32 representation.
    """
    arr = np.ascontiguousarray(batch_flat.astype(np.float32, copy=False))
    return [hashlib.sha256(arr[i].tobytes()).hexdigest() for i in range(arr.shape[0])]


def effective_rank(X: np.ndarray) -> float:
    if X.shape[0] <= 1:
        return 0.0
    Xc = X - X.mean(axis=0, keepdims=True)
    # X is only projected to <=256 dimensions, so this is cheap.
    s = np.linalg.svd(Xc, compute_uv=False)
    p = s * s
    total = float(p.sum())
    if total <= 0:
        return 0.0
    p = p / total
    return float(np.exp(-np.sum(p * np.log(p + 1e-30))))


def pairwise_sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    aa = np.sum(A * A, axis=1, keepdims=True)
    bb = np.sum(B * B, axis=1, keepdims=True).T
    return np.maximum(aa + bb - 2.0 * (A @ B.T), 0.0)


def nearest_centroid_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    classes: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    centroids = np.stack([X_train[y_train == c].mean(axis=0) for c in classes])
    d2 = pairwise_sqdist(X_test, centroids)
    pred = classes[np.argmin(d2, axis=1)]
    return pred, d2, centroids


def knn_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    k: int,
    batch_size: int = 1024,
) -> np.ndarray:
    k = min(k, len(X_train))
    preds = []
    classes = np.unique(y_train)
    for start in range(0, len(X_test), batch_size):
        stop = min(len(X_test), start + batch_size)
        d2 = pairwise_sqdist(X_test[start:stop], X_train)
        nn = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
        labels = y_train[nn]
        block_pred = np.empty(len(labels), dtype=y_train.dtype)
        for i, row in enumerate(labels):
            counts = np.array([(row == c).sum() for c in classes])
            block_pred[i] = classes[np.argmax(counts)]
        preds.append(block_pred)
    return np.concatenate(preds)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray) -> np.ndarray:
    lut = {int(c): i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[lut[int(t)], lut[int(p)]] += 1
    return cm


def save_confusion_csv(path: Path, cm: np.ndarray, classes: np.ndarray) -> None:
    rows = []
    for i, true_c in enumerate(classes):
        row = {"true_subject": int(true_c)}
        for j, pred_c in enumerate(classes):
            row[f"pred_{int(pred_c)}"] = int(cm[i, j])
        rows.append(row)
    write_csv(path, rows)


@dataclass
class H5Info:
    n: int
    emb_shape: Tuple[int, ...]
    flat_dim: int
    subjects: np.ndarray
    raw_labels: np.ndarray


def inspect_h5(path: str, embedding_key: str, subject_key: str, label_key: str) -> H5Info:
    with h5py.File(path, "r") as f:
        for key in (embedding_key, subject_key, label_key):
            if key not in f:
                raise KeyError(f"Missing H5 key: {key}. Available keys: {list(f.keys())}")
        n = len(f[embedding_key])
        if len(f[subject_key]) != n or len(f[label_key]) != n:
            raise ValueError(
                f"Length mismatch: embedding={n}, "
                f"{subject_key}={len(f[subject_key])}, {label_key}={len(f[label_key])}"
            )
        emb_shape = tuple(f[embedding_key].shape)
        flat_dim = int(np.prod(emb_shape[1:]))
        subjects = np.asarray(f[subject_key][:]).astype(np.int64).ravel()
        raw_labels = np.asarray(f[label_key][:]).astype(np.int64).ravel()
    return H5Info(n, emb_shape, flat_dim, subjects, raw_labels)


def run(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    core_suspect = set(parse_int_list(args.core_suspect))
    extended_suspect = set(parse_int_list(args.extended_suspect))

    info = inspect_h5(args.h5, args.embedding_key, args.subject_key, args.label_key)
    subjects = np.array(sorted(np.unique(info.subjects)), dtype=np.int64)
    subj_to_pos = {int(s): i for i, s in enumerate(subjects)}

    print("=" * 100)
    print(f"Model: {args.model_name}")
    print(f"H5: {args.h5}")
    print(f"Embedding shape: {info.emb_shape}, flat_dim={info.flat_dim}")
    print(f"n_windows={info.n}, n_subjects={len(subjects)}")
    print(f"projection_dim={args.projection_dim}, batch_size={args.batch_size}")
    print("=" * 100, flush=True)

    projected = np.empty((info.n, args.projection_dim), dtype=np.float32)
    exact_hash_map: Dict[str, List[Tuple[int, int, int]]] = defaultdict(list)

    with h5py.File(args.h5, "r") as f:
        ds = f[args.embedding_key]
        for start in range(0, info.n, args.batch_size):
            stop = min(info.n, start + args.batch_size)
            batch = np.asarray(ds[start:stop])
            batch_flat = batch.reshape(len(batch), -1)
            projected[start:stop] = project_batch(
                batch_flat,
                info.flat_dim,
                args.projection_dim,
                args.projection_seed,
                args.projection_block_cols,
            )
            if not args.skip_exact_hash:
                hashes = exact_hash_rows(batch_flat)
                for local_i, h in enumerate(hashes):
                    global_i = start + local_i
                    exact_hash_map[h].append(
                        (
                            global_i,
                            int(info.subjects[global_i]),
                            int(info.raw_labels[global_i]),
                        )
                    )
            print(f"[project] {stop}/{info.n}", flush=True)

    # Standardize projection globally for meaningful geometric diagnostics.
    proj_mean = projected.mean(axis=0, keepdims=True)
    proj_std = projected.std(axis=0, keepdims=True)
    proj_std = np.where(proj_std > 1e-8, proj_std, 1.0)
    Z = (projected - proj_mean) / proj_std

    # Per-subject statistics.
    per_subject_rows: List[dict] = []
    centroids = {}
    within_scales = {}
    for s in subjects:
        idx = np.flatnonzero(info.subjects == s)
        Zs = Z[idx]
        c = Zs.mean(axis=0)
        centroids[int(s)] = c
        within = np.linalg.norm(Zs - c, axis=1)
        within_scales[int(s)] = float(np.mean(within))
        per_subject_rows.append(
            {
                "model": args.model_name,
                "subject": int(s),
                "n_windows": int(len(idx)),
                "n_raw_labels": int(len(np.unique(info.raw_labels[idx]))),
                "projected_mean_norm": float(np.linalg.norm(c)),
                "projected_mean_variance": float(Zs.var(axis=0).mean()),
                "projected_effective_rank": effective_rank(Zs),
                "within_mean_radius": float(np.mean(within)),
                "within_rms_radius": float(np.sqrt(np.mean(within * within))),
                "group": (
                    "core_suspect"
                    if int(s) in core_suspect
                    else "extended_suspect"
                    if int(s) in extended_suspect
                    else "other"
                ),
            }
        )
    write_csv(outdir / "per_subject_projected_stats.csv", per_subject_rows)

    # Centroid distance block structure.
    C = np.stack([centroids[int(s)] for s in subjects])
    D = np.sqrt(pairwise_sqdist(C, C))
    global_within = float(np.mean([within_scales[int(s)] for s in subjects]))
    Dn = D / max(global_within, 1e-12)

    def group_positions(group: set[int]) -> List[int]:
        return [i for i, s in enumerate(subjects) if int(s) in group]

    core_pos = group_positions(core_suspect)
    other_pos = [i for i, s in enumerate(subjects) if int(s) not in core_suspect]

    def offdiag_mean(rows: Sequence[int], cols: Sequence[int], same_group: bool) -> float:
        vals = []
        for i in rows:
            for j in cols:
                if same_group and i == j:
                    continue
                vals.append(Dn[i, j])
        return float(np.mean(vals)) if vals else float("nan")

    distance_summary = {
        "model": args.model_name,
        "global_within_mean_radius": global_within,
        "core_core_normalized_centroid_distance": offdiag_mean(core_pos, core_pos, True),
        "other_other_normalized_centroid_distance": offdiag_mean(other_pos, other_pos, True),
        "core_other_normalized_centroid_distance": offdiag_mean(core_pos, other_pos, False),
    }
    write_csv(outdir / "centroid_block_distance_summary.csv", [distance_summary])

    # Nearest subject-centroid pairs.
    pair_rows = []
    for i in range(len(subjects)):
        for j in range(i + 1, len(subjects)):
            pair_rows.append(
                {
                    "model": args.model_name,
                    "subject_a": int(subjects[i]),
                    "subject_b": int(subjects[j]),
                    "normalized_centroid_distance": float(Dn[i, j]),
                    "a_group": per_subject_rows[i]["group"],
                    "b_group": per_subject_rows[j]["group"],
                }
            )
    pair_rows.sort(key=lambda r: r["normalized_centroid_distance"])
    write_csv(outdir / "subject_centroid_pairs.csv", pair_rows)

    # Exact duplicate rows.
    duplicate_rows = []
    cross_subject_groups = 0
    if not args.skip_exact_hash:
        for h, members in exact_hash_map.items():
            member_subjects = sorted({m[1] for m in members})
            if len(member_subjects) <= 1:
                continue
            cross_subject_groups += 1
            duplicate_rows.append(
                {
                    "model": args.model_name,
                    "sha256_float32": h,
                    "n_rows": len(members),
                    "subjects": ",".join(map(str, member_subjects)),
                    "raw_labels": ",".join(map(str, sorted({m[2] for m in members}))),
                    "indices": ",".join(map(str, [m[0] for m in members[:20]])),
                }
            )
    write_csv(outdir / "cross_subject_exact_duplicate_rows.csv", duplicate_rows)

    # Train/test identity diagnostic.
    train_idx, test_idx = stratified_split_indices(
        info.subjects,
        info.raw_labels,
        args.test_fraction,
        args.split_seed,
    )
    y_train = info.subjects[train_idx]
    y_test = info.subjects[test_idx]
    Z_train = Z[train_idx]
    Z_test = Z[test_idx]

    pred_nc, d2_nc, train_centroids = nearest_centroid_predict(
        Z_train, y_train, Z_test, subjects
    )
    pred_knn = knn_predict(
        Z_train, y_train, Z_test, args.knn_k, args.knn_batch_size
    )

    cm_nc = confusion_matrix(y_test, pred_nc, subjects)
    cm_knn = confusion_matrix(y_test, pred_knn, subjects)
    save_confusion_csv(outdir / "confusion_nearest_centroid.csv", cm_nc, subjects)
    save_confusion_csv(outdir / "confusion_knn.csv", cm_knn, subjects)

    # Per-subject prediction summaries and centroid drift.
    per_subject_identity_rows = []
    for i, s in enumerate(subjects):
        mask_test = y_test == s
        idx_train_s = train_idx[y_train == s]
        idx_test_s = test_idx[y_test == s]

        train_c = Z[idx_train_s].mean(axis=0)
        test_c = Z[idx_test_s].mean(axis=0)
        pooled_within = np.mean(
            np.linalg.norm(Z[idx_train_s] - train_c, axis=1)
        )
        drift = float(np.linalg.norm(train_c - test_c))
        norm_drift = drift / max(float(pooled_within), 1e-12)

        pred_counts_nc = np.bincount(
            np.searchsorted(subjects, pred_nc[mask_test]),
            minlength=len(subjects),
        )
        pred_counts_knn = np.bincount(
            np.searchsorted(subjects, pred_knn[mask_test]),
            minlength=len(subjects),
        )
        top_nc = int(subjects[np.argmax(pred_counts_nc)])
        top_knn = int(subjects[np.argmax(pred_counts_knn)])

        per_subject_identity_rows.append(
            {
                "model": args.model_name,
                "subject": int(s),
                "group": (
                    "core_suspect"
                    if int(s) in core_suspect
                    else "extended_suspect"
                    if int(s) in extended_suspect
                    else "other"
                ),
                "n_train": int(len(idx_train_s)),
                "n_test": int(len(idx_test_s)),
                "nearest_centroid_recall": float(cm_nc[i, i] / max(cm_nc[i].sum(), 1)),
                "nearest_centroid_top_pred_subject": top_nc,
                "nearest_centroid_top_pred_fraction": float(
                    pred_counts_nc.max() / max(pred_counts_nc.sum(), 1)
                ),
                "knn_recall": float(cm_knn[i, i] / max(cm_knn[i].sum(), 1)),
                "knn_top_pred_subject": top_knn,
                "knn_top_pred_fraction": float(
                    pred_counts_knn.max() / max(pred_counts_knn.sum(), 1)
                ),
                "train_test_centroid_drift": drift,
                "train_test_centroid_drift_normalized": norm_drift,
                "mean_nearest_centroid_true_class_distance": float(
                    np.mean(d2_nc[mask_test, i])
                ),
            }
        )
    write_csv(outdir / "per_subject_identity_diagnostics.csv", per_subject_identity_rows)

    summary = {
        "script": Path(__file__).name,
        "script_version": "1.0-streaming-subject-block-audit",
        "model": args.model_name,
        "h5": args.h5,
        "embedding_shape": list(info.emb_shape),
        "flat_dim": info.flat_dim,
        "n_windows": info.n,
        "n_subjects": len(subjects),
        "subjects": [int(x) for x in subjects],
        "core_suspect": sorted(core_suspect),
        "extended_suspect": sorted(extended_suspect),
        "projection_dim": args.projection_dim,
        "projection_seed": args.projection_seed,
        "split_seed": args.split_seed,
        "test_fraction": args.test_fraction,
        "knn_k": args.knn_k,
        "nearest_centroid_bacc": balanced_accuracy_from_confusion(cm_nc),
        "knn_bacc": balanced_accuracy_from_confusion(cm_knn),
        "cross_subject_exact_duplicate_groups": cross_subject_groups,
        **distance_summary,
    }
    with (outdir / "diagnostic_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 100)
    print("DONE")
    print(f"nearest-centroid bACC: {summary['nearest_centroid_bacc']:.6f}")
    print(f"kNN bACC:              {summary['knn_bacc']:.6f}")
    print(f"cross-subject exact duplicate groups: {cross_subject_groups}")
    print(f"output: {outdir}")
    print("=" * 100)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Streaming subject-block integrity audit for EEG embeddings."
    )
    p.add_argument("--h5", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--outdir", required=True)

    p.add_argument("--embedding-key", default="embedding")
    p.add_argument("--subject-key", default="subject_id")
    p.add_argument("--label-key", default="label")

    p.add_argument("--core-suspect", default=CORE_SUSPECT_DEFAULT)
    p.add_argument("--extended-suspect", default=EXTENDED_SUSPECT_DEFAULT)

    p.add_argument("--projection-dim", type=int, default=128)
    p.add_argument("--projection-seed", type=int, default=20260719)
    p.add_argument("--projection-block-cols", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=32)

    p.add_argument("--test-fraction", type=float, default=0.25)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--knn-k", type=int, default=5)
    p.add_argument("--knn-batch-size", type=int, default=512)

    p.add_argument(
        "--skip-exact-hash",
        action="store_true",
        help="Skip exact row hashing when runtime is a concern.",
    )
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
