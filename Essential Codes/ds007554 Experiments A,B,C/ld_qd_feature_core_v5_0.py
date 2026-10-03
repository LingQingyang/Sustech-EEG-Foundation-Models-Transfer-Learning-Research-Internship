#!/usr/bin/env python3
"""Shared rest-relative LDA/QDA feature core.

This module contains no dataset- or split-specific logic.  It implements:

    ell                       relative pooled-covariance LDA scores
    q                         relative interpolated-QDA scores
    r = q - ell               quadratic residual beyond LDA
    F0 = z(ell)
    F1 = z(q)
    F2 = z(ell) direct-sum z(r)
    F3 = F2 direct-sum z(vec(z(ell) outer z(r)))

The intended use is a strictly train-only fit inside an outer evaluation fold.
Formal high-rank runs use arithmetic pooling shrinkage

    Sigma_c(alpha) = (1-alpha) Sigma_c + alpha Sigma_pooled,  alpha > 0,

so individual class covariances may be singular while every fitted QDA
covariance remains positive definite whenever the equal-class pooled
covariance is positive definite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import linalg


ARMS = (
    "F0_LD",
    "F1_QD",
    "F2_LD_plus_QResidual",
    "F3_Tensor",
)


def finite_mean(values) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if len(arr) else float("nan")


def covariance_unbiased(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or len(X) < 2:
        raise ValueError("Need a two-dimensional array with at least two rows")
    Z = X - X.mean(axis=0, keepdims=True)
    C = (Z.T @ Z) / float(len(X) - 1)
    return 0.5 * (C + C.T)


def rank_condition(C: np.ndarray) -> Tuple[int, float, float, np.ndarray]:
    C = 0.5 * (np.asarray(C, dtype=np.float64) + np.asarray(C, dtype=np.float64).T)
    eig = np.linalg.eigvalsh(C)
    scale = max(float(np.max(np.abs(eig))), 1.0)
    tol = np.finfo(np.float64).eps * max(C.shape) * scale
    pos = eig[eig > tol]
    rank = int(len(pos))
    condition = float(np.max(pos) / np.min(pos)) if len(pos) else float("inf")
    logdet = float(np.sum(np.log(pos))) if rank == C.shape[0] else float("nan")
    return rank, condition, logdet, eig


@dataclass
class ClassStats:
    means: np.ndarray
    covs: np.ndarray
    pooled: np.ndarray
    pooled_chol: np.ndarray
    class_counts: np.ndarray
    class_covariance_ranks: np.ndarray
    class_covariance_spd: np.ndarray
    lambda_spectra: np.ndarray
    generalized_eigvecs: np.ndarray
    class_names: List[str]


def fit_class_stats(
    X: np.ndarray,
    y: np.ndarray,
    class_names: Sequence[str],
) -> ClassStats:
    """Fit empirical class covariances and an equal-class pooled covariance.

    Individual empirical class covariances are allowed to be positive
    semidefinite and rank deficient.  This is essential when the reduced
    dimension exceeds ``n_class - 1``.  The equal-class pooled covariance must
    itself be positive definite because it is the LDA metric and the shrinkage
    anchor.  No pseudoinverse, eigenvalue clipping, or silent ridge is used.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    if X.ndim != 2 or len(X) != len(y):
        raise ValueError("X/y shape mismatch")
    K = len(class_names)
    M = X.shape[1]
    observed = sorted(np.unique(y).astype(int).tolist())
    if observed != list(range(K)):
        raise ValueError(f"Expected classes 0..{K-1}, observed={observed}")

    means = np.zeros((K, M), dtype=np.float64)
    covs = np.zeros((K, M, M), dtype=np.float64)
    counts = np.zeros(K, dtype=np.int64)
    class_ranks = np.zeros(K, dtype=np.int64)
    class_spd = np.zeros(K, dtype=bool)

    for c in range(K):
        Xc = X[y == c]
        counts[c] = len(Xc)
        if len(Xc) < 2:
            raise ValueError(
                f"Need at least two observations for class covariance: "
                f"class={class_names[c]!r}, n={len(Xc)}"
            )
        means[c] = Xc.mean(axis=0)
        covs[c] = covariance_unbiased(Xc)
        class_ranks[c] = rank_condition(covs[c])[0]
        try:
            np.linalg.cholesky(covs[c])
            class_spd[c] = True
        except np.linalg.LinAlgError:
            class_spd[c] = False

    pooled = 0.5 * (np.mean(covs, axis=0) + np.mean(covs, axis=0).T)
    try:
        pooled_chol = np.linalg.cholesky(pooled)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError("Equal-class pooled covariance is not SPD") from exc

    # Compute the generalized class-covariance spectra in pooled-whitened
    # coordinates from the centered observations themselves.  Calling
    # scipy.linalg.eigh(C_c, C_pooled) directly can return large spurious
    # negative eigenvalues when C_pooled is SPD but ill-conditioned.  The
    # Gram construction below is algebraically identical, yet preserves the
    # positive-semidefinite structure by construction.
    lambda_spectra = np.zeros((K, M), dtype=np.float64)
    generalized_eigvecs = np.zeros((K, M, M), dtype=np.float64)
    for c in range(K):
        Xc = X[y == c]
        Zc = Xc - means[c][None, :]
        whitened_rows = linalg.solve_triangular(
            pooled_chol, Zc.T, lower=True, check_finite=False
        ).T
        Cwhite = (whitened_rows.T @ whitened_rows) / float(len(Xc) - 1)
        Cwhite = 0.5 * (Cwhite + Cwhite.T)
        vals, u = np.linalg.eigh(Cwhite)
        if not np.all(np.isfinite(vals)):
            raise np.linalg.LinAlgError(
                f"Non-finite generalized covariance spectrum for class {class_names[c]!r}"
            )
        scale = max(float(np.max(np.abs(vals))), 1.0)
        tol = 1000.0 * np.finfo(np.float64).eps * max(M, 1) * scale
        min_val = float(np.min(vals))
        if min_val < -tol:
            raise np.linalg.LinAlgError(
                f"Whitened empirical class covariance lost positive semidefiniteness: "
                f"class={class_names[c]!r}, min_eigenvalue={min_val}, tolerance={tol}"
            )
        # Only round-off-scale negative values are projected to the exact PSD
        # boundary.  This does not regularize the covariance or alter its rank.
        vals = np.maximum(vals, 0.0)
        vecs = linalg.solve_triangular(
            pooled_chol.T, u, lower=False, check_finite=False
        )
        lambda_spectra[c] = vals
        generalized_eigvecs[c] = vecs

    return ClassStats(
        means=means,
        covs=covs,
        pooled=pooled,
        pooled_chol=pooled_chol,
        class_counts=counts,
        class_covariance_ranks=class_ranks,
        class_covariance_spd=class_spd,
        lambda_spectra=lambda_spectra,
        generalized_eigvecs=generalized_eigvecs,
        class_names=list(class_names),
    )


def lda_relative_scores(X: np.ndarray, stats: ClassStats) -> np.ndarray:
    """Equal-prior class scores relative to class 0."""
    X = np.asarray(X, dtype=np.float64)
    beta = linalg.cho_solve(
        (stats.pooled_chol, True), stats.means.T, check_finite=False
    ).T
    constants = -0.5 * np.sum(stats.means * beta, axis=1)
    absolute = X @ beta.T + constants[None, :]
    return absolute[:, 1:] - absolute[:, [0]]


def interpolated_covariance(
    stats: ClassStats,
    class_id: int,
    alpha: float,
    mode: str,
) -> np.ndarray:
    """Interpolate class covariance toward the equal-class pooled covariance.

    ``alpha=0`` is the empirical class covariance and may be singular.
    ``alpha=1`` is the pooled-covariance LDA endpoint.  Arithmetic interpolation
    with any ``alpha>0`` is positive definite when the pooled covariance is
    positive definite.
    """
    alpha = float(alpha)
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must lie in [0,1], got {alpha}")
    Cc = stats.covs[int(class_id)]
    Cp = stats.pooled
    if alpha == 0.0:
        return Cc
    if alpha == 1.0:
        return Cp
    if mode == "arithmetic":
        S = (1.0 - alpha) * Cc + alpha * Cp
        return 0.5 * (S + S.T)
    if mode != "geometric":
        raise ValueError(f"Unknown covariance interpolation mode {mode!r}")

    vals = stats.lambda_spectra[int(class_id)]
    vecs = stats.generalized_eigvecs[int(class_id)]
    if np.any(vals <= 0) or not np.all(np.isfinite(vals)):
        raise np.linalg.LinAlgError("Geometric interpolation requires positive spectrum")
    B = Cp @ vecs
    S = (B * (vals ** (1.0 - alpha))[None, :]) @ B.T
    return 0.5 * (S + S.T)


@dataclass
class QDAModel:
    chols: List[np.ndarray]
    logdets: np.ndarray
    alpha: float
    interpolation: str


def fit_qda_model(
    stats: ClassStats,
    alpha: float,
    interpolation: str,
) -> QDAModel:
    chols: List[np.ndarray] = []
    logdets = np.zeros(len(stats.class_names), dtype=np.float64)
    for c in range(len(stats.class_names)):
        S = interpolated_covariance(stats, c, alpha, interpolation)
        try:
            L = np.linalg.cholesky(S)
        except np.linalg.LinAlgError as exc:
            raise np.linalg.LinAlgError(
                f"Interpolated covariance is not SPD: class={stats.class_names[c]!r}, "
                f"alpha={alpha}, interpolation={interpolation}"
            ) from exc
        chols.append(L)
        logdets[c] = float(2.0 * np.sum(np.log(np.diag(L))))
    return QDAModel(
        chols=chols,
        logdets=logdets,
        alpha=float(alpha),
        interpolation=str(interpolation),
    )


def qda_relative_scores(
    X: np.ndarray,
    stats: ClassStats,
    model: QDAModel,
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    absolute = np.empty((len(X), len(stats.class_names)), dtype=np.float64)
    for c in range(len(stats.class_names)):
        delta = (X - stats.means[c]).T
        solved = linalg.solve_triangular(
            model.chols[c], delta, lower=True, check_finite=False
        )
        mahal = np.sum(solved * solved, axis=0)
        absolute[:, c] = -0.5 * (model.logdets[c] + mahal)
    return absolute[:, 1:] - absolute[:, [0]]


def native_prediction(relative_scores: np.ndarray) -> np.ndarray:
    relative_scores = np.asarray(relative_scores, dtype=np.float64)
    scores = np.concatenate(
        [np.zeros((len(relative_scores), 1), dtype=np.float64), relative_scores],
        axis=1,
    )
    return np.argmax(scores, axis=1).astype(np.int64)


@dataclass
class SafeStandardizer:
    mean: np.ndarray
    scale: np.ndarray
    degenerate: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray, tol: float = 1e-10) -> "SafeStandardizer":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("Standardizer expects a two-dimensional array")
        mean = X.mean(axis=0)
        std = X.std(axis=0, ddof=0)
        degenerate = std < float(tol) * (1.0 + np.abs(mean))
        scale = std.copy()
        scale[degenerate] = 1.0
        return cls(mean=mean, scale=scale, degenerate=degenerate)

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = (np.asarray(X, dtype=np.float64) - self.mean) / self.scale
        if np.any(self.degenerate):
            Z[:, self.degenerate] = 0.0
        return Z


@dataclass
class FeatureBuilder:
    ell_scaler: SafeStandardizer
    q_scaler: SafeStandardizer
    r_scaler: SafeStandardizer
    tensor_scaler: SafeStandardizer

    @classmethod
    def fit(
        cls,
        ell: np.ndarray,
        q: np.ndarray,
        tol: float = 1e-10,
    ) -> "FeatureBuilder":
        ell = np.asarray(ell, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)
        if ell.shape != q.shape:
            raise ValueError("ell and q must have identical shapes")
        r = q - ell
        ell_scaler = SafeStandardizer.fit(ell, tol)
        q_scaler = SafeStandardizer.fit(q, tol)
        r_scaler = SafeStandardizer.fit(r, tol)
        ell_z = ell_scaler.transform(ell)
        r_z = r_scaler.transform(r)
        tensor = np.einsum("ni,nj->nij", ell_z, r_z, optimize=True).reshape(len(ell_z), -1)
        tensor_scaler = SafeStandardizer.fit(tensor, tol)
        return cls(
            ell_scaler=ell_scaler,
            q_scaler=q_scaler,
            r_scaler=r_scaler,
            tensor_scaler=tensor_scaler,
        )

    def transform(self, ell: np.ndarray, q: np.ndarray) -> Dict[str, np.ndarray]:
        ell = np.asarray(ell, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)
        r = q - ell
        ell_z = self.ell_scaler.transform(ell)
        q_z = self.q_scaler.transform(q)
        r_z = self.r_scaler.transform(r)
        tensor = np.einsum("ni,nj->nij", ell_z, r_z, optimize=True).reshape(len(ell_z), -1)
        tensor_z = self.tensor_scaler.transform(tensor)
        return {
            "F0_LD": ell_z,
            "F1_QD": q_z,
            "F2_LD_plus_QResidual": np.concatenate([ell_z, r_z], axis=1),
            "F3_Tensor": np.concatenate([ell_z, r_z, tensor_z], axis=1),
        }

    def degenerate_report(self) -> Dict[str, List[int]]:
        return {
            "ell": np.flatnonzero(self.ell_scaler.degenerate).astype(int).tolist(),
            "q": np.flatnonzero(self.q_scaler.degenerate).astype(int).tolist(),
            "r": np.flatnonzero(self.r_scaler.degenerate).astype(int).tolist(),
            "tensor": np.flatnonzero(self.tensor_scaler.degenerate).astype(int).tolist(),
        }


def feature_dimensions(n_relative_scores: int) -> Dict[str, int]:
    d = int(n_relative_scores)
    return {
        "F0_LD": d,
        "F1_QD": d,
        "F2_LD_plus_QResidual": 2 * d,
        "F3_Tensor": 2 * d + d * d,
    }


@dataclass
class RidgeReadout:
    coefficients: np.ndarray  # (p + 1) x K; final row is intercept
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


def fit_ridge_readout(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    ridge_lambda: float,
) -> RidgeReadout:
    """Closed-form class-balanced multiclass ridge readout."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    counts = np.bincount(y, minlength=int(n_classes)).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError(f"Missing readout class; counts={counts.tolist()}")
    weights = 1.0 / counts[y]
    weights *= len(weights) / np.sum(weights)
    Xa = np.concatenate([X, np.ones((len(X), 1), dtype=np.float64)], axis=1)
    Y = np.eye(int(n_classes), dtype=np.float64)[y]
    A = Xa.T @ (weights[:, None] * Xa)
    penalty = np.eye(Xa.shape[1], dtype=np.float64) * float(ridge_lambda)
    penalty[-1, -1] = 0.0
    B = Xa.T @ (weights[:, None] * Y)
    coef = linalg.solve(A + penalty, B, assume_a="sym", check_finite=False)
    return RidgeReadout(
        coefficients=coef,
        n_classes=int(n_classes),
        ridge_lambda=float(ridge_lambda),
    )


def confusion_matrix_fixed(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((int(n_classes), int(n_classes)), dtype=np.int64)
    for a, b in zip(np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)):
        if 0 <= a < n_classes and 0 <= b < n_classes:
            cm[a, b] += 1
    return cm


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
) -> Dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    recalls = []
    for c in range(int(n_classes)):
        mask = y_true == c
        recalls.append(float(np.mean(y_pred[mask] == c)) if np.any(mask) else float("nan"))
    return {
        "balanced_accuracy": finite_mean(recalls),
        "accuracy": float(np.mean(y_true == y_pred)),
        "recalls": recalls,
        "confusion_matrix": confusion_matrix_fixed(y_true, y_pred, n_classes),
    }


def self_test(seed: int = 0) -> Dict[str, float]:
    """Small deterministic algebra test; no dataset access required."""
    rng = np.random.default_rng(seed)
    K, M, n_per = 5, 6, 80
    rows, labels = [], []
    for c in range(K):
        A = np.eye(M)
        A[c % M, c % M] = 0.7 + 0.15 * c
        A[(c + 1) % M, (c + 1) % M] = 1.3 - 0.08 * c
        Xc = rng.standard_normal((n_per, M)) @ A.T
        Xc += 0.35 * c * np.eye(1, M, c % M)
        rows.append(Xc)
        labels.append(np.full(n_per, c, dtype=np.int64))
    X = np.concatenate(rows, axis=0)
    y = np.concatenate(labels, axis=0)

    stats = fit_class_stats(X, y, [f"c{i}" for i in range(K)])
    ell = lda_relative_scores(X, stats)
    q1 = qda_relative_scores(X, stats, fit_qda_model(stats, 1.0, "arithmetic"))
    endpoint_error = float(np.max(np.abs(q1 - ell)))
    if endpoint_error > 1e-8:
        raise AssertionError(f"alpha=1 endpoint failed: max error={endpoint_error}")

    q = qda_relative_scores(X, stats, fit_qda_model(stats, 0.5, "arithmetic"))
    builder = FeatureBuilder.fit(ell, q)
    features = builder.transform(ell, q)
    expected = feature_dimensions(K - 1)
    observed = {k: int(v.shape[1]) for k, v in features.items()}
    if observed != expected:
        raise AssertionError(f"Feature dimensions failed: observed={observed}, expected={expected}")

    readout = fit_ridge_readout(features["F3_Tensor"], y, K, 0.1)
    pred = readout.predict(features["F3_Tensor"])
    metric = evaluate_predictions(y, pred, K)
    if not np.isfinite(metric["balanced_accuracy"]):
        raise AssertionError("Readout returned a non-finite metric")

    # High-dimensional regression test: every class covariance is singular
    # (n_class - 1 < M), but their equal-class pooled covariance is full rank.
    K_hd, M_hd, n_hd = 5, 20, 9
    X_hd = np.concatenate([
        rng.standard_normal((n_hd, M_hd)) + 0.08 * c
        for c in range(K_hd)
    ], axis=0)
    y_hd = np.concatenate([
        np.full(n_hd, c, dtype=np.int64) for c in range(K_hd)
    ])
    stats_hd = fit_class_stats(X_hd, y_hd, [f"hd{i}" for i in range(K_hd)])
    if np.any(stats_hd.class_covariance_spd):
        raise AssertionError("High-dimensional test unexpectedly produced an SPD class covariance")
    q_hd = qda_relative_scores(
        X_hd, stats_hd, fit_qda_model(stats_hd, 0.1, "arithmetic")
    )
    if not np.all(np.isfinite(q_hd)):
        raise AssertionError("Pooled-shrinkage QDA produced non-finite high-dimensional scores")
    alpha0_rejected = False
    try:
        fit_qda_model(stats_hd, 0.0, "arithmetic")
    except np.linalg.LinAlgError:
        alpha0_rejected = True
    if not alpha0_rejected:
        raise AssertionError("alpha=0 should remain unavailable for singular class covariances")

    return {
        "alpha1_max_abs_q_minus_ell": endpoint_error,
        "tensor_dimension": float(observed["F3_Tensor"]),
        "training_balanced_accuracy": float(metric["balanced_accuracy"]),
        "high_dimensional_class_rank": float(np.max(stats_hd.class_covariance_ranks)),
        "high_dimensional_dimension": float(M_hd),
        "pooled_shrinkage_singular_class_test": 1.0,
    }


if __name__ == "__main__":
    print(self_test())
