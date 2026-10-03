"""Pure NumPy geometry of rank-one task-update operators.

For a matrix update

    Delta W = sum_i sigma_i Z_i,     Z_i = u_i v_i^T,

the immutable analysis atom is the complete rank-one operator ``Z_i``.  Its
Frobenius inner product factorises as

    <Z_i, Z_j>_F = (u_i^T u_j) (v_i^T v_j).

This identity gives exact projection energies and principal angles without
forming 40,000- or 160,000-dimensional vectorisations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


EPS = 1e-12


@dataclass(frozen=True)
class SVDRecord:
    U: np.ndarray
    s: np.ndarray
    V: np.ndarray
    d_out: int
    d_in: int
    reconstruction_relative_error: float
    orthonormality_error: float

    @property
    def rank(self) -> int:
        return int(len(self.s))

    @property
    def energy(self) -> np.ndarray:
        return self.s * self.s

    def reconstruct_prefix(
        self, k: int, shape: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        k = min(self.rank, max(0, int(k)))
        if k == 0:
            return np.zeros(shape or (self.d_out, self.d_in), dtype=np.float64)
        return self.U[:, :k] @ (self.s[:k, None] * self.V[:, :k].T)

    def reconstruct_indices(
        self, indices: Sequence[int], shape: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        indices = np.asarray(list(indices), dtype=np.int64)
        if indices.size == 0:
            return np.zeros(shape or (self.d_out, self.d_in), dtype=np.float64)
        if np.any(indices < 0) or np.any(indices >= self.rank):
            raise IndexError("SVD component index out of range")
        return self.U[:, indices] @ (self.s[indices, None] * self.V[:, indices].T)


def compute_svd(delta: np.ndarray) -> SVDRecord:
    matrix = np.asarray(delta, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"SVD update must be two-dimensional, found {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise FloatingPointError("SVD input contains non-finite values")
    U, s, Vh = np.linalg.svd(matrix, full_matrices=False)
    reconstructed = U @ (s[:, None] * Vh)
    denominator = max(float(np.linalg.norm(matrix)), EPS)
    relative_error = float(np.linalg.norm(reconstructed - matrix) / denominator)
    left_error = float(np.linalg.norm(U.T @ U - np.eye(U.shape[1]), ord="fro"))
    right_error = float(np.linalg.norm(Vh @ Vh.T - np.eye(Vh.shape[0]), ord="fro"))
    return SVDRecord(
        U=U,
        s=s,
        V=Vh.T,
        d_out=int(matrix.shape[0]),
        d_in=int(matrix.shape[1]),
        reconstruction_relative_error=relative_error,
        orthonormality_error=max(left_error, right_error),
    )


def decompose_updates(
    updates: Mapping[Tuple[str, int, str], np.ndarray]
) -> Dict[Tuple[str, int, str], SVDRecord]:
    return {key: compute_svd(value) for key, value in updates.items()}


def rank_for_fraction(full_rank: int, fraction: float) -> int:
    if fraction <= 0:
        return 0
    return min(int(full_rank), int(math.ceil(float(fraction) * int(full_rank) - 1e-15)))


def natural_fraction_grid(records: Sequence[SVDRecord]) -> Tuple[float, ...]:
    values = {0.0, 1.0}
    for record in records:
        values.update(round(k / record.rank, 12) for k in range(1, record.rank + 1))
    return tuple(sorted(values))


def nearest_natural_fraction(fraction: float, records: Sequence[SVDRecord]) -> float:
    fraction = min(1.0, max(0.0, float(fraction)))
    grid = natural_fraction_grid(records)
    return float(min(grid, key=lambda x: (abs(x - fraction), x)))


def operator_gram(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    if U.ndim != 2 or V.ndim != 2 or U.shape[1] != V.shape[1]:
        raise ValueError("Paired U/V operator bases must have the same column count")
    return (U.T @ U) * (V.T @ V)


def orthonormalizer_from_gram(
    gram: np.ndarray,
    ambient_dimension: int,
    relative_tolerance: float = 0.0,
) -> Tuple[np.ndarray, int, float]:
    """Return B such that raw_operator_basis @ B has orthonormal columns."""

    gram = np.asarray(gram, dtype=np.float64)
    if gram.size == 0:
        return np.zeros((gram.shape[0], 0), dtype=np.float64), 0, 0.0
    gram = (gram + gram.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    sigma_max = math.sqrt(float(np.max(eigenvalues))) if len(eigenvalues) else 0.0
    if relative_tolerance > 0:
        sigma_tolerance = float(relative_tolerance) * sigma_max
    else:
        sigma_tolerance = (
            max(int(ambient_dimension), gram.shape[0])
            * np.finfo(np.float64).eps
            * max(sigma_max, 1.0)
        )
    keep = eigenvalues > sigma_tolerance**2
    rank = int(np.sum(keep))
    if rank == 0:
        return np.zeros((gram.shape[0], 0), dtype=np.float64), 0, sigma_tolerance
    B = eigenvectors[:, keep] / np.sqrt(eigenvalues[keep])[None, :]
    return B, rank, float(sigma_tolerance)


def projection_fractions_rank_one(
    candidate_U: np.ndarray,
    candidate_V: np.ndarray,
    context_parts: Sequence[Tuple[np.ndarray, np.ndarray]],
    ambient_dimension: int,
    relative_tolerance: float = 0.0,
) -> Tuple[np.ndarray, int, float]:
    """Compute ``||P_C Z_i||_F^2`` exactly through the context Gram matrix."""

    candidate_U = np.asarray(candidate_U, dtype=np.float64)
    candidate_V = np.asarray(candidate_V, dtype=np.float64)
    candidate_rank = candidate_U.shape[1]
    if candidate_V.shape[1] != candidate_rank:
        raise ValueError("Candidate U/V ranks differ")
    nonempty = [part for part in context_parts if part[0].shape[1] > 0]
    if not nonempty:
        return np.zeros(candidate_rank, dtype=np.float64), 0, 0.0
    U_context = np.concatenate([part[0] for part in nonempty], axis=1)
    V_context = np.concatenate([part[1] for part in nonempty], axis=1)
    gram = operator_gram(U_context, V_context)
    B, rank, tolerance = orthonormalizer_from_gram(
        gram, ambient_dimension, relative_tolerance
    )
    if rank == 0:
        return np.zeros(candidate_rank, dtype=np.float64), 0, tolerance
    cross = (U_context.T @ candidate_U) * (V_context.T @ candidate_V)
    coordinates = B.T @ cross
    values = np.sum(coordinates * coordinates, axis=0)
    if values.size and (float(np.min(values)) < -1e-8 or float(np.max(values)) > 1 + 1e-6):
        raise RuntimeError(
            f"Projection energy escaped [0,1]: min={values.min()}, max={values.max()}"
        )
    return np.clip(values, 0.0, 1.0), rank, tolerance


@dataclass(frozen=True)
class PrincipalDecomposition:
    cosines: np.ndarray
    candidate_mix: np.ndarray
    candidate_coefficients: np.ndarray
    context_coefficients: np.ndarray
    U_context_raw: np.ndarray
    V_context_raw: np.ndarray

    @property
    def rank(self) -> int:
        return int(len(self.cosines))


def principal_decomposition_rank_one(
    candidate_U: np.ndarray,
    candidate_V: np.ndarray,
    candidate_s: np.ndarray,
    context_parts: Sequence[Tuple[np.ndarray, np.ndarray]],
    ambient_dimension: int,
    relative_tolerance: float = 0.0,
) -> PrincipalDecomposition:
    """Principal angles between candidate and context rank-one operator spans."""

    candidate_U = np.asarray(candidate_U, dtype=np.float64)
    candidate_V = np.asarray(candidate_V, dtype=np.float64)
    candidate_s = np.asarray(candidate_s, dtype=np.float64)
    candidate_rank = candidate_U.shape[1]
    if candidate_V.shape[1] != candidate_rank or len(candidate_s) != candidate_rank:
        raise ValueError("Candidate U/V/s ranks differ")
    nonempty = [part for part in context_parts if part[0].shape[1] > 0]
    if not nonempty:
        return PrincipalDecomposition(
            np.zeros(0),
            np.zeros((candidate_rank, 0)),
            np.zeros(0),
            np.zeros((0, 0)),
            np.zeros((candidate_U.shape[0], 0)),
            np.zeros((candidate_V.shape[0], 0)),
        )
    U_context = np.concatenate([part[0] for part in nonempty], axis=1)
    V_context = np.concatenate([part[1] for part in nonempty], axis=1)
    gram = operator_gram(U_context, V_context)
    B, rank, _ = orthonormalizer_from_gram(
        gram, ambient_dimension, relative_tolerance
    )
    if rank == 0 or candidate_rank == 0:
        return PrincipalDecomposition(
            np.zeros(0),
            np.zeros((candidate_rank, 0)),
            np.zeros(0),
            np.zeros((U_context.shape[1], 0)),
            U_context,
            V_context,
        )
    cross_raw = (candidate_U.T @ U_context) * (candidate_V.T @ V_context)
    cross = cross_raw @ B
    candidate_mix, cosines, context_mix_h = np.linalg.svd(cross, full_matrices=False)
    cosines = np.clip(cosines, 0.0, 1.0)
    context_mix = context_mix_h.T
    return PrincipalDecomposition(
        cosines=cosines,
        candidate_mix=candidate_mix,
        candidate_coefficients=candidate_mix.T @ candidate_s,
        context_coefficients=B @ context_mix,
        U_context_raw=U_context,
        V_context_raw=V_context,
    )


def projected_candidate_update(
    decomposition: PrincipalDecomposition,
    principal_indices: Sequence[int],
    shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Project selected candidate principal directions into the context basis."""

    indices = np.asarray(list(principal_indices), dtype=np.int64)
    if indices.size == 0:
        if shape is None:
            shape = (
                decomposition.U_context_raw.shape[0],
                decomposition.V_context_raw.shape[0],
            )
        return np.zeros(shape, dtype=np.float64)
    if np.any(indices < 0) or np.any(indices >= decomposition.rank):
        raise IndexError("Principal index out of range")
    principal_amplitudes = (
        decomposition.candidate_coefficients[indices]
        * decomposition.cosines[indices]
    )
    raw_coefficients = (
        decomposition.context_coefficients[:, indices] @ principal_amplitudes
    )
    return (
        decomposition.U_context_raw * raw_coefficients[None, :]
    ) @ decomposition.V_context_raw.T


def haar_basis(dimension: int, rank: int, rng: np.random.Generator) -> np.ndarray:
    dimension = int(dimension)
    rank = int(rank)
    if rank < 0 or rank > dimension:
        raise ValueError(f"Invalid Haar basis shape ({dimension}, {rank})")
    if rank == 0:
        return np.zeros((dimension, 0), dtype=np.float64)
    matrix = rng.standard_normal((dimension, rank))
    Q, R = np.linalg.qr(matrix, mode="reduced")
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1
    return Q * signs[None, :]


def quantile_summary(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {key: float("nan") for key in ("min", "q01", "median", "q95", "q99", "max")}
    return {
        "min": float(np.min(array)),
        "q01": float(np.quantile(array, 0.01)),
        "median": float(np.median(array)),
        "q95": float(np.quantile(array, 0.95)),
        "q99": float(np.quantile(array, 0.99)),
        "max": float(np.max(array)),
    }


def random_projection_quantile_normal(
    probability: float, subspace_rank: int, ambient_dimension: int
) -> float:
    """Large-dimensional approximation to Beta(k/2,(D-k)/2).

    This is used only for the explicitly labelled componentwise display cutoff,
    never as a replacement for the matched rank-one Monte Carlo spectrum.
    """

    k = int(subspace_rank)
    D = int(ambient_dimension)
    if k <= 0:
        return 0.0
    if k >= D:
        return 1.0
    mean = k / D
    variance = 2.0 * k * (D - k) / (D * D * (D + 2.0))
    z = NormalDist().inv_cdf(float(probability))
    return float(min(1.0, max(0.0, mean + z * math.sqrt(max(variance, 0.0)))))


def synthetic_self_test() -> Dict[str, float]:
    """Exact small-space checks against explicit vectorisation."""

    rng = np.random.default_rng(20260907)
    U, _ = np.linalg.qr(rng.standard_normal((7, 3)), mode="reduced")
    V, _ = np.linalg.qr(rng.standard_normal((5, 3)), mode="reduced")
    values, rank, _ = projection_fractions_rank_one(U, V, [(U, V)], 35)
    if rank != 3 or not np.allclose(values, 1.0, atol=1e-10):
        raise AssertionError("Identical-context projection failed")

    U2 = np.eye(4)[:, :2]
    V2 = np.eye(4)[:, :2]
    U3 = np.eye(4)[:, 2:]
    V3 = np.eye(4)[:, 2:]
    orthogonal, _, _ = projection_fractions_rank_one(U2, V2, [(U3, V3)], 16)
    if np.max(np.abs(orthogonal)) > 1e-12:
        raise AssertionError("Orthogonal-context projection failed")

    C, _ = np.linalg.qr(rng.standard_normal((7, 2)), mode="reduced")
    D, _ = np.linalg.qr(rng.standard_normal((5, 2)), mode="reduced")
    single, _, _ = projection_fractions_rank_one(U, V, [(C, D)], 35)
    full, _, _ = projection_fractions_rank_one(U, V, [(C, D), (U[:, :1], V[:, :1])], 35)
    if float(np.min(full - single)) < -1e-9:
        raise AssertionError("Full-context nesting failed")

    context_vectors = np.column_stack(
        [np.outer(C[:, i], D[:, i]).reshape(-1) for i in range(C.shape[1])]
    )
    candidate_vectors = np.column_stack(
        [np.outer(U[:, i], V[:, i]).reshape(-1) for i in range(U.shape[1])]
    )
    explicit_Q, _ = np.linalg.qr(context_vectors, mode="reduced")
    explicit = np.sum((explicit_Q.T @ candidate_vectors) ** 2, axis=0)
    if not np.allclose(single, explicit, atol=1e-10):
        raise AssertionError("Gram and explicit projection implementations differ")

    principal = principal_decomposition_rank_one(U, V, np.array([3.0, 2.0, 1.0]), [(U, V)], 35)
    if not np.allclose(principal.cosines, 1.0, atol=1e-10):
        raise AssertionError("Identical principal angles failed")
    reconstructed = projected_candidate_update(principal, range(principal.rank), (7, 5))
    expected = U @ (np.array([3.0, 2.0, 1.0])[:, None] * V.T)
    if not np.allclose(reconstructed, expected, atol=1e-10):
        raise AssertionError("Principal projection reconstruction failed")
    return {
        "identical_projection_min": float(np.min(values)),
        "orthogonal_projection_max": float(np.max(orthogonal)),
        "full_minus_single_min": float(np.min(full - single)),
        "explicit_projection_max_error": float(np.max(np.abs(single - explicit))),
        "principal_identity_max_error": float(np.max(np.abs(principal.cosines - 1.0))),
    }

