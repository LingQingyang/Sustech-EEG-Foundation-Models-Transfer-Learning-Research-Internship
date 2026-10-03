"""The six task-adaptation spectra, and only the six spectra.

Main spectra
------------
1. Individual Energy
2. Individual Functional
3. Shared Energy
4. Shared Functional

Projection-route diagnostic spectra
------------------------------------
5. Principal Angle
6. Overlapped Functional

The final pair is deliberately retained.  Together they test, and empirically
reject, the shortcut that geometric subspace alignment alone is sufficient for
functional projection-based transfer.  Heat-map and STI calculations live in
their own modules and do not appear here.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .config import (
    EXPECTED_MODULE_SHAPES,
    TASK_ORDER,
    AnalysisConfig,
    ProjectConfig,
)
from .data import load_all_task_meta
from .geometry import (
    PrincipalDecomposition,
    SVDRecord,
    haar_basis,
    nearest_natural_fraction,
    principal_decomposition_rank_one,
    projected_candidate_update,
    projection_fractions_rank_one,
    quantile_summary,
    random_projection_quantile_normal,
    rank_for_fraction,
)
from .io import (
    ProgressStore,
    RunRecord,
    dump_json,
    ensure_dir,
    load_decompositions,
    log,
    module_layer_role,
    read_csv,
    sha256_file,
    stable_seed,
    write_csv,
)
from .model import FunctionalEvaluator


EPS = 1e-12
SPECTRUM_NAMES = (
    "Individual Energy",
    "Individual Functional",
    "Shared Energy",
    "Shared Functional",
    "Principal Angle",
    "Overlapped Functional",
)
DecompositionGrid = Mapping[Tuple[str, int, str], SVDRecord]
EvaluatorFactory = Callable[[RunRecord], object]


@dataclass(frozen=True)
class SpectraPaths:
    root: Path
    individual_energy: Path
    individual_functional: Path
    functional_cutoffs: Path
    shared_energy: Path
    shared_energy_random: Path
    shared_functional: Path
    principal_angle: Path
    overlap_functional: Path
    components: Path
    random_cache: Path
    progress: Path

    @classmethod
    def under(cls, output_dir: Path) -> "SpectraPaths":
        root = ensure_dir(Path(output_dir) / "spectra")
        return cls(
            root=root,
            individual_energy=root / "individual_energy_spectrum.csv",
            individual_functional=root / "individual_functional_spectrum.csv",
            functional_cutoffs=root / "functional_cutoffs.csv",
            shared_energy=root / "shared_energy_spectrum.csv",
            shared_energy_random=root / "shared_energy_random_reference.csv",
            shared_functional=root / "shared_functional_spectrum.csv",
            principal_angle=root / "principal_angle_spectrum.csv",
            overlap_functional=root / "overlapped_functional_spectrum.csv",
            components=ensure_dir(root / "shared_components"),
            random_cache=ensure_dir(root / "random_cache"),
            progress=ensure_dir(root / "progress"),
        )


def _records_for_run(
    run: RunRecord, decompositions: DecompositionGrid
) -> List[SVDRecord]:
    return [decompositions[(run.task, run.replicate, module)] for module in EXPECTED_MODULE_SHAPES]


def _component_count(run: RunRecord, decompositions: DecompositionGrid, q: float) -> int:
    return sum(
        rank_for_fraction(decompositions[(run.task, run.replicate, module)].rank, q)
        for module in EXPECTED_MODULE_SHAPES
    )


def _individual_energy(
    run: RunRecord, decompositions: DecompositionGrid, q: float
) -> float:
    numerator = 0.0
    denominator = 0.0
    for module in EXPECTED_MODULE_SHAPES:
        record = decompositions[(run.task, run.replicate, module)]
        k = rank_for_fraction(record.rank, q)
        numerator += float(np.sum(record.energy[:k]))
        denominator += float(np.sum(record.energy))
    return numerator / max(denominator, EPS)


def _first_crossing_interval(
    rows: Sequence[Mapping[str, object]], x_key: str, y_key: str, threshold: float
) -> Optional[Tuple[float, float]]:
    ordered = sorted(rows, key=lambda row: float(row[x_key]))
    for index, row in enumerate(ordered):
        if float(row[y_key]) >= threshold:
            if index == 0:
                x = float(row[x_key])
                return x, x
            return float(ordered[index - 1][x_key]), float(row[x_key])
    return None


def _estimate_x_for_y(
    rows: Sequence[Mapping[str, object]],
    x_key: str,
    y_key: str,
    target: float,
) -> Optional[float]:
    ordered = sorted(rows, key=lambda row: float(row[x_key]))
    if not ordered:
        return None
    for left, right in zip(ordered[:-1], ordered[1:]):
        x0, x1 = float(left[x_key]), float(right[x_key])
        y0, y1 = float(left[y_key]), float(right[y_key])
        if (target - y0) * (target - y1) <= 0 and abs(y1 - y0) > 1e-15:
            return x0 + (target - y0) * (x1 - x0) / (y1 - y0)
        if abs(target - y0) <= 1e-15:
            return x0
    nearest = min(ordered, key=lambda row: (abs(float(row[y_key]) - target), float(row[x_key])))
    return float(nearest[x_key])


def _available_vertical_targets(
    rows: Sequence[Mapping[str, object]],
    y_key: str,
    grid: Sequence[float],
    extras: Sequence[float],
) -> Tuple[float, ...]:
    values = [float(row[y_key]) for row in rows if math.isfinite(float(row[y_key]))]
    if not values:
        return tuple()
    lower, upper = max(0.0, min(values)), min(1.0, max(values))
    selected = {float(value) for value in grid if lower - 1e-12 <= value <= upper + 1e-12}
    selected.update(
        min(1.0, max(0.0, float(value)))
        for value in extras
        if value is not None and math.isfinite(float(value))
    )
    return tuple(sorted(selected))


def _nearest_rows_to_targets(
    rows: Sequence[Mapping[str, object]],
    x_key: str,
    y_key: str,
    targets: Sequence[float],
) -> Dict[float, List[float]]:
    selected: Dict[float, List[float]] = defaultdict(list)
    for target in targets:
        row = min(
            rows,
            key=lambda item: (abs(float(item[y_key]) - float(target)), float(item[x_key])),
        )
        selected[float(row[x_key])].append(float(target))
    return selected


def _target_json(values: Sequence[float]) -> str:
    return json.dumps([float(value) for value in values], separators=(",", ":"))


def _fraction_lattice(step: float, lower: float, upper: float) -> Tuple[float, ...]:
    if upper <= lower + 1e-15:
        return (float(upper),)
    start = int(math.ceil((lower - 1e-12) / step))
    stop = int(math.floor((upper + 1e-12) / step))
    values = {float(lower), float(upper)}
    values.update(round(index * step, 12) for index in range(start, stop + 1))
    return tuple(sorted(value for value in values if lower - 1e-12 <= value <= upper + 1e-12))


def individual_spectra_for_run(
    run: RunRecord,
    decompositions: DecompositionGrid,
    analysis: AnalysisConfig,
    evaluator,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    """Compute Individual Energy and Individual Functional for one run."""

    records = _records_for_run(run, decompositions)
    evaluated: Dict[float, Dict[str, object]] = {}

    def ensure_q(q: float, stage: str) -> Dict[str, object]:
        q = float(round(min(1.0, max(0.0, q)), 12))
        if q not in evaluated:
            metrics = evaluator.evaluate_q(q)
            evaluated[q] = {
                "q": q,
                "component_count": _component_count(run, decompositions, q),
                "raw_bacc": float(metrics["balanced_accuracy"]),
                "accuracy": float(metrics["accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
                "probe_stages": stage,
            }
        elif stage not in str(evaluated[q]["probe_stages"]).split("+"):
            evaluated[q]["probe_stages"] = f"{evaluated[q]['probe_stages']}+{stage}"
        return evaluated[q]

    for q in analysis.coarse_fractions:
        ensure_q(q, "coarse_probe")
    b0 = float(ensure_q(0.0, "mandatory_baseline")["raw_bacc"])
    bfull = float(evaluator.full_bacc)
    q1 = ensure_q(1.0, "mandatory_endpoint")
    if abs(float(q1["raw_bacc"]) - bfull) > analysis.metric_tolerance:
        raise RuntimeError(f"{run.tag}: q=1 does not reproduce the full checkpoint")
    threshold = analysis.target_functional_retention * bfull
    coarse = sorted(evaluated.values(), key=lambda row: float(row["q"]))
    targets = _available_vertical_targets(
        coarse,
        "raw_bacc",
        analysis.bacc_vertical_targets,
        (b0, bfull, threshold, min(float(x["raw_bacc"]) for x in coarse), max(float(x["raw_bacc"]) for x in coarse)),
    )
    for target in targets:
        estimate = _estimate_x_for_y(coarse, "q", "raw_bacc", target)
        if estimate is not None:
            ensure_q(nearest_natural_fraction(estimate, records), f"bacc_target_{target:.3f}")

    crossing = _first_crossing_interval(list(evaluated.values()), "q", "raw_bacc", threshold)
    if crossing is not None:
        for q in _fraction_lattice(analysis.fine_step, crossing[0], crossing[1]):
            ensure_q(nearest_natural_fraction(q, records), "functional99_refine")
    reached = [row for row in evaluated.values() if float(row["raw_bacc"]) >= threshold]
    if not reached:
        raise RuntimeError(f"{run.tag}: functional 99% cutoff was not reached")
    q_functional = min(float(row["q"]) for row in reached)

    natural_q = sorted(
        {0.0, 1.0}
        | {
            round(k / record.rank, 12)
            for record in records
            for k in range(1, record.rank + 1)
        }
    )
    natural_energy = [_individual_energy(run, decompositions, q) for q in natural_q]
    energy_selection: Dict[float, List[float]] = defaultdict(list)
    for target in analysis.energy_vertical_targets:
        index = next(
            (index for index, value in enumerate(natural_energy) if value + 1e-15 >= target),
            len(natural_q) - 1,
        )
        energy_selection[float(natural_q[index])].append(float(target))
    energy_selection[0.0].append(0.0)
    energy_selection[1.0].append(1.0)
    q_energy = min(
        q
        for q, values in energy_selection.items()
        if any(abs(value - 0.99) < 1e-12 for value in values)
    )

    functional_rows_all = sorted(evaluated.values(), key=lambda row: float(row["q"]))
    functional_targets = _available_vertical_targets(
        functional_rows_all,
        "raw_bacc",
        analysis.bacc_vertical_targets,
        (b0, bfull, threshold),
    )
    functional_selection = _nearest_rows_to_targets(
        functional_rows_all, "q", "raw_bacc", functional_targets
    )
    functional_selection[0.0].append(b0)
    functional_selection[1.0].append(bfull)
    functional_selection[q_functional].append(threshold)
    maximum = max(functional_rows_all, key=lambda row: float(row["raw_bacc"]))
    functional_selection[float(maximum["q"])].append(float(maximum["raw_bacc"]))

    common = {
        "task": run.task,
        "replicate": run.replicate,
        "full_bacc": bfull,
        "zero_component_baseline_bacc": b0,
        "chance_bacc": 1.0 / int(run.metadata["num_classes"]),
        "q_energy_99": q_energy,
        "q_functional_99": q_functional,
        "functional99_target_bacc": threshold,
        "analysis_profile": analysis.profile,
    }
    energy_rows = []
    for q in sorted(energy_selection):
        energy_rows.append(
            {
                **common,
                "q": q,
                "component_count": _component_count(run, decompositions, q),
                "individual_energy": _individual_energy(run, decompositions, q),
                "vertical_targets": _target_json(sorted(set(energy_selection[q]))),
                "is_energy_cutoff": int(abs(q - q_energy) < 1e-12),
            }
        )
    functional_rows = []
    by_q = {float(row["q"]): row for row in functional_rows_all}
    for q in sorted(functional_selection):
        row = by_q[q]
        functional_rows.append(
            {
                **common,
                **row,
                "vertical_targets": _target_json(sorted(set(functional_selection[q]))),
                "is_functional_cutoff": int(abs(q - q_functional) < 1e-12),
            }
        )
    cutoff_row = min(functional_rows_all, key=lambda row: abs(float(row["q"]) - q_functional))
    cutoff = {
        "task": run.task,
        "replicate": run.replicate,
        "q_star": q_functional,
        "K_star": int(cutoff_row["component_count"]),
        "B_ind_at_qstar": float(cutoff_row["raw_bacc"]),
        "B0": b0,
        "Bfull": bfull,
        "q_energy_99": q_energy,
        "d_energy_99": _component_count(run, decompositions, q_energy),
    }
    return energy_rows, functional_rows, cutoff


def _runs_by_task(runs: Sequence[RunRecord]) -> Dict[str, List[RunRecord]]:
    grouped: Dict[str, List[RunRecord]] = defaultdict(list)
    for run in runs:
        grouped[run.task].append(run)
    return {task: sorted(values, key=lambda run: run.replicate) for task, values in grouped.items()}


def context_comparisons(
    runs: Sequence[RunRecord], analysis: AnalysisConfig, include_replicate: bool = False
) -> List[Tuple[RunRecord, List[RunRecord], str]]:
    grouped = _runs_by_task(runs)
    comparisons = []
    for candidate in runs:
        other_tasks = [task for task in TASK_ORDER if task != candidate.task]
        if analysis.context_pairing == "matched":
            for task in other_tasks:
                matches = [run for run in grouped[task] if run.replicate == candidate.replicate]
                comparisons.append((candidate, [matches[0]], "single"))
            full = [
                [run for run in grouped[task] if run.replicate == candidate.replicate][0]
                for task in other_tasks
            ]
            comparisons.append((candidate, full, "full"))
        else:
            for task in other_tasks:
                for context in grouped[task]:
                    comparisons.append((candidate, [context], "single"))
            for context_tuple in itertools.product(*(grouped[task] for task in other_tasks)):
                comparisons.append((candidate, list(context_tuple), "full"))
        if include_replicate:
            peer = [run for run in grouped[candidate.task] if run.replicate != candidate.replicate]
            if peer:
                comparisons.append((candidate, [peer[0]], "replicate"))
    return comparisons


def comparison_id(
    candidate: RunRecord, contexts: Sequence[RunRecord], regime: str
) -> str:
    context = "+".join(f"{run.task}r{run.replicate}" for run in contexts)
    return f"{candidate.task}r{candidate.replicate}__{regime}__{context}"


def _cutoff_map(rows: Sequence[Mapping[str, object]]) -> Dict[Tuple[str, int], Dict[str, float]]:
    return {
        (str(row["task"]), int(float(row["replicate"]))): {
            key: float(row[key])
            for key in (
                "q_star",
                "K_star",
                "B_ind_at_qstar",
                "B0",
                "Bfull",
                "q_energy_99",
                "d_energy_99",
            )
        }
        for row in rows
    }


def _context_parts(
    contexts: Sequence[RunRecord],
    module: str,
    cutoffs: Mapping[Tuple[str, int], Mapping[str, float]],
    decompositions: DecompositionGrid,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    parts = []
    for context in contexts:
        record = decompositions[(context.task, context.replicate, module)]
        k = rank_for_fraction(record.rank, cutoffs[context.key]["q_star"])
        parts.append((record.U[:, :k], record.V[:, :k]))
    return parts


def shared_components(
    candidate: RunRecord,
    contexts: Sequence[RunRecord],
    cutoffs: Mapping[Tuple[str, int], Mapping[str, float]],
    decompositions: DecompositionGrid,
    analysis: AnalysisConfig,
) -> Tuple[List[Dict[str, object]], Dict[str, int], Dict[str, float]]:
    """Rank candidate components by sigma_i^2 ||P_C Z_i||_F^2."""

    components: List[Dict[str, object]] = []
    context_dimensions = {}
    tolerances = {}
    for module, shape in EXPECTED_MODULE_SHAPES.items():
        candidate_record = decompositions[(candidate.task, candidate.replicate, module)]
        k_candidate = rank_for_fraction(
            candidate_record.rank, cutoffs[candidate.key]["q_star"]
        )
        if k_candidate == 0:
            context_dimensions[module] = 0
            tolerances[module] = 0.0
            continue
        fractions, context_rank, tolerance = projection_fractions_rank_one(
            candidate_record.U[:, :k_candidate],
            candidate_record.V[:, :k_candidate],
            _context_parts(contexts, module, cutoffs, decompositions),
            shape[0] * shape[1],
            analysis.ortho_rtol,
        )
        context_dimensions[module] = context_rank
        tolerances[module] = tolerance
        projection_null = random_projection_quantile_normal(
            analysis.display_shared_quantile, context_rank, shape[0] * shape[1]
        )
        block, role = module_layer_role(module)
        for index in range(k_candidate):
            sigma = float(candidate_record.s[index])
            observed = sigma * sigma * float(fractions[index])
            null = sigma * sigma * projection_null
            components.append(
                {
                    "module": module,
                    "block": block + 1,
                    "matrix_type": role,
                    "component_index": index + 1,
                    "component_index0": index,
                    "sigma": sigma,
                    "sigma2": sigma * sigma,
                    "projection_energy_fraction": float(fractions[index]),
                    "shared_energy": observed,
                    "component_random_projection_quantile": projection_null,
                    "component_random_shared_energy_threshold": null,
                    "shared_margin_vs_component_null": observed - null,
                    "above_component_null": int(observed > null),
                    "component_null_quantile": analysis.display_shared_quantile,
                    "component_null_method": "normal approximation to Beta(k/2,(D-k)/2)",
                }
            )
    components.sort(
        key=lambda row: (
            -float(row["shared_energy"]),
            str(row["module"]),
            int(row["component_index"]),
        )
    )
    prefix = 0
    still_prefix = True
    for rank, row in enumerate(components, start=1):
        row["shared_rank"] = rank
        if still_prefix and int(row["above_component_null"]):
            prefix += 1
        else:
            still_prefix = False
    for row in components:
        row["in_shared_prefix"] = int(int(row["shared_rank"]) <= prefix)
        row["display_shared_prefix_size"] = prefix
    return components, context_dimensions, tolerances


def shared_energy_spectrum(
    candidate: RunRecord,
    contexts: Sequence[RunRecord],
    regime: str,
    cutoffs: Mapping[Tuple[str, int], Mapping[str, float]],
    decompositions: DecompositionGrid,
    analysis: AnalysisConfig,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    components, dimensions, tolerances = shared_components(
        candidate, contexts, cutoffs, decompositions, analysis
    )
    K = len(components)
    denominator = sum(float(row["sigma2"]) for row in components)
    cumulative = np.cumsum([float(row["shared_energy"]) for row in components])
    gamma = float(cumulative[-1] / max(denominator, EPS)) if K else 0.0
    if gamma < -1e-10 or gamma > 1 + 1e-7:
        raise RuntimeError(f"Shared-energy endpoint escaped [0,1]: {gamma}")
    gamma = min(1.0, max(0.0, gamma))
    prefix = int(components[0]["display_shared_prefix_size"]) if components else 0
    counts = {0, K, prefix}
    target_map: Dict[int, List[float]] = defaultdict(list)
    target_map[0].append(0.0)
    target_map[K].append(gamma)
    for target in analysis.energy_vertical_targets:
        if target > gamma + 1e-12:
            continue
        count = 0 if target <= 0 else min(K, int(np.searchsorted(cumulative / max(denominator, EPS), target - 1e-15)) + 1)
        counts.add(count)
        target_map[count].append(float(target))
    cid = comparison_id(candidate, contexts, regime)
    common = {
        "comparison_id": cid,
        "candidate": candidate.task,
        "candidate_replicate": candidate.replicate,
        "context_regime": regime,
        "context_tasks": "+".join(run.task for run in contexts),
        "context_replicates": "+".join(str(run.replicate) for run in contexts),
        "K_star": K,
        "Gamma": gamma,
        "functional_energy_denominator": denominator,
        "context_dimension_total": sum(dimensions.values()),
        "context_dimension_by_matrix": json.dumps(dimensions, sort_keys=True),
        "orthogonalization_sigma_tolerance_max": max(tolerances.values()) if tolerances else 0.0,
        "display_shared_prefix_size": prefix,
        "analysis_profile": analysis.profile,
    }
    rows = []
    for count in sorted(counts):
        value = 0.0 if count == 0 else float(cumulative[count - 1] / max(denominator, EPS))
        rows.append(
            {
                **common,
                "component_count": count,
                "component_fraction": count / max(K, 1),
                "shared_energy_spectrum": value,
                "vertical_targets": _target_json(sorted(set(target_map[count]))),
                "is_display_prefix_endpoint": int(count == prefix),
            }
        )
    for row in components:
        row.update(common)
    return rows, components


def _random_shared_curve(
    candidate: RunRecord,
    contexts: Sequence[RunRecord],
    cutoffs: Mapping[Tuple[str, int], Mapping[str, float]],
    decompositions: DecompositionGrid,
    analysis: AnalysisConfig,
    draw: int,
    sample_counts: Sequence[int],
    regime: str,
) -> List[float]:
    rng = np.random.default_rng(
        stable_seed(
            analysis.random_seed,
            "shared",
            comparison_id(candidate, contexts, regime),
            draw,
        )
    )
    components = []
    for module, shape in EXPECTED_MODULE_SHAPES.items():
        candidate_record = decompositions[(candidate.task, candidate.replicate, module)]
        k_candidate = rank_for_fraction(
            candidate_record.rank, cutoffs[candidate.key]["q_star"]
        )
        if k_candidate == 0:
            continue
        random_context = []
        for context in contexts:
            context_record = decompositions[(context.task, context.replicate, module)]
            k_context = rank_for_fraction(
                context_record.rank, cutoffs[context.key]["q_star"]
            )
            random_context.append(
                (
                    haar_basis(shape[0], k_context, rng),
                    haar_basis(shape[1], k_context, rng),
                )
            )
        fractions, _, _ = projection_fractions_rank_one(
            candidate_record.U[:, :k_candidate],
            candidate_record.V[:, :k_candidate],
            random_context,
            shape[0] * shape[1],
            analysis.ortho_rtol,
        )
        components.extend(
            (
                float(candidate_record.s[index] ** 2 * fractions[index]),
                float(candidate_record.s[index] ** 2),
            )
            for index in range(k_candidate)
        )
    components.sort(key=lambda pair: -pair[0])
    denominator = sum(pair[1] for pair in components)
    cumulative = np.cumsum([pair[0] for pair in components])
    values = []
    for count in sample_counts:
        count = min(len(components), max(0, int(count)))
        values.append(
            0.0 if count == 0 else float(cumulative[count - 1] / max(denominator, EPS))
        )
    return values


def shared_random_reference(
    candidate: RunRecord,
    contexts: Sequence[RunRecord],
    regime: str,
    observed_rows: Sequence[Mapping[str, object]],
    cutoffs: Mapping[Tuple[str, int], Mapping[str, float]],
    decompositions: DecompositionGrid,
    analysis: AnalysisConfig,
) -> List[Dict[str, object]]:
    sample_rows = sorted(observed_rows, key=lambda row: int(row["component_count"]))
    counts = [int(row["component_count"]) for row in sample_rows]
    curves = np.asarray(
        [
            _random_shared_curve(
                candidate,
                contexts,
                cutoffs,
                decompositions,
                analysis,
                draw,
                counts,
                regime,
            )
            for draw in range(analysis.shared_random_n)
        ],
        dtype=np.float64,
    )
    rows = []
    for index, observed in enumerate(sample_rows):
        summary = quantile_summary(curves[:, index])
        random_upper = float(
            np.quantile(curves[:, index], analysis.random_upper_quantile)
        )
        rows.append(
            {
                "comparison_id": observed["comparison_id"],
                "candidate": candidate.task,
                "candidate_replicate": candidate.replicate,
                "context_regime": regime,
                "context_tasks": observed["context_tasks"],
                "context_replicates": observed["context_replicates"],
                "component_count": observed["component_count"],
                "component_fraction": observed["component_fraction"],
                "random_n": analysis.shared_random_n,
                "random_median": summary["median"],
                "random_q95": summary["q95"],
                "random_q99": summary["q99"],
                "random_q01": summary["q01"],
                "random_upper": random_upper,
                "random_upper_quantile": analysis.random_upper_quantile,
                "random_method": "shape/rank/rank-one matched Haar context; empirical candidate singular values",
            }
        )
    return rows


def shared_functional_spectrum(
    candidate: RunRecord,
    contexts: Sequence[RunRecord],
    regime: str,
    components: Sequence[Mapping[str, object]],
    energy_rows: Sequence[Mapping[str, object]],
    cutoffs: Mapping[Tuple[str, int], Mapping[str, float]],
    analysis: AnalysisConfig,
    evaluator,
) -> List[Dict[str, object]]:
    order = [
        (str(row["module"]), int(row["component_index0"])) for row in components
    ]
    K = len(order)
    prefix = int(components[0]["display_shared_prefix_size"]) if components else 0
    evaluated: Dict[int, Dict[str, object]] = {}

    def ensure_count(count: int, stage: str) -> Dict[str, object]:
        count = min(K, max(0, int(count)))
        if count not in evaluated:
            metrics = evaluator.evaluate_global_order(order, count)
            evaluated[count] = {
                "component_count": count,
                "component_fraction": count / max(K, 1),
                "raw_bacc": float(metrics["balanced_accuracy"]),
                "accuracy": float(metrics["accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
                "probe_stages": stage,
            }
        elif stage not in str(evaluated[count]["probe_stages"]).split("+"):
            evaluated[count]["probe_stages"] = f"{evaluated[count]['probe_stages']}+{stage}"
        return evaluated[count]

    for fraction in analysis.coarse_fractions:
        count = 0 if fraction <= 0 else min(K, int(math.ceil(fraction * K - 1e-15)))
        ensure_count(count, "coarse_probe")
    for row in energy_rows:
        ensure_count(int(row["component_count"]), "shared_energy_sample")
    ensure_count(prefix, "display_prefix_endpoint")
    baseline = float(ensure_count(0, "mandatory_baseline")["raw_bacc"])
    if abs(baseline - cutoffs[candidate.key]["B0"]) > analysis.metric_tolerance:
        raise RuntimeError("Shared and Individual zero-component baselines differ")
    bfull = float(evaluator.full_bacc)
    threshold = analysis.target_functional_retention * bfull
    coarse = sorted(evaluated.values(), key=lambda row: int(row["component_count"]))
    targets = _available_vertical_targets(
        coarse,
        "raw_bacc",
        analysis.bacc_vertical_targets,
        (
            baseline,
            bfull,
            threshold,
            cutoffs[candidate.key]["B_ind_at_qstar"],
            min(float(row["raw_bacc"]) for row in coarse),
            max(float(row["raw_bacc"]) for row in coarse),
        ),
    )
    for target in targets:
        estimate = _estimate_x_for_y(coarse, "component_count", "raw_bacc", target)
        if estimate is not None:
            ensure_count(int(math.floor(estimate + 0.5)), f"bacc_target_{target:.3f}")
    crossing = _first_crossing_interval(
        list(evaluated.values()), "component_count", "raw_bacc", threshold
    )
    if crossing is not None:
        step = max(1, int(math.ceil(analysis.fine_step * max(K, 1))))
        lower = max(0, int(math.floor(crossing[0])) - step)
        upper = min(K, int(math.ceil(crossing[1])) + step)
        for count in range(lower, upper + 1, step):
            ensure_count(count, "functional99_refine")
        ensure_count(upper, "functional99_refine")
    endpoint = ensure_count(K, "mandatory_endpoint")
    expected_endpoint = cutoffs[candidate.key]["B_ind_at_qstar"]
    endpoint_error = abs(float(endpoint["raw_bacc"]) - expected_endpoint)
    if endpoint_error > analysis.metric_tolerance:
        raise RuntimeError(
            f"{comparison_id(candidate, contexts, regime)}: shared endpoint does not equal Individual q*"
        )
    all_rows = sorted(evaluated.values(), key=lambda row: int(row["component_count"]))
    reached = [row for row in all_rows if float(row["raw_bacc"]) >= threshold]
    m_star = min(int(row["component_count"]) for row in reached) if reached else None
    maximum = max(all_rows, key=lambda row: float(row["raw_bacc"]))
    selected = _nearest_rows_to_targets(all_rows, "component_count", "raw_bacc", targets)
    selected[0].append(baseline)
    selected[float(K)].append(float(endpoint["raw_bacc"]))
    selected[float(prefix)].append(float(ensure_count(prefix, "display_prefix_endpoint")["raw_bacc"]))
    if m_star is not None:
        selected[float(m_star)].append(threshold)
    selected[float(maximum["component_count"])].append(float(maximum["raw_bacc"]))

    cid = comparison_id(candidate, contexts, regime)
    rows = []
    for row in all_rows:
        count = int(row["component_count"])
        if float(count) not in selected:
            continue
        rows.append(
            {
                "comparison_id": cid,
                "candidate": candidate.task,
                "candidate_replicate": candidate.replicate,
                "context_regime": regime,
                "context_tasks": "+".join(run.task for run in contexts),
                "context_replicates": "+".join(str(run.replicate) for run in contexts),
                "K_star": K,
                **row,
                "full_bacc": bfull,
                "chance_bacc": 1.0 / int(candidate.metadata["num_classes"]),
                "B_ind_at_qstar": expected_endpoint,
                "functional99_target_bacc": threshold,
                "m_star_99": "" if m_star is None else m_star,
                "reached_99": int(m_star is not None),
                "max_raw_bacc": float(maximum["raw_bacc"]),
                "max_raw_bacc_at_m": int(maximum["component_count"]),
                "endpoint_bacc_error_vs_individual_qstar": endpoint_error,
                "display_shared_prefix_size": prefix,
                "is_display_prefix_endpoint": int(count == prefix),
                "vertical_targets": _target_json(sorted(set(selected[float(count)]))),
                "reconstruction_rule": "original candidate sigma_i Z_i in shared-energy order",
                "analysis_profile": analysis.profile,
            }
        )
    return rows


def principal_geometry(
    candidate: RunRecord,
    contexts: Sequence[RunRecord],
    cutoffs: Mapping[Tuple[str, int], Mapping[str, float]],
    decompositions: DecompositionGrid,
    analysis: AnalysisConfig,
) -> Tuple[Dict[str, PrincipalDecomposition], List[Tuple[float, str, int]]]:
    decomposed = {}
    pairs = []
    for module, shape in EXPECTED_MODULE_SHAPES.items():
        candidate_record = decompositions[(candidate.task, candidate.replicate, module)]
        k_candidate = rank_for_fraction(
            candidate_record.rank, cutoffs[candidate.key]["q_star"]
        )
        principal = principal_decomposition_rank_one(
            candidate_record.U[:, :k_candidate],
            candidate_record.V[:, :k_candidate],
            candidate_record.s[:k_candidate],
            _context_parts(contexts, module, cutoffs, decompositions),
            shape[0] * shape[1],
            analysis.ortho_rtol,
        )
        decomposed[module] = principal
        pairs.extend(
            (float(cosine), module, index)
            for index, cosine in enumerate(principal.cosines)
        )
    pairs.sort(key=lambda value: (-value[0], value[1], value[2]))
    return decomposed, pairs


def _random_principal_upper(
    candidate: RunRecord,
    contexts: Sequence[RunRecord],
    regime: str,
    real_length: int,
    cutoffs: Mapping[Tuple[str, int], Mapping[str, float]],
    decompositions: DecompositionGrid,
    analysis: AnalysisConfig,
    cache_dir: Optional[Path] = None,
) -> np.ndarray:
    if real_length == 0:
        return np.zeros(0, dtype=np.float64)
    specification = []
    for module, shape in EXPECTED_MODULE_SHAPES.items():
        candidate_record = decompositions[(candidate.task, candidate.replicate, module)]
        context_ranks = [
            rank_for_fraction(
                decompositions[(context.task, context.replicate, module)].rank,
                cutoffs[context.key]["q_star"],
            )
            for context in contexts
        ]
        specification.append(
            [
                module,
                list(shape),
                rank_for_fraction(
                    candidate_record.rank, cutoffs[candidate.key]["q_star"]
                ),
                context_ranks,
            ]
        )
    signature_payload = {
        "version": "rank-one-principal-null-clean-1.0",
        "random_n": analysis.shared_random_n,
        "random_seed": analysis.random_seed,
        "upper_quantile": analysis.random_upper_quantile,
        "ortho_rtol": analysis.ortho_rtol,
        "real_length": real_length,
        "specification": specification,
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    cache_path = Path(cache_dir) / f"principal_{signature}.npz" if cache_dir else None
    if cache_path is not None and cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as archive:
            cached = np.asarray(archive["random_upper"], dtype=np.float64)
            cached_signature = str(np.asarray(archive["signature"]).item())
        if cached_signature != signature or len(cached) != real_length:
            raise RuntimeError(f"Invalid matched-random cache: {cache_path}")
        return cached
    samples = np.empty((analysis.shared_random_n, real_length), dtype=np.float32)
    for draw in range(analysis.shared_random_n):
        rng = np.random.default_rng(
            stable_seed(
                analysis.random_seed,
                "principal",
                signature,
                draw,
            )
        )
        values = []
        for module, shape in EXPECTED_MODULE_SHAPES.items():
            candidate_record = decompositions[(candidate.task, candidate.replicate, module)]
            k_candidate = rank_for_fraction(
                candidate_record.rank, cutoffs[candidate.key]["q_star"]
            )
            random_context = []
            for context in contexts:
                context_record = decompositions[(context.task, context.replicate, module)]
                k_context = rank_for_fraction(
                    context_record.rank, cutoffs[context.key]["q_star"]
                )
                random_context.append(
                    (
                        haar_basis(shape[0], k_context, rng),
                        haar_basis(shape[1], k_context, rng),
                    )
                )
            principal = principal_decomposition_rank_one(
                haar_basis(shape[0], k_candidate, rng),
                haar_basis(shape[1], k_candidate, rng),
                np.ones(k_candidate),
                random_context,
                shape[0] * shape[1],
                analysis.ortho_rtol,
            )
            values.extend(float(x) for x in principal.cosines)
        values = sorted(values, reverse=True)
        if len(values) != real_length:
            raise RuntimeError(
                f"Real/random principal spectrum lengths differ: {real_length} vs {len(values)}"
            )
        samples[draw] = np.asarray(values, dtype=np.float32)
    result = np.quantile(samples, analysis.random_upper_quantile, axis=0).astype(np.float64)
    if cache_path is not None:
        ensure_dir(cache_path.parent)
        temporary = cache_path.with_name(
            f"{cache_path.name}.tmp.{os.getpid()}.{time.time_ns()}"
        )
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                random_upper=result,
                signature=np.asarray(signature),
                random_n=np.asarray(analysis.shared_random_n),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, cache_path)
    return result


def principal_angle_spectrum(
    candidate: RunRecord,
    contexts: Sequence[RunRecord],
    regime: str,
    cutoffs: Mapping[Tuple[str, int], Mapping[str, float]],
    decompositions: DecompositionGrid,
    analysis: AnalysisConfig,
    random_cache_dir: Optional[Path] = None,
) -> List[Dict[str, object]]:
    _, pairs = principal_geometry(candidate, contexts, cutoffs, decompositions, analysis)
    random_upper = None
    if regime != "replicate":
        random_upper = _random_principal_upper(
            candidate,
            contexts,
            regime,
            len(pairs),
            cutoffs,
            decompositions,
            analysis,
            random_cache_dir,
        )
    cid = comparison_id(candidate, contexts, regime)
    rows = []
    for dimension, (cosine, module, module_index) in enumerate(pairs, start=1):
        threshold = (
            float(random_upper[dimension - 1]) if random_upper is not None else float("nan")
        )
        rows.append(
            {
                "comparison_id": cid,
                "candidate": candidate.task,
                "candidate_replicate": candidate.replicate,
                "context_regime": regime,
                "context_tasks": "+".join(run.task for run in contexts),
                "context_replicates": "+".join(str(run.replicate) for run in contexts),
                "principal_dimension": dimension,
                "cos_theta": cosine,
                "theta_degrees": float(np.degrees(np.arccos(np.clip(cosine, 0, 1)))),
                "matched_random_upper": threshold,
                "matched_random_quantile": analysis.random_upper_quantile,
                "above_matched_random": int(math.isfinite(threshold) and cosine > threshold),
                "module": module,
                "module_principal_index": module_index + 1,
                "interpretation": "geometric diagnostic; not a functional-recovery guarantee",
                "analysis_profile": analysis.profile,
            }
        )
    return rows


def overlapped_functional_spectrum(
    candidate: RunRecord,
    contexts: Sequence[RunRecord],
    regime: str,
    cutoffs: Mapping[Tuple[str, int], Mapping[str, float]],
    decompositions: DecompositionGrid,
    analysis: AnalysisConfig,
    evaluator,
) -> List[Dict[str, object]]:
    """Functional test of writing projected principal directions into the model."""

    principal_by_module, pairs = principal_geometry(
        candidate, contexts, cutoffs, decompositions, analysis
    )
    P = len(pairs)
    evaluated: Dict[int, Dict[str, object]] = {}

    def ensure_count(count: int, stage: str):
        count = min(P, max(0, int(count)))
        if count in evaluated:
            if stage not in str(evaluated[count]["probe_stages"]).split("+"):
                evaluated[count]["probe_stages"] = (
                    f"{evaluated[count]['probe_stages']}+{stage}"
                )
            return evaluated[count]
        selected: Dict[str, List[int]] = defaultdict(list)
        for _, module, index in pairs[:count]:
            selected[module].append(index)
        updates = {
            module: projected_candidate_update(
                principal_by_module[module], selected[module], shape
            )
            for module, shape in EXPECTED_MODULE_SHAPES.items()
        }
        metrics = evaluator.evaluate_updates(updates)
        row = {
            "principal_dimensions": count,
            "full_principal_dimensions": P,
            "principal_fraction": count / max(P, 1),
            "raw_bacc": float(metrics["balanced_accuracy"]),
            "accuracy": float(metrics["accuracy"]),
            "macro_f1": float(metrics["macro_f1"]),
            "full_bacc": evaluator.full_bacc,
            "functional_retention": float(metrics["balanced_accuracy"]) / max(evaluator.full_bacc, EPS),
            "probe_stages": stage,
        }
        evaluated[count] = row
        return row

    for fraction in analysis.coarse_fractions:
        count = 0 if fraction <= 0 else min(P, int(math.ceil(fraction * P - 1e-15)))
        ensure_count(count, "coarse_probe")
    threshold = analysis.target_functional_retention
    crossing = _first_crossing_interval(
        list(evaluated.values()), "principal_fraction", "functional_retention", threshold
    )
    if crossing is not None:
        step = max(1, int(math.ceil(analysis.fine_step * max(P, 1))))
        lower = max(0, int(math.floor(crossing[0] * P)) - step)
        upper = min(P, int(math.ceil(crossing[1] * P)) + step)
        for count in range(lower, upper + 1, step):
            ensure_count(count, "functional99_refine")
        ensure_count(upper, "functional99_refine")
    ensure_count(P, "mandatory_endpoint")
    ordered = sorted(evaluated.values(), key=lambda row: int(row["principal_dimensions"]))
    reached = [row for row in ordered if float(row["functional_retention"]) >= threshold]
    first_99 = min((int(row["principal_dimensions"]) for row in reached), default=None)
    maximum = max(ordered, key=lambda row: float(row["functional_retention"]))
    cid = comparison_id(candidate, contexts, regime)
    return [
        {
            "comparison_id": cid,
            "candidate": candidate.task,
            "candidate_replicate": candidate.replicate,
            "context_regime": regime,
            "context_tasks": "+".join(run.task for run in contexts),
            "context_replicates": "+".join(str(run.replicate) for run in contexts),
            **row,
            "reached_99": int(first_99 is not None),
            "first_99_principal_dimensions": "" if first_99 is None else first_99,
            "max_functional_retention": float(maximum["functional_retention"]),
            "max_retention_at_dimension": int(maximum["principal_dimensions"]),
            "reconstruction_rule": "candidate principal directions projected into context span",
            "interpretation": "tests whether geometric projection preserves candidate function",
            "analysis_profile": analysis.profile,
        }
        for row in ordered
    ]


def _default_evaluator_factory(
    project: ProjectConfig,
    runs: Sequence[RunRecord],
    decompositions: DecompositionGrid,
) -> EvaluatorFactory:
    metas = load_all_task_meta(project)
    return lambda run: FunctionalEvaluator(
        project,
        run,
        metas[run.task],
        decompositions,
        verify_full_replay=project.analysis.profile == "full",
    )


def _replace_rows(
    rows: List[Dict[str, object]],
    predicate: Callable[[Mapping[str, object]], bool],
    replacements: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    return [row for row in rows if not predicate(row)] + [dict(row) for row in replacements]


def _existing(path: Path, resume: bool) -> List[Dict[str, object]]:
    return [dict(row) for row in read_csv(path)] if resume and path.is_file() else []


def _delta_signatures(runs: Sequence[RunRecord]) -> Dict[str, str]:
    return {run.tag: sha256_file(run.delta_path) for run in runs}


def generate_individual_spectra(
    project: ProjectConfig,
    runs: Sequence[RunRecord],
    decompositions: DecompositionGrid,
    paths: SpectraPaths,
    evaluator_factory: EvaluatorFactory,
    resume: bool,
) -> Dict[Tuple[str, int], Dict[str, float]]:
    signature = {
        "stage": "individual_spectra",
        "analysis": asdict(project.analysis),
        "delta_sha256": _delta_signatures(runs),
    }
    progress = ProgressStore(paths.progress / "individual.json", signature, resume)
    energy_rows = _existing(paths.individual_energy, resume)
    functional_rows = _existing(paths.individual_functional, resume)
    cutoff_rows = _existing(paths.functional_cutoffs, resume)
    for run in runs:
        if progress.has(run.tag):
            continue
        log(f"Individual spectra: {run.tag}")
        evaluator = evaluator_factory(run)
        try:
            energy, functional, cutoff = individual_spectra_for_run(
                run, decompositions, project.analysis, evaluator
            )
        finally:
            evaluator.close()
        same_run = lambda row, run=run: (
            str(row.get("task")) == run.task
            and int(float(row.get("replicate", -1))) == run.replicate
        )
        energy_rows = _replace_rows(energy_rows, same_run, energy)
        functional_rows = _replace_rows(functional_rows, same_run, functional)
        cutoff_rows = _replace_rows(cutoff_rows, same_run, [cutoff])
        write_csv(paths.individual_energy, energy_rows)
        write_csv(paths.individual_functional, functional_rows)
        write_csv(paths.functional_cutoffs, cutoff_rows)
        progress.mark(run.tag)
    return _cutoff_map(cutoff_rows)


def generate_shared_spectra(
    project: ProjectConfig,
    runs: Sequence[RunRecord],
    decompositions: DecompositionGrid,
    paths: SpectraPaths,
    cutoffs: Mapping[Tuple[str, int], Mapping[str, float]],
    evaluator_factory: EvaluatorFactory,
    resume: bool,
) -> None:
    signature = {
        "stage": "shared_spectra",
        "analysis": asdict(project.analysis),
        "cutoffs": cutoffs,
        "delta_sha256": _delta_signatures(runs),
    }
    progress = ProgressStore(paths.progress / "shared.json", signature, resume)
    energy_master = _existing(paths.shared_energy, resume)
    random_master = _existing(paths.shared_energy_random, resume)
    functional_master = _existing(paths.shared_functional, resume)
    comparisons = context_comparisons(runs, project.analysis, include_replicate=False)
    by_candidate: Dict[Tuple[str, int], List[Tuple[RunRecord, List[RunRecord], str]]] = defaultdict(list)
    for item in comparisons:
        by_candidate[item[0].key].append(item)
    for candidate in runs:
        pending = [
            item
            for item in by_candidate[candidate.key]
            if not progress.has(comparison_id(item[0], item[1], item[2]))
        ]
        if not pending:
            continue
        evaluator = evaluator_factory(candidate)
        try:
            for candidate, contexts, regime in pending:
                cid = comparison_id(candidate, contexts, regime)
                log(f"Shared spectra: {cid}")
                energy, components = shared_energy_spectrum(
                    candidate,
                    contexts,
                    regime,
                    cutoffs,
                    decompositions,
                    project.analysis,
                )
                random_rows = shared_random_reference(
                    candidate,
                    contexts,
                    regime,
                    energy,
                    cutoffs,
                    decompositions,
                    project.analysis,
                )
                functional = shared_functional_spectrum(
                    candidate,
                    contexts,
                    regime,
                    components,
                    energy,
                    cutoffs,
                    project.analysis,
                    evaluator,
                )
                same = lambda row, cid=cid: str(row.get("comparison_id")) == cid
                energy_master = _replace_rows(energy_master, same, energy)
                random_master = _replace_rows(random_master, same, random_rows)
                functional_master = _replace_rows(functional_master, same, functional)
                write_csv(paths.shared_energy, energy_master)
                write_csv(paths.shared_energy_random, random_master)
                write_csv(paths.shared_functional, functional_master)
                safe = hashlib.sha256(cid.encode("utf-8")).hexdigest()[:16]
                write_csv(paths.components / f"{safe}.csv", components)
                progress.mark(cid)
        finally:
            evaluator.close()


def generate_principal_angle_spectrum(
    project: ProjectConfig,
    runs: Sequence[RunRecord],
    decompositions: DecompositionGrid,
    paths: SpectraPaths,
    cutoffs: Mapping[Tuple[str, int], Mapping[str, float]],
    resume: bool,
) -> None:
    signature = {
        "stage": "principal_angle_spectrum",
        "analysis": asdict(project.analysis),
        "cutoffs": cutoffs,
        "delta_sha256": _delta_signatures(runs),
    }
    progress = ProgressStore(paths.progress / "principal_angle.json", signature, resume)
    master = _existing(paths.principal_angle, resume)
    for candidate, contexts, regime in context_comparisons(
        runs, project.analysis, include_replicate=True
    ):
        cid = comparison_id(candidate, contexts, regime)
        if progress.has(cid):
            continue
        log(f"Principal Angle Spectrum: {cid}")
        rows = principal_angle_spectrum(
            candidate,
            contexts,
            regime,
            cutoffs,
            decompositions,
            project.analysis,
            paths.random_cache,
        )
        master = _replace_rows(
            master, lambda row, cid=cid: str(row.get("comparison_id")) == cid, rows
        )
        write_csv(paths.principal_angle, master)
        progress.mark(cid)


def generate_overlap_functional_spectrum(
    project: ProjectConfig,
    runs: Sequence[RunRecord],
    decompositions: DecompositionGrid,
    paths: SpectraPaths,
    cutoffs: Mapping[Tuple[str, int], Mapping[str, float]],
    evaluator_factory: EvaluatorFactory,
    resume: bool,
) -> None:
    signature = {
        "stage": "overlapped_functional_spectrum",
        "analysis": asdict(project.analysis),
        "cutoffs": cutoffs,
        "delta_sha256": _delta_signatures(runs),
    }
    progress = ProgressStore(paths.progress / "overlap_functional.json", signature, resume)
    master = _existing(paths.overlap_functional, resume)
    comparisons = context_comparisons(runs, project.analysis, include_replicate=True)
    grouped: Dict[Tuple[str, int], List[Tuple[RunRecord, List[RunRecord], str]]] = defaultdict(list)
    for item in comparisons:
        grouped[item[0].key].append(item)
    for candidate in runs:
        pending = [
            item
            for item in grouped[candidate.key]
            if not progress.has(comparison_id(item[0], item[1], item[2]))
        ]
        if not pending:
            continue
        evaluator = evaluator_factory(candidate)
        try:
            for candidate, contexts, regime in pending:
                cid = comparison_id(candidate, contexts, regime)
                log(f"Overlapped Functional Spectrum: {cid}")
                rows = overlapped_functional_spectrum(
                    candidate,
                    contexts,
                    regime,
                    cutoffs,
                    decompositions,
                    project.analysis,
                    evaluator,
                )
                master = _replace_rows(
                    master,
                    lambda row, cid=cid: str(row.get("comparison_id")) == cid,
                    rows,
                )
                write_csv(paths.overlap_functional, master)
                progress.mark(cid)
        finally:
            evaluator.close()


def generate_spectra(
    project: ProjectConfig,
    runs: Sequence[RunRecord],
    group: str = "all",
    resume: bool = True,
    decompositions: Optional[DecompositionGrid] = None,
    evaluator_factory: Optional[EvaluatorFactory] = None,
) -> Dict[str, object]:
    """Generate the requested main and/or projection-diagnostic spectra."""

    if group not in {"main", "projection", "all"}:
        raise ValueError("group must be main, projection, or all")
    paths = SpectraPaths.under(project.output_dir)
    if decompositions is None:
        decompositions = load_decompositions(runs)
    maximum_reconstruction_error = max(
        record.reconstruction_relative_error for record in decompositions.values()
    )
    maximum_orthonormality_error = max(
        record.orthonormality_error for record in decompositions.values()
    )
    if maximum_reconstruction_error > 1e-8 or maximum_orthonormality_error > 1e-8:
        raise RuntimeError("SVD sanity check failed")
    if evaluator_factory is None:
        evaluator_factory = _default_evaluator_factory(project, runs, decompositions)

    cutoffs = generate_individual_spectra(
        project,
        runs,
        decompositions,
        paths,
        evaluator_factory,
        resume,
    )
    if group in {"main", "all"}:
        generate_shared_spectra(
            project,
            runs,
            decompositions,
            paths,
            cutoffs,
            evaluator_factory,
            resume,
        )
    if group in {"projection", "all"}:
        generate_principal_angle_spectrum(
            project, runs, decompositions, paths, cutoffs, resume
        )
        generate_overlap_functional_spectrum(
            project,
            runs,
            decompositions,
            paths,
            cutoffs,
            evaluator_factory,
            resume,
        )

    expected = {
        "individual_energy": paths.individual_energy,
        "individual_functional": paths.individual_functional,
        "functional_cutoffs": paths.functional_cutoffs,
    }
    if group in {"main", "all"}:
        expected.update(
            {
                "shared_energy": paths.shared_energy,
                "shared_energy_random": paths.shared_energy_random,
                "shared_functional": paths.shared_functional,
            }
        )
    if group in {"projection", "all"}:
        expected.update(
            {
                "principal_angle": paths.principal_angle,
                "overlapped_functional": paths.overlap_functional,
            }
        )
    missing = [str(path) for path in expected.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"Spectra stage ended with missing outputs: {missing}")
    manifest = {
        "version": "clean-1.0",
        "group": group,
        "six_spectra": list(SPECTRUM_NAMES),
        "main_spectra": [
            "Individual Energy",
            "Individual Functional",
            "Shared Energy",
            "Shared Functional",
        ],
        "projection_diagnostic_spectra": [
            "Principal Angle",
            "Overlapped Functional",
        ],
        "analysis_config": asdict(project.analysis),
        "svd_sanity": {
            "max_reconstruction_relative_error": maximum_reconstruction_error,
            "max_orthonormality_error": maximum_orthonormality_error,
        },
        "boundaries": [
            "Context is built only within identical matrix positions.",
            "Shared Functional writes original candidate sigma_i Z_i, never projected context operators.",
            "Overlapped Functional deliberately writes projected principal directions and is a diagnostic of that route.",
            "All bACC values are pooled fitted-data functional-retention audits, not held-out generalisation.",
        ],
        "outputs": {name: str(path) for name, path in expected.items()},
        "output_sha256": {name: sha256_file(path) for name, path in expected.items()},
    }
    dump_json(paths.root / "spectra_manifest.json", manifest)
    dump_json(
        paths.root / f"_SPECTRA_{group.upper()}_SUCCESS.json",
        {
            "status": "passed",
            "manifest_sha256": sha256_file(paths.root / "spectra_manifest.json"),
        },
    )
    return manifest
