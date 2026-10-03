"""Paper-aligned Singular Task Interference (STI), separate from spectra.

For each of the 72 matrix positions and each of the 2^5 replicate choices, the
five tasks contribute ``k=floor(r/T)`` singular triplets, with ``T=5``.  The
raw statistic follows the Task Singular Vectors construction used by the
legacy analysis:

    STI = || (U^T U - I) diag(s) (V^T V - I) ||_1,

where ``||.||_1`` is the entrywise L1 norm.  No plotting lives here.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .config import EXPECTED_MODULE_SHAPES, TASK_ORDER, ProjectConfig
from .geometry import SVDRecord, haar_basis, quantile_summary
from .io import (
    ProgressStore,
    RunRecord,
    dump_json,
    ensure_dir,
    load_decompositions,
    module_layer_role,
    read_csv,
    sha256_file,
    stable_seed,
    write_csv,
)


DecompositionGrid = Mapping[Tuple[str, int, str], SVDRecord]


@dataclass(frozen=True)
class STIPaths:
    root: Path
    observed: Path
    random_reference: Path
    summary: Path
    random_draws: Path
    progress: Path
    manifest: Path

    @classmethod
    def under(cls, output_dir: Path) -> "STIPaths":
        root = ensure_dir(Path(output_dir) / "sti")
        return cls(
            root=root,
            observed=root / "sti_observed.csv",
            random_reference=root / "sti_random_reference.csv",
            summary=root / "sti_summary.csv",
            random_draws=root / "sti_random_draws.csv",
            progress=ensure_dir(root / "progress"),
            manifest=root / "sti_manifest.json",
        )


def sti_from_parts(
    left_bases: Sequence[np.ndarray],
    singular_values: Sequence[np.ndarray],
    right_bases: Sequence[np.ndarray],
) -> float:
    """Compute raw entrywise-L1 STI from task-wise truncated SVD factors."""

    if not (len(left_bases) == len(singular_values) == len(right_bases)):
        raise ValueError("STI U/s/V task counts differ")
    if not left_bases:
        raise ValueError("STI needs at least one task")
    ranks = []
    for U, s, V in zip(left_bases, singular_values, right_bases):
        U = np.asarray(U, dtype=np.float64)
        s = np.asarray(s, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        if U.ndim != 2 or V.ndim != 2 or s.ndim != 1:
            raise ValueError("STI factors must be matrices, vector, matrices")
        if U.shape[1] != len(s) or V.shape[1] != len(s):
            raise ValueError("STI factor ranks differ")
        if not (np.isfinite(U).all() and np.isfinite(s).all() and np.isfinite(V).all()):
            raise FloatingPointError("STI factors contain non-finite values")
        ranks.append(len(s))
    if len(set(ranks)) != 1:
        raise ValueError(f"STI requires equal per-task truncation ranks, found {ranks}")
    U_all = np.concatenate(left_bases, axis=1).astype(np.float64, copy=False)
    V_all = np.concatenate(right_bases, axis=1).astype(np.float64, copy=False)
    s_all = np.concatenate(singular_values).astype(np.float64, copy=False)
    left_cross = U_all.T @ U_all - np.eye(U_all.shape[1], dtype=np.float64)
    right_cross = V_all.T @ V_all - np.eye(V_all.shape[1], dtype=np.float64)
    interaction = (left_cross * s_all[None, :]) @ right_cross
    value = float(np.sum(np.abs(interaction), dtype=np.float64))
    if not np.isfinite(value) or value < 0:
        raise FloatingPointError(f"Invalid STI value: {value}")
    return value


def _runs_by_task(runs: Sequence[RunRecord]) -> Dict[str, List[RunRecord]]:
    grouped: Dict[str, List[RunRecord]] = defaultdict(list)
    for run in runs:
        grouped[run.task].append(run)
    return {
        task: sorted(grouped[task], key=lambda run: run.replicate)
        for task in TASK_ORDER
    }


def replicate_combinations(runs: Sequence[RunRecord]) -> List[Tuple[RunRecord, ...]]:
    grouped = _runs_by_task(runs)
    missing = [task for task in TASK_ORDER if len(grouped.get(task, [])) != 2]
    if missing:
        raise ValueError(f"STI requires two runs for every task; invalid tasks={missing}")
    combinations = list(itertools.product(*(grouped[task] for task in TASK_ORDER)))
    if len(combinations) != 32:
        raise AssertionError(f"Expected 32 replicate combinations, found {len(combinations)}")
    return combinations


def _paper_rank(records: Sequence[SVDRecord]) -> int:
    k = min(record.rank for record in records) // len(TASK_ORDER)
    if k < 1:
        raise RuntimeError("Paper STI floor(r/T) produced k < 1")
    return int(k)


def observed_sti_rows(
    runs: Sequence[RunRecord], decompositions: DecompositionGrid
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for combination_index, combination in enumerate(replicate_combinations(runs), start=1):
        combination_id = "+".join(run.tag for run in combination)
        for module in EXPECTED_MODULE_SHAPES:
            records = [
                decompositions[(run.task, run.replicate, module)] for run in combination
            ]
            k = _paper_rank(records)
            value = sti_from_parts(
                [record.U[:, :k] for record in records],
                [record.s[:k] for record in records],
                [record.V[:, :k] for record in records],
            )
            block, role = module_layer_role(module)
            rows.append(
                {
                    "replicate_combination_index": combination_index,
                    "replicate_combination": combination_id,
                    "module": module,
                    "block": block + 1,
                    "matrix_type": role,
                    "n_tasks": len(TASK_ORDER),
                    "k_paper": k,
                    "sti": value,
                    "norm": "entrywise_L1",
                }
            )
    expected = 32 * len(EXPECTED_MODULE_SHAPES)
    if len(rows) != expected:
        raise AssertionError(f"Expected {expected} observed STI rows, found {len(rows)}")
    return rows


def random_sti_for_module(
    module: str,
    combinations: Sequence[Tuple[RunRecord, ...]],
    decompositions: DecompositionGrid,
    random_n: int,
    random_seed: int,
    save_draws: bool = False,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """Matched null: retain singular values, randomize only U/V orientations."""

    if module not in EXPECTED_MODULE_SHAPES:
        raise KeyError(module)
    if random_n < 1:
        raise ValueError("random_n must be positive")
    record_sets = [
        [decompositions[(run.task, run.replicate, module)] for run in combination]
        for combination in combinations
    ]
    rng = np.random.default_rng(stable_seed(random_seed, "sti-null", module))
    values = np.empty(random_n, dtype=np.float64)
    draw_rows: List[Dict[str, object]] = []
    ranks = []
    for draw in range(random_n):
        combination_index = int(rng.integers(0, len(record_sets)))
        records = record_sets[combination_index]
        k = _paper_rank(records)
        ranks.append(k)
        left = [haar_basis(record.d_out, k, rng) for record in records]
        right = [haar_basis(record.d_in, k, rng) for record in records]
        singulars = [record.s[:k] for record in records]
        values[draw] = sti_from_parts(left, singulars, right)
        if save_draws:
            draw_rows.append(
                {
                    "module": module,
                    "random_draw": draw + 1,
                    "sampled_combination_index": combination_index + 1,
                    "k_paper": k,
                    "sti_random": float(values[draw]),
                }
            )
    if len(set(ranks)) != 1:
        raise RuntimeError(f"{module}: matched-null paper rank changed across draws")
    block, role = module_layer_role(module)
    summary = quantile_summary(values)
    row = {
        "module": module,
        "block": block + 1,
        "matrix_type": role,
        "k_paper": ranks[0],
        "random_n": random_n,
        "random_combo_sampling": "uniform_over_32_empirical_replicate_combinations",
        "orientation_null": "independent Haar U/V; empirical singular values preserved",
        "random_min": summary["min"],
        "random_q01": summary["q01"],
        "random_median": summary["median"],
        "random_q95": summary["q95"],
        "random_q99": summary["q99"],
        "random_max": summary["max"],
    }
    return row, draw_rows


def summarize_sti(
    observed: Sequence[Mapping[str, object]],
    random_reference: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    observed_by_module: Dict[str, List[float]] = defaultdict(list)
    for row in observed:
        observed_by_module[str(row["module"])].append(float(row["sti"]))
    random_by_module = {str(row["module"]): row for row in random_reference}
    rows: List[Dict[str, object]] = []
    for module in EXPECTED_MODULE_SHAPES:
        values = np.asarray(observed_by_module[module], dtype=np.float64)
        if len(values) != 32:
            raise RuntimeError(f"{module}: expected 32 observed STI values")
        random = random_by_module[module]
        random_median = float(random["random_median"])
        observed_median = float(np.median(values))
        block, role = module_layer_role(module)
        rows.append(
            {
                "module": module,
                "block": block + 1,
                "matrix_type": role,
                "k_paper": int(float(random["k_paper"])),
                "observed_n": len(values),
                "observed_min": float(np.min(values)),
                "observed_median": observed_median,
                "observed_max": float(np.max(values)),
                "random_n": int(float(random["random_n"])),
                "random_median": random_median,
                "random_q99": float(random["random_q99"]),
                "observed_median_over_random_median": (
                    observed_median / random_median if random_median > 0 else float("nan")
                ),
                "observed_median_above_random_q99": int(
                    observed_median > float(random["random_q99"])
                ),
            }
        )
    return rows


def _replace_module_rows(
    rows: Sequence[Mapping[str, object]],
    module: str,
    replacements: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    return [dict(row) for row in rows if str(row.get("module")) != module] + [
        dict(row) for row in replacements
    ]


def generate_sti(
    project: ProjectConfig,
    runs: Sequence[RunRecord],
    resume: bool = True,
    decompositions: Optional[DecompositionGrid] = None,
) -> Dict[str, object]:
    """Generate observed STI and its matched random reference."""

    paths = STIPaths.under(project.output_dir)
    if decompositions is None:
        decompositions = load_decompositions(runs)
    combinations = replicate_combinations(runs)
    delta_sha256 = {run.tag: sha256_file(run.delta_path) for run in runs}
    observed_progress = ProgressStore(
        paths.progress / "observed.json",
        {"stage": "sti_observed", "delta_sha256": delta_sha256},
        resume,
    )
    if resume and observed_progress.has("all") and paths.observed.is_file():
        observed = [dict(row) for row in read_csv(paths.observed)]
        if len(observed) != 32 * len(EXPECTED_MODULE_SHAPES):
            raise RuntimeError("Cached observed STI table is incomplete")
    else:
        observed = observed_sti_rows(runs, decompositions)
        write_csv(paths.observed, observed)
        observed_progress.mark("all")

    signature = {
        "stage": "sti_random_reference",
        "analysis": asdict(project.analysis),
        "delta_sha256": delta_sha256,
    }
    progress = ProgressStore(paths.progress / "random.json", signature, resume)
    random_rows: List[Dict[str, object]] = (
        [dict(row) for row in read_csv(paths.random_reference)]
        if resume and paths.random_reference.is_file()
        else []
    )
    draw_rows: List[Dict[str, object]] = (
        [dict(row) for row in read_csv(paths.random_draws)]
        if resume and project.analysis.save_random_draws and paths.random_draws.is_file()
        else []
    )
    for module in EXPECTED_MODULE_SHAPES:
        if progress.has(module):
            continue
        row, draws = random_sti_for_module(
            module,
            combinations,
            decompositions,
            project.analysis.sti_random_n,
            project.analysis.random_seed,
            project.analysis.save_random_draws,
        )
        random_rows = _replace_module_rows(random_rows, module, [row])
        write_csv(paths.random_reference, random_rows)
        if project.analysis.save_random_draws:
            draw_rows = _replace_module_rows(draw_rows, module, draws)
            write_csv(paths.random_draws, draw_rows)
        progress.mark(module)

    if {str(row["module"]) for row in random_rows} != set(EXPECTED_MODULE_SHAPES):
        raise RuntimeError("STI random stage ended with an incomplete 72-matrix grid")
    summary = summarize_sti(observed, random_rows)
    write_csv(paths.summary, summary)
    outputs = {
        "observed": paths.observed,
        "random_reference": paths.random_reference,
        "summary": paths.summary,
    }
    if project.analysis.save_random_draws:
        outputs["random_draws"] = paths.random_draws
    manifest = {
        "version": "clean-1.0",
        "formula": "||(U^T U-I) diag(s) (V^T V-I)||_entrywise_L1",
        "task_count": len(TASK_ORDER),
        "replicate_combinations": len(combinations),
        "paper_rank_rule": "k=floor(min full matrix rank / T)",
        "observed_rows": len(observed),
        "random_repetitions_per_matrix": project.analysis.sti_random_n,
        "random_aggregation": (
            "one uniformly sampled empirical replicate combination per draw and matrix; "
            "empirical singular values retained; U/V orientations Haar-randomized"
        ),
        "outputs": {name: str(path) for name, path in outputs.items()},
        "output_sha256": {name: sha256_file(path) for name, path in outputs.items()},
    }
    dump_json(paths.manifest, manifest)
    dump_json(
        paths.root / "_STI_SUCCESS.json",
        {"status": "passed", "manifest_sha256": sha256_file(paths.manifest)},
    )
    return manifest
