"""Weight-update heat-map tables.

This module owns only the numerical heat-map statistic.  It does not compute a
spectrum, STI, or a figure.  The reported cell value is

    H = ||Delta W||_F^2 / ||W0||_F^2,

for each task, replicate, Transformer block, and analysed matrix role.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .config import EXPECTED_MODULE_SHAPES, ProjectConfig
from .io import (
    RunRecord,
    dump_json,
    ensure_dir,
    load_delta_weights,
    module_layer_role,
    sha256_file,
    write_csv,
)


EPS = 1e-12


@dataclass(frozen=True)
class HeatmapPaths:
    root: Path
    observations: Path
    summary: Path
    manifest: Path

    @classmethod
    def under(cls, output_dir: Path) -> "HeatmapPaths":
        root = ensure_dir(Path(output_dir) / "heatmap")
        return cls(
            root=root,
            observations=root / "weight_update_heatmap.csv",
            summary=root / "weight_update_heatmap_summary.csv",
            manifest=root / "heatmap_manifest.json",
        )


def _validate_base_weights(base_weights: Mapping[str, np.ndarray]) -> None:
    if set(base_weights) != set(EXPECTED_MODULE_SHAPES):
        missing = sorted(set(EXPECTED_MODULE_SHAPES) - set(base_weights))
        extra = sorted(set(base_weights) - set(EXPECTED_MODULE_SHAPES))
        raise ValueError(f"Invalid W0 matrix grid; missing={missing}, extra={extra}")
    for module, shape in EXPECTED_MODULE_SHAPES.items():
        value = np.asarray(base_weights[module])
        if value.shape != shape:
            raise ValueError(f"{module}: expected W0 shape {shape}, found {value.shape}")
        if not np.isfinite(value).all():
            raise FloatingPointError(f"{module}: non-finite W0 values")


def compute_heatmap_rows(
    runs: Sequence[RunRecord],
    base_weights: Mapping[str, np.ndarray],
) -> List[Dict[str, object]]:
    """Return one relative-update-energy observation per run and matrix."""

    _validate_base_weights(base_weights)
    rows: List[Dict[str, object]] = []
    for run in runs:
        updates = load_delta_weights(run.delta_path)
        if set(updates) != set(EXPECTED_MODULE_SHAPES):
            raise ValueError(f"{run.tag}: incomplete delta matrix grid")
        for module, expected_shape in EXPECTED_MODULE_SHAPES.items():
            delta = np.asarray(updates[module], dtype=np.float64)
            base = np.asarray(base_weights[module], dtype=np.float64)
            if delta.shape != expected_shape:
                raise ValueError(
                    f"{run.tag}/{module}: expected {expected_shape}, found {delta.shape}"
                )
            delta_norm = float(np.linalg.norm(delta, ord="fro"))
            base_norm = float(np.linalg.norm(base, ord="fro"))
            if base_norm <= EPS:
                raise RuntimeError(f"{module}: zero-norm W0 matrix")
            relative_energy = (delta_norm * delta_norm) / (base_norm * base_norm)
            block, role = module_layer_role(module)
            rows.append(
                {
                    "task": run.task,
                    "replicate": run.replicate,
                    "module": module,
                    "block": block + 1,
                    "matrix_type": role,
                    "delta_frobenius_norm": delta_norm,
                    "w0_frobenius_norm": base_norm,
                    "relative_update_energy": relative_energy,
                    "relative_update_norm": delta_norm / base_norm,
                    "log10_relative_update_energy": float(
                        np.log10(max(relative_energy, 1e-300))
                    ),
                }
            )
    expected = len(runs) * len(EXPECTED_MODULE_SHAPES)
    if len(rows) != expected:
        raise AssertionError(f"Expected {expected} heat-map rows, found {len(rows)}")
    return rows


def summarize_replicates(
    rows: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    """Aggregate replicate observations without averaging in logarithmic space."""

    grouped: Dict[Tuple[str, str], List[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["task"]), str(row["module"])), []).append(row)
    summary: List[Dict[str, object]] = []
    for (task, module), values in sorted(grouped.items()):
        energies = np.asarray(
            [float(row["relative_update_energy"]) for row in values], dtype=np.float64
        )
        block, role = module_layer_role(module)
        median = float(np.median(energies))
        summary.append(
            {
                "task": task,
                "module": module,
                "block": block + 1,
                "matrix_type": role,
                "n_replicates": len(values),
                "relative_update_energy_mean": float(np.mean(energies)),
                "relative_update_energy_median": median,
                "relative_update_energy_min": float(np.min(energies)),
                "relative_update_energy_max": float(np.max(energies)),
                "log10_median_relative_update_energy": float(
                    np.log10(max(median, 1e-300))
                ),
                "aggregation_note": "median is computed in linear energy space",
            }
        )
    return summary


def _load_base_weights(project: ProjectConfig) -> Tuple[Dict[str, np.ndarray], str]:
    from .model import build_labram_backbone, extract_tsv_weights, require_torch

    backbone, _, digest = build_labram_backbone(project)
    try:
        values = {
            module: tensor.numpy().astype(np.float64, copy=True)
            for module, tensor in extract_tsv_weights(backbone).items()
        }
    finally:
        del backbone
        gc.collect()
        torch, _, _ = require_torch()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return values, digest


def generate_heatmap(
    project: ProjectConfig,
    runs: Sequence[RunRecord],
    base_weights: Optional[Mapping[str, np.ndarray]] = None,
) -> Dict[str, object]:
    """Compute, audit, and save heat-map tables; never draw the figure here."""

    paths = HeatmapPaths.under(project.output_dir)
    base_digest = "injected-for-test"
    if base_weights is None:
        base_weights, base_digest = _load_base_weights(project)
        declared = {
            str(run.metadata.get("base_full_backbone_digest", "")) for run in runs
        } - {""}
        if len(declared) > 1 or (declared and base_digest not in declared):
            raise RuntimeError("Heat-map W0 differs from the W0 recorded by training")
    rows = compute_heatmap_rows(runs, base_weights)
    summary = summarize_replicates(rows)
    write_csv(paths.observations, rows)
    write_csv(paths.summary, summary)
    manifest = {
        "version": "clean-1.0",
        "statistic": "||Delta W||_F^2 / ||W0||_F^2",
        "base_full_backbone_digest": base_digest,
        "n_runs": len(runs),
        "n_observations": len(rows),
        "replicate_summary": "median and descriptive range in linear energy space",
        "outputs": {
            "observations": str(paths.observations),
            "summary": str(paths.summary),
        },
        "output_sha256": {
            "observations": sha256_file(paths.observations),
            "summary": sha256_file(paths.summary),
        },
    }
    dump_json(paths.manifest, manifest)
    dump_json(
        paths.root / "_HEATMAP_SUCCESS.json",
        {"status": "passed", "manifest_sha256": sha256_file(paths.manifest)},
    )
    return manifest
