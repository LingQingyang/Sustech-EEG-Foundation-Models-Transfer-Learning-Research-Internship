"""CSV-only figure generation for training QC, heat map, STI, and six spectra.

The plotting layer never performs SVD, model inference, projection, or random
sampling.  Deleting the figures and rerunning this module therefore cannot
change any numerical result table.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from .config import MODULE_ROLES, TASK_ORDER, ProjectConfig
from .io import dump_json, ensure_dir, read_csv, sha256_file


TASK_COLORS = {
    "Rest": "#4C78A8",
    "Motor": "#F58518",
    "P300": "#54A24B",
    "SSS": "#E45756",
    "TS": "#B279A2",
}


def _matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required only for the plot command") from exc
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "figure.dpi": 130,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt


def _save(fig, base: Path) -> List[Path]:
    base = Path(base)
    ensure_dir(base.parent)
    png = base.with_suffix(".png")
    pdf = base.with_suffix(".pdf")
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    return [png, pdf]


def _float(row: Mapping[str, object], key: str, default: float = float("nan")) -> float:
    value = row.get(key, "")
    try:
        return float(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _int(row: Mapping[str, object], key: str, default: int = 0) -> int:
    value = _float(row, key, float(default))
    return int(value) if math.isfinite(value) else default


def _group(
    rows: Sequence[Mapping[str, object]], key: str
) -> Dict[str, List[Mapping[str, object]]]:
    grouped: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return grouped


def plot_training_curves(project: ProjectConfig, figure_dir: Path) -> List[Path]:
    plt = _matplotlib()
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.7), sharex=True)
    axes = axes.ravel()
    found = 0
    for task_index, task in enumerate(TASK_ORDER):
        ax = axes[task_index]
        for replicate in (1, 2):
            path = project.root / "runs" / task / f"rep{replicate:02d}" / "epoch_metrics.csv"
            if not path.is_file():
                continue
            rows = sorted(read_csv(path), key=lambda row: _int(row, "epoch"))
            ax.plot(
                [_int(row, "epoch") for row in rows],
                [_float(row, "balanced_accuracy") for row in rows],
                marker="o",
                ms=2.5,
                lw=1.2,
                label=f"rep {replicate}",
            )
            found += 1
        ax.set_title(task)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Online train bACC")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.2)
        if ax.lines:
            ax.legend()
    axes[-1].axis("off")
    if not found:
        plt.close(fig)
        return []
    fig.suptitle("Full fine-tuning: 20-epoch training traces")
    fig.tight_layout()
    outputs = _save(fig, Path(figure_dir) / "training_curves")
    plt.close(fig)
    return outputs


def _heatmap_arrays(rows: Sequence[Mapping[str, object]], value_key: str):
    arrays = {}
    for task in TASK_ORDER:
        array = np.full((12, len(MODULE_ROLES)), np.nan, dtype=np.float64)
        for row in rows:
            if str(row.get("task")) != task:
                continue
            block = _int(row, "block") - 1
            role = str(row.get("matrix_type"))
            if 0 <= block < 12 and role in MODULE_ROLES:
                array[block, MODULE_ROLES.index(role)] = _float(row, value_key)
        arrays[task] = array
    return arrays


def _plot_heatmap_set(
    rows: Sequence[Mapping[str, object]],
    figure_dir: Path,
    value_key: str,
    label: str,
    stem: str,
) -> List[Path]:
    plt = _matplotlib()
    arrays = _heatmap_arrays(rows, value_key)
    finite = np.concatenate([array[np.isfinite(array)] for array in arrays.values()])
    if not len(finite):
        return []
    vmin, vmax = float(np.min(finite)), float(np.max(finite))
    if abs(vmax - vmin) < 1e-15:
        vmax = vmin + 1e-15
    fig, axes = plt.subplots(2, 3, figsize=(12.3, 7.4), sharex=True, sharey=True)
    axes = axes.ravel()
    image = None
    for index, task in enumerate(TASK_ORDER):
        ax = axes[index]
        image = ax.imshow(
            arrays[task], aspect="auto", origin="lower", vmin=vmin, vmax=vmax, cmap="viridis"
        )
        ax.set_title(task)
        ax.set_xticks(range(len(MODULE_ROLES)), MODULE_ROLES, rotation=35, ha="right")
        ax.set_yticks(range(12), range(1, 13))
        ax.set_xlabel("Matrix")
        ax.set_ylabel("Transformer block")
    axes[-1].axis("off")
    cbar = fig.colorbar(image, ax=axes.tolist(), shrink=0.82, pad=0.02)
    cbar.set_label(label)
    fig.suptitle("Weight-update heat map (median over two replicates)")
    fig.subplots_adjust(left=0.07, right=0.90, bottom=0.09, top=0.91, wspace=0.22, hspace=0.28)
    outputs = _save(fig, Path(figure_dir) / stem)
    plt.close(fig)
    return outputs


def plot_heatmaps(project: ProjectConfig, figure_dir: Path) -> List[Path]:
    path = project.output_dir / "heatmap" / "weight_update_heatmap_summary.csv"
    if not path.is_file():
        return []
    rows = read_csv(path)
    outputs = _plot_heatmap_set(
        rows,
        figure_dir,
        "log10_median_relative_update_energy",
        r"$\log_{10}(\mathrm{median}\; ||\Delta W||_F^2/||W_0||_F^2)$",
        "heatmap_log10",
    )
    outputs += _plot_heatmap_set(
        rows,
        figure_dir,
        "relative_update_energy_median",
        r"median $||\Delta W||_F^2/||W_0||_F^2$",
        "heatmap_linear",
    )
    return outputs


def plot_sti(project: ProjectConfig, figure_dir: Path) -> List[Path]:
    path = project.output_dir / "sti" / "sti_summary.csv"
    if not path.is_file():
        return []
    rows = read_csv(path)
    plt = _matplotlib()
    raw = np.full((12, len(MODULE_ROLES)), np.nan)
    ratio = np.full_like(raw, np.nan)
    for row in rows:
        block = _int(row, "block") - 1
        role = str(row["matrix_type"])
        if role not in MODULE_ROLES or not 0 <= block < 12:
            continue
        role_index = MODULE_ROLES.index(role)
        raw[block, role_index] = _float(row, "observed_median")
        ratio[block, role_index] = _float(row, "observed_median_over_random_median")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharey=True)
    for ax, values, title, cbar_label in (
        (axes[0], raw, "Observed STI", "median raw STI"),
        (axes[1], ratio, "Observed / matched-random", "median ratio"),
    ):
        image = ax.imshow(values, aspect="auto", origin="lower", cmap="magma")
        ax.set_title(title)
        ax.set_xticks(range(len(MODULE_ROLES)), MODULE_ROLES, rotation=35, ha="right")
        ax.set_yticks(range(12), range(1, 13))
        ax.set_xlabel("Matrix")
        ax.set_ylabel("Transformer block")
        fig.colorbar(image, ax=ax, shrink=0.86, label=cbar_label)
    fig.suptitle("Singular Task Interference (median over 32 replicate combinations)")
    fig.tight_layout()
    outputs = _save(fig, Path(figure_dir) / "sti_heatmaps")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    blocks = np.arange(1, 13)
    observed_median, observed_min, observed_max, random_median = [], [], [], []
    for block in blocks:
        current = [row for row in rows if _int(row, "block") == block]
        observed_median.append(np.nanmedian([_float(row, "observed_median") for row in current]))
        observed_min.append(np.nanmin([_float(row, "observed_min") for row in current]))
        observed_max.append(np.nanmax([_float(row, "observed_max") for row in current]))
        random_median.append(np.nanmedian([_float(row, "random_median") for row in current]))
    ax.plot(blocks, observed_median, marker="o", label="observed median across matrices")
    ax.fill_between(blocks, observed_min, observed_max, alpha=0.14, label="observed range")
    ax.plot(blocks, random_median, ls="--", marker="s", ms=3, label="matched-random median")
    ax.set_xlabel("Transformer block")
    ax.set_ylabel("Raw entrywise-L1 STI")
    ax.set_xticks(blocks)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    outputs += _save(fig, Path(figure_dir) / "sti_depth_profile")
    plt.close(fig)
    return outputs


def _task_grid(plt, ylabel: str, title: str):
    fig, axes = plt.subplots(2, 3, figsize=(12.3, 7.2))
    axes = axes.ravel()
    for index, task in enumerate(TASK_ORDER):
        axes[index].set_title(task)
        axes[index].set_xlabel("Retained fraction")
        axes[index].set_ylabel(ylabel)
        axes[index].grid(alpha=0.2)
    axes[-1].axis("off")
    fig.suptitle(title)
    return fig, axes


def plot_individual_spectra(project: ProjectConfig, figure_dir: Path) -> List[Path]:
    root = project.output_dir / "spectra"
    specs = (
        (
            root / "individual_energy_spectrum.csv",
            "individual_energy",
            "Individual Energy",
            "Retained update energy",
            "individual_energy_spectrum",
        ),
        (
            root / "individual_functional_spectrum.csv",
            "raw_bacc",
            "Individual Functional",
            "Pooled fitted-data bACC",
            "individual_functional_spectrum",
        ),
    )
    plt = _matplotlib()
    outputs: List[Path] = []
    for path, y_key, title, ylabel, stem in specs:
        if not path.is_file():
            continue
        rows = read_csv(path)
        fig, axes = _task_grid(plt, ylabel, title)
        for task_index, task in enumerate(TASK_ORDER):
            ax = axes[task_index]
            for replicate in (1, 2):
                current = sorted(
                    [
                        row
                        for row in rows
                        if str(row.get("task")) == task
                        and _int(row, "replicate") == replicate
                    ],
                    key=lambda row: _float(row, "q"),
                )
                if current:
                    ax.plot(
                        [_float(row, "q") for row in current],
                        [_float(row, y_key) for row in current],
                        marker="o",
                        ms=3,
                        lw=1.2,
                        label=f"rep {replicate}",
                    )
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(0, 1.02)
            if ax.lines:
                ax.legend()
        fig.tight_layout()
        outputs += _save(fig, Path(figure_dir) / stem)
        plt.close(fig)
    return outputs


def _comparison_curves(
    rows: Sequence[Mapping[str, object]], candidate: str, regime: str
) -> Dict[str, List[Mapping[str, object]]]:
    selected = [
        row
        for row in rows
        if str(row.get("candidate")) == candidate
        and str(row.get("context_regime")) == regime
    ]
    return _group(selected, "comparison_id")


def _plot_comparison_spectrum(
    rows: Sequence[Mapping[str, object]],
    figure_dir: Path,
    regime: str,
    x_key: str,
    y_key: str,
    title: str,
    ylabel: str,
    stem: str,
    random_rows: Optional[Sequence[Mapping[str, object]]] = None,
) -> List[Path]:
    plt = _matplotlib()
    fig, axes = _task_grid(plt, ylabel, f"{title} — {regime} context")
    random_grouped = _group(random_rows or [], "comparison_id")
    curves_found = 0
    for task_index, task in enumerate(TASK_ORDER):
        ax = axes[task_index]
        curves = _comparison_curves(rows, task, regime)
        for curve_id, curve in sorted(curves.items()):
            curve = sorted(curve, key=lambda row: _float(row, x_key))
            x = [_float(row, x_key) for row in curve]
            y = [_float(row, y_key) for row in curve]
            label = str(curve[0].get("context_tasks", curve_id))
            ax.plot(x, y, color=TASK_COLORS[task], alpha=0.30, lw=0.9, label=label)
            curves_found += 1
            random = sorted(
                random_grouped.get(curve_id, []), key=lambda row: _float(row, x_key)
            )
            if random:
                ax.plot(
                    [_float(row, x_key) for row in random],
                    [_float(row, "random_upper") for row in random],
                    color="#666666",
                    alpha=0.22,
                    lw=0.7,
                    ls="--",
                )
        if x_key in {"component_fraction", "principal_fraction"}:
            ax.set_xlim(-0.02, 1.02)
        if y_key in {"shared_energy_spectrum", "raw_bacc", "functional_retention"}:
            ax.set_ylim(0, 1.05)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            unique = dict(zip(labels, handles))
            ax.legend(unique.values(), unique.keys(), loc="best", ncol=2)
    if not curves_found:
        plt.close(fig)
        return []
    fig.tight_layout()
    outputs = _save(fig, Path(figure_dir) / stem)
    plt.close(fig)
    return outputs


def plot_shared_spectra(project: ProjectConfig, figure_dir: Path) -> List[Path]:
    root = project.output_dir / "spectra"
    energy_path = root / "shared_energy_spectrum.csv"
    functional_path = root / "shared_functional_spectrum.csv"
    random_path = root / "shared_energy_random_reference.csv"
    energy = read_csv(energy_path) if energy_path.is_file() else []
    functional = read_csv(functional_path) if functional_path.is_file() else []
    random = read_csv(random_path) if random_path.is_file() else []
    outputs: List[Path] = []
    for regime in ("single", "full"):
        if energy:
            outputs += _plot_comparison_spectrum(
                energy,
                figure_dir,
                regime,
                "component_fraction",
                "shared_energy_spectrum",
                "Shared Energy",
                "Cumulative shared energy (absolute)",
                f"shared_energy_{regime}",
                random_rows=random,
            )
        if functional:
            outputs += _plot_comparison_spectrum(
                functional,
                figure_dir,
                regime,
                "component_fraction",
                "raw_bacc",
                "Shared Functional",
                "Pooled fitted-data bACC",
                f"shared_functional_{regime}",
            )
    return outputs


def plot_principal_angle(project: ProjectConfig, figure_dir: Path) -> List[Path]:
    path = project.output_dir / "spectra" / "principal_angle_spectrum.csv"
    if not path.is_file():
        return []
    rows = read_csv(path)
    plt = _matplotlib()
    outputs: List[Path] = []
    for regime in ("single", "full", "replicate"):
        fig, axes = _task_grid(
            plt, r"$\cos(\theta)$", f"Principal Angle — {regime} context"
        )
        found = 0
        for task_index, task in enumerate(TASK_ORDER):
            ax = axes[task_index]
            for curve_id, curve in sorted(_comparison_curves(rows, task, regime).items()):
                curve = sorted(curve, key=lambda row: _int(row, "principal_dimension"))
                length = max(1, len(curve))
                x = [(_int(row, "principal_dimension") / length) for row in curve]
                ax.plot(
                    x,
                    [_float(row, "cos_theta") for row in curve],
                    color=TASK_COLORS[task],
                    alpha=0.30,
                    lw=0.9,
                )
                random = [_float(row, "matched_random_upper") for row in curve]
                if any(math.isfinite(value) for value in random):
                    ax.plot(x, random, color="#555555", alpha=0.20, lw=0.7, ls="--")
                found += 1
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(0, 1.02)
            ax.set_xlabel("Normalized principal dimension")
        if found:
            fig.tight_layout()
            outputs += _save(fig, Path(figure_dir) / f"principal_angle_{regime}")
        plt.close(fig)
    return outputs


def plot_overlap_functional(project: ProjectConfig, figure_dir: Path) -> List[Path]:
    path = project.output_dir / "spectra" / "overlapped_functional_spectrum.csv"
    if not path.is_file():
        return []
    rows = read_csv(path)
    outputs: List[Path] = []
    for regime in ("single", "full", "replicate"):
        outputs += _plot_comparison_spectrum(
            rows,
            figure_dir,
            regime,
            "principal_fraction",
            "functional_retention",
            "Overlapped Functional",
            "Functional retention / full bACC",
            f"overlapped_functional_{regime}",
        )
    return outputs


def plot_all(project: ProjectConfig) -> Dict[str, object]:
    """Regenerate every available figure from saved CSV tables only."""

    figure_dir = ensure_dir(project.output_dir / "figures")
    outputs: List[Path] = []
    outputs += plot_training_curves(project, figure_dir)
    outputs += plot_heatmaps(project, figure_dir)
    outputs += plot_sti(project, figure_dir)
    outputs += plot_individual_spectra(project, figure_dir)
    outputs += plot_shared_spectra(project, figure_dir)
    outputs += plot_principal_angle(project, figure_dir)
    outputs += plot_overlap_functional(project, figure_dir)
    if not outputs:
        raise FileNotFoundError("No numerical CSV outputs were available to plot")
    manifest = {
        "version": "clean-1.0",
        "contract": "CSV-only plotting; no SVD, inference, projection, or random sampling",
        "figure_count": len(outputs),
        "figures": [str(path) for path in outputs],
        "sha256": {str(path.name): sha256_file(path) for path in outputs},
    }
    manifest_path = figure_dir / "figure_manifest.json"
    dump_json(manifest_path, manifest)
    dump_json(
        figure_dir / "_PLOTTING_SUCCESS.json",
        {"status": "passed", "manifest_sha256": sha256_file(manifest_path)},
    )
    return manifest
