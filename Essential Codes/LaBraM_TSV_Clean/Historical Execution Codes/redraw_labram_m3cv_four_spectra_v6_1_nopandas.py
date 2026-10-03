#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LaBraM x M3CV four-spectra post-processing/redraw, v6.1-nopandas.

Pure post-processing. No training, no inference, no SVD recomputation.
Dependencies: Python stdlib + numpy + matplotlib only.

Outputs PNG only plus compact CSV/JSON post-processed tables.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

VERSION = "v6.1-nopandas"
TASK_ORDER = ["Rest", "Motor", "P300", "SSS", "TS"]
MATRIX_ORDER = ["Q", "K", "V", "O", "fc1", "fc2"]
TOTAL_WHOLE_MODEL_COMPONENTS = 14400


def parse_args():
    p = argparse.ArgumentParser(description="Post-process/redraw four spectra without pandas/scipy.")
    p.add_argument("--analysis-dir", required=True)
    p.add_argument("--outdir", default=None)
    p.add_argument("--shared-q", type=float, default=0.95)
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_csv(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str] | None = None):
    rows = list(rows)
    ensure_dir(path.parent)
    if fieldnames is None:
        fields = []
        seen = set()
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    fields.append(k)
        fieldnames = fields
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def fnum(x, default=float("nan")) -> float:
    try:
        if x is None or str(x).strip() == "":
            return default
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def inum(x, default=0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def bval(x) -> bool:
    return str(x).strip().lower() in {"1", "true", "t", "yes", "y"}


def group_rows(rows: Iterable[dict], keys: Sequence[str]) -> Dict[Tuple[str, ...], List[dict]]:
    out = defaultdict(list)
    for r in rows:
        out[tuple(str(r.get(k, "")) for k in keys)].append(r)
    return dict(out)


def ambient_dim_for_matrix_type(matrix_type: str) -> int:
    if matrix_type in {"Q", "K", "V", "O"}:
        return 200 * 200
    if matrix_type in {"fc1", "fc2"}:
        return 800 * 200
    raise KeyError(matrix_type)


def projection_q_approx(p: float, k: int, d: int) -> float:
    """Normal approximation to Beta(k/2,(d-k)/2) projection-energy null.

    This deliberately avoids scipy. In our large ambient spaces the approximation
    is adequate for the lightweight q95 cutoff used only in post-processing.
    """
    if k <= 0:
        return 0.0
    if k >= d:
        return 1.0
    mean = k / d
    var = 2.0 * k * (d - k) / (d * d * (d + 2.0))
    z = NormalDist().inv_cdf(p)
    return float(min(1.0, max(0.0, mean + z * math.sqrt(max(var, 0.0)))))


def dedupe_xy(xs: Sequence[float], ys: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    best = {}
    for x, y in zip(xs, ys):
        if not (math.isfinite(float(x)) and math.isfinite(float(y))):
            continue
        x = float(x); y = float(y)
        if x not in best or y > best[x]:
            best[x] = y
    if not best:
        return np.array([0.0]), np.array([np.nan])
    x = np.array(sorted(best), dtype=float)
    y = np.array([best[v] for v in x], dtype=float)
    return x, y


def interp_curve(x: Sequence[float], y: Sequence[float], xgrid: Sequence[float]) -> np.ndarray:
    x, y = dedupe_xy(x, y)
    xg = np.asarray(xgrid, dtype=float)
    if len(x) == 0 or np.all(~np.isfinite(y)):
        return np.full_like(xg, np.nan, dtype=float)
    return np.interp(xg, x, y, left=y[0], right=y[-1])


def mean_arrays(arrays: Sequence[np.ndarray]) -> np.ndarray:
    if not arrays:
        return np.array([], dtype=float)
    return np.nanmean(np.vstack(arrays), axis=0)


def load_main_tables(analysis_dir: Path) -> dict:
    names = [
        "individual_spectra.csv",
        "individual_summary.csv",
        "shared_energy_spectra.csv",
        "shared_functional_spectra.csv",
        "shared_summary.csv",
        "weight_update_heatmap.csv",
    ]
    return {name[:-4]: read_csv(analysis_dir / name) for name in names}


def component_files_by_id(components_dir: Path) -> Dict[str, Path]:
    out = {}
    for fp in sorted(components_dir.glob("*.csv")):
        rows = read_csv(fp)
        if not rows:
            continue
        cid = str(rows[0].get("comparison_id", ""))
        if cid:
            out[cid] = fp
    return out


def make_heatmap_compare(rows: List[dict], fig_dir: Path, data_dir: Path, dpi: int):
    # replicate mean per task/block/matrix
    bucket = defaultdict(lambda: [0.0, 0.0, 0])
    for r in rows:
        key = (str(r["task"]), inum(r["block"]), str(r["matrix_type"]))
        lin = fnum(r["relative_update_energy"])
        lg = fnum(r["log10_relative_update_energy"])
        if math.isfinite(lin) and math.isfinite(lg):
            bucket[key][0] += lin
            bucket[key][1] += lg
            bucket[key][2] += 1

    agg_rows = []
    for (task, block, mt), (slin, slog, n) in sorted(bucket.items()):
        agg_rows.append({
            "task": task,
            "block": block,
            "matrix_type": mt,
            "relative_update_energy_mean": slin / n,
            "log10_relative_update_energy_mean": slog / n,
            "n_replicates": n,
        })
    write_csv(data_dir / "weight_update_heatmap_replicate_mean.csv", agg_rows)

    lin_vals = [100.0 * fnum(r["relative_update_energy_mean"]) for r in agg_rows]
    log_vals = [fnum(r["log10_relative_update_energy_mean"]) for r in agg_rows]
    lin_min, lin_max = min(lin_vals), max(lin_vals)
    log_min, log_max = min(log_vals), max(log_vals)

    lookup = {(r["task"], inum(r["block"]), r["matrix_type"]): r for r in agg_rows}
    fig, axes = plt.subplots(
        2, len(TASK_ORDER), figsize=(14.2, 8.4), sharey=True, constrained_layout=True
    )
    fig.suptitle(
        "Weight-update heat map: log scale vs linear scale (replicate mean; common colour scales)",
        fontsize=17,
    )
    mlog = mlin = None
    for col, task in enumerate(TASK_ORDER):
        arr_log = np.full((12, len(MATRIX_ORDER)), np.nan)
        arr_lin = np.full((12, len(MATRIX_ORDER)), np.nan)
        for block in range(1, 13):
            for j, mt in enumerate(MATRIX_ORDER):
                r = lookup.get((task, block, mt))
                if r is None:
                    continue
                arr_log[block - 1, j] = fnum(r["log10_relative_update_energy_mean"])
                arr_lin[block - 1, j] = 100.0 * fnum(r["relative_update_energy_mean"])

        ax = axes[0, col]
        mlog = ax.imshow(arr_log, origin="lower", aspect="auto", vmin=log_min, vmax=log_max)
        ax.set_title(task, fontsize=14)
        ax.set_xticks(range(len(MATRIX_ORDER)))
        ax.set_xticklabels(MATRIX_ORDER, rotation=35, ha="right")
        ax.set_yticks(range(12)); ax.set_yticklabels(range(1, 13))
        if col == 0:
            ax.set_ylabel("Transformer block\n(log scale)", fontsize=12)

        ax = axes[1, col]
        mlin = ax.imshow(arr_lin, origin="lower", aspect="auto", vmin=lin_min, vmax=lin_max)
        ax.set_title(task, fontsize=14)
        ax.set_xticks(range(len(MATRIX_ORDER)))
        ax.set_xticklabels(MATRIX_ORDER, rotation=35, ha="right")
        ax.set_yticks(range(12)); ax.set_yticklabels(range(1, 13))
        ax.set_xlabel("Matrix type", fontsize=11)
        if col == 0:
            ax.set_ylabel("Transformer block\n(linear scale)", fontsize=12)

    c1 = fig.colorbar(mlog, ax=axes[0, :], fraction=0.022, pad=0.02)
    c1.set_label(r"$\log_{10}(\|\Delta W\|_F^2 / \|W_0\|_F^2)$", fontsize=11)
    c2 = fig.colorbar(mlin, ax=axes[1, :], fraction=0.022, pad=0.02)
    c2.set_label("Relative update energy (%)", fontsize=11)

    out = fig_dir / "02_weight_update_heatmap_log_vs_linear.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def make_individual_avg(rows: List[dict], fig_dir: Path, data_dir: Path, dpi: int):
    by_task_rep = group_rows(rows, ["task", "replicate"])
    energy_out, func_out = [], []
    xgrid = np.linspace(0, TOTAL_WHOLE_MODEL_COMPONENTS, 500)

    task_curves = {}
    for task in TASK_ORDER:
        e_arrays, f_arrays = [], []
        for rep in (1, 2):
            sub = by_task_rep.get((task, str(rep)), [])
            ex, ey, fx, fy = [], [], [], []
            for r in sub:
                cc = fnum(r.get("component_count"))
                if bval(r.get("is_energy_sample")):
                    v = fnum(r.get("individual_energy"))
                    if math.isfinite(cc) and math.isfinite(v):
                        ex.append(cc); ey.append(v)
                if bval(r.get("is_functional_sample")):
                    v = fnum(r.get("raw_bacc"))
                    if math.isfinite(cc) and math.isfinite(v):
                        fx.append(cc); fy.append(v)
            if ex:
                e_arrays.append(interp_curve(ex, ey, xgrid))
            if fx:
                f_arrays.append(interp_curve(fx, fy, xgrid))
        eavg = mean_arrays(e_arrays)
        favg = mean_arrays(f_arrays)
        task_curves[task] = (eavg, favg)
        for x, y in zip(xgrid, eavg):
            energy_out.append({"task": task, "component_count": int(round(x)), "individual_energy_avg": float(y)})
        for x, y in zip(xgrid, favg):
            func_out.append({"task": task, "component_count": int(round(x)), "raw_bacc_avg": float(y)})

    write_csv(data_dir / "individual_energy_average_curve.csv", energy_out)
    write_csv(data_dir / "individual_functional_average_curve.csv", func_out)

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for task in TASK_ORDER:
        eavg, favg = task_curves[task]
        axes[0].plot(xgrid, eavg, linewidth=2.2, label=task)
        axes[1].plot(xgrid, favg, linewidth=2.2, label=task)
    axes[0].axhline(0.99, linestyle="--", linewidth=1.2, color="grey")
    axes[0].set_title("Individual Energy (replicate mean)", fontsize=17)
    axes[0].set_ylabel("Retained update-energy fraction", fontsize=13)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(ncol=5, fontsize=11, loc="lower right")
    axes[1].set_title("Individual Functional (replicate mean)", fontsize=17)
    axes[1].set_ylabel("Raw balanced accuracy", fontsize=13)
    axes[1].set_xlabel("Retained rank-one components (whole model)", fontsize=13)
    axes[1].grid(True, alpha=0.25)
    axes[1].set_ylim(0.3, 1.02)
    fig.tight_layout()
    out = fig_dir / "04_individual_spectra_replicate_mean.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def build_shared_meta(shared_energy_rows: List[dict]) -> dict:
    out = {}
    for r in shared_energy_rows:
        cid = str(r.get("comparison_id", ""))
        if not cid or cid in out:
            continue
        bymat_raw = r.get("context_dimension_by_matrix", "")
        try:
            bymat = json.loads(bymat_raw) if bymat_raw else {}
        except Exception:
            bymat = {}
        out[cid] = {
            "candidate": str(r.get("candidate", "")),
            "candidate_replicate": inum(r.get("candidate_replicate")),
            "context_regime": str(r.get("context_regime", "")),
            "context_tasks": str(r.get("context_tasks", "")),
            "context_replicates": str(r.get("context_replicates", "")),
            "K_star": inum(r.get("K_star")),
            "denominator": fnum(r.get("functional_energy_denominator"), 1.0),
            "Gamma": fnum(r.get("Gamma"), 0.0),
            "context_dimension_by_matrix": bymat,
        }
    return out


def compute_shared_threshold_tables(
    shared_energy_rows: List[dict],
    shared_func_rows: List[dict],
    component_files: Dict[str, Path],
    data_dir: Path,
    q: float,
):
    meta = build_shared_meta(shared_energy_rows)
    func_by_id = group_rows(shared_func_rows, ["comparison_id"])
    summary_rows, energy_curve_rows, func_curve_rows, all_component_rows = [], [], [], []

    for cid, fp in sorted(component_files.items()):
        if cid not in meta:
            continue
        info = meta[cid]
        crows = read_csv(fp)
        crows.sort(key=lambda r: inum(r.get("shared_rank"), 10**9))
        denom = max(info["denominator"], 1e-30)

        prefix = 0
        cum_obs = 0.0
        cum_null = 0.0
        processed = []
        still_prefix = True
        for r in crows:
            mt = str(r.get("matrix_type", ""))
            module = str(r.get("module", ""))
            sigma2 = fnum(r.get("sigma2"), 0.0)
            obs = fnum(r.get("shared_energy"), 0.0)
            d = ambient_dim_for_matrix_type(mt)
            k = inum(info["context_dimension_by_matrix"].get(module, 0))
            proj_q = projection_q_approx(q, k, d)
            null_q = sigma2 * proj_q
            is_shared = obs > null_q
            if still_prefix and is_shared:
                prefix += 1
            else:
                still_prefix = False
            cum_obs += obs / denom
            cum_null += null_q / denom
            rr = dict(r)
            rr.update({
                "ambient_dimension": d,
                "context_dimension_module": k,
                "projection_random_q": proj_q,
                "shared_energy_random_q": null_q,
                "shared_margin_vs_random_q": obs - null_q,
                "is_shared_by_threshold": int(is_shared),
                "is_shared_prefix": int(inum(r.get("shared_rank")) <= prefix),
                "q_threshold": q,
                "cum_shared_energy_spectrum": cum_obs,
                "cum_random_q_spectrum": cum_null,
                "shared_component_fraction": inum(r.get("shared_rank")) / max(info["K_star"], 1),
            })
            processed.append(rr)
        # Correct prefix flag after final prefix is known.
        for rr in processed:
            rr["is_shared_prefix"] = int(1 <= inum(rr.get("shared_rank")) <= prefix)
            all_component_rows.append(rr)

        # Shared Energy curve: explicit origin then prefix only.
        energy_curve_rows.append({
            "comparison_id": cid,
            "candidate": info["candidate"],
            "candidate_replicate": info["candidate_replicate"],
            "context_regime": info["context_regime"],
            "context_tasks": info["context_tasks"],
            "x_fraction": 0.0,
            "x_component_count": 0,
            "shared_energy_spectrum": 0.0,
            "random_q_spectrum": 0.0,
        })
        for rr in processed[:prefix]:
            energy_curve_rows.append({
                "comparison_id": cid,
                "candidate": info["candidate"],
                "candidate_replicate": info["candidate_replicate"],
                "context_regime": info["context_regime"],
                "context_tasks": info["context_tasks"],
                "x_fraction": fnum(rr["shared_component_fraction"]),
                "x_component_count": inum(rr["shared_rank"]),
                "shared_energy_spectrum": fnum(rr["cum_shared_energy_spectrum"]),
                "random_q_spectrum": fnum(rr["cum_random_q_spectrum"]),
            })

        # Existing functional inference points only, truncated at prefix.
        frows = func_by_id.get((cid,), [])
        frows = [r for r in frows if math.isfinite(fnum(r.get("raw_bacc")))]
        frows.sort(key=lambda r: inum(r.get("component_count")))
        kept_f = [r for r in frows if inum(r.get("component_count")) <= prefix]
        if not kept_f and frows:
            kept_f = [frows[0]]
        for r in kept_f:
            func_curve_rows.append({
                "comparison_id": cid,
                "candidate": info["candidate"],
                "candidate_replicate": info["candidate_replicate"],
                "context_regime": info["context_regime"],
                "context_tasks": info["context_tasks"],
                "x_fraction": inum(r.get("component_count")) / max(info["K_star"], 1),
                "x_component_count": inum(r.get("component_count")),
                "raw_bacc": fnum(r.get("raw_bacc")),
                "full_bacc": fnum(r.get("full_bacc")),
                "chance_bacc": fnum(r.get("chance_bacc")),
            })

        if prefix > 0:
            ep = processed[prefix - 1]
            eobs = fnum(ep["cum_shared_energy_spectrum"], 0.0)
            enull = fnum(ep["cum_random_q_spectrum"], 0.0)
        else:
            eobs = enull = 0.0
        if kept_f:
            fend = kept_f[-1]
            fm = inum(fend.get("component_count"))
            fb = fnum(fend.get("raw_bacc"))
        else:
            fm, fb = 0, float("nan")
        summary_rows.append({
            "comparison_id": cid,
            "candidate": info["candidate"],
            "candidate_replicate": info["candidate_replicate"],
            "context_regime": info["context_regime"],
            "context_tasks": info["context_tasks"],
            "context_replicates": info["context_replicates"],
            "K_star": info["K_star"],
            "Gamma_full_untruncated": info["Gamma"],
            "K_shared_prefix": prefix,
            "shared_component_fraction": prefix / max(info["K_star"], 1),
            "shared_energy_at_prefix": eobs,
            "random_q_energy_at_prefix": enull,
            "functional_endpoint_component_displayed": fm,
            "functional_endpoint_bacc_displayed": fb,
            "shared_q": q,
            "cutoff_rule": "contiguous_prefix_until_component_shared_energy<=matched_random_q",
        })

    write_csv(data_dir / "shared_components_postprocessed.csv", all_component_rows)
    write_csv(data_dir / "shared_prefix_summary.csv", summary_rows)
    write_csv(data_dir / "shared_energy_prefix_curves.csv", energy_curve_rows)
    write_csv(data_dir / "shared_functional_prefix_curves.csv", func_curve_rows)
    return summary_rows, energy_curve_rows, func_curve_rows


def average_shared_group(
    curve_rows: List[dict], ycols: Sequence[str], summary_rows: List[dict]
) -> List[dict]:
    sum_by_cid = {str(r["comparison_id"]): r for r in summary_rows}
    groups = group_rows(curve_rows, ["candidate", "context_regime", "context_tasks"])
    out = []
    for (candidate, regime, ctx), rows in groups.items():
        by_cid = group_rows(rows, ["comparison_id"])
        cids = sorted(cid[0] for cid in by_cid.keys())
        if not cids:
            continue
        cutfracs = [fnum(sum_by_cid[cid]["shared_component_fraction"], 0.0) for cid in cids]
        kstars = [inum(sum_by_cid[cid]["K_star"]) for cid in cids]
        common_end = max(0.0, min(cutfracs))
        mean_kstar = float(np.mean(kstars)) if kstars else 0.0

        xs = {0.0, round(common_end, 6)}
        for cid in cids:
            for r in by_cid[(cid,)]:
                xf = fnum(r.get("x_fraction"))
                if 0.0 <= xf <= common_end + 1e-12:
                    xs.add(round(xf, 6))
        xgrid = np.array(sorted(xs), dtype=float)
        col_means = {}
        for yc in ycols:
            arrays = []
            for cid in cids:
                sub = [r for r in by_cid[(cid,)] if fnum(r.get("x_fraction")) <= common_end + 1e-12]
                if not sub:
                    continue
                xx = [fnum(r.get("x_fraction")) for r in sub]
                yy = [fnum(r.get(yc)) for r in sub]
                arrays.append(interp_curve(xx, yy, xgrid))
            col_means[yc] = mean_arrays(arrays) if arrays else np.full_like(xgrid, np.nan)
        for j, xf in enumerate(xgrid):
            rr = {
                "candidate": candidate,
                "context_regime": regime,
                "context_tasks": ctx,
                "x_fraction": float(xf),
                "x_component_count_display": int(round(xf * mean_kstar)),
                "common_end_fraction": common_end,
                "mean_K_star_display": mean_kstar,
            }
            for yc in ycols:
                rr[yc] = float(col_means[yc][j]) if j < len(col_means[yc]) else float("nan")
            out.append(rr)
    return out


def make_shared_figures(
    summary_rows: List[dict], energy_rows: List[dict], func_rows: List[dict],
    fig_dir: Path, data_dir: Path, dpi: int
):
    avg_energy = average_shared_group(energy_rows, ["shared_energy_spectrum", "random_q_spectrum"], summary_rows)
    avg_func = average_shared_group(func_rows, ["raw_bacc", "full_bacc", "chance_bacc"], summary_rows)
    write_csv(data_dir / "shared_energy_average_curves.csv", avg_energy)
    write_csv(data_dir / "shared_functional_average_curves.csv", avg_func)

    egroup = group_rows(avg_energy, ["candidate", "context_regime"])
    fgroup = group_rows(avg_func, ["candidate", "context_regime"])
    outputs = []
    numbering = {
        ("single", "Rest"): "05_single_context_Rest_redraw.png",
        ("single", "Motor"): "06_single_context_Motor_redraw.png",
        ("single", "P300"): "07_single_context_P300_redraw.png",
        ("single", "SSS"): "08_single_context_SSS_redraw.png",
        ("single", "TS"): "09_single_context_TS_redraw.png",
        ("full", "Rest"): "10_full_context_Rest_redraw.png",
        ("full", "Motor"): "11_full_context_Motor_redraw.png",
        ("full", "P300"): "12_full_context_P300_redraw.png",
        ("full", "SSS"): "13_full_context_SSS_redraw.png",
        ("full", "TS"): "14_full_context_TS_redraw.png",
    }
    for regime in ("single", "full"):
        for candidate in TASK_ORDER:
            esub = egroup.get((candidate, regime), [])
            fsub = fgroup.get((candidate, regime), [])
            if not esub or not fsub:
                continue
            contexts = sorted(
                {str(r["context_tasks"]) for r in esub},
                key=lambda x: TASK_ORDER.index(x) if x in TASK_ORDER else 999,
            )
            fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            for ctx in contexts:
                ge = sorted([r for r in esub if r["context_tasks"] == ctx], key=lambda r: inum(r["x_component_count_display"]))
                x = [inum(r["x_component_count_display"]) for r in ge]
                y = [fnum(r["shared_energy_spectrum"]) for r in ge]
                yq = [fnum(r["random_q_spectrum"]) for r in ge]
                (line,) = axes[0].plot(x, y, linewidth=2.2, label=ctx)
                axes[0].plot(x, yq, linewidth=1.1, linestyle="--", alpha=0.45, color=line.get_color())

                gf = sorted([r for r in fsub if r["context_tasks"] == ctx], key=lambda r: inum(r["x_component_count_display"]))
                axes[1].plot(
                    [inum(r["x_component_count_display"]) for r in gf],
                    [fnum(r["raw_bacc"]) for r in gf],
                    linewidth=2.2,
                    label=ctx,
                )

            full_vals = [fnum(r["full_bacc"]) for r in fsub if math.isfinite(fnum(r["full_bacc"]))]
            chance_vals = [fnum(r["chance_bacc"]) for r in fsub if math.isfinite(fnum(r["chance_bacc"]))]
            if full_vals:
                axes[1].axhline(float(np.mean(full_vals)), linestyle="--", linewidth=1.2, color="grey", label="full FT ref")
            if chance_vals:
                axes[1].axhline(float(np.mean(chance_vals)), linestyle=":", linewidth=1.2, color="grey", label="chance")

            axes[0].set_title(f"{candidate} | {regime.capitalize()} Context: Shared Energy (random-q95 truncated)", fontsize=15)
            axes[0].set_ylabel("Shared Energy Spectrum", fontsize=12)
            axes[0].grid(True, alpha=0.25)
            axes[0].legend(fontsize=9, loc="best", ncol=2)
            axes[1].set_title(f"{candidate} | {regime.capitalize()} Context: Shared Functional (same cutoff)", fontsize=15)
            axes[1].set_ylabel("Raw balanced accuracy", fontsize=12)
            axes[1].set_xlabel("Candidate functional components added", fontsize=12)
            axes[1].grid(True, alpha=0.25)
            axes[1].set_ylim(0.0 if not chance_vals else max(0.0, min(chance_vals) - 0.02), 1.02)
            axes[1].legend(fontsize=9, loc="best")
            fig.tight_layout()
            out = fig_dir / numbering[(regime, candidate)]
            fig.savefig(out, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            outputs.append(out)
    return outputs


def main():
    args = parse_args()
    analysis_dir = Path(args.analysis_dir).expanduser().resolve()
    if not analysis_dir.is_dir():
        raise FileNotFoundError(analysis_dir)
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else analysis_dir / "redraw_v6_1_png_only"
    if outdir.exists() and args.overwrite:
        shutil.rmtree(outdir)
    elif outdir.exists() and any(outdir.iterdir()):
        raise RuntimeError(f"Output exists: {outdir}. Use --overwrite.")
    fig_dir = ensure_dir(outdir / "figures_png")
    data_dir = ensure_dir(outdir / "postprocessed_data")

    tables = load_main_tables(analysis_dir)
    component_files = component_files_by_id(analysis_dir / "components")

    heatmap = make_heatmap_compare(tables["weight_update_heatmap"], fig_dir, data_dir, args.dpi)
    individual = make_individual_avg(tables["individual_spectra"], fig_dir, data_dir, args.dpi)
    summary, ec, fc = compute_shared_threshold_tables(
        tables["shared_energy_spectra"],
        tables["shared_functional_spectra"],
        component_files,
        data_dir,
        args.shared_q,
    )
    shared = make_shared_figures(summary, ec, fc, fig_dir, data_dir, args.dpi)

    manifest = {
        "version": VERSION,
        "analysis_dir": str(analysis_dir),
        "output_dir": str(outdir),
        "shared_quantile_threshold": args.shared_q,
        "dependencies": ["python stdlib", "numpy", "matplotlib"],
        "no_pandas": True,
        "no_scipy": True,
        "no_training": True,
        "no_inference": True,
        "png_only": True,
        "generated_figures": [str(heatmap), str(individual)] + [str(p) for p in shared],
        "cutoff_rule": "contiguous prefix of shared-energy-ranked candidate components; stop at first component not exceeding module-matched random q95",
        "random_threshold_implementation": "large-dimensional normal approximation to Beta(k/2,(D-k)/2) projection-energy null",
    }
    with open(outdir / "redraw_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("DONE")
    print("Output:", outdir)
    print("PNG figures:", fig_dir)
    print("Postprocessed data:", data_dir)


if __name__ == "__main__":
    main()
