#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot full fine-tuning training curves for LaBraM x M3CV runs.

What it does
------------
1. Scan ROOT/runs/<task>/repXX/ for per-epoch history CSV files.
2. Auto-detect common metric columns such as:
      objective_loss / loss
      balanced_accuracy / bacc
      seconds / epoch_time
   and also common train/val/eval variants.
3. Produce:
      - per-run plots
      - per-task combined plots (all replicates overlaid)
      - cross-task summary plots grouped by metric
4. Save all figures under:
      ROOT/figures_full_finetuning_curves

This script is intentionally defensive because different trainer versions may
use slightly different history CSV filenames / column names.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt


DEFAULT_ROOT = Path(
    "/omni-eeg-01/task calibration/results/tsv_labram_m3cv/tsv_labram_m3cv_v5_0"
)

HISTORY_CANDIDATE_FILES = (
    "epoch_history.csv",
    "history.csv",
    "train_history.csv",
    "training_history.csv",
    "metrics_history.csv",
    "epoch_metrics.csv",
    "metrics.csv",
)

# Canonical metrics we actively try to plot.
METRIC_ALIASES = {
    "objective_loss": [
        "objective_loss",
        "loss",
        "train_loss",
        "training_loss",
        "objective",
    ],
    "eval_objective_loss": [
        "eval_objective_loss",
        "validation_loss",
        "val_loss",
        "valid_loss",
        "eval_loss",
        "test_loss",
    ],
    "balanced_accuracy": [
        "balanced_accuracy",
        "bacc",
        "eval_balanced_accuracy",
        "validation_balanced_accuracy",
        "val_balanced_accuracy",
        "valid_balanced_accuracy",
        "test_balanced_accuracy",
    ],
    "train_balanced_accuracy": [
        "train_balanced_accuracy",
        "training_balanced_accuracy",
        "train_bacc",
    ],
    "seconds": [
        "seconds",
        "elapsed_seconds",
        "epoch_seconds",
        "time_seconds",
        "wall_seconds",
    ],
    "learning_rate": [
        "learning_rate",
        "lr",
        "backbone_lr",
        "current_lr",
    ],
}

# Text that often appears in columns we do NOT want as line plots.
EXCLUDE_COLUMN_SUBSTRINGS = (
    "sha",
    "digest",
    "path",
    "file",
    "task",
    "replicate",
    "protocol",
    "status",
)


def log(msg: str) -> None:
    print(msg, flush=True)


def normalize(name: str) -> str:
    return (
        name.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("[", "")
        .replace("]", "")
    )


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def maybe_float(x: str) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    try:
        v = float(s)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def find_history_csv(run_dir: Path) -> Optional[Path]:
    for name in HISTORY_CANDIDATE_FILES:
        p = run_dir / name
        if p.is_file():
            return p
    # Fallback: pick the only CSV if there is exactly one, otherwise the CSV
    # whose header looks most like epoch history.
    csvs = sorted(run_dir.glob("*.csv"))
    if not csvs:
        return None
    if len(csvs) == 1:
        return csvs[0]

    best = None
    best_score = -1
    for p in csvs:
        try:
            rows = read_csv_rows(p)
        except Exception:
            continue
        if not rows:
            continue
        cols = [normalize(c) for c in rows[0].keys()]
        score = 0
        if "epoch" in cols:
            score += 10
        score += sum(
            1 for c in cols
            if any(alias == c for aliases in METRIC_ALIASES.values() for alias in aliases)
        )
        if score > best_score:
            best_score = score
            best = p
    return best


def detect_epoch_column(rows: List[Dict[str, str]]) -> str:
    if not rows:
        raise ValueError("empty CSV")
    cols = list(rows[0].keys())
    norm = {c: normalize(c) for c in cols}
    for c, nc in norm.items():
        if nc == "epoch":
            return c

    # Fallback: choose the first numeric integer-like column with monotone
    # nondecreasing values.
    best = None
    for c in cols:
        vals = [maybe_float(r.get(c, "")) for r in rows]
        if any(v is None for v in vals):
            continue
        if all(abs(v - round(v)) < 1e-9 for v in vals):
            mono = all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))
            if mono:
                best = c
                break
    if best is None:
        raise RuntimeError("Could not detect epoch column")
    return best


def canonical_metric_columns(rows: List[Dict[str, str]], epoch_col: str) -> Dict[str, str]:
    cols = list(rows[0].keys())
    norm_map = {c: normalize(c) for c in cols}
    out = {}

    for canonical, aliases in METRIC_ALIASES.items():
        for c, nc in norm_map.items():
            if c == epoch_col:
                continue
            if nc in aliases:
                out[canonical] = c
                break

    return out


def find_extra_numeric_columns(rows: List[Dict[str, str]], epoch_col: str,
                               already_used: Sequence[str]) -> List[str]:
    cols = list(rows[0].keys())
    extra = []
    for c in cols:
        if c == epoch_col or c in already_used:
            continue
        nc = normalize(c)
        if any(tok in nc for tok in EXCLUDE_COLUMN_SUBSTRINGS):
            continue
        vals = [maybe_float(r.get(c, "")) for r in rows]
        if sum(v is not None for v in vals) >= max(2, int(0.7 * len(vals))):
            extra.append(c)
    return extra


def extract_series(rows: List[Dict[str, str]], xcol: str, ycol: str) -> Tuple[List[float], List[float]]:
    xs = []
    ys = []
    for r in rows:
        x = maybe_float(r.get(xcol, ""))
        y = maybe_float(r.get(ycol, ""))
        if x is None or y is None:
            continue
        xs.append(x)
        ys.append(y)
    return xs, ys


def plot_single_metric(series_dict: Dict[str, Tuple[List[float], List[float]]],
                       title: str, xlabel: str, ylabel: str, outpath: Path) -> None:
    if not series_dict:
        return
    plt.figure(figsize=(8, 5))
    for label, (xs, ys) in series_dict.items():
        if xs and ys:
            plt.plot(xs, ys, marker="o", linewidth=1.5, markersize=3, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    if len(series_dict) > 1:
        plt.legend()
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=180)
    plt.close()


def task_rep_sort_key(tag: str) -> Tuple[str, int]:
    # tag like "Motor/rep02"
    if "/rep" in tag:
        task, rep = tag.split("/rep")
        try:
            return task, int(rep)
        except Exception:
            return task, 999
    return tag, 999


def main():
    parser = argparse.ArgumentParser(
        description="Plot full fine-tuning curves from run histories",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--outdir", type=Path, default=None,
        help="Output directory. Default: ROOT/figures_full_finetuning_curves"
    )
    parser.add_argument(
        "--include-extra-numeric", action="store_true",
        help="Also plot extra numeric columns not in the canonical alias table."
    )
    args = parser.parse_args()

    root = args.root
    outdir = args.outdir or (root / "figures_full_finetuning_curves")
    runs_root = root / "runs"

    if not runs_root.is_dir():
        raise FileNotFoundError(f"Missing runs directory: {runs_root}")

    discovered = []
    for task_dir in sorted(runs_root.iterdir()):
        if not task_dir.is_dir():
            continue
        task = task_dir.name
        for rep_dir in sorted(task_dir.glob("rep*")):
            if not rep_dir.is_dir():
                continue
            hist = find_history_csv(rep_dir)
            if hist is None:
                log(f"[WARN] No history CSV found in {rep_dir}")
                continue
            discovered.append((task, rep_dir.name, rep_dir, hist))

    if not discovered:
        raise RuntimeError("No run histories were found.")

    log(f"[INFO] Found {len(discovered)} run histories.")
    for task, rep, _, hist in discovered:
        log(f"  - {task}/{rep}: {hist.name}")

    # Per-run parsed cache
    per_run = {}
    task_group = defaultdict(list)

    for task, rep, rep_dir, hist in discovered:
        rows = read_csv_rows(hist)
        if not rows:
            log(f"[WARN] Empty history CSV: {hist}")
            continue

        epoch_col = detect_epoch_column(rows)
        metric_cols = canonical_metric_columns(rows, epoch_col)
        extra_cols = []
        if args.include_extra_numeric:
            extra_cols = find_extra_numeric_columns(rows, epoch_col, list(metric_cols.values()))

        per_run[(task, rep)] = {
            "task": task,
            "rep": rep,
            "run_dir": rep_dir,
            "hist": hist,
            "rows": rows,
            "epoch_col": epoch_col,
            "metric_cols": metric_cols,
            "extra_cols": extra_cols,
        }
        task_group[task].append(rep)

    # 1) Per-run figures
    for (task, rep), info in per_run.items():
        rows = info["rows"]
        epoch_col = info["epoch_col"]
        metric_cols = dict(info["metric_cols"])

        run_out = outdir / "per_run" / task / rep
        run_out.mkdir(parents=True, exist_ok=True)

        # Canonical metrics
        for canonical, col in metric_cols.items():
            xs, ys = extract_series(rows, epoch_col, col)
            if not xs:
                continue
            plot_single_metric(
                {f"{task}/{rep}": (xs, ys)},
                title=f"{task}/{rep}: {canonical}",
                xlabel=epoch_col,
                ylabel=canonical,
                outpath=run_out / f"{canonical}.png",
            )

        # Train-vs-eval loss on one figure if both exist
        if "objective_loss" in metric_cols or "eval_objective_loss" in metric_cols:
            series = {}
            if "objective_loss" in metric_cols:
                xs, ys = extract_series(rows, epoch_col, metric_cols["objective_loss"])
                if xs:
                    series["train/objective_loss"] = (xs, ys)
            if "eval_objective_loss" in metric_cols:
                xs, ys = extract_series(rows, epoch_col, metric_cols["eval_objective_loss"])
                if xs:
                    series["eval/objective_loss"] = (xs, ys)
            if series:
                plot_single_metric(
                    series,
                    title=f"{task}/{rep}: loss curves",
                    xlabel=epoch_col,
                    ylabel="loss",
                    outpath=run_out / "loss_curves_combined.png",
                )

        # Train-vs-eval bACC on one figure if both exist
        bacc_series = {}
        if "train_balanced_accuracy" in metric_cols:
            xs, ys = extract_series(rows, epoch_col, metric_cols["train_balanced_accuracy"])
            if xs:
                bacc_series["train/bACC"] = (xs, ys)
        if "balanced_accuracy" in metric_cols:
            xs, ys = extract_series(rows, epoch_col, metric_cols["balanced_accuracy"])
            if xs:
                bacc_series["eval/bACC"] = (xs, ys)
        if bacc_series:
            plot_single_metric(
                bacc_series,
                title=f"{task}/{rep}: balanced accuracy",
                xlabel=epoch_col,
                ylabel="balanced accuracy",
                outpath=run_out / "balanced_accuracy_combined.png",
            )

        # Optional extra numeric columns
        for col in info["extra_cols"]:
            xs, ys = extract_series(rows, epoch_col, col)
            if xs:
                safe = normalize(col)
                plot_single_metric(
                    {f"{task}/{rep}": (xs, ys)},
                    title=f"{task}/{rep}: {col}",
                    xlabel=epoch_col,
                    ylabel=col,
                    outpath=run_out / f"extra_{safe}.png",
                )

    # 2) Per-task combined figures (overlay replicates)
    for task, reps in sorted(task_group.items()):
        infos = [per_run[(task, rep)] for rep in sorted(reps)]
        metric_to_series = defaultdict(dict)

        for info in infos:
            epoch_col = info["epoch_col"]
            rows = info["rows"]
            label = f"{task}/{info['rep']}"
            for canonical, col in info["metric_cols"].items():
                xs, ys = extract_series(rows, epoch_col, col)
                if xs:
                    metric_to_series[canonical][label] = (xs, ys)

        task_out = outdir / "per_task" / task
        task_out.mkdir(parents=True, exist_ok=True)

        for canonical, series in metric_to_series.items():
            plot_single_metric(
                dict(sorted(series.items(), key=lambda kv: task_rep_sort_key(kv[0]))),
                title=f"{task}: {canonical} across replicates",
                xlabel="epoch",
                ylabel=canonical,
                outpath=task_out / f"{canonical}_replicates.png",
            )

        # Combined task-level loss plot
        loss_series = {}
        for info in infos:
            label = f"{task}/{info['rep']}"
            rows = info["rows"]
            epoch_col = info["epoch_col"]
            if "objective_loss" in info["metric_cols"]:
                xs, ys = extract_series(rows, epoch_col, info["metric_cols"]["objective_loss"])
                if xs:
                    loss_series[f"{label} train"] = (xs, ys)
            if "eval_objective_loss" in info["metric_cols"]:
                xs, ys = extract_series(rows, epoch_col, info["metric_cols"]["eval_objective_loss"])
                if xs:
                    loss_series[f"{label} eval"] = (xs, ys)
        if loss_series:
            plot_single_metric(
                dict(sorted(loss_series.items())),
                title=f"{task}: loss curves",
                xlabel="epoch",
                ylabel="loss",
                outpath=task_out / "loss_curves_all.png",
            )

        # Combined task-level bACC plot
        bacc_series = {}
        for info in infos:
            label = f"{task}/{info['rep']}"
            rows = info["rows"]
            epoch_col = info["epoch_col"]
            if "balanced_accuracy" in info["metric_cols"]:
                xs, ys = extract_series(rows, epoch_col, info["metric_cols"]["balanced_accuracy"])
                if xs:
                    bacc_series[f"{label} eval"] = (xs, ys)
            if "train_balanced_accuracy" in info["metric_cols"]:
                xs, ys = extract_series(rows, epoch_col, info["metric_cols"]["train_balanced_accuracy"])
                if xs:
                    bacc_series[f"{label} train"] = (xs, ys)
        if bacc_series:
            plot_single_metric(
                dict(sorted(bacc_series.items())),
                title=f"{task}: balanced accuracy curves",
                xlabel="epoch",
                ylabel="balanced accuracy",
                outpath=task_out / "balanced_accuracy_all.png",
            )

    # 3) Cross-task summary figures: one figure per canonical metric, all runs
    all_metric_series = defaultdict(dict)
    for (task, rep), info in per_run.items():
        label = f"{task}/{rep}"
        rows = info["rows"]
        epoch_col = info["epoch_col"]
        for canonical, col in info["metric_cols"].items():
            xs, ys = extract_series(rows, epoch_col, col)
            if xs:
                all_metric_series[canonical][label] = (xs, ys)

    summary_out = outdir / "summary"
    summary_out.mkdir(parents=True, exist_ok=True)
    for canonical, series in all_metric_series.items():
        plot_single_metric(
            dict(sorted(series.items(), key=lambda kv: task_rep_sort_key(kv[0]))),
            title=f"All runs: {canonical}",
            xlabel="epoch",
            ylabel=canonical,
            outpath=summary_out / f"all_runs_{canonical}.png",
        )

    # 4) Simple manifest
    manifest_lines = []
    manifest_lines.append(f"root: {root}")
    manifest_lines.append(f"outdir: {outdir}")
    manifest_lines.append("")
    manifest_lines.append("discovered runs:")
    for task, rep, rep_dir, hist in discovered:
        key = (task, rep)
        if key not in per_run:
            continue
        info = per_run[key]
        manifest_lines.append(
            f"- {task}/{rep}: history={hist.name}, epoch_col={info['epoch_col']}, "
            f"metrics={sorted(info['metric_cols'].keys())}"
        )
    (outdir / "plot_manifest.txt").write_text("\n".join(manifest_lines), encoding="utf-8")

    log(f"[DONE] Figures written to: {outdir}")
    log(f"[DONE] Manifest written to: {outdir / 'plot_manifest.txt'}")


if __name__ == "__main__":
    main()
