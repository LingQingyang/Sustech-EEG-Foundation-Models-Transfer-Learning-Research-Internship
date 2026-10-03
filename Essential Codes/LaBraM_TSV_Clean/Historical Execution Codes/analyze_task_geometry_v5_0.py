#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LaBraM x M3CV Task Adaptation Geometry: V5.0 analysis
===================================================

Implements the mature V5.0 analysis protocol:

1) Individual Energy Spectrum
2) Individual Functional Spectrum
   - 5% coarse retained-rank grid
   - 1% refinement around the stable 99% functional-retention cutoff
3) Principal Angle Spectrum
   - functionally retained rank-1 update directions Z_j = u_j v_j^T
   - Single Context and Full Context
4) Matched Random Reference
   - exact ambient matrix shapes and retained ranks
   - random rank-1 directions u v^T
   - 1000 repetitions by default
   - 99th percentile by whole-model principal-dimension order
5) Replicate Reference
   - same-task rep1 <-> rep2, through the exact same geometry/function pipeline
6) Overlapped Functional Spectrum
   - principal-pair projection, sorted by cos(theta)
   - writes projected context directions back to their corresponding 72 matrices
   - candidate head and all non-72 adapted parameters are preserved
7) Paper-aligned STI
   - original TSV compression rule k = floor(r_full / T)

The central implementation detail is that principal angles are computed in the
Frobenius space of full rank-1 matrices Z = u v^T.  We never approximate that
geometry by separately comparing U-side and V-side subspaces.

For two rank-1 directions:
    <u_i v_i^T, u_j v_j^T>_F
      = (u_i^T u_j) (v_i^T v_j)

Therefore all principal-angle calculations can be done from Hadamard products
of small Gram matrices, without explicitly vectorizing an 800 x 200 matrix
into a 160,000-dimensional vector.

Keep this file next to:
    run_tsv_labram_m3cv_v0_3_4.py

Training is expected to have been certified by:
    run_task_geometry_v5_0.py
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import itertools
import json
import math
import os
import random
import sys
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

VERSION = "analysis-v5.0"
EPS = 1e-12

DEFAULT_TASKS = ("Rest", "Motor", "P300", "SSS", "TS")
TASK_FILES = {
    "Rest": "m3cv_resting_session1_4s.h5",
    "Motor": "m3cv_Motor_Session1_4s.h5",
    "P300": "m3cv_P300_Session1_4s.h5",
    "SSS": "m3cv_SSS_Session1_4s.h5",
    "TS": "m3cv_TS_Session1_4s.h5",
}
TASK_CLASS_NAMES = {
    "Rest": {0: "Beg_EC (Task 1)", 2: "Beg_EO (Task 2)"},
    "Motor": {0: "FT", 1: "RH", 2: "LH"},
    "P300": {0: "Non-target", 1: "Target"},
    "SSS": {0: "SSVEP", 1: "SSAEP", 2: "SSSEP"},
    "TS": {0: "VEP", 1: "AEP", 2: "SEP"},
}
MODULE_ROLES = ("Q", "K", "V", "O", "fc1", "fc2")
N_BLOCKS = 12
N_MODULES = 72


# =============================================================================
# Generic utilities
# =============================================================================

def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def parse_csv_strs(x: str) -> List[str]:
    return [z.strip() for z in str(x).split(",") if z.strip()]


def atomic_json(path: Path | str, payload) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    tmp = p.with_name(p.name + f".tmp.{os.getpid()}.{time.time_ns()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def load_json(path: Path | str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_csv(path: Path | str, rows: Sequence[Mapping[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    p = Path(path)
    ensure_dir(p.parent)
    fields: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                fields.append(str(k))
                seen.add(k)

    # Existing CSVs keep their established schema.  Every stage in this script
    # writes a fixed schema, so a mismatch is a fail-closed programming error.
    if p.is_file() and p.stat().st_size > 0:
        with open(p, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            existing = next(reader)
        if existing != fields:
            raise RuntimeError(
                f"CSV schema mismatch for {p}: existing={existing}, new={fields}"
            )
        mode = "a"
        write_header = False
    else:
        mode = "w"
        write_header = True

    with open(p, mode, encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        for row in rows:
            w.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def read_csv(path: Path | str) -> List[Dict[str, str]]:
    p = Path(path)
    if not p.is_file():
        return []
    with open(p, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def remove_csv_rows(path: Path | str, predicate) -> None:
    """Remove stale/partial rows before recomputing one resumable unit."""
    p = Path(path)
    if not p.is_file():
        return
    rows = read_csv(p)
    kept = [r for r in rows if not predicate(r)]
    if len(kept) == len(rows):
        return
    p.unlink()
    if kept:
        append_csv(p, kept)


def stable_seed(base: int, *parts: object) -> int:
    text = f"{int(base)}|" + "|".join(str(x) for x in parts)
    h = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little") % (2**31 - 1)


class Progress:
    def __init__(self, path: Path, resume: bool, signature: str):
        self.path = path
        self.resume = bool(resume)
        self.signature = str(signature)
        if self.resume and path.is_file():
            payload = load_json(path)
            old_sig = str(payload.get("signature", ""))
            if old_sig != self.signature:
                raise RuntimeError(
                    f"Stale resume marker {path}: signature {old_sig} != {self.signature}. "
                    "Rerun this stage with --no-resume rather than mixing analysis versions."
                )
            self.done = set(payload.get("done", []))
        else:
            self.done = set()

    def has(self, key: str) -> bool:
        return self.resume and key in self.done

    def mark(self, key: str) -> None:
        self.done.add(key)
        atomic_json(
            self.path,
            {"signature": self.signature, "done": sorted(self.done)},
        )


def analysis_code_sha256() -> str:
    return hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()


def stage_signature(args, stage: str) -> str:
    stage = str(stage)
    payload = {
        "analysis_version": VERSION,
        "analysis_code_sha256": getattr(args, "_analysis_code_sha256", analysis_code_sha256()),
        "final_manifest_sha256": getattr(args, "_final_manifest_sha256", ""),
        "stage": stage,
        "tasks": list(args.tasks),
    }
    if stage in ("individual", "cross", "overlap"):
        payload.update({
            "coarse_step": float(args.coarse_step),
            "refine_step": float(args.refine_step),
            "target": float(args.target_functional_retention),
        })
    if stage in ("individual", "overlap"):
        payload["model_init_seed"] = int(args.model_init_seed)
    if stage == "cross":
        payload.update({
            "random_n": int(args.random_n),
            "random_seed": int(args.random_seed),
            "full_context_pairings": str(args.full_context_pairings),
        })
    if stage == "overlap":
        payload["full_context_pairings"] = str(args.full_context_pairings)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def import_core():
    """
    Import the mature trainer lazily so pure geometry self-tests do not need
    PyTorch/LaBraM.
    """
    try:
        import run_tsv_labram_m3cv_v0_3_4 as core
    except Exception as e:
        raise RuntimeError(
            "Cannot import run_tsv_labram_m3cv_v0_3_4.py. "
            "Place this analyzer in the same directory as the mature trainer."
        ) from e

    # Patch the V5.0 registry into the reused training implementation.
    core.TASK_FILES = dict(TASK_FILES)
    core.TASK_CLASS_NAMES = {k: dict(v) for k, v in TASK_CLASS_NAMES.items()}
    core.EXPECTED_CLASSES = {k: len(v) for k, v in TASK_CLASS_NAMES.items()}
    return core


# =============================================================================
# Run discovery / SVD store
# =============================================================================

@dataclass(frozen=True)
class RunDesc:
    task: str
    replicate: int
    run_dir: Path
    metadata_path: Path
    delta_path: Path
    checkpoint_path: Path
    protocol_fingerprint: str
    base_full_backbone_digest: str
    input_chans: Tuple[int, ...]
    provenance_source: str

    @property
    def key(self) -> Tuple[str, int]:
        return self.task, self.replicate

    @property
    def tag(self) -> str:
        return f"{self.task}_rep{self.replicate:02d}"


@dataclass
class SVDRec:
    U: np.ndarray
    s: np.ndarray
    V: np.ndarray  # right singular vectors as columns
    d_out: int
    d_in: int

    @property
    def r_full(self) -> int:
        return int(len(self.s))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def discover_runs(root: Path, tasks: Sequence[str]) -> Tuple[List[RunDesc], dict]:
    """Discover the exact V5.0 10-run grid and validate per-run provenance.

    V5.0 provenance is per RUN, not per task, because Motor/rep01 is inherited
    while Motor/rep02 is newly retrained.  This function therefore refuses the
    older task-level provenance shortcut.
    """
    manifest_path = root / "run_manifest.json"
    success_path = root / "_FINAL_SUCCESS.json"
    if not manifest_path.is_file() or not success_path.is_file():
        raise RuntimeError(
            "V5.0 final is not certified: run_manifest.json and _FINAL_SUCCESS.json are required."
        )
    manifest = load_json(manifest_path)
    final_success = load_json(success_path)
    manifest_sha = _sha256_file(manifest_path)
    if str(final_success.get("run_manifest_sha256", "")) != manifest_sha:
        raise RuntimeError("V5.0 final manifest hash mismatch")
    if str(manifest.get("version")) != "5.0":
        raise RuntimeError(f"Expected V5.0 final manifest, found {manifest.get('version')!r}")
    if int(final_success.get("n_runs", -1)) != 10:
        raise RuntimeError("V5.0 _FINAL_SUCCESS must certify exactly 10 runs")
    if str(final_success.get("composite_protocol_fingerprint", "")) != str(
        manifest.get("composite_protocol_fingerprint", "")
    ):
        raise RuntimeError("Composite protocol fingerprint mismatch")

    manifest_tasks = [str(x) for x in manifest.get("tasks", [])]
    if manifest_tasks != [str(x) for x in tasks]:
        raise RuntimeError(
            f"Manifest task order {manifest_tasks} != requested tasks {list(tasks)}"
        )
    reps = int(manifest.get("replicates", 0))
    if reps != 2:
        raise RuntimeError(f"V5.0 requires exactly two replicates, found {reps}")
    common_epochs = int(manifest.get("epochs", 0))
    if common_epochs != 20:
        raise RuntimeError(f"V5.0 final endpoint must be 20, found {common_epochs}")
    base_digest = str(manifest.get("base_full_backbone_digest", ""))
    if not base_digest:
        raise RuntimeError("V5.0 manifest lacks W0 digest")
    input_chans = tuple(int(x) for x in manifest.get("protocol", {}).get("input_chans", []))
    if len(input_chans) != 65 or input_chans[0] != 0:
        raise RuntimeError("V5.0 manifest has invalid LaBraM input_chans")

    expected_grid = [(t, r) for t in tasks for r in range(1, reps + 1)]
    manifest_grid = [
        (str(x.get("task")), int(x.get("replicate", -1)))
        for x in manifest.get("expected_run_grid", [])
    ]
    if manifest_grid != expected_grid:
        raise RuntimeError(
            f"Manifest expected_run_grid mismatch: {manifest_grid} != {expected_grid}"
        )

    provenance = manifest.get("run_provenance", {}) or {}
    expected_keys = {f"{t}/rep{r:02d}" for t, r in expected_grid}
    if set(provenance) != expected_keys:
        raise RuntimeError(
            f"Per-run provenance grid mismatch. missing={sorted(expected_keys-set(provenance))}, "
            f"extra={sorted(set(provenance)-expected_keys)}"
        )

    runs: List[RunDesc] = []
    expected_sources = {
        "Motor/rep01": "legacy_v0.3.4_reused",
        "Motor/rep02": "v5.0_new_training",
        "P300/rep01": "legacy_v0.3.4_reused",
        "P300/rep02": "legacy_v0.3.4_reused",
        "SSS/rep01": "legacy_v0.3.4_reused",
        "SSS/rep02": "legacy_v0.3.4_reused",
        "TS/rep01": "legacy_v0.3.4_reused",
        "TS/rep02": "legacy_v0.3.4_reused",
        "Rest/rep01": "v5.0_new_training",
        "Rest/rep02": "v5.0_new_training",
    }
    for task, rep in expected_grid:
        key = f"{task}/rep{rep:02d}"
        run_dir = root / "runs" / task / f"rep{rep:02d}"
        mp = run_dir / "metadata.json"
        dp = run_dir / "delta_weights.npz"
        cp = run_dir / "adapted_checkpoint.pth"
        sp = run_dir / "_SUCCESS.json"
        for p in (mp, dp, cp, sp):
            if not p.is_file():
                raise FileNotFoundError(p)

        prov = provenance[key]
        source = str(prov.get("source", ""))
        if source != expected_sources[key]:
            raise RuntimeError(
                f"{key}: provenance source {source!r} != expected {expected_sources[key]!r}"
            )
        origin_text = str(prov.get("origin_path", ""))
        if not origin_text:
            raise RuntimeError(f"{key}: provenance origin_path is empty")
        origin = Path(origin_text).expanduser()
        if source == "legacy_v0.3.4_reused":
            if not run_dir.is_symlink():
                raise RuntimeError(f"{key}: legacy run must be a replicate-level symlink")
            if run_dir.resolve() != origin.resolve():
                raise RuntimeError(
                    f"{key}: symlink target {run_dir.resolve()} != provenance origin {origin}"
                )
        elif source == "v5.0_new_training":
            if run_dir.is_symlink():
                raise RuntimeError(f"{key}: newly trained V5 run must not be a symlink")
            if run_dir.resolve() != origin.resolve():
                raise RuntimeError(f"{key}: new-run origin_path mismatch")
        else:
            raise RuntimeError(f"{key}: unknown provenance source {source!r}")

        meta = load_json(mp)
        run_success = load_json(sp)
        actual_hashes = {
            "metadata_sha256": _sha256_file(mp),
            "delta_weights_sha256": _sha256_file(dp),
            "adapted_checkpoint_sha256": _sha256_file(cp),
        }
        for hfield, actual in actual_hashes.items():
            if str(run_success.get(hfield, "")) != actual:
                raise RuntimeError(f"{key}: {hfield} mismatch against _SUCCESS")
            expected_h = str((prov.get("artifacts", {}) or {}).get(hfield, ""))
            if expected_h != actual:
                raise RuntimeError(f"{key}: {hfield} mismatch against V5 provenance")

        if str(meta.get("task")) != task or int(meta.get("replicate", -1)) != rep:
            raise RuntimeError(f"{key}: metadata identity mismatch")
        if int(meta.get("epochs", -1)) != common_epochs:
            raise RuntimeError(f"{key}: endpoint mismatch")
        if str(meta.get("base_full_backbone_digest", "")) != base_digest:
            raise RuntimeError(f"{key}: W0 digest mismatch")
        observed_fp = str(meta.get("protocol_fingerprint", ""))
        expected_fp = str(prov.get("protocol_fingerprint", ""))
        if not expected_fp or observed_fp != expected_fp:
            raise RuntimeError(f"{key}: protocol fingerprint mismatch")
        if str(run_success.get("protocol_fingerprint", "")) != expected_fp:
            raise RuntimeError(f"{key}: _SUCCESS fingerprint mismatch")
        if str(run_success.get("base_full_backbone_digest", "")) != base_digest:
            raise RuntimeError(f"{key}: _SUCCESS W0 mismatch")
        if [int(x) for x in meta.get("input_chans", [])] != list(input_chans):
            raise RuntimeError(f"{key}: input_chans mismatch")

        runs.append(RunDesc(
            task=task,
            replicate=rep,
            run_dir=run_dir,
            metadata_path=mp,
            delta_path=dp,
            checkpoint_path=cp,
            protocol_fingerprint=expected_fp,
            base_full_backbone_digest=base_digest,
            input_chans=input_chans,
            provenance_source=source,
        ))

    # Refuse hidden extra replicate directories, including an accidental link to
    # the rejected old Motor/rep02 or any stale rep03.
    extras = set()
    runs_root = root / "runs"
    for task_dir in runs_root.iterdir():
        if not task_dir.is_dir():
            continue
        for rep_dir in task_dir.iterdir():
            if not rep_dir.is_dir() or not rep_dir.name.startswith("rep"):
                continue
            try:
                rep = int(rep_dir.name[3:])
                key = f"{task_dir.name}/rep{rep:02d}"
            except Exception:
                key = f"{task_dir.name}/{rep_dir.name}"
            if key not in expected_keys:
                extras.add(key)
    if extras:
        raise RuntimeError(f"Unexpected run directories in V5 root: {sorted(extras)}")

    return runs, manifest


def verify_current_inputs(core, manifest: Mapping[str, object], args) -> None:
    """Ensure functional inference uses the exact inputs certified by V5 final."""
    protocol = manifest.get("protocol", {}) or {}
    core_path = Path(core.__file__).resolve()
    expected_core_sha = str(protocol.get("runner_sha256", ""))
    if expected_core_sha and _sha256_file(core_path) != expected_core_sha:
        raise RuntimeError(
            "Mature training core changed since V5 preflight; refusing model reconstruction."
        )

    checkpoint = Path(args.checkpoint)
    modeling = Path(args.modeling_file)
    if _sha256_file(checkpoint) != str(protocol.get("checkpoint", {}).get("sha256", "")):
        raise RuntimeError("LaBraM checkpoint changed since V5 preflight")
    if _sha256_file(modeling) != str(protocol.get("modeling_file", {}).get("sha256", "")):
        raise RuntimeError("modeling_finetune.py changed since V5 preflight")
    if int(args.model_init_seed) != int(protocol.get("model_init_seed", args.model_init_seed)):
        raise RuntimeError("Analyzer model_init_seed differs from V5 training protocol")

    data = protocol.get("data", {}) or {}
    for task in args.tasks:
        meta = core.load_task_meta(task, Path(args.data_root) / TASK_FILES[task])
        audit = data.get(task, {}) or {}
        st = meta.path.stat()
        if int(audit.get("file_size_bytes", -1)) != int(st.st_size):
            raise RuntimeError(f"{task}: H5 size changed since V5 preflight")
        if int(audit.get("file_mtime_ns", -1)) != int(st.st_mtime_ns):
            raise RuntimeError(f"{task}: H5 mtime changed since V5 preflight")
        if str(audit.get("labels_sha256", "")) != core._sha256_ndarray(meta.labels):
            raise RuntimeError(f"{task}: labels changed since V5 preflight")
        if str(audit.get("subject_ids_sha256", "")) != core._sha256_ndarray(meta.subjects):
            raise RuntimeError(f"{task}: subject IDs changed since V5 preflight")


def discover_modules(delta_path: Path, core) -> List[str]:
    with np.load(delta_path, allow_pickle=False) as z:
        mods = sorted(core.unsanitize_module_key(k) for k in z.files)
    if len(mods) != N_MODULES:
        raise RuntimeError(f"Expected 72 analyzed matrices, found {len(mods)}")
    return mods


class SVDStore:
    """
    Lazy SVD store with bounded in-memory cache.

    It deliberately reads the saved Delta-W archive rather than trusting any
    previous analysis product.  The mature trainer already certified the archive
    against the adapted checkpoint.
    """
    def __init__(self, runs: Sequence[RunDesc], core, max_cache: int = 256):
        self.core = core
        self.by_key = {r.key: r for r in runs}
        self.max_cache = int(max_cache)
        self.cache: OrderedDict[Tuple[str, int, str], SVDRec] = OrderedDict()

    def get(self, run: RunDesc, module: str) -> SVDRec:
        key = (run.task, run.replicate, module)
        if key in self.cache:
            rec = self.cache.pop(key)
            self.cache[key] = rec
            return rec

        skey = self.core.sanitize_module_key(module)
        with np.load(run.delta_path, allow_pickle=False) as z:
            x = np.asarray(z[skey], dtype=np.float64)
        if x.ndim != 2 or not np.isfinite(x).all():
            raise RuntimeError(f"{run.tag} {module}: invalid Delta-W")
        U, s, Vh = np.linalg.svd(x, full_matrices=False)
        rec = SVDRec(U=U, s=s, V=Vh.T, d_out=x.shape[0], d_in=x.shape[1])
        self.cache[key] = rec
        while len(self.cache) > self.max_cache:
            self.cache.popitem(last=False)
        return rec


def rank_for_q(r_full: int, q: float) -> int:
    if q <= 0:
        return 0
    return min(int(r_full), max(1, int(math.ceil(float(q) * int(r_full)))))


def q_grid(step: float) -> List[float]:
    n = int(round(1.0 / step))
    return [round(i * step, 10) for i in range(n + 1)]


def stable_cutoff(points: Sequence[Tuple[float, float]], target: float) -> Optional[float]:
    pts = sorted((float(q), float(v)) for q, v in points)
    for i, (q, _) in enumerate(pts):
        if all(v >= target for _, v in pts[i:]):
            return q
    return None


# =============================================================================
# Rank-1 Frobenius geometry
# =============================================================================

@dataclass
class PrincipalDecomp:
    module: str
    cos: np.ndarray                  # [p]
    candidate_mix: np.ndarray        # [r_candidate, p]
    candidate_coeff: np.ndarray      # [p], <a_j, Delta-W_candidate-retained>
    context_coeff: np.ndarray        # [r_context_raw, p], b_j in raw context Z basis
    U_context_raw: np.ndarray        # [d_out, r_context_raw]
    V_context_raw: np.ndarray        # [d_in, r_context_raw]


def _orthonormalize_context_from_gram(H: np.ndarray) -> np.ndarray:
    """
    If K contains raw context Z columns, H=K^T K.
    Return B such that Q=K B has orthonormal columns.
    """
    if H.size == 0:
        return np.zeros((H.shape[0], 0), dtype=np.float64)
    H = (H + H.T) * 0.5
    lam, R = np.linalg.eigh(H)
    order = np.argsort(lam)[::-1]
    lam = lam[order]
    R = R[:, order]
    if not len(lam) or lam[0] <= 0:
        return np.zeros((H.shape[0], 0), dtype=np.float64)
    tol = max(H.shape) * np.finfo(np.float64).eps * float(lam[0]) * 10.0
    keep = lam > tol
    if not np.any(keep):
        return np.zeros((H.shape[0], 0), dtype=np.float64)
    return R[:, keep] / np.sqrt(lam[keep])[None, :]


def principal_from_uv(
    Uc: np.ndarray,
    Vc: np.ndarray,
    sc: np.ndarray,
    context_uv: Sequence[Tuple[np.ndarray, np.ndarray]],
    module: str = "",
) -> PrincipalDecomp:
    """
    Principal angles between:
      span{u_i v_i^T}_candidate
    and
      span{u_j v_j^T}_context

    Candidate Z directions are orthonormal by SVD.  Context directions may come
    from multiple tasks and therefore are orthonormalized implicitly through
    their Gram matrix.
    """
    Uc = np.asarray(Uc, dtype=np.float64)
    Vc = np.asarray(Vc, dtype=np.float64)
    sc = np.asarray(sc, dtype=np.float64)

    if Uc.shape[1] != Vc.shape[1] or Uc.shape[1] != len(sc):
        raise ValueError("Candidate U/V/s rank mismatch")

    if not context_uv:
        return PrincipalDecomp(
            module, np.zeros(0), np.zeros((Uc.shape[1], 0)),
            np.zeros(0), np.zeros((0, 0)),
            np.zeros((Uc.shape[0], 0)), np.zeros((Vc.shape[0], 0)),
        )

    Uall = np.concatenate([np.asarray(x[0], float) for x in context_uv], axis=1)
    Vall = np.concatenate([np.asarray(x[1], float) for x in context_uv], axis=1)
    if Uall.shape[1] != Vall.shape[1]:
        raise ValueError("Context U/V rank mismatch")

    rc = Uc.shape[1]
    rr = Uall.shape[1]
    if rc == 0 or rr == 0:
        return PrincipalDecomp(
            module, np.zeros(0), np.zeros((rc, 0)), np.zeros(0),
            np.zeros((rr, 0)), Uall, Vall,
        )

    # Frobenius Gram of paired rank-1 matrices:
    # <u_i v_i^T, u_j v_j^T> = <u_i,u_j><v_i,v_j>
    Hcc = (Uall.T @ Uall) * (Vall.T @ Vall)
    B = _orthonormalize_context_from_gram(Hcc)
    if B.shape[1] == 0:
        return PrincipalDecomp(
            module, np.zeros(0), np.zeros((rc, 0)), np.zeros(0),
            np.zeros((rr, 0)), Uall, Vall,
        )

    Htc_raw = (Uc.T @ Uall) * (Vc.T @ Vall)
    M = Htc_raw @ B  # candidate orthonormal basis vs orthonormalized context basis

    P, c, Qh = np.linalg.svd(M, full_matrices=False)
    c = np.clip(c, 0.0, 1.0)
    Q = Qh.T

    # a_j = sum_i P_ij Z_candidate_i
    # <a_j, Delta W_candidate> = P[:,j]^T sigma
    candidate_coeff = P.T @ sc

    # b_j = (K B) Q[:,j] = K (B Q[:,j])
    context_coeff = B @ Q

    return PrincipalDecomp(
        module=module,
        cos=c,
        candidate_mix=P,
        candidate_coeff=candidate_coeff,
        context_coeff=context_coeff,
        U_context_raw=Uall,
        V_context_raw=Vall,
    )


def haar_basis(d: int, r: int, rng: np.random.Generator) -> np.ndarray:
    if r == 0:
        return np.zeros((d, 0), dtype=np.float64)
    if r > d:
        raise ValueError(f"Cannot draw {r} orthonormal columns in R^{d}")
    A = rng.normal(size=(d, r))
    Q, R = np.linalg.qr(A, mode="reduced")
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    return Q * signs[None, :]


def pure_geometry_selftest() -> Dict[str, float]:
    rng = np.random.default_rng(7)

    # Identical rank-1 subspaces.
    U = haar_basis(8, 3, rng)
    V = haar_basis(7, 3, rng)
    s = np.array([3.0, 2.0, 1.0])
    d = principal_from_uv(U, V, s, [(U, V)], "identical")
    if not np.allclose(d.cos, 1.0, atol=1e-10):
        raise AssertionError(("identical", d.cos))

    # Exactly orthogonal via U-side.
    U2 = np.eye(8)[:, 3:6]
    U1 = np.eye(8)[:, :3]
    V1 = np.eye(7)[:, :3]
    V2 = np.eye(7)[:, :3]
    d2 = principal_from_uv(U1, V1, s, [(U2, V2)], "orthogonal")
    if not np.allclose(d2.cos, 0.0, atol=1e-12):
        raise AssertionError(("orthogonal", d2.cos))

    # Full context containing the candidate must cover it exactly.
    U3 = haar_basis(8, 2, rng)
    V3 = haar_basis(7, 2, rng)
    d3 = principal_from_uv(U, V, s, [(U, V), (U3, V3)], "nested")
    if not np.allclose(d3.cos, 1.0, atol=1e-9):
        raise AssertionError(("nested", d3.cos))

    # Same-subspace overlap projection must reconstruct retained Delta-W.
    alpha = d.candidate_coeff * d.cos
    raw = d.context_coeff @ alpha
    recon = (d.U_context_raw * raw[None, :]) @ d.V_context_raw.T
    truth = (U * s[None, :]) @ V.T
    rel = np.linalg.norm(recon - truth) / np.linalg.norm(truth)
    if rel > 1e-9:
        raise AssertionError(("projection reconstruction", rel))

    return {
        "identical_min_cos": float(np.min(d.cos)),
        "orthogonal_max_cos": float(np.max(d2.cos)),
        "nested_min_cos": float(np.min(d3.cos)),
        "same_subspace_projection_relative_error": float(rel),
    }


# =============================================================================
# Individual spectra
# =============================================================================

def individual_energy_rows(
    runs: Sequence[RunDesc],
    modules: Sequence[str],
    store: SVDStore,
    coarse_step: float,
) -> List[Dict[str, object]]:
    rows = []
    for run in runs:
        recs = [store.get(run, m) for m in modules]
        total = float(sum(np.sum(r.s * r.s) for r in recs))
        for q in q_grid(coarse_step):
            kept = 0.0
            dims = 0
            for r in recs:
                k = rank_for_q(r.r_full, q)
                kept += float(np.sum(r.s[:k] * r.s[:k]))
                dims += k
            rows.append({
                "task": run.task,
                "replicate": run.replicate,
                "q": q,
                "retained_dimensions": dims,
                "full_dimensions": int(sum(r.r_full for r in recs)),
                "energy_retention": kept / max(total, EPS),
            })
    return rows


def assign_tsv_matrices(model, matrices: Mapping[str, np.ndarray]) -> None:
    import torch
    with torch.no_grad():
        for l, block in enumerate(model.blocks):
            q = torch.as_tensor(
                matrices[f"L{l:02d}/Q"],
                device=block.attn.qkv.weight.device,
                dtype=block.attn.qkv.weight.dtype,
            )
            k = torch.as_tensor(
                matrices[f"L{l:02d}/K"],
                device=block.attn.qkv.weight.device,
                dtype=block.attn.qkv.weight.dtype,
            )
            v = torch.as_tensor(
                matrices[f"L{l:02d}/V"],
                device=block.attn.qkv.weight.device,
                dtype=block.attn.qkv.weight.dtype,
            )
            block.attn.qkv.weight.copy_(torch.cat([q, k, v], dim=0))
            for role, param in (
                ("O", block.attn.proj.weight),
                ("fc1", block.mlp.fc1.weight),
                ("fc2", block.mlp.fc2.weight),
            ):
                x = torch.as_tensor(
                    matrices[f"L{l:02d}/{role}"],
                    device=param.device,
                    dtype=param.dtype,
                )
                param.copy_(x)


class CandidateEvaluator:
    """
    Candidate checkpoint evaluator.

    Non-72 adapted backbone parameters and the candidate task head stay at the
    adapted checkpoint throughout.  Only the 72 analyzed matrices are replaced.
    This is exactly the V5.0 functional-retention boundary.
    """
    def __init__(
        self,
        core,
        run: RunDesc,
        meta,
        modules: Sequence[str],
        store: SVDStore,
        modeling_file: Path,
        checkpoint: Path,
        device,
        eval_batch_size: int,
        ram_reserve_gb: float,
        model_init_seed: int,
    ):
        import torch

        self.core = core
        self.run = run
        self.meta = meta
        self.modules = list(modules)
        self.store = store
        self.device = device

        run_meta = load_json(run.metadata_path)
        backbone, audit, digest = core.build_labram_backbone(
            modeling_file, checkpoint, device, model_init_seed
        )
        if str(digest) != str(run.base_full_backbone_digest):
            raise RuntimeError(
                f"{run.tag}: rebuilt W0 digest {digest} != certified {run.base_full_backbone_digest}"
            )
        self.base_tsv = {
            k: v.numpy().astype(np.float64, copy=True)
            for k, v in core.extract_tsv_weights(backbone).items()
        }

        saved = torch.load(run.checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(saved, Mapping):
            raise RuntimeError(f"{run.tag}: checkpoint is not a mapping")
        if str(saved.get("task")) != run.task or int(saved.get("replicate", -1)) != run.replicate:
            raise RuntimeError(f"{run.tag}: checkpoint internal identity mismatch")
        if str(saved.get("protocol_fingerprint", "")) != run.protocol_fingerprint:
            raise RuntimeError(f"{run.tag}: checkpoint internal fingerprint mismatch")
        if str(saved.get("base_full_backbone_digest", "")) != run.base_full_backbone_digest:
            raise RuntimeError(f"{run.tag}: checkpoint internal W0 mismatch")
        input_chans = tuple(int(x) for x in saved.get("input_chans", []))
        if input_chans != run.input_chans:
            raise RuntimeError(f"{run.tag}: checkpoint input_chans mismatch")
        if not isinstance(saved.get("backbone"), Mapping) or not isinstance(saved.get("head"), Mapping):
            raise RuntimeError(f"{run.tag}: checkpoint lacks backbone/head mappings")

        backbone.load_state_dict(saved["backbone"], strict=True)
        classifier = core.TaskClassifier(
            backbone, len(meta.classes), input_chans, head_seed=0
        ).to(device)
        classifier.head.load_state_dict(saved["head"], strict=True)

        self.backbone = backbone
        self.classifier = classifier
        self.task_data = core.load_task_into_memory(
            meta, reserve_gb=float(ram_reserve_gb)
        )
        self.loader = core.make_eval_loader(
            meta, int(eval_batch_size), device, task_data=self.task_data
        )
        self.weights = core.class_weights_for_meta(meta, device)
        full = core.evaluate(
            classifier, self.loader, device, len(meta.classes), self.weights
        )
        self.full_bacc = float(full["balanced_accuracy"])

        stored = float(run_meta["metrics_final"]["balanced_accuracy"])
        if abs(stored - self.full_bacc) > 1e-6:
            raise RuntimeError(
                f"{run.tag}: full-checkpoint bACC mismatch: "
                f"reeval={self.full_bacc}, stored={stored}"
            )

    def close(self):
        try:
            del self.task_data
        except Exception:
            pass
        try:
            del self.classifier
            del self.backbone
        except Exception:
            pass
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def eval_matrices(self, matrices: Mapping[str, np.ndarray]) -> float:
        assign_tsv_matrices(self.backbone, matrices)
        metrics = self.core.evaluate(
            self.classifier, self.loader, self.device,
            len(self.meta.classes), self.weights
        )
        return float(metrics["balanced_accuracy"])

    def truncated_matrices(self, q: float) -> Tuple[Dict[str, np.ndarray], int, int]:
        mats = {}
        dims = 0
        full_dims = 0
        for m in self.modules:
            r = self.store.get(self.run, m)
            k = rank_for_q(r.r_full, q)
            if k:
                delta = (r.U[:, :k] * r.s[:k][None, :]) @ r.V[:, :k].T
            else:
                delta = np.zeros((r.d_out, r.d_in), dtype=np.float64)
            mats[m] = self.base_tsv[m] + delta
            dims += k
            full_dims += r.r_full
        return mats, dims, full_dims


def functional_curve_for_run(
    evaluator: CandidateEvaluator,
    coarse_step: float,
    refine_step: float,
    target: float,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    run = evaluator.run
    values: Dict[float, Tuple[float, int, int]] = {}

    def evaluate_q(q: float):
        q = round(float(q), 10)
        if q in values:
            return
        mats, dims, full_dims = evaluator.truncated_matrices(q)
        bacc = evaluator.eval_matrices(mats)
        values[q] = (bacc, dims, full_dims)

    for q in q_grid(coarse_step):
        evaluate_q(q)

    coarse_points = [
        (q, values[q][0] / max(evaluator.full_bacc, EPS))
        for q in sorted(values)
    ]
    q_coarse = stable_cutoff(coarse_points, target)

    if q_coarse is not None and q_coarse > 0:
        lo = max(0.0, q_coarse - coarse_step)
        n = int(round((q_coarse - lo) / refine_step))
        for i in range(n + 1):
            evaluate_q(lo + i * refine_step)

    points = [
        (q, values[q][0] / max(evaluator.full_bacc, EPS))
        for q in sorted(values)
    ]
    q_star = stable_cutoff(points, target)
    cutoff_source = "stable_99pct"
    if q_star is None:
        q_star = 1.0
        cutoff_source = "full_rank_fallback"

    rows = []
    for q in sorted(values):
        bacc, dims, full_dims = values[q]
        rows.append({
            "task": run.task,
            "replicate": run.replicate,
            "q": q,
            "retained_dimensions": dims,
            "full_dimensions": full_dims,
            "balanced_accuracy": bacc,
            "full_balanced_accuracy": evaluator.full_bacc,
            "functional_retention": bacc / max(evaluator.full_bacc, EPS),
            "is_refinement_point": int(
                abs((q / coarse_step) - round(q / coarse_step)) > 1e-8
            ),
        })

    _, dims_star, full_dims = values.get(
        q_star,
        (None, None, None),
    )
    if dims_star is None:
        mats, dims_star, full_dims = evaluator.truncated_matrices(q_star)

    cutoff = {
        "task": run.task,
        "replicate": run.replicate,
        "q_star": q_star,
        "retained_dimensions": int(dims_star),
        "full_dimensions": int(full_dims),
        "target_functional_retention": target,
        "source": cutoff_source,
        "full_balanced_accuracy": evaluator.full_bacc,
        "full_rank_functional_retention": float(
            values[1.0][0] / max(evaluator.full_bacc, EPS)
        ),
    }
    return rows, cutoff


# =============================================================================
# Cross-task geometry helpers
# =============================================================================

def cutoff_map(cutoff_csv: Path) -> Dict[Tuple[str, int], float]:
    rows = read_csv(cutoff_csv)
    out = {}
    for r in rows:
        out[(r["task"], int(r["replicate"]))] = float(r["q_star"])
    return out


def retained_uvs(
    store: SVDStore,
    run: RunDesc,
    module: str,
    qcuts: Mapping[Tuple[str, int], float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rec = store.get(run, module)
    q = float(qcuts[run.key])
    k = rank_for_q(rec.r_full, q)
    return rec.U[:, :k], rec.V[:, :k], rec.s[:k]


def decomp_for_module(
    store: SVDStore,
    candidate: RunDesc,
    contexts: Sequence[RunDesc],
    module: str,
    qcuts: Mapping[Tuple[str, int], float],
) -> PrincipalDecomp:
    Uc, Vc, sc = retained_uvs(store, candidate, module, qcuts)
    ctx = []
    for r in contexts:
        U, V, s = retained_uvs(store, r, module, qcuts)
        ctx.append((U, V))
    return principal_from_uv(Uc, Vc, sc, ctx, module)


def geometry_for_comparison(
    store: SVDStore,
    candidate: RunDesc,
    contexts: Sequence[RunDesc],
    modules: Sequence[str],
    qcuts: Mapping[Tuple[str, int], float],
) -> Tuple[Dict[str, PrincipalDecomp], List[Tuple[float, str, int]]]:
    decs = {}
    all_pairs: List[Tuple[float, str, int]] = []
    for m in modules:
        d = decomp_for_module(store, candidate, contexts, m, qcuts)
        decs[m] = d
        for j, c in enumerate(d.cos):
            all_pairs.append((float(c), m, int(j)))
    all_pairs.sort(key=lambda x: x[0], reverse=True)
    return decs, all_pairs


def comparison_signature(
    store: SVDStore,
    candidate: RunDesc,
    contexts: Sequence[RunDesc],
    modules: Sequence[str],
    qcuts: Mapping[Tuple[str, int], float],
    random_n: int,
    seed_base: int,
) -> str:
    # The random law depends only on ambient dimensions and retained ranks,
    # not on task names or replicate IDs.  Omitting run tags lets identical
    # matched configurations reuse the same expensive 1000-repeat null.
    payload = {
        "geometry_version": "rank1-Z-v5.0",
        "random_n": int(random_n),
        "seed_base": int(seed_base),
        "modules": [],
    }
    for m in modules:
        cr = store.get(candidate, m)
        ck = rank_for_q(cr.r_full, qcuts[candidate.key])
        rr = []
        for r in contexts:
            x = store.get(r, m)
            rr.append(rank_for_q(x.r_full, qcuts[r.key]))
        payload["modules"].append([m, cr.d_out, cr.d_in, ck, rr])
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def random_cos_one_module(
    d_out: int,
    d_in: int,
    r_candidate: int,
    r_contexts: Sequence[int],
    rng: np.random.Generator,
) -> np.ndarray:
    Uc = haar_basis(d_out, r_candidate, rng)
    Vc = haar_basis(d_in, r_candidate, rng)
    ctx = []
    for r in r_contexts:
        ctx.append((
            haar_basis(d_out, int(r), rng),
            haar_basis(d_in, int(r), rng),
        ))
    sc = np.ones(r_candidate, dtype=np.float64)
    return principal_from_uv(Uc, Vc, sc, ctx, "random").cos


def matched_random_q99(
    cache_dir: Path,
    store: SVDStore,
    candidate: RunDesc,
    contexts: Sequence[RunDesc],
    modules: Sequence[str],
    qcuts: Mapping[Tuple[str, int], float],
    random_n: int,
    seed_base: int,
) -> np.ndarray:
    sig = comparison_signature(
        store, candidate, contexts, modules, qcuts, random_n, seed_base
    )
    p = ensure_dir(cache_dir) / f"{sig}.npz"
    if p.is_file():
        with np.load(p, allow_pickle=False) as z:
            return np.asarray(z["q99"], dtype=np.float64)

    # Determine the whole-model principal-spectrum length from the real ranks.
    lengths = []
    specs = []
    for m in modules:
        cr = store.get(candidate, m)
        ck = rank_for_q(cr.r_full, qcuts[candidate.key])
        ctx_ranks = []
        for r in contexts:
            rr = store.get(r, m)
            ctx_ranks.append(rank_for_q(rr.r_full, qcuts[r.key]))
        specs.append((cr.d_out, cr.d_in, ck, ctx_ranks))
        # Random concatenated rank-1 directions are in general position.
        lengths.append(min(ck, sum(ctx_ranks)))
    L = int(sum(lengths))
    if L == 0:
        return np.zeros(0, dtype=np.float64)

    samples = np.empty((int(random_n), L), dtype=np.float32)
    for b in range(int(random_n)):
        rng = np.random.default_rng(
            stable_seed(seed_base, sig, "matched_random", b)
        )
        vals = []
        for d_out, d_in, ck, ctx_ranks in specs:
            if ck == 0 or sum(ctx_ranks) == 0:
                continue
            vals.extend(
                random_cos_one_module(
                    d_out, d_in, ck, ctx_ranks, rng
                ).tolist()
            )
        vals = np.sort(np.asarray(vals, dtype=np.float64))[::-1]
        if len(vals) != L:
            raise RuntimeError(
                f"Random principal length changed: expected {L}, got {len(vals)}"
            )
        samples[b] = vals.astype(np.float32)

    q99 = np.quantile(samples, 0.99, axis=0).astype(np.float64)
    tmp = p.with_name(p.name + f".tmp.{os.getpid()}")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, q99=q99, random_n=np.array([random_n]))
    os.replace(tmp, p)
    return q99


def context_id(contexts: Sequence[RunDesc]) -> str:
    return "+".join(r.tag for r in contexts)


def pa_rows_for_curve(
    curve_id: str,
    context_type: str,
    candidate: RunDesc,
    contexts: Sequence[RunDesc],
    pairs: Sequence[Tuple[float, str, int]],
    random_q99: Optional[np.ndarray],
) -> List[Dict[str, object]]:
    rows = []
    for i, (c, m, j) in enumerate(pairs, 1):
        rq = float(random_q99[i - 1]) if random_q99 is not None and i <= len(random_q99) else float("nan")
        rows.append({
            "curve_id": curve_id,
            "context_type": context_type,
            "candidate_task": candidate.task,
            "candidate_replicate": candidate.replicate,
            "context_tasks": "+".join(r.task for r in contexts),
            "context_replicates": "+".join(str(r.replicate) for r in contexts),
            "principal_dimension": i,
            "cos_theta": c,
            "matched_random_q99": rq,
            "above_random_q99": int(np.isfinite(rq) and c > rq),
            "module": m,
            "module_principal_index": j + 1,
        })
    return rows


# =============================================================================
# Overlapped functional spectrum
# =============================================================================

def overlap_matrices(
    evaluator: CandidateEvaluator,
    decs: Mapping[str, PrincipalDecomp],
    global_pairs: Sequence[Tuple[float, str, int]],
    m_keep: int,
) -> Dict[str, np.ndarray]:
    selected: Dict[str, List[int]] = defaultdict(list)
    for _, mod, j in global_pairs[: int(m_keep)]:
        selected[mod].append(int(j))

    mats = {}
    for mod in evaluator.modules:
        d = decs[mod]
        js = selected.get(mod, [])
        if not js:
            delta = np.zeros_like(evaluator.base_tsv[mod], dtype=np.float64)
        else:
            jj = np.asarray(js, dtype=np.int64)
            alpha = d.candidate_coeff[jj] * d.cos[jj]
            raw_coeff = d.context_coeff[:, jj] @ alpha
            delta = (
                d.U_context_raw * raw_coeff[None, :]
            ) @ d.V_context_raw.T
        mats[mod] = evaluator.base_tsv[mod] + delta
    return mats


def overlap_curve(
    evaluator: CandidateEvaluator,
    store: SVDStore,
    contexts: Sequence[RunDesc],
    modules: Sequence[str],
    qcuts: Mapping[Tuple[str, int], float],
    coarse_step: float,
    refine_step: float,
    target: float,
) -> List[Dict[str, object]]:
    decs, pairs = geometry_for_comparison(
        store, evaluator.run, contexts, modules, qcuts
    )
    P = len(pairs)
    values: Dict[int, float] = {}

    def eval_m(m: int):
        m = max(0, min(P, int(m)))
        if m in values:
            return
        mats = overlap_matrices(evaluator, decs, pairs, m)
        values[m] = evaluator.eval_matrices(mats)

    coarse_qs = q_grid(coarse_step)
    for q in coarse_qs:
        m = 0 if q <= 0 or P == 0 else min(P, max(1, int(math.ceil(q * P))))
        eval_m(m)

    coarse_points = sorted(
        (m / max(P, 1), b / max(evaluator.full_bacc, EPS))
        for m, b in values.items()
    )
    q_coarse = stable_cutoff(coarse_points, target)

    if q_coarse is not None and q_coarse > 0 and P > 0:
        lo = max(0.0, q_coarse - coarse_step)
        n = int(round((q_coarse - lo) / refine_step))
        for i in range(n + 1):
            q = lo + i * refine_step
            m = 0 if q <= 0 else min(P, max(1, int(math.ceil(q * P))))
            eval_m(m)

    out = []
    for m in sorted(values):
        b = values[m]
        out.append({
            "principal_dimensions": int(m),
            "full_principal_dimensions": int(P),
            "principal_fraction": float(m / max(P, 1)),
            "balanced_accuracy": float(b),
            "full_balanced_accuracy": float(evaluator.full_bacc),
            "functional_retention": float(
                b / max(evaluator.full_bacc, EPS)
            ),
        })
    return out


# =============================================================================
# Paper STI
# =============================================================================

def sti_metrics(
    Us: Sequence[np.ndarray],
    Ss: Sequence[np.ndarray],
    Vs: Sequence[np.ndarray],
) -> Dict[str, float]:
    U = np.concatenate(Us, axis=1)
    V = np.concatenate(Vs, axis=1)
    s = np.concatenate(Ss)
    A = U.T @ U - np.eye(U.shape[1])
    B = V.T @ V - np.eye(V.shape[1])
    M = (A * s[None, :]) @ B
    return {
        "sti_l1_entrywise": float(np.sum(np.abs(M))),
        "sti_l1_induced": float(np.linalg.norm(M, ord=1)),
        "sti_fro": float(np.linalg.norm(M, ord="fro")),
        "retained_sigma_l1": float(np.sum(np.abs(s))),
        "retained_sigma_l2": float(np.linalg.norm(s)),
    }


def paper_sti_rows(
    tasks: Sequence[str],
    runs_by_task: Mapping[str, Sequence[RunDesc]],
    modules: Sequence[str],
    store: SVDStore,
) -> List[Dict[str, object]]:
    T = len(tasks)
    rows = []
    combinations = itertools.product(*(runs_by_task[t] for t in tasks))
    for combo_idx, combo in enumerate(combinations, 1):
        combo_tag = "+".join(r.tag for r in combo)
        vals = []
        local_rows = []
        for m in modules:
            recs = [store.get(r, m) for r in combo]
            # Original TSV paper-aligned compression rule used in the previous
            # implementation: exact k=floor(r_full/T) per task/module.
            ks = [int(math.floor(rec.r_full / T)) for rec in recs]
            if any(k < 1 for k in ks):
                raise RuntimeError(
                    f"Paper STI floor(r_full/T) produced k<1 for module {m}: {ks}"
                )
            Us = [rec.U[:, :k] for rec, k in zip(recs, ks)]
            Vs = [rec.V[:, :k] for rec, k in zip(recs, ks)]
            Ss = [rec.s[:k] for rec, k in zip(recs, ks)]
            met = sti_metrics(Us, Ss, Vs)
            vals.append(met)
            local_rows.append({
                "combo_id": combo_tag,
                "scope": "module",
                "module": m,
                "sti_l1_entrywise": met["sti_l1_entrywise"],
                "sti_l1_induced": met["sti_l1_induced"],
                "sti_fro": met["sti_fro"],
                "retained_sigma_l1": met["retained_sigma_l1"],
                "retained_sigma_l2": met["retained_sigma_l2"],
                "n_tasks": T,
                "paper_rank_rule": "floor(r_full/T)",
            })
        rows.extend(local_rows)
        rows.append({
            "combo_id": combo_tag,
            "scope": "whole_model_summary",
            "module": "ALL72",
            "sti_l1_entrywise": float(sum(v["sti_l1_entrywise"] for v in vals)),
            "sti_l1_induced": float(sum(v["sti_l1_induced"] for v in vals)),
            "sti_fro": float(sum(v["sti_fro"] for v in vals)),
            "retained_sigma_l1": float(sum(v["retained_sigma_l1"] for v in vals)),
            "retained_sigma_l2": float(math.sqrt(sum(v["retained_sigma_l2"]**2 for v in vals))),
            "n_tasks": T,
            "paper_rank_rule": "floor(r_full/T)",
        })
        log(f"STI replicate-combination {combo_idx}: {combo_tag}")
    return rows


# =============================================================================
# Plotting
# =============================================================================

def _import_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _median_range_by_x(rows, xkey, ykey):
    by = defaultdict(list)
    for r in rows:
        try:
            x = float(r[xkey])
            y = float(r[ykey])
        except Exception:
            continue
        if np.isfinite(x) and np.isfinite(y):
            by[x].append(y)
    xs = sorted(by)
    med = [float(np.median(by[x])) for x in xs]
    lo = [float(np.min(by[x])) for x in xs]
    hi = [float(np.max(by[x])) for x in xs]
    return np.asarray(xs), np.asarray(med), np.asarray(lo), np.asarray(hi)


def plot_individual(out_fig: Path, energy_csv: Path, functional_csv: Path):
    plt = _import_plt()
    erows = read_csv(energy_csv)
    frows = read_csv(functional_csv)

    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    for task in DEFAULT_TASKS:
        rr = [r for r in erows if r["task"] == task]
        x, med, lo, hi = _median_range_by_x(rr, "retained_dimensions", "energy_retention")
        if len(x):
            line, = ax.plot(x, med, marker="o", label=task)
            ax.fill_between(x, lo, hi, alpha=0.12)
    ax.set_xlabel("Actual retained dimensions across 72 matrices")
    ax.set_ylabel("Whole-model update-energy retention")
    ax.set_ylim(0, 1.02)
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_fig / "01_individual_energy_spectrum.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    for task in DEFAULT_TASKS:
        rr = [r for r in frows if r["task"] == task]
        x, med, lo, hi = _median_range_by_x(rr, "retained_dimensions", "functional_retention")
        if len(x):
            line, = ax.plot(x, med, marker="o", label=task)
            ax.fill_between(x, lo, hi, alpha=0.12)
    ax.axhline(0.99, linestyle="--", linewidth=1.2)
    ax.set_xlabel("Actual retained dimensions across 72 matrices")
    ax.set_ylabel("bACC / full-checkpoint bACC")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_fig / "02_individual_functional_spectrum.png", dpi=180)
    plt.close(fig)


def plot_pa_for_type(
    out_fig: Path, pa_csv: Path, context_type: str, prefix: str
):
    plt = _import_plt()
    rows = read_csv(pa_csv)
    for candidate in DEFAULT_TASKS:
        rr0 = [
            r for r in rows
            if r["candidate_task"] == candidate and r["context_type"] == context_type
        ]
        if not rr0:
            continue
        fig, ax = plt.subplots(figsize=(8.7, 5.5))
        if context_type == "single":
            context_labels = [t for t in DEFAULT_TASKS if t != candidate]
            for ctx in context_labels:
                rr = [r for r in rr0 if r["context_tasks"] == ctx]
                x, med, lo, hi = _median_range_by_x(rr, "principal_dimension", "cos_theta")
                if len(x):
                    line, = ax.plot(x, med, label=ctx)
                    ax.fill_between(x, lo, hi, alpha=0.10)
                    _, rq, _, _ = _median_range_by_x(rr, "principal_dimension", "matched_random_q99")
                    if len(rq) == len(x):
                        ax.plot(x, rq, linestyle=":", linewidth=1.0, color=line.get_color())
        else:
            rr = rr0
            x, med, lo, hi = _median_range_by_x(rr, "principal_dimension", "cos_theta")
            if len(x):
                line, = ax.plot(x, med, label="Full Context")
                ax.fill_between(x, lo, hi, alpha=0.10)
                _, rq, _, _ = _median_range_by_x(rr, "principal_dimension", "matched_random_q99")
                if len(rq) == len(x):
                    ax.plot(x, rq, linestyle=":", linewidth=1.0, color=line.get_color())
        # Same-task replicate reference is the empirical geometry ceiling.
        rep_rows = [
            r for r in rows
            if r["candidate_task"] == candidate and r["context_type"] == "replicate"
        ]
        xr, mr, lr, hr = _median_range_by_x(
            rep_rows, "principal_dimension", "cos_theta"
        )
        if len(xr):
            ax.plot(xr, mr, linestyle="--", linewidth=1.6, label="Replicate Reference")
            ax.fill_between(xr, lr, hr, alpha=0.08)

        ax.set_title(f"{candidate}: Principal Angle Spectrum ({context_type})")
        ax.set_xlabel("Whole-model principal dimension")
        ax.set_ylabel(r"$\cos\theta$")
        ax.set_ylim(-0.02, 1.02)
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_fig / f"{prefix}_{candidate}.png", dpi=180)
        plt.close(fig)


def plot_overlap_for_type(
    out_fig: Path, overlap_csv: Path, context_type: str, prefix: str
):
    plt = _import_plt()
    rows = read_csv(overlap_csv)
    for candidate in DEFAULT_TASKS:
        rr0 = [
            r for r in rows
            if r["candidate_task"] == candidate and r["context_type"] == context_type
        ]
        if not rr0:
            continue
        fig, ax = plt.subplots(figsize=(8.7, 5.5))
        if context_type == "single":
            for ctx in [t for t in DEFAULT_TASKS if t != candidate]:
                rr = [r for r in rr0 if r["context_tasks"] == ctx]
                x, med, lo, hi = _median_range_by_x(
                    rr, "principal_dimensions", "functional_retention"
                )
                if len(x):
                    ax.plot(x, med, label=ctx)
                    ax.fill_between(x, lo, hi, alpha=0.10)
        else:
            x, med, lo, hi = _median_range_by_x(
                rr0, "principal_dimensions", "functional_retention"
            )
            if len(x):
                ax.plot(x, med, label="Full Context")
                ax.fill_between(x, lo, hi, alpha=0.10)
        # Same-task projection through the other replicate gives the empirical
        # functional upper reference under training randomness.
        rep_rows = [
            r for r in rows
            if r["candidate_task"] == candidate and r["context_type"] == "replicate"
        ]
        xr, mr, lr, hr = _median_range_by_x(
            rep_rows, "principal_dimensions", "functional_retention"
        )
        if len(xr):
            ax.plot(xr, mr, linestyle="--", linewidth=1.6, label="Replicate Reference")
            ax.fill_between(xr, lr, hr, alpha=0.08)

        ax.axhline(0.99, linestyle=":", linewidth=1.2)
        ax.set_title(f"{candidate}: Overlapped Functional Spectrum ({context_type})")
        ax.set_xlabel("Accumulated principal dimensions")
        ax.set_ylabel("bACC / candidate full-checkpoint bACC")
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_fig / f"{prefix}_{candidate}.png", dpi=180)
        plt.close(fig)


# =============================================================================
# Pipeline stages
# =============================================================================

def run_individual(args, core, root, out_analysis, out_fig, runs, modules, store, manifest):
    energy_csv = out_analysis / "individual_energy_spectrum.csv"
    functional_csv = out_analysis / "individual_functional_spectrum.csv"
    cutoff_csv = out_analysis / "functional_cutoffs_99pct.csv"

    # Energy is pure algebra and cheap, so rewrite atomically rather than append.
    erows = individual_energy_rows(runs, modules, store, args.coarse_step)
    if energy_csv.exists():
        energy_csv.unlink()
    append_csv(energy_csv, erows)

    progress_path = out_analysis / "_progress_individual.json"
    if not args.resume:
        for p in (functional_csv, cutoff_csv, progress_path):
            p.unlink(missing_ok=True)
    progress = Progress(progress_path, args.resume, stage_signature(args, "individual"))

    import torch
    device = torch.device(args.device)
    metas = {
        t: core.load_task_meta(t, Path(args.data_root) / TASK_FILES[t])
        for t in args.tasks
    }
    modeling_file = Path(args.modeling_file)
    checkpoint = Path(args.checkpoint)

    for idx, run in enumerate(runs, 1):
        key = run.tag
        if progress.has(key):
            log(f"Individual resume: skip {key}")
            continue
        # If a prior process died after writing rows but before marking the
        # unit complete, remove those partial rows before recomputation.
        remove_csv_rows(
            functional_csv,
            lambda r, run=run: (
                r.get("task") == run.task
                and int(r.get("replicate", -1)) == run.replicate
            ),
        )
        remove_csv_rows(
            cutoff_csv,
            lambda r, run=run: (
                r.get("task") == run.task
                and int(r.get("replicate", -1)) == run.replicate
            ),
        )
        log(f"Individual functional {idx}/{len(runs)}: {key}")
        ev = CandidateEvaluator(
            core, run, metas[run.task], modules, store,
            modeling_file, checkpoint, device,
            args.eval_batch_size, args.ram_reserve_gb, args.model_init_seed
        )
        try:
            rows, cutoff = functional_curve_for_run(
                ev, args.coarse_step, args.refine_step,
                args.target_functional_retention
            )
            append_csv(functional_csv, rows)
            append_csv(cutoff_csv, [cutoff])
            progress.mark(key)
        finally:
            ev.close()

    plot_individual(out_fig, energy_csv, functional_csv)
    return cutoff_csv


def _runs_by_task(runs: Sequence[RunDesc]) -> Dict[str, List[RunDesc]]:
    out = defaultdict(list)
    for r in runs:
        out[r.task].append(r)
    return {k: sorted(v, key=lambda x: x.replicate) for k, v in out.items()}


def _single_specs(candidate: RunDesc, by_task, tasks):
    for t in tasks:
        if t == candidate.task:
            continue
        for r in by_task[t]:
            yield [r]


def _full_specs(candidate: RunDesc, by_task, tasks, pairing_mode: str):
    other = [t for t in tasks if t != candidate.task]
    if pairing_mode == "matched":
        # Fast smoke option only: same replicate index where available.
        ctx = []
        for t in other:
            match = [r for r in by_task[t] if r.replicate == candidate.replicate]
            ctx.append(match[0] if match else by_task[t][0])
        yield ctx
    else:
        for combo in itertools.product(*(by_task[t] for t in other)):
            yield list(combo)


def _replicate_spec(candidate: RunDesc, by_task):
    peers = [r for r in by_task[candidate.task] if r.replicate != candidate.replicate]
    return peers[:1]


def run_cross(args, core, out_analysis, out_fig, runs, modules, store, cutoff_csv):
    qcuts = cutoff_map(cutoff_csv)
    missing = [r.key for r in runs if r.key not in qcuts]
    if missing:
        raise RuntimeError(f"Missing functional cutoffs for {missing}")

    by_task = _runs_by_task(runs)
    pa_csv = out_analysis / "principal_angle_spectra.csv"
    progress_path = out_analysis / "_progress_principal.json"
    if not args.resume:
        pa_csv.unlink(missing_ok=True)
        progress_path.unlink(missing_ok=True)
    progress = Progress(progress_path, args.resume, stage_signature(args, "cross"))
    cache_dir = out_analysis / "random_cache"

    def one_curve(candidate, contexts, ctype, use_random):
        cid = f"{ctype}|{candidate.tag}|{context_id(contexts)}"
        if progress.has(cid):
            return
        remove_csv_rows(pa_csv, lambda r, cid=cid: r.get("curve_id") == cid)
        log(f"Principal: {cid}")
        decs, pairs = geometry_for_comparison(
            store, candidate, contexts, modules, qcuts
        )
        rq = None
        if use_random:
            rq = matched_random_q99(
                cache_dir, store, candidate, contexts, modules, qcuts,
                args.random_n, args.random_seed
            )
            if len(rq) != len(pairs):
                # Context raw directions can occasionally have exact numerical
                # dependencies.  Fail rather than silently misalign order.
                raise RuntimeError(
                    f"{cid}: real/random principal length mismatch "
                    f"{len(pairs)} vs {len(rq)}"
                )
        rows = pa_rows_for_curve(cid, ctype, candidate, contexts, pairs, rq)
        append_csv(pa_csv, rows)
        progress.mark(cid)

    for candidate in runs:
        for contexts in _single_specs(candidate, by_task, args.tasks):
            one_curve(candidate, contexts, "single", True)
        for contexts in _full_specs(
            candidate, by_task, args.tasks, args.full_context_pairings
        ):
            one_curve(candidate, contexts, "full", True)
        peers = _replicate_spec(candidate, by_task)
        if peers:
            one_curve(candidate, peers, "replicate", False)

    plot_pa_for_type(out_fig, pa_csv, "single", "03_07_principal_single")
    plot_pa_for_type(out_fig, pa_csv, "full", "08_12_principal_full")
    return pa_csv


def run_overlap(args, core, out_analysis, out_fig, runs, modules, store, cutoff_csv):
    qcuts = cutoff_map(cutoff_csv)
    by_task = _runs_by_task(runs)
    overlap_csv = out_analysis / "overlapped_functional_spectra.csv"
    progress_path = out_analysis / "_progress_overlap.json"
    if not args.resume:
        overlap_csv.unlink(missing_ok=True)
        progress_path.unlink(missing_ok=True)
    progress = Progress(progress_path, args.resume, stage_signature(args, "overlap"))

    import torch
    device = torch.device(args.device)
    metas = {
        t: core.load_task_meta(t, Path(args.data_root) / TASK_FILES[t])
        for t in args.tasks
    }
    modeling_file = Path(args.modeling_file)
    checkpoint = Path(args.checkpoint)

    for ci, candidate in enumerate(runs, 1):
        log(f"Overlap candidate {ci}/{len(runs)}: {candidate.tag}")
        ev = CandidateEvaluator(
            core, candidate, metas[candidate.task], modules, store,
            modeling_file, checkpoint, device,
            args.eval_batch_size, args.ram_reserve_gb, args.model_init_seed
        )
        try:
            specs = []
            specs.extend(("single", x) for x in _single_specs(candidate, by_task, args.tasks))
            specs.extend(
                ("full", x) for x in _full_specs(
                    candidate, by_task, args.tasks, args.full_context_pairings
                )
            )
            peers = _replicate_spec(candidate, by_task)
            if peers:
                specs.append(("replicate", peers))

            for ctype, contexts in specs:
                cid = f"{ctype}|{candidate.tag}|{context_id(contexts)}"
                if progress.has(cid):
                    continue
                remove_csv_rows(
                    overlap_csv, lambda r, cid=cid: r.get("curve_id") == cid
                )
                log(f"Overlap: {cid}")
                curve = overlap_curve(
                    ev, store, contexts, modules, qcuts,
                    args.coarse_step, args.refine_step,
                    args.target_functional_retention
                )
                rows = []
                for r in curve:
                    rows.append({
                        "curve_id": cid,
                        "context_type": ctype,
                        "candidate_task": candidate.task,
                        "candidate_replicate": candidate.replicate,
                        "context_tasks": "+".join(x.task for x in contexts),
                        "context_replicates": "+".join(str(x.replicate) for x in contexts),
                        **r,
                    })
                append_csv(overlap_csv, rows)
                progress.mark(cid)
        finally:
            ev.close()

    # Compact decision table retained in V5.0: for every curve report the
    # first stable 99% point if reached, otherwise the maximum retention.
    all_rows = read_csv(overlap_csv)
    by_curve = defaultdict(list)
    for r in all_rows:
        by_curve[r["curve_id"]].append(r)
    summary_rows = []
    for cid, rr in sorted(by_curve.items()):
        rr = sorted(rr, key=lambda x: int(x["principal_dimensions"]))
        pts = [
            (float(x["principal_fraction"]), float(x["functional_retention"]))
            for x in rr
        ]
        qstar = stable_cutoff(pts, args.target_functional_retention)
        if qstar is not None:
            eligible = [
                x for x in rr
                if abs(float(x["principal_fraction"]) - qstar) < 1e-10
            ]
            hit = eligible[0]
            source = "stable_99pct"
        else:
            hit = max(rr, key=lambda x: float(x["functional_retention"]))
            source = "max_retention"
        first = rr[0]
        summary_rows.append({
            "curve_id": cid,
            "context_type": first["context_type"],
            "candidate_task": first["candidate_task"],
            "candidate_replicate": first["candidate_replicate"],
            "context_tasks": first["context_tasks"],
            "context_replicates": first["context_replicates"],
            "source": source,
            "principal_dimensions": hit["principal_dimensions"],
            "full_principal_dimensions": hit["full_principal_dimensions"],
            "functional_retention": hit["functional_retention"],
        })
    summary_csv = out_analysis / "overlapped_functional_summary.csv"
    if summary_csv.exists():
        summary_csv.unlink()
    append_csv(summary_csv, summary_rows)

    plot_overlap_for_type(out_fig, overlap_csv, "single", "13_17_overlap_single")
    plot_overlap_for_type(out_fig, overlap_csv, "full", "18_22_overlap_full")
    return overlap_csv


def run_sti(args, out_analysis, runs, modules, store):
    sti_csv = out_analysis / "paper_sti.csv"
    marker = out_analysis / "_progress_sti.json"
    sig = stage_signature(args, "sti")
    if sti_csv.is_file() and args.resume and marker.is_file():
        payload = load_json(marker)
        if str(payload.get("signature", "")) != sig:
            raise RuntimeError(
                "Stale STI resume marker. Rerun --mode sti --no-resume rather than mixing versions."
            )
        log("STI resume: certified existing paper_sti.csv kept")
        return sti_csv
    if not args.resume:
        sti_csv.unlink(missing_ok=True)
        marker.unlink(missing_ok=True)
    by_task = _runs_by_task(runs)
    rows = paper_sti_rows(args.tasks, by_task, modules, store)
    sti_csv.unlink(missing_ok=True)
    append_csv(sti_csv, rows)
    atomic_json(marker, {"signature": sig, "n_rows": len(rows)})
    return sti_csv


# =============================================================================
# Manifest / CLI
# =============================================================================

def build_parser():
    p = argparse.ArgumentParser(
        description="V5.0 task-adaptation geometry analyzer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--mode",
        choices=["selftest", "individual", "cross", "overlap", "sti", "all"],
        default="all",
    )
    p.add_argument(
        "--root",
        default="/omni-eeg-01/task calibration/results/tsv_labram_m3cv/tsv_labram_m3cv_v5_0",
    )
    p.add_argument(
        "--data-root",
        default="/omni-eeg-01/task calibration/dataset/m3cv",
    )
    p.add_argument(
        "--checkpoint",
        default="/omni-eeg-01/task calibration/LaBraM/checkpoints/labram-base.pth",
    )
    p.add_argument(
        "--modeling-file",
        default="/omni-eeg-01/task calibration/LaBraM/modeling_finetune.py",
    )
    p.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--ram-reserve-gb", type=float, default=4.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--model-init-seed", type=int, default=314159)

    p.add_argument("--coarse-step", type=float, default=0.05)
    p.add_argument("--refine-step", type=float, default=0.01)
    p.add_argument("--target-functional-retention", type=float, default=0.99)

    p.add_argument("--random-n", type=int, default=1000)
    p.add_argument("--random-seed", type=int, default=20260828)
    p.add_argument(
        "--full-context-pairings",
        choices=["all", "matched"],
        default="all",
        help="'all' is the V5.0 scientific protocol; 'matched' is only a faster smoke option.",
    )
    p.add_argument("--svd-cache-size", type=int, default=256)
    p.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    return p


def main():
    args = build_parser().parse_args()
    args.tasks = parse_csv_strs(args.tasks)

    if tuple(args.tasks) != DEFAULT_TASKS:
        raise ValueError(
            f"V5.0 scientific analysis requires the exact task order {DEFAULT_TASKS}; "
            f"received {tuple(args.tasks)}"
        )
    if abs(1.0 / args.coarse_step - round(1.0 / args.coarse_step)) > 1e-9:
        raise ValueError("--coarse-step must divide 1 exactly")
    if abs(1.0 / args.refine_step - round(1.0 / args.refine_step)) > 1e-9:
        raise ValueError("--refine-step must divide 1 exactly")
    if args.refine_step >= args.coarse_step:
        raise ValueError("--refine-step must be smaller than --coarse-step")
    if args.random_n < 1:
        raise ValueError("--random-n must be >=1")

    test = pure_geometry_selftest()
    log(f"Pure geometry self-test PASSED: {test}")
    if args.mode == "selftest":
        return

    core = import_core()
    root = Path(args.root)
    out_analysis = ensure_dir(root / "analysis_v5_0")
    out_fig = ensure_dir(root / "figures_v5_0")

    runs, manifest = discover_runs(root, args.tasks)
    verify_current_inputs(core, manifest, args)
    args._analysis_code_sha256 = analysis_code_sha256()
    args._final_manifest_sha256 = _sha256_file(root / "run_manifest.json")
    modules = discover_modules(runs[0].delta_path, core)
    store = SVDStore(runs, core, max_cache=args.svd_cache_size)

    atomic_json(out_analysis / "geometry_selftest.json", test)

    cutoff_csv = out_analysis / "functional_cutoffs_99pct.csv"

    if args.mode in ("individual", "all"):
        cutoff_csv = run_individual(
            args, core, root, out_analysis, out_fig,
            runs, modules, store, manifest
        )

    if args.mode in ("cross", "all"):
        if not cutoff_csv.is_file():
            raise RuntimeError(
                "Cross-task geometry requires functional_cutoffs_99pct.csv. "
                "Run --mode individual first."
            )
        run_cross(
            args, core, out_analysis, out_fig,
            runs, modules, store, cutoff_csv
        )

    if args.mode in ("overlap", "all"):
        if not cutoff_csv.is_file():
            raise RuntimeError(
                "Overlapped functional analysis requires functional cutoffs. "
                "Run --mode individual first."
            )
        run_overlap(
            args, core, out_analysis, out_fig,
            runs, modules, store, cutoff_csv
        )

    if args.mode in ("sti", "all"):
        run_sti(args, out_analysis, runs, modules, store)

    # Re-render figures when all prerequisite CSVs are available.
    e = out_analysis / "individual_energy_spectrum.csv"
    f = out_analysis / "individual_functional_spectrum.csv"
    pa = out_analysis / "principal_angle_spectra.csv"
    ov = out_analysis / "overlapped_functional_spectra.csv"
    if e.is_file() and f.is_file():
        plot_individual(out_fig, e, f)
    if pa.is_file():
        plot_pa_for_type(out_fig, pa, "single", "03_07_principal_single")
        plot_pa_for_type(out_fig, pa, "full", "08_12_principal_full")
    if ov.is_file():
        plot_overlap_for_type(out_fig, ov, "single", "13_17_overlap_single")
        plot_overlap_for_type(out_fig, ov, "full", "18_22_overlap_full")

    artifacts = {}
    for p in sorted(out_analysis.glob("*.csv")):
        artifacts[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()

    analysis_manifest = {
        "version": VERSION,
        "root": str(root),
        "analysis_code_sha256": args._analysis_code_sha256,
        "final_manifest_sha256": args._final_manifest_sha256,
        "composite_protocol_fingerprint": manifest.get("composite_protocol_fingerprint"),
        "run_provenance": manifest.get("run_provenance", {}),
        "tasks": args.tasks,
        "n_runs": len(runs),
        "n_modules": len(modules),
        "coarse_step": args.coarse_step,
        "refine_step": args.refine_step,
        "target_functional_retention": args.target_functional_retention,
        "matched_random_n": args.random_n,
        "full_context_pairings": args.full_context_pairings,
        "geometry": "Frobenius rank-1 Z=u v^T subspaces",
        "important_boundaries": [
            "All ten task/replicate runs are independent fine-tunes from one W0; V5.0 provenance reuses seven certified legacy runs and newly trains three runs.",
            "Functional spectra evaluate the same pooled fitted task dataset; they are retention audits, not held-out generalization.",
            "Only the 72 analyzed matrices are reconstructed; candidate head and non-72 adapted parameters are preserved.",
            "q=0 places all 72 analyzed matrices at W0.",
            "Principal angles are computed between full rank-1 matrix directions Z=u v^T, not separate U-side/V-side spaces.",
            "Single Context has exactly one other task; Full Context has all four other tasks.",
            "Matched Random Reference preserves matrix ambient shapes and exact retained ranks.",
            "Replicate Reference uses the same-task second independent training run.",
            "STI uses the original paper-aligned k=floor(r_full/T) compression rule.",
        ],
        "artifacts_sha256": artifacts,
        "selftest": test,
    }
    atomic_json(out_analysis / "analysis_manifest_v5_0.json", analysis_manifest)
    atomic_json(out_analysis / "_ANALYSIS_V5_0_SUCCESS.json", {
        "version": VERSION,
        "manifest_sha256": hashlib.sha256(
            (out_analysis / "analysis_manifest_v5_0.json").read_bytes()
        ).hexdigest(),
    })
    log(f"V5.0 analysis complete: {out_analysis}")


if __name__ == "__main__":
    main()
