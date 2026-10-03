#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LaBraM x M3CV revised four-spectrum analysis (analysis only)
============================================================
Version: v6.0.4-fast-profile

Implements the 2026-09-07 Research Proposal:
  Individual / Shared x Energy / Function

IMPORTANT
---------
This script NEVER fine-tunes.  It consumes the already-certified v5.0 final
artifacts (adapted_checkpoint.pth + delta_weights.npz + metadata.json) and only
performs post-hoc decomposition, reconstruction/inference, random baselines,
STI, tables and figures.

Scientific contract implemented here
------------------------------------
1. Atomic component: Z_i = u_i v_i^T, always kept intact.
2. Reported spectrum points are sampled on the ABSOLUTE VERTICAL axis, not on
   a fixed component-fraction grid. Energy curves use absolute targets
   0,0.05,...,1 (plus 0.99 where reachable); functional curves use absolute
   raw-bACC targets 0,0.05,...,1 within the observed range, plus mandatory
   baseline/full/cutoff extrema. A sparse 5% x-grid is used only as an internal
   bracketing/probing device, never as the reported spectrum sampling.
3. Functional reconstruction changes ONLY the 72 analyzed matrices.  The
   candidate head and every non-analyzed fine-tuned parameter stay adapted.
4. Context is constructed only within the identical matrix position.
5. Componentwise shared energy is
       eps_sh_i = sigma_i^2 * ||P_context Z_i||_F^2.
   No independent "sharedness" score is introduced.
6. Shared Energy globally sorts the candidate functional components by eps_sh.
   Its denominator is the ORIGINAL candidate functional energy sum sigma_i^2,
   so the endpoint Gamma is not forced to 1.
7. Shared Functional uses exactly the same ordering, but reconstructs with the
   candidate's ORIGINAL sigma_i Z_i.  Projected operators NEVER enter the model.
8. Full context is the union of the other four tasks' functional Z operators,
   orthogonalized in Frobenius matrix space.  We compute this projection through
   the operator Gram matrix, which is algebraically identical to explicit
   vectorization + SVD/QR, but avoids 40k/160k-dimensional flattened matrices.
9. STI is the paper-aligned layer/matrix statistic with k=floor(r/T), T=5.
10. Pooled bACC is a functional-retention audit, not held-out generalization.

Expected final-run layout (same contract as the existing v0.3.x/v5.0 runner):
  ROOT/
    _FINAL_SUCCESS.json
    run_manifest.json
    preflight.json
    runs/<Task>/repXX/
      metadata.json
      _SUCCESS.json
      delta_weights.npz
      adapted_checkpoint.pth
      epoch_metrics.csv

The exact training runner used to create v5.0 must be supplied through
--core-script (or discovered automatically).  The core is used only for exact
LaBraM construction/data/evaluation compatibility; no training entry point is
called.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import gzip
import hashlib
import importlib.util
import inspect
import itertools
import json
import math
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

VERSION = "v6.0.6-v5-core-compatible"
EPS = 1e-12
TASK_ORDER_DEFAULT = ("Rest", "Motor", "P300", "SSS", "TS")
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
# Runtime sampling policy is configured from --profile.
# "fast" is designed for an overnight + half-day analysis window: sparse
# hidden x-probes, 0.10 raw-bACC vertical targets, and 2% local cutoff
# refinement. "full" restores the denser 5% probes / 0.05 bACC / 1% refinement.
FULL_COARSE_FRACS = tuple(round(i / 20, 8) for i in range(21))
FAST_COARSE_FRACS = (0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.85, 1.0)
FULL_FINE_FRACS = tuple(round(i / 100, 8) for i in range(101))
FAST_FINE_FRACS = tuple(round(i / 50, 8) for i in range(51))
ENERGY_VERTICAL_TARGETS = tuple(sorted(set(tuple(round(i * 0.05, 8) for i in range(21)) + (0.99,))))
COARSE_FRACS = FAST_COARSE_FRACS
FINE_FRACS = FAST_FINE_FRACS
BACC_VERTICAL_TARGETS = tuple(round(i * 0.10, 8) for i in range(11))
SAMPLING_POLICY = "absolute-vertical-axis-v3-fast"

def configure_runtime_profile(args):
    """Configure speed/rigor trade-offs without changing the four-spectrum definitions."""
    global COARSE_FRACS, FINE_FRACS, BACC_VERTICAL_TARGETS, SAMPLING_POLICY
    if args.profile == "full":
        COARSE_FRACS = FULL_COARSE_FRACS
        FINE_FRACS = FULL_FINE_FRACS
        BACC_VERTICAL_TARGETS = tuple(round(i * 0.05, 8) for i in range(21))
        SAMPLING_POLICY = "absolute-vertical-axis-v3-full"
        if args.eval_batch_size is None: args.eval_batch_size = 64
        if args.shared_random_n is None: args.shared_random_n = 1000
        if args.sti_random_n is None: args.sti_random_n = 1000
    else:
        COARSE_FRACS = FAST_COARSE_FRACS
        FINE_FRACS = FAST_FINE_FRACS
        BACC_VERTICAL_TARGETS = tuple(round(i * 0.10, 8) for i in range(11))
        SAMPLING_POLICY = "absolute-vertical-axis-v3-fast"
        if args.eval_batch_size is None: args.eval_batch_size = 128
        if args.shared_random_n is None: args.shared_random_n = 128
        if args.sti_random_n is None: args.sti_random_n = 128
    return args


# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------

def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def ensure_dir(p: Path | str) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_json(p: Path | str):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def json_safe(x):
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [json_safe(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def dump_json(p: Path | str, payload) -> None:
    p = Path(p)
    ensure_dir(p.parent)
    tmp = p.with_name(p.name + f".tmp.{os.getpid()}.{time.time_ns()}")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(json_safe(payload), f, ensure_ascii=False, indent=2)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, p)
    finally:
        tmp.unlink(missing_ok=True)


def read_csv(p: Path | str) -> List[Dict[str, str]]:
    with open(p, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(p: Path | str, rows: Sequence[Mapping[str, object]],
              fieldnames: Optional[Sequence[str]] = None) -> None:
    p = Path(p); ensure_dir(p.parent); rows = list(rows)
    if fieldnames is None:
        fields, seen = [], set()
        for r in rows:
            for k in r:
                if k not in seen:
                    seen.add(k); fields.append(k)
        fieldnames = fields
    tmp = p.with_name(p.name + f".tmp.{os.getpid()}.{time.time_ns()}")
    try:
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({k: json_safe(r.get(k, "")) for k in fieldnames})
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, p)
    finally:
        tmp.unlink(missing_ok=True)



def sampling_rows_compatible(rows:Sequence[Mapping])->bool:
    return bool(rows) and all(
        str(r.get("sampling_policy", ""))==SAMPLING_POLICY and
        str(r.get("analysis_version", ""))==VERSION
        for r in rows
    )

def write_csv_gz(p: Path | str, rows: Sequence[Mapping[str, object]],
                 fieldnames: Optional[Sequence[str]] = None) -> None:
    p = Path(p); ensure_dir(p.parent); rows = list(rows)
    if fieldnames is None:
        fields, seen = [], set()
        for r in rows:
            for k in r:
                if k not in seen:
                    seen.add(k); fields.append(k)
        fieldnames = fields
    tmp = p.with_name(p.name + f".tmp.{os.getpid()}.{time.time_ns()}")
    try:
        with gzip.open(tmp, "wt", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({k: json_safe(r.get(k, "")) for k in fieldnames})
        os.replace(tmp, p)
    finally:
        tmp.unlink(missing_ok=True)


def sha256_file(p: Path | str, block=1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(block)
            if not b: break
            h.update(b)
    return h.hexdigest()


def stable_seed(base: int, *parts) -> int:
    text = f"{int(base)}|" + "|".join(str(x) for x in parts)
    d = hashlib.sha256(text.encode()).digest()
    return int.from_bytes(d[:8], "little") % (2**31 - 1)


def parse_csv_strs(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def sanitize_module_key(module: str) -> str:
    return module.replace("/", "__")


def unsanitize_module_key(key: str) -> str:
    return key.replace("__", "/")


def module_layer_role(module: str) -> Tuple[int, str]:
    left, role = module.split("/")
    return int(left[1:]), role


def qstats(a: Sequence[float]) -> Dict[str, float]:
    x = np.asarray(a, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {k: float("nan") for k in ("min","q01","q05","median","q95","q99","max","mean")}
    return {
        "min": float(np.min(x)), "q01": float(np.quantile(x, .01)),
        "q05": float(np.quantile(x, .05)), "median": float(np.median(x)),
        "q95": float(np.quantile(x, .95)), "q99": float(np.quantile(x, .99)),
        "max": float(np.max(x)), "mean": float(np.mean(x)),
    }


def import_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# -----------------------------------------------------------------------------
# Exact training-core adapter
# -----------------------------------------------------------------------------

def _candidate_core_paths(root: Path) -> List[Path]:
    """Locate the frozen mature numerical core used by the certified V5 final.

    V5.0 did not have a separate unified five-task trainer.  It reused
    run_tsv_labram_m3cv_v0_3_4.py and patched the five-task registry in the
    orchestrator/analyzer.  Keep the same compatibility contract here.
    """
    home = Path.home()
    cands = [
        home / "run_tsv_labram_m3cv_v0_3_4.py",
        Path.cwd() / "run_tsv_labram_m3cv_v0_3_4.py",
        root.parent / "run_tsv_labram_m3cv_v0_3_4.py",
    ]
    out, seen = [], set()
    for p in cands:
        p = p.expanduser().resolve()
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def load_core(explicit: str, root: Path, tasks: Sequence[str]):
    """Load the frozen v0.3.4 numerical core and patch the V5 task registry.

    This mirrors analyze_task_geometry_v5_0.py exactly: the old core supplies
    model loading, data loaders, evaluation and 72-matrix extraction; Rest is
    enabled by patching TASK_FILES/TASK_CLASS_NAMES/EXPECTED_CLASSES.
    """
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(p)
    else:
        p = next((x for x in _candidate_core_paths(root) if x.is_file()), None)
        if p is None:
            raise FileNotFoundError(
                "Could not find the frozen mature core run_tsv_labram_m3cv_v0_3_4.py. "
                "Place it in ~/ or pass --core-script explicitly."
            )

    spec = importlib.util.spec_from_file_location("labram_training_core_v034", p)
    if spec is None or spec.loader is None:
        raise ImportError(p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    required = [
        "build_labram_backbone", "TaskClassifier", "extract_tsv_weights",
        "class_weights_for_meta", "evaluate", "load_task_meta",
    ]
    missing = [x for x in required if not hasattr(mod, x)]
    if missing:
        raise RuntimeError(f"Core {p} lacks required analysis API: {missing}")

    # Exact V5 compatibility mechanism used by the previous certified analyzer.
    mod.TASK_FILES = dict(TASK_FILES)
    mod.TASK_CLASS_NAMES = {k: dict(v) for k, v in TASK_CLASS_NAMES.items()}
    mod.EXPECTED_CLASSES = {k: len(v) for k, v in TASK_CLASS_NAMES.items()}

    unknown = [t for t in tasks if t not in mod.TASK_CLASS_NAMES]
    if unknown:
        raise RuntimeError(f"Patched training core still lacks task definitions: {unknown}")

    log(f"Using frozen V5 mature core + five-task registry patch: {p}")
    return mod, p


def verify_core_provenance(core_path: Path, manifest: Mapping[str, object], preflight: Mapping[str, object]) -> None:
    """Check that analysis uses the same mature core bytes certified by V5."""
    protocol = manifest.get("protocol", preflight.get("protocol", {})) or {}
    expected = str(protocol.get("runner_sha256", ""))
    if not expected:
        cp = manifest.get("code_provenance", {}) or {}
        expected = str(cp.get("mature_core_sha256", ""))
    if expected:
        actual = sha256_file(Path(core_path))
        if actual != expected:
            raise RuntimeError(
                f"Mature core hash mismatch: analysis core {actual} != V5-certified {expected}"
            )


def make_eval_loader_compat(core, meta, batch_size: int, device, reserve_gb: float):
    task_data = None
    if hasattr(core, "load_task_into_memory"):
        task_data = core.load_task_into_memory(meta, reserve_gb=reserve_gb)
    if hasattr(core, "make_eval_loader"):
        fn = core.make_eval_loader
        try:
            return fn(meta, batch_size, device, task_data=task_data), task_data
        except TypeError:
            try: return fn(meta, batch_size, device), task_data
            except TypeError: pass
    if task_data is not None and hasattr(core, "InMemoryBatchLoader"):
        return core.InMemoryBatchLoader(task_data, batch_size=batch_size, shuffle=False, seed=0), task_data
    if hasattr(core, "M3CV4sDataset"):
        import torch
        from torch.utils.data import DataLoader
        ds = core.M3CV4sDataset(meta)
        return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0,
                          pin_memory=(device.type == "cuda")), ds
    raise RuntimeError("Training core provides no compatible evaluation loader")


def assign_tsv_matrices(model, matrices: Mapping[str, np.ndarray]) -> None:
    import torch
    with torch.no_grad():
        for l, block in enumerate(model.blocks):
            q = torch.as_tensor(matrices[f"L{l:02d}/Q"], device=block.attn.qkv.weight.device,
                                dtype=block.attn.qkv.weight.dtype)
            k = torch.as_tensor(matrices[f"L{l:02d}/K"], device=block.attn.qkv.weight.device,
                                dtype=block.attn.qkv.weight.dtype)
            v = torch.as_tensor(matrices[f"L{l:02d}/V"], device=block.attn.qkv.weight.device,
                                dtype=block.attn.qkv.weight.dtype)
            block.attn.qkv.weight.copy_(torch.cat([q,k,v], dim=0))
            for role, param in (("O", block.attn.proj.weight),
                                ("fc1", block.mlp.fc1.weight),
                                ("fc2", block.mlp.fc2.weight)):
                x = torch.as_tensor(matrices[f"L{l:02d}/{role}"], device=param.device, dtype=param.dtype)
                param.copy_(x)


# -----------------------------------------------------------------------------
# Run discovery / integrity
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class RunDesc:
    task: str
    replicate: int
    run_dir: Path
    metadata_path: Path
    delta_path: Path
    checkpoint_path: Path
    success_path: Path
    protocol_fingerprint: str
    @property
    def key(self): return (self.task, self.replicate)
    @property
    def tag(self): return f"{self.task}_rep{self.replicate:02d}"


@dataclass
class SVDRec:
    U: np.ndarray
    s: np.ndarray
    V: np.ndarray
    d_out: int
    d_in: int
    recon_relerr: float
    orth_err: float
    @property
    def r(self): return len(self.s)


def _manifest_run_index(manifest: dict) -> Dict[Tuple[str,int], dict]:
    """Index per-run provenance rows from a homogeneous or composite manifest."""
    out={}
    for row in manifest.get("runs", []) or []:
        try:
            key=(str(row.get("task")), int(row.get("replicate")))
        except Exception:
            continue
        if key in out:
            raise RuntimeError(f"Duplicate per-run manifest entry {key}")
        out[key]=row
    return out


def _declared_run_fingerprint(row: Optional[dict]) -> str:
    if not row:
        return ""
    for k in ("protocol_fingerprint", "source_protocol_fingerprint",
              "run_protocol_fingerprint", "fingerprint"):
        v=row.get(k)
        if v not in (None, ""):
            return str(v)
    return ""


def _all_declared_fingerprints(obj) -> set[str]:
    """Collect fingerprint-like string values explicitly recorded in a manifest."""
    out=set()
    def walk(x):
        if isinstance(x, dict):
            for k,v in x.items():
                if "fingerprint" in str(k).lower() and isinstance(v,(str,int,float)) and str(v):
                    out.add(str(v))
                walk(v)
        elif isinstance(x, list):
            for v in x: walk(v)
    walk(obj)
    return out


def discover_runs(root: Path, tasks: Sequence[str]) -> Tuple[List[RunDesc], dict, dict]:
    mp = root / "run_manifest.json"; sp = root / "_FINAL_SUCCESS.json"; pp = root / "preflight.json"
    for p in (mp, sp, pp):
        if not p.is_file(): raise FileNotFoundError(p)
    manifest = load_json(mp); success = load_json(sp); preflight = load_json(pp)
    if success.get("run_manifest_sha256") and success["run_manifest_sha256"] != sha256_file(mp):
        raise RuntimeError("Top-level run_manifest hash mismatch")

    # IMPORTANT: v5.0 is a certified COMPOSITE final.  Its top-level fingerprint
    # identifies the composite contract, while inherited Motor/P300/SSS/TS runs
    # can legitimately retain their legacy per-run fingerprints.  Therefore the
    # top-level fingerprint is checked only against the top-level success marker.
    # Per-run metadata/_SUCCESS/checkpoint fingerprints are checked against each
    # other and, when available, against the corresponding manifest['runs'] row.
    fp = str(manifest.get("protocol_fingerprint", ""))
    if not fp or str(success.get("protocol_fingerprint", fp)) != fp:
        raise RuntimeError("Top-level protocol fingerprint mismatch")

    mtasks = [str(x) for x in manifest.get("tasks", [])]
    if not set(tasks).issubset(set(mtasks)):
        raise RuntimeError(f"Requested tasks {tasks} are not in manifest tasks {mtasks}")
    nrep = int(manifest.get("replicates", 0))
    expected = {(t,r) for t in tasks for r in range(1,nrep+1)}
    mrun = _manifest_run_index(manifest)
    declared_fps = _all_declared_fingerprints(manifest)

    runs=[]; seen=set()
    for mpath in sorted((root/"runs").glob("*/rep*/metadata.json")):
        meta=load_json(mpath); task=str(meta.get("task")); rep=int(meta.get("replicate",-1))
        if task not in tasks: continue
        key=(task,rep)
        if key in seen: raise RuntimeError(f"Duplicate run {key}")
        seen.add(key); rd=mpath.parent
        delta=rd/"delta_weights.npz"; ckpt=rd/"adapted_checkpoint.pth"; suc=rd/"_SUCCESS.json"
        for p in (delta,ckpt,suc):
            if not p.is_file(): raise FileNotFoundError(p)
        sr=load_json(suc)

        meta_fp=str(meta.get("protocol_fingerprint", ""))
        success_fp=str(sr.get("protocol_fingerprint", ""))
        if not meta_fp or not success_fp:
            raise RuntimeError(f"{key}: missing per-run protocol fingerprint")
        if meta_fp != success_fp:
            raise RuntimeError(f"{key}: metadata/_SUCCESS fingerprint disagreement: {meta_fp} != {success_fp}")

        declared_fp=_declared_run_fingerprint(mrun.get(key))
        if declared_fp and meta_fp != declared_fp:
            raise RuntimeError(f"{key}: per-run fingerprint disagrees with composite manifest: {meta_fp} != {declared_fp}")
        if not declared_fp and meta_fp not in declared_fps:
            raise RuntimeError(f"{key}: run fingerprint {meta_fp} is not declared anywhere in the final manifest")
        # Homogeneous manifests historically used the top-level fingerprint for
        # every run.  If there is no per-run provenance row, retain that strict
        # behavior rather than silently accepting an arbitrary mismatch.
        if not mrun and meta_fp != fp:
            raise RuntimeError(f"{key}: run fingerprint {meta_fp} != top-level {fp} and no per-run composite provenance is declared")

        if sr.get("metadata_sha256") and sr["metadata_sha256"] != sha256_file(mpath): raise RuntimeError(f"{key}: metadata hash mismatch")
        if sr.get("delta_weights_sha256") and sr["delta_weights_sha256"] != sha256_file(delta): raise RuntimeError(f"{key}: delta hash mismatch")
        if sr.get("adapted_checkpoint_sha256") and sr["adapted_checkpoint_sha256"] != sha256_file(ckpt): raise RuntimeError(f"{key}: checkpoint hash mismatch")

        # If the composite manifest carries artifact digests too, cross-check
        # them against the certified per-run success marker.
        row=mrun.get(key) or {}
        artifacts=row.get("artifacts", {}) if isinstance(row,dict) else {}
        for name, success_key in (("delta_weights_sha256","delta_weights_sha256"),
                                  ("adapted_checkpoint_sha256","adapted_checkpoint_sha256"),
                                  ("metadata_sha256","metadata_sha256")):
            mv=artifacts.get(name) if isinstance(artifacts,dict) else None
            sv=sr.get(success_key)
            if mv and sv and str(mv)!=str(sv):
                raise RuntimeError(f"{key}: manifest/per-run artifact hash disagreement for {name}")

        runs.append(RunDesc(task,rep,rd,mpath,delta,ckpt,suc,meta_fp))

    observed={r.key for r in runs}
    if observed != expected:
        raise RuntimeError(f"Final run grid mismatch. Missing={sorted(expected-observed)} extra={sorted(observed-expected)}")
    if mrun and not expected.issubset(set(mrun)):
        raise RuntimeError(f"Composite manifest lacks per-run provenance rows for {sorted(expected-set(mrun))}")
    order={t:i for i,t in enumerate(tasks)}
    runs.sort(key=lambda r:(order[r.task],r.replicate))
    return runs, manifest, preflight


def discover_modules(delta_path: Path) -> List[str]:
    with np.load(delta_path, allow_pickle=False) as z:
        modules = sorted(unsanitize_module_key(k) for k in z.files)
    if len(modules) != N_MODULES:
        raise RuntimeError(f"Expected 72 matrices, found {len(modules)}")
    expected={f"L{l:02d}/{r}" for l in range(12) for r in MODULE_ROLES}
    if set(modules) != expected:
        raise RuntimeError(f"72-matrix map mismatch: missing={sorted(expected-set(modules))[:10]}")
    return modules


def audit_delta_archives(runs:Sequence[RunDesc], modules:Sequence[str])->dict:
    """Fail-closed structural audit of all ten delta archives.

    Confirms identical 72-matrix keys, proposal-specified shapes, finite values,
    and a common matrix map across every task/replicate before any analysis.
    """
    expected_shapes={"Q":(200,200),"K":(200,200),"V":(200,200),"O":(200,200),
                     "fc1":(800,200),"fc2":(200,800)}
    ref={m:expected_shapes[module_layer_role(m)[1]] for m in modules}
    dtypes=defaultdict(set)
    for run in runs:
        with np.load(run.delta_path,allow_pickle=False) as z:
            keys={unsanitize_module_key(k) for k in z.files}
            if keys!=set(modules):
                raise RuntimeError(f"{run.key}: delta matrix map mismatch")
            for m in modules:
                a=np.asarray(z[sanitize_module_key(m)])
                if tuple(a.shape)!=tuple(ref[m]):
                    raise RuntimeError(f"{run.key} {m}: shape {a.shape} != expected {ref[m]}")
                if not np.issubdtype(a.dtype,np.floating):
                    raise RuntimeError(f"{run.key} {m}: non-floating delta dtype {a.dtype}")
                if not np.isfinite(a).all():
                    raise FloatingPointError(f"{run.key} {m}: non-finite delta values")
                dtypes[m].add(str(a.dtype))
    return {
        "status":"PASSED",
        "n_runs":len(runs),
        "n_matrices_per_run":len(modules),
        "matrix_shapes":{m:list(ref[m]) for m in modules},
        "dtypes_by_matrix":{m:sorted(v) for m,v in dtypes.items()},
    }


def protocol_paths(manifest: dict, preflight: dict, tasks: Sequence[str]):
    protocol = manifest.get("protocol", preflight.get("protocol", {}))
    data = protocol.get("data", {})
    paths={}
    for t in tasks:
        p = data.get(t, {}).get("path")
        if not p: raise RuntimeError(f"Protocol does not record dataset path for {t}")
        paths[t]=Path(p)
    ckpt = protocol.get("checkpoint", {}).get("path")
    mf = protocol.get("modeling_file", {})
    modeling = mf.get("path") if isinstance(mf, dict) else mf
    if not ckpt or not modeling:
        raise RuntimeError("Protocol does not record pretrained checkpoint/modeling file paths")
    chans=[int(x) for x in protocol.get("input_chans", [])]
    if not chans: raise RuntimeError("Protocol does not record input_chans")
    model_init_seed=int(protocol.get("model_init_seed",314159))
    expected_w0=str(protocol.get("checkpoint",{}).get("base_full_backbone_digest", manifest.get("base_full_backbone_digest","")))
    for t,p in paths.items():
        if not p.is_file():raise FileNotFoundError(f"Dataset path for {t} does not exist: {p}")
    if not Path(ckpt).is_file():raise FileNotFoundError(f"Pretrained checkpoint missing: {ckpt}")
    if not Path(modeling).is_file():raise FileNotFoundError(f"Modeling file missing: {modeling}")
    return paths, Path(ckpt), Path(modeling), chans, model_init_seed, expected_w0


# -----------------------------------------------------------------------------
# SVD and operator-space projection algebra
# -----------------------------------------------------------------------------

def compute_svd(delta: np.ndarray) -> SVDRec:
    A=np.asarray(delta,dtype=np.float64)
    U,s,Vh=np.linalg.svd(A,full_matrices=False)
    recon=U @ (s[:,None]*Vh)
    denom=max(float(np.linalg.norm(A)),EPS)
    rel=float(np.linalg.norm(recon-A)/denom)
    ou=float(np.linalg.norm(U.T@U-np.eye(U.shape[1]),ord="fro"))
    ov=float(np.linalg.norm(Vh@Vh.T-np.eye(Vh.shape[0]),ord="fro"))
    return SVDRec(U,s,Vh.T,A.shape[0],A.shape[1],rel,max(ou,ov))


def load_all_svds(runs: Sequence[RunDesc], modules: Sequence[str]) -> Dict[Tuple[str,int,str],SVDRec]:
    out={}
    for ri,r in enumerate(runs,1):
        with np.load(r.delta_path,allow_pickle=False) as z:
            for m in modules:
                A=np.asarray(z[sanitize_module_key(m)],dtype=np.float64)
                out[(r.task,r.replicate,m)] = compute_svd(A)
        log(f"SVD cache {ri}/{len(runs)}: {r.tag}")
    return out


def k_for_frac(r: int, q: float) -> int:
    if q <= 0: return 0
    return min(int(r), int(math.ceil(float(q)*int(r)-1e-15)))


def actual_count(modules: Sequence[str], recs: Mapping[str,SVDRec], q: float) -> int:
    return sum(k_for_frac(recs[m].r,q) for m in modules)


def operator_gram(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Gram of columns vec(u_j v_j^T): G=(U^T U) Hadamard (V^T V)."""
    if U.shape[1] != V.shape[1]: raise ValueError("U/V context column mismatch")
    return (U.T @ U) * (V.T @ V)


def projection_fractions_rank1(candidate_U: np.ndarray, candidate_V: np.ndarray,
                               context_parts: Sequence[Tuple[np.ndarray,np.ndarray]],
                               ambient_dim: int, ortho_rtol: float = 0.0):
    """Return ||P_S Z_i||_F^2 for candidate rank-one unit operators.

    This is exactly b_i^T G^+ b_i in Frobenius operator space.  It is equivalent
    to explicitly flattening every Z into R^(d_out*d_in), constructing the union
    subspace and projecting, but uses only the much smaller operator Gram matrix.
    """
    kc=candidate_U.shape[1]
    if candidate_V.shape[1]!=kc: raise ValueError("candidate U/V mismatch")
    if not context_parts or sum(x[0].shape[1] for x in context_parts)==0:
        return np.zeros(kc,dtype=np.float64),0,0.0
    Uctx=np.concatenate([x[0] for x in context_parts],axis=1)
    Vctx=np.concatenate([x[1] for x in context_parts],axis=1)
    G=operator_gram(Uctx,Vctx)
    G=(G+G.T)*0.5
    ew,EV=np.linalg.eigh(G)
    ew=np.maximum(ew,0.0)
    smax=math.sqrt(float(np.max(ew))) if len(ew) else 0.0
    if ortho_rtol>0:
        sigma_tol=float(ortho_rtol)*smax
    else:
        sigma_tol=max(int(ambient_dim),G.shape[0]) * np.finfo(np.float64).eps * max(smax,1.0)
    eval_tol=sigma_tol*sigma_tol
    keep=ew>eval_tol
    rank=int(np.sum(keep))
    if rank==0:
        return np.zeros(kc,dtype=np.float64),0,float(sigma_tol)
    E=EV[:,keep]; inv=1.0/ew[keep]
    B=(Uctx.T@candidate_U)*(Vctx.T@candidate_V)
    C=E.T@B
    vals=np.sum((C*C)*inv[:,None],axis=0)
    if np.min(vals)<-1e-8 or np.max(vals)>1+1e-6:
        raise RuntimeError(f"Projection energy fraction escaped [0,1]: min={vals.min()} max={vals.max()}")
    vals=np.clip(vals,0.0,1.0)
    return vals,rank,float(sigma_tol)


def synthetic_algebra_selftest() -> dict:
    rng=np.random.default_rng(20260907)
    # identical single-context operator basis -> score 1 for its own columns
    A,_=np.linalg.qr(rng.standard_normal((7,3)),mode="reduced")
    B,_=np.linalg.qr(rng.standard_normal((5,3)),mode="reduced")
    vals,rank,_=projection_fractions_rank1(A,B,[(A,B)],ambient_dim=35)
    if not np.allclose(vals,1,atol=1e-10): raise AssertionError(vals)
    if rank!=3: raise AssertionError(rank)
    # orthogonal candidate/context rank-one operators -> 0
    U=np.eye(4)[:,:2]; V=np.eye(4)[:,:2]
    U2=np.eye(4)[:,2:4]; V2=np.eye(4)[:,2:4]
    vals2,_,_=projection_fractions_rank1(U,V,[(U2,V2)],ambient_dim=16)
    if np.max(np.abs(vals2))>1e-12: raise AssertionError(vals2)
    # full union nests its single constituent
    C,_=np.linalg.qr(rng.standard_normal((7,2)),mode="reduced")
    D,_=np.linalg.qr(rng.standard_normal((5,2)),mode="reduced")
    s1,_,_=projection_fractions_rank1(A,B,[(C,D)],ambient_dim=35)
    sf,_,_=projection_fractions_rank1(A,B,[(C,D),(A[:,:1],B[:,:1])],ambient_dim=35)
    if np.min(sf-s1)<-1e-9: raise AssertionError("nesting failed")
    return {"status":"passed","identical_min":float(vals.min()),"orthogonal_max":float(vals2.max()),
            "nesting_min_gain":float(np.min(sf-s1))}


# -----------------------------------------------------------------------------
# Model evaluator / reconstruction
# -----------------------------------------------------------------------------

class CandidateEvaluator:
    def __init__(self, core, run: RunDesc, meta, manifest: dict, checkpoint: Path,
                 modeling_file: Path, input_chans: Sequence[int], model_init_seed: int,
                 expected_w0: str, modules: Sequence[str], svds: Mapping[Tuple[str,int,str],SVDRec],
                 device, eval_batch_size: int, reserve_gb: float, verify_full_replay: bool=True):
        import torch
        self.core=core; self.run=run; self.meta=meta; self.modules=list(modules); self.device=device
        self.svds=svds; self.task_data=None
        backbone,_,base_digest=core.build_labram_backbone(modeling_file,checkpoint,device,model_init_seed)
        if expected_w0 and str(base_digest)!=str(expected_w0):
            raise RuntimeError(f"Current W0 digest != training W0 digest for {run.key}")
        self.base_tsv={k:v.numpy().astype(np.float64,copy=True) for k,v in core.extract_tsv_weights(backbone).items()}
        saved=torch.load(run.checkpoint_path,map_location="cpu",weights_only=False)
        # Check the checkpoint against THIS run's provenance fingerprint.
        # v5.0 is composite, so legacy runs need not equal the top-level FP.
        fp=str(run.protocol_fingerprint)
        if fp and saved.get("protocol_fingerprint") and str(saved["protocol_fingerprint"])!=fp:
            raise RuntimeError(f"{run.key}: checkpoint fingerprint mismatch: {saved.get('protocol_fingerprint')} != {fp}")
        if expected_w0 and saved.get("base_full_backbone_digest") and str(saved["base_full_backbone_digest"])!=str(expected_w0):
            raise RuntimeError(f"{run.key}: checkpoint W0 digest mismatch")
        saved_chans=[int(x) for x in saved.get("input_chans",input_chans)]
        if list(saved_chans)!=list(input_chans): raise RuntimeError(f"{run.key}: channel mapping mismatch")
        backbone.load_state_dict(saved["backbone"],strict=True)
        self.backbone=backbone
        self.classifier=core.TaskClassifier(backbone,int(load_json(run.metadata_path)["num_classes"]),saved_chans,head_seed=0).to(device)
        self.classifier.head.load_state_dict(saved["head"],strict=True)
        self.adapted_tsv={k:v.numpy().astype(np.float64,copy=True) for k,v in core.extract_tsv_weights(backbone).items()}
        self.loader,self.task_data=make_eval_loader_compat(core,meta,eval_batch_size,device,reserve_gb)
        self.weights=core.class_weights_for_meta(meta,device)
        stored_metrics=dict(load_json(run.metadata_path).get("metrics_final",{}))
        if verify_full_replay:
            self.full_metrics=core.evaluate(self.classifier,self.loader,device,int(load_json(run.metadata_path)["num_classes"]),self.weights)
            stored=stored_metrics.get("balanced_accuracy")
            if stored is not None and abs(float(stored)-float(self.full_metrics["balanced_accuracy"]))>1e-5:
                raise RuntimeError(f"{run.key}: full checkpoint bACC replay mismatch: {self.full_metrics['balanced_accuracy']} vs {stored}")
        else:
            if stored_metrics.get("balanced_accuracy") is None:
                raise RuntimeError(f"{run.key}: fast profile requires stored final bACC")
            self.full_metrics=stored_metrics
        # prove delta archive reconstructs checkpoint 72 matrices
        with np.load(run.delta_path,allow_pickle=False) as z:
            errs=[]
            for m in modules:
                d=np.asarray(z[sanitize_module_key(m)],dtype=np.float64)
                errs.append(float(np.linalg.norm(self.base_tsv[m]+d-self.adapted_tsv[m]) / max(np.linalg.norm(d),EPS)))
        self.delta_checkpoint_recon_err=max(errs)
        if self.delta_checkpoint_recon_err>2e-5:
            raise RuntimeError(f"{run.key}: delta->checkpoint reconstruction error {self.delta_checkpoint_recon_err:.3e}")

    @property
    def bfull(self): return float(self.full_metrics["balanced_accuracy"])

    def evaluate_matrix_updates(self, updates: Mapping[str,np.ndarray]) -> dict:
        mats={m:self.base_tsv[m]+np.asarray(updates[m],dtype=np.float64) for m in self.modules}
        assign_tsv_matrices(self.backbone,mats)
        nclass=int(load_json(self.run.metadata_path)["num_classes"])
        return self.core.evaluate(self.classifier,self.loader,self.device,nclass,self.weights)

    def evaluate_q(self,q:float) -> dict:
        updates={}
        for m in self.modules:
            r=self.svds[(self.run.task,self.run.replicate,m)]
            k=k_for_frac(r.r,q)
            updates[m]=np.zeros((r.d_out,r.d_in),dtype=np.float64) if k==0 else r.U[:,:k]@(r.s[:k,None]*r.V[:,:k].T)
        return self.evaluate_matrix_updates(updates)

    def evaluate_global_order(self, order: Sequence[Tuple[str,int]], mcount: int) -> dict:
        selected=defaultdict(list)
        for mod,i in order[:int(mcount)]: selected[mod].append(int(i))
        updates={}
        for m in self.modules:
            r=self.svds[(self.run.task,self.run.replicate,m)]
            idx=selected.get(m,[])
            if not idx: updates[m]=np.zeros((r.d_out,r.d_in),dtype=np.float64)
            else:
                I=np.asarray(idx,dtype=int)
                updates[m]=r.U[:,I]@(r.s[I,None]*r.V[:,I].T)
        return self.evaluate_matrix_updates(updates)

    def close(self):
        import gc
        try: del self.classifier, self.backbone, self.loader, self.task_data
        except Exception: pass
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except Exception: pass


# -----------------------------------------------------------------------------
# Stage: training QC + update heat map
# -----------------------------------------------------------------------------

def collect_training_summary(runs: Sequence[RunDesc]) -> List[dict]:
    rows=[]
    for r in runs:
        meta=load_json(r.metadata_path); fm=meta.get("metrics_final",{})
        epoch_rows=read_csv(r.run_dir/"epoch_metrics.csv") if (r.run_dir/"epoch_metrics.csv").is_file() else []
        total_time=sum(float(x.get("seconds",0) or 0) for x in epoch_rows)
        rows.append({
            "task":r.task,"replicate":r.replicate,"n_samples":meta.get("n_samples"),
            "num_classes":meta.get("num_classes"),"epochs":meta.get("epochs"),
            "full_accuracy":fm.get("accuracy"),"full_bacc":fm.get("balanced_accuracy"),
            "full_macro_f1":fm.get("macro_f1"),"full_objective_loss":fm.get("objective_loss"),
            "training_time_seconds":total_time,
        })
    return rows


def heatmap_rows(runs,modules,svds,base_tsv):
    rows=[]
    for r in runs:
        for m in modules:
            rec=svds[(r.task,r.replicate,m)]
            dn=float(np.linalg.norm(rec.s)); bn=float(np.linalg.norm(base_tsv[m])); H=(dn*dn)/max(bn*bn,EPS)
            layer,role=module_layer_role(m)
            rows.append({"task":r.task,"replicate":r.replicate,"module":m,"block":layer+1,"matrix_type":role,
                         "delta_frobenius_norm":dn,"w0_frobenius_norm":bn,"relative_update_energy":H,
                         "log10_relative_update_energy":float(np.log10(max(H,1e-300)))})
    return rows


# -----------------------------------------------------------------------------
# Stage: Individual Energy + Individual Functional
# -----------------------------------------------------------------------------

def energy_fraction_for_q(run:RunDesc,modules,svds,q:float)->float:
    num=den=0.0
    for m in modules:
        r=svds[(run.task,run.replicate,m)]; k=k_for_frac(r.r,q); e=r.s*r.s
        num+=float(e[:k].sum()); den+=float(e.sum())
    return num/max(den,EPS)


def first_crossing_interval(rows: Sequence[Mapping], xkey: str, ykey: str, threshold: float):
    """Earliest observed x interval ending at y >= threshold."""
    rr=sorted((r for r in rows if r.get(ykey) not in (None,"")),key=lambda x:float(x[xkey]))
    for j,r in enumerate(rr):
        if float(r[ykey])>=threshold:
            if j==0:return float(r[xkey]),float(r[xkey])
            return float(rr[j-1][xkey]),float(r[xkey])
    return None


def refine_fracs(lo:float,hi:float)->List[float]:
    if hi<=lo+1e-12:return [round(hi,8)]
    return [x for x in FINE_FRACS if x>=lo-1e-12 and x<=hi+1e-12]


def natural_q_grid(run:RunDesc, modules, svds)->List[float]:
    """All q breakpoints at which at least one matrix-wise retained rank changes."""
    vals={0.0,1.0}
    for m in modules:
        r=int(svds[(run.task,run.replicate,m)].r)
        vals.update(round(k/r,12) for k in range(1,r+1))
    return sorted(vals)


def quantize_q_natural(q:float, run:RunDesc, modules, svds)->float:
    """Nearest q breakpoint that actually changes at least one retained SVD rank."""
    q=min(1.0,max(0.0,float(q)))
    grid=natural_q_grid(run,modules,svds)
    return float(min(grid,key=lambda z:(abs(float(z)-q),float(z))))


def absolute_bacc_targets(rows:Sequence[Mapping], ykey:str, extras:Sequence[float]=()):
    """Absolute raw-bACC targets on the active profile's common vertical grid.

    Only grid levels inside the currently observed y-range are requested. Exact
    scientific/reference values (B0, Bfull, 0.99*Bfull, observed extrema, etc.)
    can be added through ``extras``. The first tuple element is the absolute grid
    level itself, not a normalized progress fraction.
    """
    rr=_finite_rows(rows,ykey)
    if not rr:return []
    ys=[float(r[ykey]) for r in rr]
    lo=max(0.0,min(ys));hi=min(1.0,max(ys))
    vals={float(v) for v in BACC_VERTICAL_TARGETS if v>=lo-1e-12 and v<=hi+1e-12}
    for v in extras:
        if v is None:continue
        v=float(v)
        if math.isfinite(v):vals.add(min(1.0,max(0.0,v)))
    return [(float(v),float(v)) for v in sorted(vals)]


def m_lattice_1pct(K:int)->List[int]:
    K=max(0,int(K))
    vals={0,K}
    for p in FINE_FRACS:
        vals.add(0 if p<=0 else min(K,int(math.ceil(p*K-1e-15))))
    return sorted(vals)


def _finite_rows(rows:Sequence[Mapping], ykey:str)->List[Mapping]:
    out=[]
    for r in rows:
        v=r.get(ykey)
        if v in (None,""):continue
        try:
            f=float(v)
        except Exception:
            continue
        if math.isfinite(f):out.append(r)
    return out


def estimate_x_for_y(rows:Sequence[Mapping], xkey:str, ykey:str, target:float,
                     quantizer=None):
    """Estimate an x whose observed y should be near target.

    Uses the earliest adjacent observed segment that brackets the requested y.
    This does NOT assume global monotonicity. If no segment brackets the target,
    it falls back to the observed point with nearest y. The returned x is merely
    a probe location; the reported spectrum point is chosen later from ACTUAL
    evaluated y values.
    """
    rr=sorted(_finite_rows(rows,ykey),key=lambda r:float(r[xkey]))
    if not rr:return None
    for a,b in zip(rr[:-1],rr[1:]):
        x0,x1=float(a[xkey]),float(b[xkey]); y0,y1=float(a[ykey]),float(b[ykey])
        if (target-y0)*(target-y1)<=0 and abs(y1-y0)>1e-15:
            w=(target-y0)/(y1-y0)
            x=x0+w*(x1-x0)
            return quantizer(x) if quantizer else x
        if abs(target-y0)<=1e-15:
            return quantizer(x0) if quantizer else x0
    best=min(rr,key=lambda r:(abs(float(r[ykey])-target),float(r[xkey])))
    x=float(best[xkey])
    return quantizer(x) if quantizer else x


def nearest_rows_to_vertical_targets(rows:Sequence[Mapping], xkey:str, ykey:str,
                                     targets:Sequence[Tuple[float,float]]):
    """Map requested vertical levels to nearest ACTUAL evaluated rows.

    Returns dict x -> list[(sampling_level,target_y)]. Duplicate x selections
    are deliberately collapsed so plotted component counts remain clean.
    """
    rr=_finite_rows(rows,ykey)
    selected=defaultdict(list)
    for frac,target in targets:
        if not rr:continue
        best=min(rr,key=lambda r:(abs(float(r[ykey])-float(target)),float(r[xkey])))
        selected[float(best[xkey])].append((float(frac),float(target)))
    return selected


def energy_q_samples(run:RunDesc, modules, svds):
    """Exact absolute-y sampling for Individual Energy on the natural SVD q lattice."""
    grid=natural_q_grid(run,modules,svds)
    vals=[energy_fraction_for_q(run,modules,svds,q) for q in grid]
    chosen=defaultdict(list)
    for target in ENERGY_VERTICAL_TARGETS:
        target=float(target)  # Individual Energy endpoint is exactly one.
        j=next((j for j,v in enumerate(vals) if v+1e-15>=target),len(vals)-1)
        chosen[float(grid[j])].append((target,target))
    # Endpoints are mandatory even under pathological numerical edge cases.
    chosen[0.0].append((0.0,0.0));chosen[1.0].append((1.0,1.0))
    for k in list(chosen):chosen[k]=list(dict.fromkeys(chosen[k]))
    return chosen


def _target_json(items):
    payload=[]
    for a,b in items:
        af=None if a is None else float(a)
        if af is not None and not math.isfinite(af):af=None
        payload.append({"sampling_level":af,"target":float(b)})
    return json.dumps(payload,sort_keys=True,allow_nan=False)


def _dedup_targets(mapping):
    for k in list(mapping):mapping[k]=list(dict.fromkeys(mapping[k]))
    return mapping


def _append_stage(old, stage:str)->str:
    parts=[x for x in str(old or "").split("+") if x]
    if stage not in parts:parts.append(stage)
    return "+".join(parts)


def individual_for_run(args, core, run, meta, modules, svds, manifest, checkpoint, modeling,
                       chans,model_init_seed,expected_w0,device,outdir):
    """Individual Energy/Functional with ABSOLUTE vertical-axis sampling.

    A sparse q-grid is used only to bracket functional targets. Reported Energy
    points are selected by absolute retained-energy levels; Functional points use
    absolute raw-bACC levels on the active profile grid. The 99% functional cutoff
    retains a local profile-dependent refinement.
    """
    outpath=outdir/"progress"/"individual"/f"{run.tag}.csv"
    if args.resume and outpath.is_file():
        old=read_csv(outpath)
        compatible=old and all(str(x.get("sampling_policy",""))==SAMPLING_POLICY and str(x.get("analysis_version",""))==VERSION for x in old)
        complete=compatible and any(str(x.get("complete","0")).lower() in ("1","true") for x in old)
        if complete:
            log(f"Individual {run.tag}: RESUME SKIP ({SAMPLING_POLICY})")
            return [x for x in old if str(x.get("is_reported_sample","0")).lower() in ("1","true")]

    ev=CandidateEvaluator(core,run,meta,manifest,checkpoint,modeling,chans,model_init_seed,expected_w0,
                          modules,svds,device,args.eval_batch_size,args.ram_reserve_gb,
                          verify_full_replay=(args.profile=="full"))
    nclass=int(load_json(run.metadata_path)["num_classes"])
    chance=1.0/nclass
    byq={}

    def ensure_q(q:float, stage:str, evaluate_function:bool=True):
        q=round(float(q),12)
        row=byq.get(q)
        if row is None:
            E=energy_fraction_for_q(run,modules,svds,q)
            row={"task":run.task,"replicate":run.replicate,"analysis_version":VERSION,"sampling_policy":SAMPLING_POLICY,
                 "probe_stages":stage,"q":q,"scan_fraction":q,
                 "component_count":sum(k_for_frac(svds[(run.task,run.replicate,m)].r,q) for m in modules),
                 "individual_energy":E,"raw_bacc":None,"accuracy":None,"macro_f1":None,
                 "full_bacc":ev.bfull,"chance_bacc":chance,"zero_component_baseline_bacc":None,
                 "delta_checkpoint_recon_error":ev.delta_checkpoint_recon_err}
            byq[q]=row
        else:
            row["probe_stages"]=_append_stage(row.get("probe_stages"),stage)
        if evaluate_function and row.get("raw_bacc") in (None,""):
            met=ev.evaluate_q(q)
            row["raw_bacc"]=float(met["balanced_accuracy"])
            row["accuracy"]=float(met["accuracy"])
            row["macro_f1"]=float(met["macro_f1"])
            log(f"  Individual {run.tag} q={q:.3f}: E={float(row['individual_energy']):.4f} bACC={met['balanced_accuracy']:.4f}")
        return row

    try:
        # Internal sparse bracketing probes. These are not automatically plotted.
        for q in COARSE_FRACS:ensure_q(q,"coarse_probe",True)
        b0=float(byq[0.0]["raw_bacc"]); bfull=float(ev.bfull)
        for v in byq.values():v["zero_component_baseline_bacc"]=b0

        # Reported Functional targets are ABSOLUTE raw-bACC levels (profile grid),
        # not fractions of the B0->Bfull span. Fixed-q values below are only
        # internal bracketing probes. Target-guided q uses the natural SVD-rank
        # breakpoint lattice, so reported x values are not forced onto 1% q bins.
        coarse=[byq[float(q)] for q in COARSE_FRACS]
        thr=.99*bfull
        coarse_min=min(float(r["raw_bacc"]) for r in coarse)
        coarse_max=max(float(r["raw_bacc"]) for r in coarse)
        btargets=absolute_bacc_targets(coarse,"raw_bacc",extras=(b0,bfull,thr,coarse_min,coarse_max))
        for level,target in btargets:
            qest=estimate_x_for_y(
                coarse,"q","raw_bacc",target,
                quantizer=lambda x:quantize_q_natural(x,run,modules,svds))
            if qest is not None:ensure_q(qest,f"bacc_target_probe_{level:.3f}",True)

        # Functional 99% cutoff: use all target-guided probes to find the earliest
        # observed crossing, then exhaust the 1%-q lattice in that local interval.
        fint=first_crossing_interval(list(byq.values()),"q","raw_bacc",thr)
        if fint:
            for q in refine_fracs(*fint):ensure_q(q,"functional99_refine",True)
        function_rows=_finite_rows(list(byq.values()),"raw_bacc")
        reached=[r for r in function_rows if float(r["raw_bacc"])>=thr]
        if not reached:
            raise RuntimeError(f"{run.key}: no functional-99 crossing although q=1 must equal full")
        qF=min(float(r["q"]) for r in reached)

        # q=1 must reproduce the saved full checkpoint.
        q1=ensure_q(1.0,"mandatory_endpoint",True)
        if abs(float(q1["raw_bacc"])-bfull)>args.metric_tol:
            raise RuntimeError(f"{run.key}: Individual q=1 does not reproduce full checkpoint")

        # Energy sampling is exact and inference-free: choose the first natural q
        # breakpoint reaching each requested vertical energy level.
        energy_selected=energy_q_samples(run,modules,svds)
        for q in energy_selected:ensure_q(q,"energy_y_sample",False)
        energy99_candidates=[q for q,items in energy_selected.items() if any(abs(a-.99)<1e-12 for a,_ in items)]
        if not energy99_candidates:raise RuntimeError(f"{run.key}: Energy-99 target missing")
        qE=min(energy99_candidates)

        # Choose ACTUAL evaluated functional rows nearest each raw-bACC target.
        # Duplicate x selections are collapsed, eliminating repeated component counts.
        function_selected=nearest_rows_to_vertical_targets(function_rows,"q","raw_bacc",btargets)
        function_selected[0.0].append((None,b0));function_selected[1.0].append((None,bfull))
        function_selected[qF].append((None,thr))
        maxrow=max(function_rows,key=lambda r:float(r["raw_bacc"]))
        function_selected[float(maxrow["q"])].append((None,float(maxrow["raw_bacc"])))
        _dedup_targets(function_selected)

        # Ensure every selected q has complete common metadata, and mark exactly
        # which points belong to each reported spectrum.
        for v in byq.values():v["zero_component_baseline_bacc"]=b0
        for q,row in byq.items():
            eitems=energy_selected.get(float(q),[])
            fitems=function_selected.get(float(q),[])
            row["q_energy_99"]=qE;row["q_functional_99"]=qF
            row["is_energy_cutoff"]=abs(float(q)-qE)<1e-12
            row["is_functional_cutoff"]=abs(float(q)-qF)<1e-12
            row["is_energy_sample"]=bool(eitems);row["is_functional_sample"]=bool(fitems)
            row["is_reported_sample"]=bool(eitems or fitems)
            row["energy_vertical_targets"]=_target_json(eitems) if eitems else "[]"
            row["bacc_vertical_targets"]=_target_json(fitems) if fitems else "[]"
            row["functional99_target_bacc"]=thr
            row["complete"]=True

        progress_rows=sorted(byq.values(),key=lambda x:float(x["q"]))
        write_csv(outpath,progress_rows)
        return [x for x in progress_rows if bool(x["is_reported_sample"])]
    finally:
        ev.close()


# -----------------------------------------------------------------------------
# Functional sets and Shared Energy geometry
# -----------------------------------------------------------------------------

def cutoffs_from_individual(individual_rows: Sequence[Mapping]) -> Dict[Tuple[str,int],dict]:
    g=defaultdict(list)
    for r in individual_rows:g[(str(r["task"]),int(r["replicate"]))].append(r)
    out={}
    for key,rr in g.items():
        qF=float(rr[0]["q_functional_99"]); qE=float(rr[0]["q_energy_99"])
        atF=min(rr,key=lambda x:abs(float(x["q"])-qF)); atE=min(rr,key=lambda x:abs(float(x["q"])-qE))
        out[key]={"q_star":qF,"q_energy99":qE,"K_star":int(float(atF["component_count"])),
                  "d_energy99":int(float(atE["component_count"])),"B_ind_at_qstar":float(atF["raw_bacc"]),
                  "B0":float(rr[0]["zero_component_baseline_bacc"]),"Bfull":float(rr[0]["full_bacc"])}
    return out


def context_parts_for_module(context_runs: Sequence[RunDesc], module: str, cutoffs, svds):
    parts=[]
    for cr in context_runs:
        rec=svds[(cr.task,cr.replicate,module)]
        k=k_for_frac(rec.r,float(cutoffs[cr.key]["q_star"]))
        parts.append((rec.U[:,:k],rec.V[:,:k]))
    return parts


def shared_components(candidate:RunDesc,context_runs:Sequence[RunDesc],modules,cutoffs,svds,ortho_rtol:float):
    comps=[]; dims={}; tol_used={}
    for m in modules:
        c=svds[(candidate.task,candidate.replicate,m)]
        kc=k_for_frac(c.r,float(cutoffs[candidate.key]["q_star"]))
        if kc==0:
            dims[m]=0; continue
        parts=context_parts_for_module(context_runs,m,cutoffs,svds)
        frac,rank,tol=projection_fractions_rank1(c.U[:,:kc],c.V[:,:kc],parts,c.d_out*c.d_in,ortho_rtol)
        dims[m]=rank; tol_used[m]=tol
        for i in range(kc):
            sigma=float(c.s[i]); eps=(sigma*sigma)*float(frac[i])
            layer,role=module_layer_role(m)
            comps.append({"module":m,"block":layer+1,"matrix_type":role,"component_index":i+1,
                          "component_index0":i,"sigma":sigma,"sigma2":sigma*sigma,
                          "projection_energy_fraction":float(frac[i]),"shared_energy":eps})
    comps.sort(key=lambda x:(-float(x["shared_energy"]),x["module"],int(x["component_index"])))
    return comps,dims,tol_used


def comparison_id(candidate:RunDesc,context_runs:Sequence[RunDesc],regime:str)->str:
    ctxt="+".join(f"{r.task}r{r.replicate}" for r in context_runs)
    return f"{candidate.task}r{candidate.replicate}__{regime}__{ctxt}"


def cumulative_vertical_samples(cum:np.ndarray, denominator:float, K:int, endpoint:float):
    """Choose component counts at ABSOLUTE cumulative shared-energy targets.

    Example: if Gamma=0.73, report targets 0, .05, .10, ..., .70 and the
    exact endpoint .73. The curve is never rescaled to [0,1].
    """
    chosen=defaultdict(list)
    if K<=0:
        chosen[0].append((0.0,0.0));return chosen
    den=max(float(denominator),EPS)
    vals=np.asarray(cum,dtype=float)/den
    targets=[float(v) for v in ENERGY_VERTICAL_TARGETS if float(v)<=float(endpoint)+1e-12]
    targets.append(float(endpoint))
    for target in sorted(set(targets)):
        if target<=0:
            m=0
        else:
            j=int(np.searchsorted(vals,target-1e-15,side="left"))
            m=min(K,j+1)
        chosen[int(m)].append((target,target))
    chosen[0].append((0.0,0.0));chosen[K].append((float(endpoint),float(endpoint)))
    for k in list(chosen):chosen[k]=list(dict.fromkeys(chosen[k]))
    return chosen


def shared_energy_curve(candidate,context_runs,regime,modules,cutoffs,svds,ortho_rtol):
    comps,dims,tols=shared_components(candidate,context_runs,modules,cutoffs,svds,ortho_rtol)
    den=sum(float(x["sigma2"]) for x in comps)
    cum=np.cumsum([float(x["shared_energy"]) for x in comps]) if comps else np.zeros(0)
    K=len(comps); cid=comparison_id(candidate,context_runs,regime)
    gamma=float(cum[-1]/max(den,EPS)) if K else 0.0
    if gamma < -1e-10 or gamma > 1+1e-7:
        raise RuntimeError(f"{cid}: Gamma outside [0,1]: {gamma}")
    gamma=min(1.0,max(0.0,gamma))
    selected=cumulative_vertical_samples(cum,den,K,gamma)
    rows=[]
    for m in sorted(selected):
        e=0.0 if m==0 else float(cum[m-1]/max(den,EPS))
        items=selected[m]
        rows.append({"comparison_id":cid,"candidate":candidate.task,"candidate_replicate":candidate.replicate,
                     "context_regime":regime,"context_tasks":"+".join(r.task for r in context_runs),
                     "context_replicates":"+".join(str(r.replicate) for r in context_runs),
                     "analysis_version":VERSION,"sampling_policy":SAMPLING_POLICY,"sampling_axis":"shared_energy",
                     "component_count":m,"K_star":K,"component_fraction":0.0 if K==0 else float(m/K),
                     "scan_fraction":0.0 if K==0 else float(m/K),
                     "shared_energy_spectrum":e,"Gamma":gamma,
                     "vertical_targets":_target_json(items),
                     "functional_energy_denominator":den,"context_dimension_total":sum(dims.values()),
                     "context_dimension_by_matrix":json.dumps(dims,sort_keys=True),
                     "orthogonalization_sigma_tolerance_max":max(tols.values()) if tols else 0.0})
    return rows,comps,dims


def all_comparisons(runs:Sequence[RunDesc],tasks:Sequence[str],profile:str="full"):
    """Yield directed candidate/context comparisons.

    full profile: proposal-complete 4 single-context replicate pairings and 32
    full-context combinations.

    fast profile: keep the two aligned replicate tracks only (rep01 with rep01,
    rep02 with rep02). This preserves replicate-level observations while cutting
    shared functional inference from 240 comparisons to 50.
    """
    by=defaultdict(list)
    for r in runs:by[r.task].append(r)
    for t in tasks:
        others=[x for x in tasks if x!=t]
        for cand in by[t]:
            if profile=="fast":
                for c in others:
                    matches=[cr for cr in by[c] if cr.replicate==cand.replicate]
                    if not matches:
                        matches=[sorted(by[c],key=lambda x:x.replicate)[0]]
                    yield cand,[matches[0]],"single"
                reps=[]
                for c in others:
                    matches=[cr for cr in by[c] if cr.replicate==cand.replicate]
                    reps.append(matches[0] if matches else sorted(by[c],key=lambda x:x.replicate)[0])
                yield cand,reps,"full"
            else:
                for c in others:
                    for cr in by[c]:
                        yield cand,[cr],"single"
                for reps in itertools.product(*[by[x] for x in others]):
                    yield cand,list(reps),"full"


# -----------------------------------------------------------------------------
# Matched-random Shared Energy baseline
# -----------------------------------------------------------------------------

def haar_basis(d:int,k:int,rng:np.random.Generator)->np.ndarray:
    if k==0:return np.zeros((d,0),dtype=np.float64)
    if k>d:raise ValueError((d,k))
    A=rng.standard_normal((d,k)); Q,R=np.linalg.qr(A,mode="reduced")
    signs=np.sign(np.diag(R)); signs[signs==0]=1
    return Q*signs[None,:]


def random_shared_curve_one(args,candidate,context_runs,regime,modules,cutoffs,svds,draw:int,
                            sample_counts:Sequence[int]):
    rng=np.random.default_rng(stable_seed(args.random_seed,"shared",comparison_id(candidate,context_runs,regime),draw))
    comps=[]
    for m in modules:
        c=svds[(candidate.task,candidate.replicate,m)]
        kc=k_for_frac(c.r,float(cutoffs[candidate.key]["q_star"]))
        if kc==0:continue
        parts=[]
        for cr in context_runs:
            rr=svds[(cr.task,cr.replicate,m)]
            kctx=k_for_frac(rr.r,float(cutoffs[cr.key]["q_star"]))
            # matched rank-one uv^T context: independent Haar U and V bases
            parts.append((haar_basis(rr.d_out,kctx,rng),haar_basis(rr.d_in,kctx,rng)))
        frac,_,_=projection_fractions_rank1(c.U[:,:kc],c.V[:,:kc],parts,c.d_out*c.d_in,args.ortho_rtol)
        for i in range(kc):
            s2=float(c.s[i]**2); comps.append((s2*float(frac[i]),s2))
    comps.sort(key=lambda x:-x[0]); K=len(comps); den=sum(x[1] for x in comps)
    cum=np.cumsum([x[0] for x in comps]) if comps else np.zeros(0)
    vals=[]
    for m in sample_counts:
        m=min(K,max(0,int(m)))
        vals.append(0.0 if m==0 else float(cum[m-1]/max(den,EPS)))
    return vals


def random_shared_baseline_comparison(args,candidate,context_runs,regime,modules,cutoffs,svds,outdir,
                                      observed_energy_rows:Sequence[Mapping]):
    cid=comparison_id(candidate,context_runs,regime)
    safe=hashlib.sha256(cid.encode()).hexdigest()[:16]
    qpath=outdir/"progress"/"random_shared"/f"{safe}_quantiles.csv"
    rawpath=outdir/"random"/f"shared_random_{safe}.csv.gz"
    sample_rows=sorted(observed_energy_rows,key=lambda r:int(float(r["component_count"])))
    sample_counts=[int(float(r["component_count"])) for r in sample_rows]
    sig_payload={"version":VERSION,"sampling_policy":SAMPLING_POLICY,"sample_counts":sample_counts,
                 "shared_random_n":int(args.shared_random_n),"random_seed":int(args.random_seed),
                 "ortho_rtol":float(args.ortho_rtol)}
    expected_signature=hashlib.sha256(json.dumps(sig_payload,sort_keys=True).encode()).hexdigest()
    if args.resume and qpath.is_file():
        old=read_csv(qpath)
        if old and all(str(r.get("sampling_signature",""))==expected_signature for r in old):
            return old
    curves=[]; raw=[]
    for d in range(args.shared_random_n):
        vals=random_shared_curve_one(args,candidate,context_runs,regime,modules,cutoffs,svds,d,sample_counts)
        curves.append(vals)
        if args.save_random_draws:
            for obs,v in zip(sample_rows,vals):
                raw.append({"comparison_id":cid,"random_replicate":d,
                            "component_count":int(float(obs["component_count"])),
                            "component_fraction":float(obs.get("component_fraction",0) or 0),
                            "scan_fraction":float(obs.get("component_fraction",0) or 0),
                            "shared_energy":v})
        if (d+1)%max(1,args.shared_random_n//10)==0:log(f"  random shared {cid}: {d+1}/{args.shared_random_n}")
    A=np.asarray(curves,dtype=float)
    rows=[]
    for j,obs in enumerate(sample_rows):
        st=qstats(A[:,j])
        rows.append({"comparison_id":cid,"candidate":candidate.task,"candidate_replicate":candidate.replicate,
                     "context_regime":regime,"context_tasks":"+".join(r.task for r in context_runs),
                     "context_replicates":"+".join(str(r.replicate) for r in context_runs),
                     "analysis_version":VERSION,"sampling_policy":SAMPLING_POLICY,"sampling_signature":expected_signature,
                     "component_count":int(float(obs["component_count"])),
                     "K_star":int(float(obs.get("K_star",0) or 0)),
                     "component_fraction":float(obs.get("component_fraction",0) or 0),
                     "scan_fraction":float(obs.get("component_fraction",0) or 0),
                     "vertical_targets":obs.get("vertical_targets","[]"),
                     "random_n":args.shared_random_n,
                     "random_median":st["median"],"random_q95":st["q95"],"random_q99":st["q99"],
                     "random_upper":st["q95"] if args.profile=="fast" else st["q99"],
                     "random_upper_quantile":0.95 if args.profile=="fast" else 0.99,
                     "random_q01":st["q01"]})
    write_csv(qpath,rows)
    if raw:write_csv_gz(rawpath,raw)
    return rows


# -----------------------------------------------------------------------------
# Shared Functional inference, resumable per comparison
# -----------------------------------------------------------------------------

def shared_functional_comparison(args,core,candidate,context_runs,regime,meta,modules,cutoffs,svds,
                                 manifest,checkpoint,modeling,chans,model_init_seed,expected_w0,device,outdir,
                                 evaluator=None):
    cid=comparison_id(candidate,context_runs,regime); safe=hashlib.sha256(cid.encode()).hexdigest()[:16]
    pth=outdir/"progress"/"shared_functional"/f"{safe}.csv"
    comps,_,_=shared_components(candidate,context_runs,modules,cutoffs,svds,args.ortho_rtol)
    order=[(str(x["module"]),int(x["component_index0"])) for x in comps]
    K=len(order)
    order_payload={"version":VERSION,"candidate":candidate.key,
                   "contexts":[r.key for r in context_runs],
                   "qstars":{f"{k[0]}r{k[1]}":float(v["q_star"]) for k,v in cutoffs.items()},
                   "order":[f"{m}:{i}" for m,i in order]}
    order_signature=hashlib.sha256(json.dumps(order_payload,sort_keys=True).encode()).hexdigest()
    if args.resume and pth.is_file():
        old=read_csv(pth)
        compatible=old and all(str(x.get("sampling_policy",""))==SAMPLING_POLICY and
                               str(x.get("analysis_version",""))==VERSION and
                               str(x.get("order_signature",""))==order_signature for x in old)
        complete=compatible and any(str(x.get("complete","0")).lower() in ("1","true") for x in old)
        if complete:
            log(f"Shared Functional {cid}: RESUME SKIP ({SAMPLING_POLICY})")
            return [x for x in old if str(x.get("is_reported_sample","0")).lower() in ("1","true")]

    own_evaluator = evaluator is None
    ev = evaluator if evaluator is not None else CandidateEvaluator(
        core,candidate,meta,manifest,checkpoint,modeling,chans,model_init_seed,expected_w0,
        modules,svds,device,args.eval_batch_size,args.ram_reserve_gb,
        verify_full_replay=(args.profile=="full"))
    chance=1.0/int(load_json(candidate.metadata_path)["num_classes"])
    bym={}

    def ensure_m(m:int,stage:str):
        m=min(K,max(0,int(m)))
        row=bym.get(m)
        if row is None:
            met=ev.evaluate_global_order(order,m)
            row={"comparison_id":cid,"candidate":candidate.task,"candidate_replicate":candidate.replicate,
                 "context_regime":regime,"context_tasks":"+".join(r.task for r in context_runs),
                 "context_replicates":"+".join(str(r.replicate) for r in context_runs),
                 "analysis_version":VERSION,"sampling_policy":SAMPLING_POLICY,"order_signature":order_signature,"probe_stages":stage,
                 "component_count":m,"K_star":K,"component_fraction":0.0 if K==0 else float(m/K),
                 "scan_fraction":0.0 if K==0 else float(m/K),
                 "raw_bacc":float(met["balanced_accuracy"]),"accuracy":float(met["accuracy"]),
                 "macro_f1":float(met["macro_f1"]),"full_bacc":ev.bfull,"chance_bacc":chance,
                 "B_ind_qstar":float(cutoffs[candidate.key]["B_ind_at_qstar"])}
            bym[m]=row
            log(f"  SharedFn {cid} m={m}/{K}: bACC={met['balanced_accuracy']:.4f}")
        else:
            row["probe_stages"]=_append_stage(row.get("probe_stages"),stage)
        return row

    try:
        # Sparse x probes only bracket the absolute raw-bACC targets.
        for p in COARSE_FRACS:
            m=0 if p<=0 else min(K,int(math.ceil(p*K-1e-15)))
            ensure_m(m,"coarse_probe")
        b0=float(ensure_m(0,"mandatory_baseline")["raw_bacc"]); bfull=float(ev.bfull)
        expected_b0=float(cutoffs[candidate.key]["B0"])
        if abs(b0-expected_b0)>args.metric_tol:
            raise RuntimeError(f"{cid}: shared m=0 baseline != Individual baseline: {b0} vs {expected_b0}")

        # Absolute raw-bACC vertical targets. Coarse component-fraction probes
        # only bracket the targets; target-guided sampling is quantized to the
        # nearest INTEGER component count, not to a percentage-of-K grid.
        thr=.99*bfull
        coarse=sorted(list(bym.values()),key=lambda x:int(x["component_count"]))
        coarse_min=min(float(r["raw_bacc"]) for r in coarse)
        coarse_max=max(float(r["raw_bacc"]) for r in coarse)
        endpoint_ref=float(cutoffs[candidate.key]["B_ind_at_qstar"])
        btargets=absolute_bacc_targets(coarse,"raw_bacc",extras=(b0,bfull,thr,endpoint_ref,coarse_min,coarse_max))
        for level,target in btargets:
            mest=estimate_x_for_y(
                coarse,"component_count","raw_bacc",target,
                quantizer=lambda x:min(K,max(0,int(math.floor(float(x)+0.5)))))
            if mest is not None:ensure_m(int(mest),f"bacc_target_probe_{level:.3f}")

        # Explicit 99%-of-full cutoff. Refine only the local observed crossing
        # interval using the active profile's refinement lattice.
        fint=first_crossing_interval(list(bym.values()),"component_count","raw_bacc",thr)
        if fint:
            lo,hi=fint
            for m in m_lattice_1pct(K):
                if m>=int(math.floor(lo))-1 and m<=int(math.ceil(hi))+1:
                    ensure_m(m,"functional99_refine")
        rows_all=sorted(bym.values(),key=lambda x:int(x["component_count"]))
        reached=[x for x in rows_all if float(x["raw_bacc"])>=thr]
        mstar=min(int(x["component_count"]) for x in reached) if reached else None
        maxrow=max(rows_all,key=lambda x:float(x["raw_bacc"]))

        endpoint=ensure_m(K,"mandatory_endpoint")
        expected_endpoint=float(cutoffs[candidate.key]["B_ind_at_qstar"])
        endpoint_err=abs(float(endpoint["raw_bacc"])-expected_endpoint)
        if endpoint_err>args.metric_tol:
            raise RuntimeError(f"{cid}: shared endpoint != individual q*: err={endpoint_err}")

        # Reported points are selected by raw-bACC target, not by x fraction.
        selected=nearest_rows_to_vertical_targets(list(bym.values()),"component_count","raw_bacc",btargets)
        selected[0].append((None,b0));selected[K].append((None,float(endpoint["raw_bacc"])))
        if mstar is not None:selected[mstar].append((None,thr))
        selected[int(maxrow["component_count"])].append((None,float(maxrow["raw_bacc"])))
        _dedup_targets(selected)

        rows_all=sorted(bym.values(),key=lambda x:int(x["component_count"]))
        for x in rows_all:
            m=int(x["component_count"]);items=selected.get(m,[])
            x["vertical_bacc_targets"]=_target_json(items) if items else "[]"
            x["functional99_target_bacc"]=thr
            x["m_star_99"]=mstar;x["reached_99"]=mstar is not None
            x["max_raw_bacc"]=float(maxrow["raw_bacc"]);x["max_raw_bacc_at_m"]=int(maxrow["component_count"])
            x["endpoint_bacc_error_vs_individual_qstar"]=endpoint_err
            x["is_functional_sample"]=bool(items);x["is_reported_sample"]=bool(items);x["complete"]=True
        write_csv(pth,rows_all)
        return [x for x in rows_all if bool(x["is_reported_sample"])]
    finally:
        if own_evaluator:
            ev.close()


# -----------------------------------------------------------------------------
# STI paper-aligned + matched random
# -----------------------------------------------------------------------------

def sti_from_parts(Us:Sequence[np.ndarray],Ss:Sequence[np.ndarray],Vs:Sequence[np.ndarray])->float:
    U=np.concatenate(Us,axis=1);V=np.concatenate(Vs,axis=1);s=np.concatenate(Ss)
    A=U.T@U-np.eye(U.shape[1]);B=V.T@V-np.eye(V.shape[1])
    M=(A*s[None,:])@B
    return float(np.sum(np.abs(M)))


def sti_rows_observed(runs,tasks,modules,svds):
    by=defaultdict(list)
    for r in runs:by[r.task].append(r)
    rows=[]
    for combo in itertools.product(*[by[t] for t in tasks]):
        ctag="+".join(f"{r.task}r{r.replicate}" for r in combo)
        for m in modules:
            recs=[svds[(r.task,r.replicate,m)] for r in combo]; k=min(r.r for r in recs)//len(tasks)
            Us=[r.U[:,:k] for r in recs];Vs=[r.V[:,:k] for r in recs];Ss=[r.s[:k] for r in recs]
            layer,role=module_layer_role(m)
            rows.append({"replicate_combination":ctag,"module":m,"block":layer+1,"matrix_type":role,
                         "k_paper":k,"sti":sti_from_parts(Us,Ss,Vs)})
    return rows


def sti_random_summary(args,runs,tasks,modules,svds):
    """Matched-random STI null at the SAME aggregation level as the figure.

    Observed STI keeps all 32 five-task replicate combinations. For each matrix,
    the null distribution draws one of those 32 empirical replicate combinations
    uniformly, preserves its five empirical singular-value vectors, randomizes
    only U/V orientations with Haar orthonormal bases, and recomputes STI.

    This yields ``sti_random_n`` null draws PER MATRIX, not per matrix x observed
    combination. It is scientifically matched to the reported 32-combination
    aggregate while avoiding an unnecessary 32x Monte-Carlo explosion.
    """
    by=defaultdict(list)
    for r in runs:by[r.task].append(r)
    combos=list(itertools.product(*[by[t] for t in tasks]))
    out=[]
    for mi,m in enumerate(modules,1):
        recsets=[[svds[(r.task,r.replicate,m)] for r in combo] for combo in combos]
        k=min(recsets[0][0].r for _ in [0])//len(tasks)
        vals=[]
        rng=np.random.default_rng(stable_seed(args.random_seed,"sti-null",m))
        for d in range(args.sti_random_n):
            recs=recsets[int(rng.integers(0,len(recsets)))]
            # All analyzed matrices have the same r=200 contract; still compute k
            # from the selected records fail-closed in case that ever changes.
            kd=min(r.r for r in recs)//len(tasks)
            Us=[haar_basis(r.d_out,kd,rng) for r in recs]
            Vs=[haar_basis(r.d_in,kd,rng) for r in recs]
            Ss=[r.s[:kd] for r in recs]
            vals.append(sti_from_parts(Us,Ss,Vs))
        st=qstats(vals);layer,role=module_layer_role(m)
        out.append({
            "module":m,"block":layer+1,"matrix_type":role,"k_paper":k,
            "random_n":args.sti_random_n,
            "random_combo_sampling":"uniform_over_32_empirical_replicate_combinations",
            "random_median":st["median"],"random_q99":st["q99"],
            "random_q01":st["q01"],"random_min":st["min"],"random_max":st["max"]})
        if mi%12==0:log(f"STI matched-random null: {mi}/{len(modules)} matrices")
    return out


# -----------------------------------------------------------------------------
# Sanity checks / summaries
# -----------------------------------------------------------------------------

def nesting_sanity(runs,tasks,modules,cutoffs,svds,ortho_rtol,profile="full"):
    by=defaultdict(list)
    for r in runs:by[r.task].append(r)
    worst=0.0; checked=0
    for cand in runs:
        others=[t for t in tasks if t!=cand.task]
        if profile=="fast":
            aligned=[]
            for t in others:
                mm=[r for r in by[t] if r.replicate==cand.replicate]
                aligned.append(mm[0] if mm else sorted(by[t],key=lambda x:x.replicate)[0])
            rep_sets=[tuple(aligned)]
        else:
            rep_sets=itertools.product(*[by[t] for t in others])
        for reps in rep_sets:
            for m in modules:
                c=svds[(cand.task,cand.replicate,m)];kc=k_for_frac(c.r,cutoffs[cand.key]["q_star"])
                if kc==0:continue
                full_parts=context_parts_for_module(reps,m,cutoffs,svds)
                f,_,_=projection_fractions_rank1(c.U[:,:kc],c.V[:,:kc],full_parts,c.d_out*c.d_in,ortho_rtol)
                for cr in reps:
                    s,_,_=projection_fractions_rank1(c.U[:,:kc],c.V[:,:kc],context_parts_for_module([cr],m,cutoffs,svds),c.d_out*c.d_in,ortho_rtol)
                    worst=min(worst,float(np.min(f-s)));checked+=kc
    if worst<-1e-7:raise RuntimeError(f"Context nesting violated: worst Full-Single={worst}")
    return {"checked_component_comparisons":checked,"worst_full_minus_single":worst,"passed":True}


def individual_summary(ind_rows):
    g=defaultdict(list)
    for r in ind_rows:g[(r["task"],int(r["replicate"]))].append(r)
    out=[]
    for (t,rep),rr in g.items():
        qE=float(rr[0]["q_energy_99"]);qF=float(rr[0]["q_functional_99"])
        e=min(rr,key=lambda x:abs(float(x["q"])-qE));f=min(rr,key=lambda x:abs(float(x["q"])-qF))
        out.append({"task":t,"replicate":rep,"B0":float(rr[0]["zero_component_baseline_bacc"]),
                    "Bfull":float(rr[0]["full_bacc"]),"q_energy_99":qE,"d_energy_99":int(float(e["component_count"])),
                    "q_functional_99":qF,"K_star":int(float(f["component_count"])),
                    "B_ind_at_qstar":float(f["raw_bacc"]),"endpoint_reconstruction_bacc_error":abs(float(next(x for x in rr if abs(float(x["q"])-1)<1e-12)["raw_bacc"])-float(rr[0]["full_bacc"]))})
    return out


def shared_summary(energy_rows,function_rows,random_rows):
    ge=defaultdict(list);gf=defaultdict(list);gr=defaultdict(list)
    for r in energy_rows:ge[r["comparison_id"]].append(r)
    for r in function_rows:gf[r["comparison_id"]].append(r)
    for r in random_rows:gr[r["comparison_id"]].append(r)
    out=[]
    for cid,ee in ge.items():
        ff=gf.get(cid,[]);rr=gr.get(cid,[]);e0=ee[0]
        gamma=float(e0["Gamma"])
        rand_end=None
        if rr:
            rend=max(rr,key=lambda x:int(float(x["component_count"])));rand_end=float(rend.get("random_upper",rend.get("random_q95",rend.get("random_q99","nan"))))
        if ff:
            f0=ff[0];mstar=f0.get("m_star_99");reached=str(f0.get("reached_99")).lower() in ("true","1")
            maxb=float(f0.get("max_raw_bacc","nan"));maxm=int(float(f0.get("max_raw_bacc_at_m",0)))
        else:mstar=None;reached=False;maxb=float("nan");maxm=0
        out.append({"comparison_id":cid,"candidate":e0["candidate"],"candidate_replicate":e0["candidate_replicate"],
                    "context_regime":e0["context_regime"],"context_tasks":e0["context_tasks"],
                    "context_replicates":e0["context_replicates"],"context_dimension_total":e0["context_dimension_total"],
                    "Gamma":gamma,"random_endpoint_upper":rand_end,
                    "random_upper_quantile":None if not rr else rr[0].get("random_upper_quantile"),
                    "Gamma_above_random_upper":None if rand_end is None else gamma>rand_end,
                    "m_star_99":mstar,"reached_99":reached,"max_raw_bacc":maxb,"max_raw_bacc_at_m":maxm})
    return out


def sti_summary(sti_rows, random_rows):
    byobs=defaultdict(list)
    byrand={}
    for r in sti_rows:byobs[str(r["module"])].append(r)
    for r in random_rows:byrand[str(r["module"])]=r
    out=[]
    for module,rr in sorted(byobs.items()):
        vals=[float(r["sti"]) for r in rr]; q=byrand.get(module,{})
        layer,role=module_layer_role(module)
        med=float(np.median(vals));rmed=float(q.get("random_median","nan"));rq99=float(q.get("random_q99","nan"))
        out.append({
            "module":module,"block":layer+1,"matrix_type":role,
            "k_paper":int(float(rr[0]["k_paper"])),"n_observed_combinations":len(vals),
            "observed_median":med,"observed_min":float(np.min(vals)),"observed_max":float(np.max(vals)),
            "random_n":q.get("random_n"),"random_median":rmed,"random_q99":rq99,
            "observed_median_over_random_median":med/max(rmed,EPS) if math.isfinite(rmed) else None,
            "observed_median_above_random_q99":med>rq99 if math.isfinite(rq99) else None,
        })
    return out


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def savefig(fig,path:Path):
    ensure_dir(path.parent);fig.savefig(path.with_suffix(".png"),dpi=220,bbox_inches="tight");fig.savefig(path.with_suffix(".pdf"),bbox_inches="tight")


def plot_training_qc(runs,outfig):
    plt=import_plt(); metrics=[("balanced_accuracy","bACC"),("objective_loss","Objective loss"),("seconds","Epoch seconds")]
    fig,axs=plt.subplots(1,3,figsize=(16,4.6))
    for ax,(key,ylabel) in zip(axs,metrics):
        for r in runs:
            p=r.run_dir/"epoch_metrics.csv"
            if not p.is_file():continue
            rr=read_csv(p);x=[int(z["epoch"]) for z in rr];y=[float(z[key]) for z in rr]
            ax.plot(x,y,alpha=.72,lw=1.2,label=f"{r.task} r{r.replicate}")
        ax.set_xlabel("Epoch");ax.set_ylabel(ylabel);ax.grid(alpha=.2)
    handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc="lower center",ncol=5,fontsize=8)
    fig.suptitle("Full fine-tuning quality control (reused final runs)");fig.tight_layout(rect=(0,.12,1,.94));savefig(fig,outfig/"01_full_finetuning_qc");plt.close(fig)


def plot_heatmaps(rows,outfig,tasks):
    plt=import_plt(); roles=list(MODULE_ROLES)
    vals=np.asarray([float(r["log10_relative_update_energy"]) for r in rows]);vmin=float(np.min(vals));vmax=float(np.max(vals))
    fig,axs=plt.subplots(1,len(tasks),figsize=(4*len(tasks),5),sharey=True)
    if len(tasks)==1:axs=[axs]
    im=None
    for ax,t in zip(axs,tasks):
        arr=np.full((12,6),np.nan)
        for b in range(1,13):
            for j,role in enumerate(roles):
                x=[float(r["log10_relative_update_energy"]) for r in rows if r["task"]==t and int(r["block"])==b and r["matrix_type"]==role]
                if x:arr[b-1,j]=np.median(x)
        im=ax.imshow(arr,aspect="auto",vmin=vmin,vmax=vmax,origin="lower")
        ax.set_title(t);ax.set_xticks(range(6),roles,rotation=45,ha="right");ax.set_yticks(range(12),range(1,13));ax.set_xlabel("Matrix type")
    axs[0].set_ylabel("Transformer block");fig.colorbar(im,ax=axs,label=r"$\log_{10}(||\Delta W||_F^2/||W_0||_F^2)$",shrink=.8)
    fig.suptitle("Weight-update heat map (replicate median; common colour scale)");savefig(fig,outfig/"02_weight_update_heatmap");plt.close(fig)
    # replicate-specific QC maps
    qc=ensure_dir(outfig/"heatmap_qc")
    for t in tasks:
        reps=sorted(set(int(r["replicate"]) for r in rows if r["task"]==t))
        for rep in reps:
            fig,ax=plt.subplots(figsize=(5,5));arr=np.full((12,6),np.nan)
            for r in rows:
                if r["task"]==t and int(r["replicate"])==rep:
                    arr[int(r["block"])-1,roles.index(r["matrix_type"])]=float(r["log10_relative_update_energy"])
            im=ax.imshow(arr,aspect="auto",vmin=vmin,vmax=vmax,origin="lower");ax.set_title(f"{t} rep{rep}")
            ax.set_xticks(range(6),roles,rotation=45,ha="right");ax.set_yticks(range(12),range(1,13));fig.colorbar(im,ax=ax)
            savefig(fig,qc/f"{t}_rep{rep:02d}");plt.close(fig)


def _truthy(v)->bool:
    return str(v).lower() in ("1","true","yes")


def plot_individual(ind_rows,outfig,tasks):
    plt=import_plt();fig,axs=plt.subplots(1,2,figsize=(13,5))
    for t in tasks:
        reps=sorted(set(int(r["replicate"]) for r in ind_rows if r["task"]==t))
        for rep in reps:
            allr=[r for r in ind_rows if r["task"]==t and int(r["replicate"])==rep]
            er=sorted([r for r in allr if _truthy(r.get("is_energy_sample",False))],key=lambda x:int(float(x["component_count"])))
            fr=sorted([r for r in allr if _truthy(r.get("is_functional_sample",False))],key=lambda x:int(float(x["component_count"])))
            if er:
                axs[0].plot([int(float(z["component_count"])) for z in er],
                            [float(z["individual_energy"]) for z in er],alpha=.72,lw=1.3,marker="o",ms=2.5,
                            label=f"{t} r{rep}")
            if fr:
                axs[1].plot([int(float(z["component_count"])) for z in fr],
                            [float(z["raw_bacc"]) for z in fr],alpha=.72,lw=1.3,marker="o",ms=2.5,
                            label=f"{t} r{rep}")
    axs[0].axhline(.99,ls="--",lw=1,color="0.5");axs[0].set_ylabel("Retained update-energy fraction")
    axs[1].set_ylabel("Raw balanced accuracy")
    for ax in axs:ax.set_xlabel("Retained rank-one components (whole model)");ax.grid(alpha=.2)
    handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc="lower center",ncol=5,fontsize=8)
    axs[0].set_title("Individual Energy (absolute-y sampled)")
    axs[1].set_title("Individual Functional (absolute-y sampled)")
    fig.tight_layout(rect=(0,.12,1,1));savefig(fig,outfig/"04_individual_spectra");plt.close(fig)


def _aggregate_curves(rows, groupkey, ykey):
    # returns group -> x -> list y
    out=defaultdict(lambda:defaultdict(list))
    for r in rows:out[str(r[groupkey])][int(float(r["component_count"]))].append(float(r[ykey]))
    return out


def _interp_comparison_curves(rows:Sequence[Mapping], ykey:str, grid:np.ndarray):
    curves=[]; Ks=[]
    for cid in sorted(set(str(r["comparison_id"]) for r in rows)):
        rr=sorted([r for r in rows if str(r["comparison_id"])==cid],key=lambda x:int(float(x["component_count"])))
        if len(rr)<2:continue
        K=max(1,int(float(rr[0].get("K_star",rr[-1]["component_count"]))))
        x=np.asarray([int(float(r["component_count"]))/K for r in rr],dtype=float)
        y=np.asarray([float(r[ykey]) for r in rr],dtype=float)
        # Collapse accidental duplicate x before interpolation.
        ux=[];uy=[]
        for xv in sorted(set(x.tolist())):
            vals=y[np.isclose(x,xv)]
            ux.append(xv);uy.append(float(np.mean(vals)))
        curves.append(np.interp(grid,np.asarray(ux),np.asarray(uy)));Ks.append(K)
    return curves,Ks


def plot_shared_candidate(candidate_task,regime,energy_rows,function_rows,random_rows,outpath):
    plt=import_plt();fig,axs=plt.subplots(2,1,figsize=(9,8),sharex=True)
    ee=[r for r in energy_rows if r["candidate"]==candidate_task and r["context_regime"]==regime]
    ff=[r for r in function_rows if r["candidate"]==candidate_task and r["context_regime"]==regime]
    contexts=sorted(set(r["context_tasks"] for r in ee))
    if regime=="full":contexts=[contexts[0]] if contexts else []
    grid=np.linspace(0.0,1.0,101)
    for ctx in contexts:
        ectx=[r for r in ee if r["context_tasks"]==ctx]
        # Actual vertically sampled comparison curves remain visible as thin lines.
        for cid in sorted(set(r["comparison_id"] for r in ectx)):
            rr=sorted([r for r in ectx if r["comparison_id"]==cid],key=lambda x:int(float(x["component_count"])))
            axs[0].plot([int(float(x["component_count"])) for x in rr],
                        [float(x["shared_energy_spectrum"]) for x in rr],alpha=.13,lw=.8,marker="o",ms=1.5)
        curves,Ks=_interp_comparison_curves(ectx,"shared_energy_spectrum",grid)
        if curves:
            A=np.asarray(curves);Kmed=float(np.median(Ks));xx=grid*Kmed
            axs[0].plot(xx,np.median(A,axis=0),lw=2,label=ctx)
            axs[0].fill_between(xx,np.min(A,axis=0),np.max(A,axis=0),alpha=.12)

        fctx=[r for r in ff if r["context_tasks"]==ctx]
        for cid in sorted(set(r["comparison_id"] for r in fctx)):
            rr=sorted([r for r in fctx if r["comparison_id"]==cid],key=lambda x:int(float(x["component_count"])))
            axs[1].plot([int(float(x["component_count"])) for x in rr],
                        [float(x["raw_bacc"]) for x in rr],alpha=.13,lw=.8,marker="o",ms=1.5)
        curves,Ks=_interp_comparison_curves(fctx,"raw_bacc",grid)
        if curves:
            A=np.asarray(curves);Kmed=float(np.median(Ks));xx=grid*Kmed
            axs[1].plot(xx,np.median(A,axis=0),lw=2,label=ctx)
            axs[1].fill_between(xx,np.min(A,axis=0),np.max(A,axis=0),alpha=.12)

    # Matched-random envelope is evaluated at each observed absolute-y sampling x;
    # interpolation below is display-only aggregation across replicate comparisons.
    rr=[r for r in random_rows if r["candidate"]==candidate_task and r["context_regime"]==regime]
    if rr:
        med_curves,Ks=_interp_comparison_curves(rr,"random_median",grid)
        upper_curves,_=_interp_comparison_curves(rr,"random_upper",grid)
        if med_curves and upper_curves:
            M=np.asarray(med_curves);Q=np.asarray(upper_curves);Kmed=float(np.median(Ks));x=grid*Kmed
            med=np.median(M,axis=0);upper=np.max(Q,axis=0)
            uq=rr[0].get("random_upper_quantile",0.95)
            try: uq=float(uq)
            except Exception: uq=0.95
            axs[0].plot(x,med,ls="--",lw=1.2,label="matched random median")
            axs[0].fill_between(x,med,upper,alpha=.1,label=f"random to q{int(round(100*uq))}")

    if ff:
        fulls=[float(r["full_bacc"]) for r in ff];ch=[float(r["chance_bacc"]) for r in ff]
        axs[1].axhline(float(np.median(fulls)),ls="--",lw=1,color="0.45",label="full FT ref")
        axs[1].axhline(float(np.median(ch)),ls=":",lw=1,color="0.55",label="chance")
    axs[0].set_ylabel("Shared Energy Spectrum");axs[1].set_ylabel("Raw balanced accuracy")
    axs[1].set_xlabel("Candidate functional components added")
    for ax in axs:ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=2)
    axs[0].set_title(f"{candidate_task} | {regime.capitalize()} Context: Shared Energy (absolute-y sampled)")
    axs[1].set_title(f"{candidate_task} | {regime.capitalize()} Context: Shared Functional (absolute-y sampled)")
    fig.tight_layout();savefig(fig,outpath);plt.close(fig)


def plot_sti(sti_rows,random_rows,outfig):
    plt=import_plt();roles=list(MODULE_ROLES)
    # matrix map: median over 32 combos
    arr=np.full((12,6),np.nan);ratio=np.full((12,6),np.nan)
    for b in range(1,13):
        for j,role in enumerate(roles):
            vals=[float(r["sti"]) for r in sti_rows if int(r["block"])==b and r["matrix_type"]==role]
            arr[b-1,j]=np.median(vals)
            null=[float(r["random_median"]) for r in random_rows if int(r["block"])==b and r["matrix_type"]==role]
            if null:ratio[b-1,j]=arr[b-1,j]/max(np.median(null),EPS)
    fig,axs=plt.subplots(1,2,figsize=(12,5))
    im=axs[0].imshow(arr,aspect="auto",origin="lower");axs[0].set_title("Raw STI: median over 32 replicate combinations");fig.colorbar(im,ax=axs[0])
    im2=axs[1].imshow(ratio,aspect="auto",origin="lower");axs[1].set_title("STI / matched-random median");fig.colorbar(im2,ax=axs[1])
    for ax in axs:ax.set_xticks(range(6),roles,rotation=45,ha="right");ax.set_yticks(range(12),range(1,13));ax.set_xlabel("Matrix type");ax.set_ylabel("Block")
    fig.tight_layout();savefig(fig,outfig/"03_sti_map");plt.close(fig)
    # depth profile
    fig,ax=plt.subplots(figsize=(8,4.5));x=range(1,13);med=[];lo=[];hi=[];rmed=[]
    for b in x:
        vals=[]
        for combo in sorted(set(r["replicate_combination"] for r in sti_rows)):
            vv=[float(r["sti"]) for r in sti_rows if int(r["block"])==b and r["replicate_combination"]==combo]
            vals.append(np.median(vv))
        med.append(np.median(vals));lo.append(np.min(vals));hi.append(np.max(vals))
        nv=[float(r["random_median"]) for r in random_rows if int(r["block"])==b];rmed.append(np.median(nv) if nv else np.nan)
    ax.plot(list(x),med,lw=2,label="observed median");ax.fill_between(list(x),lo,hi,alpha=.15,label="32-combination range");ax.plot(list(x),rmed,ls="--",label="matched-random median")
    ax.set_xlabel("Transformer block");ax.set_ylabel("STI (raw entrywise L1)");ax.grid(alpha=.2);ax.legend();fig.tight_layout();savefig(fig,outfig/"03_sti_depth_profile");plt.close(fig)


# -----------------------------------------------------------------------------
# Manifest / output certification
# -----------------------------------------------------------------------------

def artifact_hashes(root:Path):
    out={}
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.stat().st_size >= 2_000_000_000:continue
        # Never hash the manifest into itself, and exclude completion/stage markers
        # whose contents depend on the final manifest hash.
        if p.name=="analysis_manifest_v6_0.json" or p.name.startswith("_STAGE_") or p.name.startswith("_ANALYSIS_") or p.name.startswith("_SUCCESS"):
            continue
        try:out[str(p.relative_to(root))]=sha256_file(p)
        except Exception:pass
    return out


def build_manifest(args,root,outdir,core_path,manifest,preflight,modules,selftest,sanity,runs=None):
    protocol=manifest.get("protocol",preflight.get("protocol",{}))
    return {
        "analysis_version":VERSION,"analysis_profile":args.profile,"generated_at":now(),"source_final_root":str(root),
        "source_protocol_fingerprint":manifest.get("protocol_fingerprint"),
        "source_run_protocol_fingerprints":({r.tag:r.protocol_fingerprint for r in runs} if runs is not None else {}),
        "source_runner":str(core_path),"source_runner_sha256":sha256_file(core_path),
        "pretrained_checkpoint":protocol.get("checkpoint"),"modeling_file":protocol.get("modeling_file"),
        "input_chans":protocol.get("input_chans"),"data":protocol.get("data"),
        "training_protocol_reused_not_rerun":True,"E_star":manifest.get("epochs",20),
        "matrix_map":modules,"n_matrices":len(modules),"maximum_component_slots":sum(200 for _ in modules),
        "svd_dtype":"float64","ortho_rtol_cli":args.ortho_rtol,
        "ortho_tolerance_rule":"if ortho_rtol=0: max(ambient_matrix_dimension,n_context)*eps64*max(sigma_max,1); applied to singular values of context operator matrix via Gram eigenvalues",
        "sampling_policy":SAMPLING_POLICY,
        "energy_absolute_vertical_targets":list(ENERGY_VERTICAL_TARGETS),
        "bacc_absolute_vertical_grid":list(BACC_VERTICAL_TARGETS),
        "sampling_rule":{
            "individual_energy":"absolute y targets 0,.05,...,1 plus .99; choose first natural matrix-wise q breakpoint reaching each retained-energy target",
            "individual_functional":f"absolute raw-bACC targets on {'0.10' if args.profile=='fast' else '0.05'} grid within observed range, plus B0/Bfull/0.99*Bfull/extrema; sparse hidden q probes only bracket targets; target-guided q uses natural SVD-rank breakpoints; Functional-99 refinement uses {'2%' if args.profile=='fast' else '1%'} q lattice",
            "shared_energy":"absolute cumulative shared-energy targets 0,.05,... up to Gamma, plus exact Gamma endpoint; no alpha*Gamma rescaling",
            "shared_functional":f"absolute raw-bACC targets on {'0.10' if args.profile=='fast' else '0.05'} grid within observed range, plus baseline/full/cutoff/extrema; sparse hidden component-fraction probes only bracket targets; target-guided m is nearest integer component; Functional-99 refinement uses {'2%' if args.profile=='fast' else '1%'}-of-K lattice",
            "plot_aggregation":"observed vertical-sampled curves are primary; normalized-x interpolation is display-only for median/range across replicate comparisons"
        },
        "internal_bracketing_probe_grid":list(COARSE_FRACS),"functional99_refinement_grid":list(FINE_FRACS),
        "shared_replicate_policy":("aligned replicate tracks only: r1->r1 and r2->r2" if args.profile=="fast" else "proposal-complete single 4 pairings / full 32 combinations"),
        "shared_random_repetitions":args.shared_random_n,
        "sti_random_repetitions_per_matrix":args.sti_random_n,
        "sti_random_aggregation":"each null draw samples one of 32 empirical replicate combinations uniformly, preserves its Sigma, and randomizes U/V only",
        "random_seed":args.random_seed,
        "selection_reconstruction_contract":{
            "selection":"epsilon_shared = sigma^2 * ||P_context Z||_F^2",
            "reconstruction":"candidate original sigma*Z only",
            "projected_operator_written_to_model":False,
        },
        "algebra_selftest":selftest,"sanity":sanity,
        "important_boundary":"Pooled bACC is fitted-task functional retention, not held-out generalization.",
        "artifacts_sha256":artifact_hashes(outdir),
    }


# -----------------------------------------------------------------------------
# Main orchestration
# -----------------------------------------------------------------------------

def build_argparser():
    p=argparse.ArgumentParser(description="Analysis-only LaBraM x M3CV revised four-spectrum pipeline",
                              formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mode",choices=["preflight","heatmap","individual","shared-energy","random","shared-functional","sti","plot","main","all"],default="preflight")
    p.add_argument("--root",default="/omni-eeg-01/task calibration/results/tsv_labram_m3cv/tsv_labram_m3cv_v5_0")
    p.add_argument("--outdir",default="")
    p.add_argument("--core-script",default="")
    p.add_argument("--tasks",default=",".join(TASK_ORDER_DEFAULT))
    p.add_argument("--device",default="cuda")
    p.add_argument("--profile",choices=["fast","full"],default="fast",help="fast = overnight-oriented reduced replicate combinations and lighter probes; full = proposal-complete replicate grid")
    p.add_argument("--eval-batch-size",type=int,default=None)
    p.add_argument("--ram-reserve-gb",type=float,default=4.0)
    p.add_argument("--ortho-rtol",type=float,default=0.0,help="0 = machine-precision rank tolerance from proposal")
    p.add_argument("--shared-random-n",type=int,default=None)
    p.add_argument("--sti-random-n",type=int,default=None)
    p.add_argument("--random-seed",type=int,default=20260907)
    p.add_argument("--save-random-draws",action="store_true")
    p.add_argument("--resume",action="store_true",default=True)
    p.add_argument("--no-resume",dest="resume",action="store_false")
    p.add_argument("--metric-tol",type=float,default=2e-5)
    return p


def main():
    args=configure_runtime_profile(build_argparser().parse_args())
    root=Path(args.root).expanduser().resolve()
    log(f"Analysis profile={args.profile}; eval_batch_size={args.eval_batch_size}; shared_random_n={args.shared_random_n}; sti_random_n={args.sti_random_n}")
    outdir=Path(args.outdir).expanduser().resolve() if args.outdir else root/"analysis_four_spectra_v6_0"
    ensure_dir(outdir);ensure_dir(outdir/"figures");ensure_dir(outdir/"progress")
    tasks=parse_csv_strs(args.tasks)
    if tasks!=list(TASK_ORDER_DEFAULT):log(f"WARNING: task order differs from proposal: {tasks}")
    selftest=synthetic_algebra_selftest();dump_json(outdir/"algebra_selftest.json",selftest);log(f"Algebra self-test PASSED: {selftest}")
    runs,manifest,preflight=discover_runs(root,tasks);modules=discover_modules(runs[0].delta_path)
    delta_archive_audit=audit_delta_archives(runs,modules)
    core,core_path=load_core(args.core_script,root,tasks)
    verify_core_provenance(core_path,manifest,preflight)
    data_paths,checkpoint,modeling,chans,model_init_seed,expected_w0=protocol_paths(manifest,preflight,tasks)
    if args.mode=="preflight":
        payload={"status":"PASSED","tasks":tasks,"runs":[r.tag for r in runs],
                 "run_protocol_fingerprints":{r.tag:r.protocol_fingerprint for r in runs},"modules":modules,
                 "core_script":str(core_path),"checkpoint":str(checkpoint),"modeling_file":str(modeling),
                 "data_paths":{k:str(v) for k,v in data_paths.items()},"algebra_selftest":selftest,
                 "delta_archive_audit":delta_archive_audit,
                 "analysis_version":VERSION,"analysis_profile":args.profile,"sampling_policy":SAMPLING_POLICY,
                 "energy_absolute_vertical_targets":list(ENERGY_VERTICAL_TARGETS),
                 "bacc_absolute_vertical_grid":list(BACC_VERTICAL_TARGETS)}
        dump_json(outdir/"preflight_analysis.json",payload);log(f"Analysis preflight PASSED -> {outdir}");return
    import torch
    device=torch.device(args.device)
    if device.type=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA requested but unavailable")
    # exact task metas from training protocol
    metas={t:core.load_task_meta(t,data_paths[t]) for t in tasks}
    svds=load_all_svds(runs,modules)
    svd_checks={"max_reconstruction_relerr":max(r.recon_relerr for r in svds.values()),
                "max_uv_orthonormality_fro_error":max(r.orth_err for r in svds.values())}
    if svd_checks["max_reconstruction_relerr"]>1e-8 or svd_checks["max_uv_orthonormality_fro_error"]>1e-8:
        raise RuntimeError(svd_checks)
    dump_json(outdir/"svd_sanity.json",svd_checks)
    # Build W0 TSV once for heat map and exact W0 digest check
    w0model,_,bd=core.build_labram_backbone(modeling,checkpoint,device,model_init_seed)
    if expected_w0 and str(bd)!=str(expected_w0):raise RuntimeError("W0 digest mismatch in analysis")
    base_tsv={k:v.numpy().astype(np.float64,copy=True) for k,v in core.extract_tsv_weights(w0model).items()}
    del w0model
    # stages
    main_stages={"heatmap","individual","shared-energy","random","shared-functional","sti","plot"}
    want=lambda *m: args.mode=="all" or args.mode in m or (args.mode=="main" and any(x in main_stages for x in m))
    training=collect_training_summary(runs);write_csv(outdir/"training_summary.csv",training)
    if want("heatmap"):
        hrows=heatmap_rows(runs,modules,svds,base_tsv);write_csv(outdir/"weight_update_heatmap.csv",hrows);plot_training_qc(runs,outdir/"figures");plot_heatmaps(hrows,outdir/"figures",tasks)
    # Individual is prerequisite for any shared stage; load/recompute as needed
    individual_rows=[]
    ind_master=outdir/"individual_spectra.csv"
    if args.mode in ("individual","all","main") or args.mode in ("shared-energy","random","shared-functional","plot"):
        if ind_master.is_file() and args.resume and args.mode!="individual":
            cached=read_csv(ind_master)
            if sampling_rows_compatible(cached):
                individual_rows=cached
            else:
                log("Individual master cache uses an older sampling policy; rebuilding")
                for r in runs:
                    individual_rows.extend(individual_for_run(args,core,r,metas[r.task],modules,svds,manifest,checkpoint,modeling,chans,model_init_seed,expected_w0,device,outdir))
                write_csv(ind_master,individual_rows);write_csv(outdir/"individual_summary.csv",individual_summary(individual_rows))
        else:
            for r in runs:
                individual_rows.extend(individual_for_run(args,core,r,metas[r.task],modules,svds,manifest,checkpoint,modeling,chans,model_init_seed,expected_w0,device,outdir))
            write_csv(ind_master,individual_rows);write_csv(outdir/"individual_summary.csv",individual_summary(individual_rows))
        if args.mode in ("individual","all","main"):plot_individual(individual_rows,outdir/"figures",tasks)
    cutoffs=cutoffs_from_individual(individual_rows) if individual_rows else {}
    energy_rows=[];function_rows=[];random_rows=[]
    energy_master=outdir/"shared_energy_spectra.csv"
    if want("shared-energy") or args.mode in ("random","shared-functional","plot"):
        if energy_master.is_file() and args.resume and args.mode!="shared-energy":
            cached=read_csv(energy_master)
            if sampling_rows_compatible(cached):energy_rows=cached
            else:log("Shared Energy master cache uses an older sampling policy; rebuilding")
        if not energy_rows:
            comparisons=list(all_comparisons(runs,tasks,args.profile)); ncomp=len(comparisons)
            for ci,(cand,ctx,regime) in enumerate(comparisons,1):
                rows,comps,dims=shared_energy_curve(cand,ctx,regime,modules,cutoffs,svds,args.ortho_rtol);energy_rows.extend(rows)
                cpath=outdir/"components"/f"{hashlib.sha256(comparison_id(cand,ctx,regime).encode()).hexdigest()[:16]}.csv"
                for rank,x in enumerate(comps,1):x.update({"comparison_id":comparison_id(cand,ctx,regime),"shared_rank":rank,"candidate":cand.task,"candidate_replicate":cand.replicate,"context_regime":regime,"context_tasks":"+".join(r.task for r in ctx),"context_replicates":"+".join(str(r.replicate) for r in ctx)})
                write_csv(cpath,comps)
                if ci%10==0 or ci==ncomp:log(f"Shared Energy geometry {ci}/{ncomp}")
            write_csv(energy_master,energy_rows)
    if want("random") or args.mode=="plot":
        rand_master=outdir/"shared_energy_random_baseline.csv"
        if rand_master.is_file() and args.resume and args.mode!="random":
            cached=read_csv(rand_master)
            if sampling_rows_compatible(cached):random_rows=cached
            else:log("Random Shared Energy cache uses an older sampling policy; rebuilding")
        if not random_rows:
            comparisons=list(all_comparisons(runs,tasks,args.profile)); ncomp=len(comparisons)
            for ci,(cand,ctx,regime) in enumerate(comparisons,1):
                cid=comparison_id(cand,ctx,regime)
                observed=[r for r in energy_rows if r["comparison_id"]==cid]
                if not observed:raise RuntimeError(f"Missing observed Shared Energy samples for {cid}")
                random_rows.extend(random_shared_baseline_comparison(args,cand,ctx,regime,modules,cutoffs,svds,outdir,observed))
                log(f"Random Shared Energy comparison {ci}/{ncomp} complete")
            write_csv(rand_master,random_rows)
    if want("shared-functional") or args.mode=="plot":
        fn_master=outdir/"shared_functional_spectra.csv"
        if fn_master.is_file() and args.resume and args.mode!="shared-functional":
            cached=read_csv(fn_master)
            if sampling_rows_compatible(cached):function_rows=cached
            else:log("Shared Functional cache uses an older sampling policy; rebuilding")
        if not function_rows:
            comparisons=list(all_comparisons(runs,tasks,args.profile)); ncomp=len(comparisons)
            grouped=defaultdict(list)
            for item in comparisons: grouped[item[0].key].append(item)
            done=0
            for cand_key,items in grouped.items():
                cand=items[0][0]
                # Reuse one model + dataloader for all contexts of the same candidate.
                # Every evaluation rewrites all 72 analyzed matrices, so there is no
                # state leakage between context orderings.
                ev=CandidateEvaluator(core,cand,metas[cand.task],manifest,checkpoint,modeling,chans,model_init_seed,expected_w0,
                                      modules,svds,device,args.eval_batch_size,args.ram_reserve_gb,
                                      verify_full_replay=(args.profile=="full"))
                try:
                    for cand,ctx,regime in items:
                        function_rows.extend(shared_functional_comparison(args,core,cand,ctx,regime,metas[cand.task],modules,cutoffs,svds,manifest,checkpoint,modeling,chans,model_init_seed,expected_w0,device,outdir,evaluator=ev))
                        done+=1
                        write_csv(fn_master,function_rows);log(f"Shared Functional comparison {done}/{ncomp} complete")
                finally:
                    ev.close()
    sti_rows=[];sti_rand=[]
    if want("sti") or args.mode=="plot":
        stip=outdir/"sti_paper.csv";strp=outdir/"sti_random_baseline.csv"
        if stip.is_file() and args.resume:sti_rows=read_csv(stip)
        else:sti_rows=sti_rows_observed(runs,tasks,modules,svds);write_csv(stip,sti_rows)
        if strp.is_file() and args.resume:
            cached=read_csv(strp)
            if cached and all(int(float(r.get("random_n",-1)))==int(args.sti_random_n) for r in cached):sti_rand=cached
            else:sti_rand=sti_random_summary(args,runs,tasks,modules,svds);write_csv(strp,sti_rand)
        else:sti_rand=sti_random_summary(args,runs,tasks,modules,svds);write_csv(strp,sti_rand)
        write_csv(outdir/"sti_summary.csv",sti_summary(sti_rows,sti_rand))
    sanity={"svd":svd_checks}
    if cutoffs:
        sanity["context_nesting"]=nesting_sanity(runs,tasks,modules,cutoffs,svds,args.ortho_rtol,args.profile)
    if energy_rows and function_rows:
        write_csv(outdir/"shared_summary.csv",shared_summary(energy_rows,function_rows,random_rows))
    if args.mode in ("plot","all","main"):
        if not (outdir/"weight_update_heatmap.csv").is_file():
            hrows=heatmap_rows(runs,modules,svds,base_tsv);write_csv(outdir/"weight_update_heatmap.csv",hrows)
        else:hrows=read_csv(outdir/"weight_update_heatmap.csv")
        plot_training_qc(runs,outdir/"figures");plot_heatmaps(hrows,outdir/"figures",tasks)
        if not individual_rows and ind_master.is_file():individual_rows=read_csv(ind_master)
        if individual_rows:plot_individual(individual_rows,outdir/"figures",tasks)
        if not energy_rows and energy_master.is_file():energy_rows=read_csv(energy_master)
        if not function_rows and (outdir/"shared_functional_spectra.csv").is_file():function_rows=read_csv(outdir/"shared_functional_spectra.csv")
        if not random_rows and (outdir/"shared_energy_random_baseline.csv").is_file():random_rows=read_csv(outdir/"shared_energy_random_baseline.csv")
        if energy_rows and function_rows:
            for i,t in enumerate(tasks,5):plot_shared_candidate(t,"single",energy_rows,function_rows,random_rows,outdir/"figures"/f"{i:02d}_single_context_{t}")
            for i,t in enumerate(tasks,10):plot_shared_candidate(t,"full",energy_rows,function_rows,random_rows,outdir/"figures"/f"{i:02d}_full_context_{t}")
        if not sti_rows and (outdir/"sti_paper.csv").is_file():sti_rows=read_csv(outdir/"sti_paper.csv")
        if not sti_rand and (outdir/"sti_random_baseline.csv").is_file():sti_rand=read_csv(outdir/"sti_random_baseline.csv")
        if sti_rows and sti_rand:plot_sti(sti_rows,sti_rand,outdir/"figures")
    # Manifest is emitted for every completed stage.  A *global* success marker is
    # emitted only when all proposal-required artifacts exist; partial runs receive
    # an explicit stage marker so they cannot be mistaken for a complete analysis.
    am=build_manifest(args,root,outdir,core_path,manifest,preflight,modules,selftest,sanity,runs=runs)
    dump_json(outdir/"analysis_manifest_v6_0.json",am)
    required_complete=[
        "training_summary.csv","weight_update_heatmap.csv","individual_spectra.csv",
        "individual_summary.csv","shared_energy_spectra.csv",
        "shared_energy_random_baseline.csv","shared_functional_spectra.csv",
        "sti_paper.csv","sti_random_baseline.csv","sti_summary.csv","shared_summary.csv",
        "figures/01_full_finetuning_qc.png","figures/01_full_finetuning_qc.pdf",
        "figures/02_weight_update_heatmap.png","figures/02_weight_update_heatmap.pdf",
        "figures/03_sti_map.png","figures/03_sti_map.pdf",
        "figures/03_sti_depth_profile.png","figures/03_sti_depth_profile.pdf",
        "figures/04_individual_spectra.png","figures/04_individual_spectra.pdf",
    ]
    for i,t in enumerate(tasks,5):
        required_complete.extend([f"figures/{i:02d}_single_context_{t}.png",f"figures/{i:02d}_single_context_{t}.pdf"])
    for i,t in enumerate(tasks,10):
        required_complete.extend([f"figures/{i:02d}_full_context_{t}.png",f"figures/{i:02d}_full_context_{t}.pdf"])
    complete=all((outdir/x).is_file() for x in required_complete)
    marker_payload={"version":VERSION,"analysis_manifest_sha256":sha256_file(outdir/"analysis_manifest_v6_0.json"),
                    "mode":args.mode,"timestamp":now(),"proposal_complete":bool(complete)}
    stage_name="".join(ch if ch.isalnum() else "_" for ch in args.mode.upper())
    dump_json(outdir/f"_STAGE_{stage_name}_SUCCESS.json",marker_payload)
    if complete:
        dump_json(outdir/"_ANALYSIS_V6_0_SUCCESS.json",marker_payload)
    log(f"Completed {args.mode}. proposal_complete={complete}. Outputs: {outdir}")


if __name__=="__main__":
    main()
