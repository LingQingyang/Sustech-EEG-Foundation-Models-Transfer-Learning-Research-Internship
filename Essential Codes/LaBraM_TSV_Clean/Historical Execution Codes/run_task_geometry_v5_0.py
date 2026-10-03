#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LaBraM x M3CV Task Adaptation Geometry, Version 5.0
===================================================
Unified training / provenance orchestrator.

V5.0 is intentionally a thin scientific orchestration layer over the mature
`run_tsv_labram_m3cv_v0_3_4.py` implementation.  The numerical training code,
model loading, optimizer, data staging, checkpoint writing and 72-matrix
Delta-W extraction remain in that audited core.

V5.0 final-grid policy
----------------------
* Five tasks, two replicates each: Rest, Motor, P300, SSS, TS.
* Common endpoint E*=20 is frozen from the certified legacy 8-run pilot:
  `pilot_max_epoch_fallback`, with all eight strict plateau certifications
  missing.  V5.0 does NOT pretend that a new ten-run pilot was run.
* Reuse exactly seven certified legacy final runs:
    Motor/rep01
    P300/rep01, P300/rep02
    SSS/rep01,  SSS/rep02
    TS/rep01,   TS/rep02
* Reject the legacy Motor/rep02 artifact by policy and retrain it.
* Train exactly three new runs from the identical W0 for exactly 20 epochs:
    Motor/rep02, Rest/rep01, Rest/rep02
* Legacy runs are exposed in the V5 root with REP-LEVEL symlinks.  The entire
  Motor task directory is never symlinked, so the rejected legacy rep02 cannot
  leak into the V5 grid.

Rest contract verified from the actual H5 metadata:
    raw 0 = Beg_EC (Task 1)
    raw 2 = Beg_EO (Task 2)

Typical workflow
----------------
    python3 run_task_geometry_v5_0.py --mode preflight
    python3 run_task_geometry_v5_0.py --mode inspect
    python3 run_task_geometry_v5_0.py --mode final

There is deliberately no V5 pilot mode.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    import run_tsv_labram_m3cv_v0_3_4 as core
except Exception as exc:
    raise RuntimeError(
        "Cannot import run_tsv_labram_m3cv_v0_3_4.py. "
        "Keep that mature core trainer next to this V5.0 script."
    ) from exc


VERSION = "5.0"
ALL_TASKS = ["Rest", "Motor", "P300", "SSS", "TS"]
REPLICATES = 2
EXPECTED_E_STAR = 20

LEGACY_APPROVED: Tuple[Tuple[str, int], ...] = (
    ("Motor", 1),
    ("P300", 1), ("P300", 2),
    ("SSS", 1), ("SSS", 2),
    ("TS", 1), ("TS", 2),
)
NEW_RUNS: Tuple[Tuple[str, int], ...] = (
    ("Motor", 2),
    ("Rest", 1), ("Rest", 2),
)
REJECTED_LEGACY: Tuple[Tuple[str, int], ...] = (("Motor", 2),)

DEFAULT_ROOT = Path(
    "/omni-eeg-01/task calibration/results/tsv_labram_m3cv/"
    "tsv_labram_m3cv_v5_0"
)
DEFAULT_LEGACY_ROOT = Path(
    "/omni-eeg-01/task calibration/results/tsv_labram_m3cv/"
    "tsv_labram_m3cv_v0_3_4"
)
DEFAULT_DATA_ROOT = Path("/omni-eeg-01/task calibration/dataset/m3cv")
DEFAULT_CHECKPOINT = Path(
    "/omni-eeg-01/task calibration/LaBraM/checkpoints/labram-base.pth"
)
DEFAULT_MODELING_FILE = Path(
    "/omni-eeg-01/task calibration/LaBraM/modeling_finetune.py"
)

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

# Patch only the public scientific registry/version.  All numerical routines
# continue to come from the mature core file.
core.VERSION = VERSION
core.TASK_FILES = dict(TASK_FILES)
core.TASK_CLASS_NAMES = {k: dict(v) for k, v in TASK_CLASS_NAMES.items()}
core.EXPECTED_CLASSES = {k: len(v) for k, v in TASK_CLASS_NAMES.items()}


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def log(msg: str) -> None:
    core.log(f"[V5.0] {msg}")


def sha256_file(path: Path) -> str:
    return core.sha256_file(Path(path))


def load_json(path: Path):
    return core.load_json(Path(path))


def require_file(path: Path, what: str) -> Path:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Missing {what}: {p}")
    return p


def run_key(task: str, rep: int) -> str:
    return f"{task}/rep{int(rep):02d}"


def script_sha256() -> str:
    return sha256_file(Path(__file__).resolve())


def expected_module_shapes() -> Dict[str, Tuple[int, int]]:
    out: Dict[str, Tuple[int, int]] = {}
    for layer in range(12):
        for role in ("Q", "K", "V", "O"):
            out[f"L{layer:02d}/{role}"] = (200, 200)
        out[f"L{layer:02d}/fc1"] = (800, 200)
        out[f"L{layer:02d}/fc2"] = (200, 800)
    return out


EXPECTED_MODULE_SHAPES = expected_module_shapes()


def _artifact_hashes(run_dir: Path) -> Dict[str, str]:
    return {
        "metadata_sha256": sha256_file(run_dir / "metadata.json"),
        "delta_weights_sha256": sha256_file(run_dir / "delta_weights.npz"),
        "adapted_checkpoint_sha256": sha256_file(run_dir / "adapted_checkpoint.pth"),
    }


def verify_success_pair(payload_path: Path, success_path: Path,
                        hash_field: str, what: str):
    payload = load_json(require_file(payload_path, what))
    success = load_json(require_file(success_path, f"{what} success marker"))
    actual = sha256_file(payload_path)
    if str(success.get(hash_field, "")) != actual:
        raise RuntimeError(
            f"{what}: {hash_field} mismatch; expected={success.get(hash_field)}, "
            f"actual={actual}"
        )
    return payload, success


# -----------------------------------------------------------------------------
# Core argument construction / V5 preflight
# -----------------------------------------------------------------------------

def make_core_args(args) -> argparse.Namespace:
    cargs = core.build_argparser().parse_args([])
    cargs.mode = "preflight"
    cargs.data_root = str(args.data_root)
    cargs.checkpoint = str(args.checkpoint)
    cargs.modeling_file = str(args.modeling_file)
    cargs.outdir = str(args.root)
    cargs.tasks = list(ALL_TASKS)
    cargs.replicates = REPLICATES
    cargs.seed_base = int(args.seed_base)
    cargs.model_init_seed = int(args.model_init_seed)

    cargs.batch_size = 32
    cargs.eval_batch_size = 64
    cargs.workers = int(args.workers)
    cargs.backbone_lr = 1e-4
    cargs.head_lr = 1e-3
    cargs.weight_decay = 0.0
    cargs.beta1 = 0.9
    cargs.beta2 = 0.999
    cargs.adam_eps = 1e-8
    cargs.amp = False

    cargs.pilot_max_epochs = EXPECTED_E_STAR
    cargs.epochs = 0
    cargs.conv_window = 3
    cargs.conv_patience = 2
    cargs.conv_loss_rel = 0.03
    cargs.conv_bacc_abs = 0.01
    cargs.conv_min_bacc_margin = 0.05

    # Keep mature-core defaults for the remaining preflight thresholds and
    # probes, but normalize the CLI string to the list expected by run_task().
    if isinstance(cargs.energy_probes, str):
        cargs.energy_probes = core.parse_csv_ints(cargs.energy_probes)

    cargs.ram_reserve_gb = float(args.ram_reserve_gb)
    cargs.overwrite_run = False
    cargs.device = str(args.device)
    cargs.input_chans = ",".join(str(i) for i in range(1, 65))
    cargs.input_chans_file = ""
    cargs.allow_provisional_channels = False

    # No unsafe scientific overrides in V5.0.
    for name in (
        "unsafe_allow_missing_label_mapping",
        "unsafe_allow_subject_set_mismatch",
        "unsafe_allow_subject_class_gaps",
        "unsafe_allow_manual_final_epochs",
    ):
        if hasattr(cargs, name):
            setattr(cargs, name, False)

    core.validate_args(cargs)
    return cargs


def _seal_v5_protocol(payload: dict, args) -> dict:
    """Replace the mature core's generic final-epoch clause with the real V5 policy.

    The core preflight correctly audits data/model/optimizer/etc.  Its generic
    protocol, however, says that a matching current-root pilot is required.
    V5.0 deliberately inherits E*=20 from the certified historical pilot, so
    leaving that field untouched would make the provenance statement false.
    """
    protocol = dict(payload["protocol"])
    protocol["runner_version"] = VERSION
    protocol["final_epoch_contract"] = {
        "policy": "frozen_E_star_from_certified_legacy_pilot",
        "matching_current_root_pilot_required": False,
        "legacy_pilot_required": True,
        "legacy_root": str(Path(args.legacy_root).resolve()),
        "expected_E_star": EXPECTED_E_STAR,
        "required_legacy_E_star_source": "pilot_max_epoch_fallback",
        "required_legacy_n_uncertified_runs": 8,
        "manual_cli_epoch_override_used": False,
    }
    protocol["orchestration_contract"] = {
        "version": VERSION,
        "orchestrator_sha256": script_sha256(),
        "mature_core_path": str(Path(core.__file__).resolve()),
        "mature_core_sha256": sha256_file(Path(core.__file__).resolve()),
        "legacy_reused_runs": [run_key(*x) for x in LEGACY_APPROVED],
        "new_runs": [run_key(*x) for x in NEW_RUNS],
        "rejected_legacy_runs": [run_key(*x) for x in REJECTED_LEGACY],
        "legacy_link_granularity": "replicate_directory",
        "destructive_edit_of_legacy_root": False,
    }
    fp = core.canonical_json_sha256(protocol)
    sealed = dict(payload)
    sealed["version"] = VERSION
    sealed["protocol"] = protocol
    sealed["protocol_fingerprint"] = fp
    sealed["v5_sealed_from_core_preflight"] = True
    return sealed


def run_v5_preflight(args) -> dict:
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    cargs = make_core_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    checkpoint = require_file(Path(args.checkpoint), "LaBraM checkpoint")
    modeling_file = require_file(Path(args.modeling_file), "modeling_finetune.py")
    input_chans, source, provisional = core.resolve_input_chans(cargs)
    if provisional:
        raise RuntimeError("V5.0 forbids provisional channel mapping")
    metas = core.load_all_metas(cargs)

    core.dump_json(root / "subject_metadata.json", core.subject_metadata(metas))
    core.dump_json(root / "final_config.json", {
        "version": VERSION,
        "root": str(root),
        "legacy_root": str(Path(args.legacy_root)),
        "tasks": ALL_TASKS,
        "replicates": REPLICATES,
        "frozen_E_star": EXPECTED_E_STAR,
        "legacy_reused_runs": [run_key(*x) for x in LEGACY_APPROVED],
        "new_runs": [run_key(*x) for x in NEW_RUNS],
        "rejected_legacy_runs": [run_key(*x) for x in REJECTED_LEGACY],
        "input_chans": input_chans,
        "checkpoint": str(checkpoint),
        "modeling_file": str(modeling_file),
        "orchestrator_sha256": script_sha256(),
        "mature_core_sha256": sha256_file(Path(core.__file__).resolve()),
    })

    log("Running fresh five-task V5.0 preflight")
    payload = core.run_preflight(
        cargs, metas, input_chans, source, provisional,
        modeling_file, checkpoint, device,
    )
    sealed = _seal_v5_protocol(payload, args)
    core.dump_json(root / "preflight.json", sealed)
    core.dump_json(root / "_PREFLIGHT_SUCCESS.json", {
        "version": VERSION,
        "protocol_fingerprint": sealed["protocol_fingerprint"],
        "preflight_sha256": sha256_file(root / "preflight.json"),
        "orchestrator_sha256": script_sha256(),
    })
    log(
        "V5.0 preflight PASSED and sealed. "
        f"Protocol fingerprint={sealed['protocol_fingerprint']}"
    )
    return sealed


def verify_v5_preflight(args) -> dict:
    root = Path(args.root)
    pf, ok = verify_success_pair(
        root / "preflight.json",
        root / "_PREFLIGHT_SUCCESS.json",
        "preflight_sha256",
        "V5.0 five-task preflight",
    )
    if str(pf.get("status")) != "PASSED":
        raise RuntimeError("V5.0 preflight status is not PASSED")
    fp = str(pf.get("protocol_fingerprint", ""))
    if not fp or fp != str(ok.get("protocol_fingerprint", "")):
        raise RuntimeError("V5.0 preflight fingerprint mismatch")
    if core.canonical_json_sha256(pf.get("protocol", {})) != fp:
        raise RuntimeError("V5.0 preflight canonical protocol hash mismatch")
    protocol = pf.get("protocol", {})
    if [str(x) for x in protocol.get("tasks", [])] != ALL_TASKS:
        raise RuntimeError("V5.0 preflight does not certify the exact five-task order")
    if int(protocol.get("replicates", -1)) != REPLICATES:
        raise RuntimeError("V5.0 preflight must certify exactly two replicates")
    final_contract = protocol.get("final_epoch_contract", {}) or {}
    if final_contract.get("policy") != "frozen_E_star_from_certified_legacy_pilot":
        raise RuntimeError("V5.0 preflight has the wrong final-epoch contract")
    if int(final_contract.get("expected_E_star", -1)) != EXPECTED_E_STAR:
        raise RuntimeError("V5.0 preflight has the wrong frozen E*")
    if str(protocol.get("orchestration_contract", {}).get("orchestrator_sha256", "")) != script_sha256():
        raise RuntimeError(
            "V5.0 orchestrator bytes changed since preflight. Rerun preflight before training."
        )

    # Same fail-closed reuse logic as the mature core: avoid rescanning every EEG
    # byte, but require file size/mtime plus label and subject hashes to remain
    # identical to the certified preflight.  Rest is included here explicitly.
    data_audits = protocol.get("data", {}) or {}
    for task in ALL_TASKS:
        meta = core.load_task_meta(task, Path(args.data_root) / TASK_FILES[task])
        audit = data_audits.get(task, {}) or {}
        st = meta.path.stat()
        if int(audit.get("file_size_bytes", -1)) != int(st.st_size):
            raise RuntimeError(f"{task}: H5 size changed since V5 preflight")
        if int(audit.get("file_mtime_ns", -1)) != int(st.st_mtime_ns):
            raise RuntimeError(f"{task}: H5 mtime changed since V5 preflight")
        if str(audit.get("labels_sha256", "")) != core._sha256_ndarray(meta.labels):
            raise RuntimeError(f"{task}: labels changed since V5 preflight")
        if str(audit.get("subject_ids_sha256", "")) != core._sha256_ndarray(meta.subjects):
            raise RuntimeError(f"{task}: subject IDs changed since V5 preflight")

    checkpoint = Path(args.checkpoint)
    modeling_file = Path(args.modeling_file)
    if sha256_file(checkpoint) != str(protocol.get("checkpoint", {}).get("sha256", "")):
        raise RuntimeError("LaBraM checkpoint changed since V5 preflight")
    if sha256_file(modeling_file) != str(protocol.get("modeling_file", {}).get("sha256", "")):
        raise RuntimeError("modeling_finetune.py changed since V5 preflight")
    return pf


# -----------------------------------------------------------------------------
# Legacy certification and compatibility
# -----------------------------------------------------------------------------

def verify_legacy_pilot(legacy_root: Path):
    summary, ok = verify_success_pair(
        legacy_root / "pilot" / "pilot_summary.json",
        legacy_root / "pilot" / "_SUCCESS.json",
        "pilot_summary_sha256",
        "legacy v0.3.4 pilot",
    )
    fp = str(summary.get("protocol_fingerprint", ""))
    if not fp or fp != str(ok.get("protocol_fingerprint", "")):
        raise RuntimeError("Legacy pilot fingerprint mismatch")
    if int(summary.get("recommended_E_star", -1)) != EXPECTED_E_STAR:
        raise RuntimeError(
            f"Legacy pilot E*={summary.get('recommended_E_star')} != {EXPECTED_E_STAR}"
        )
    if str(summary.get("E_star_source", "")) != "pilot_max_epoch_fallback":
        raise RuntimeError("Legacy pilot E* was not produced by pilot_max_epoch_fallback")
    if int(summary.get("n_uncertified_runs", -1)) != 8:
        raise RuntimeError("Legacy pilot must record n_uncertified_runs=8")
    runs = summary.get("runs", [])
    if len(runs) != 8:
        raise RuntimeError(f"Legacy pilot should contain eight runs, found {len(runs)}")
    return summary, ok


def verify_delta_archive(path: Path, tag: str) -> None:
    with np.load(path, allow_pickle=False) as z:
        modules = {core.unsanitize_module_key(k): tuple(int(x) for x in z[k].shape)
                   for k in z.files}
    if set(modules) != set(EXPECTED_MODULE_SHAPES):
        missing = sorted(set(EXPECTED_MODULE_SHAPES) - set(modules))
        extra = sorted(set(modules) - set(EXPECTED_MODULE_SHAPES))
        raise RuntimeError(f"{tag}: Delta-W module grid mismatch; missing={missing}, extra={extra}")
    bad_shapes = {
        m: (modules[m], EXPECTED_MODULE_SHAPES[m])
        for m in modules if modules[m] != EXPECTED_MODULE_SHAPES[m]
    }
    if bad_shapes:
        raise RuntimeError(f"{tag}: Delta-W shapes mismatch: {bad_shapes}")


def _verify_checkpoint_payload(path: Path, task: str, rep: int,
                               expected_fp: str, expected_w0: str,
                               expected_input_chans: Optional[Sequence[int]]) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"{run_key(task, rep)}: checkpoint payload is not a mapping")
    if str(payload.get("task")) != task or int(payload.get("replicate", -1)) != rep:
        raise RuntimeError(f"{run_key(task, rep)}: checkpoint internal identity mismatch")
    if str(payload.get("protocol_fingerprint", "")) != expected_fp:
        raise RuntimeError(f"{run_key(task, rep)}: checkpoint internal fingerprint mismatch")
    if str(payload.get("base_full_backbone_digest", "")) != expected_w0:
        raise RuntimeError(f"{run_key(task, rep)}: checkpoint internal W0 mismatch")
    if expected_input_chans is not None:
        got = [int(x) for x in payload.get("input_chans", [])]
        if got != [int(x) for x in expected_input_chans]:
            raise RuntimeError(f"{run_key(task, rep)}: checkpoint input_chans mismatch")
    if not isinstance(payload.get("backbone"), Mapping) or not isinstance(payload.get("head"), Mapping):
        raise RuntimeError(f"{run_key(task, rep)}: checkpoint lacks backbone/head mappings")
    del payload
    gc.collect()


def verify_run_dir(run_dir: Path, task: str, rep: int,
                   expected_epochs: int, expected_w0: str,
                   expected_fp: str,
                   expected_input_chans: Optional[Sequence[int]] = None,
                   expected_seed_base: Optional[int] = None,
                   verify_checkpoint_content: bool = True) -> dict:
    tag = run_key(task, rep)
    run_dir = Path(run_dir)
    mp = require_file(run_dir / "metadata.json", f"{tag} metadata")
    dp = require_file(run_dir / "delta_weights.npz", f"{tag} Delta-W")
    cp = require_file(run_dir / "adapted_checkpoint.pth", f"{tag} checkpoint")
    sp = require_file(run_dir / "_SUCCESS.json", f"{tag} success marker")

    meta = load_json(mp)
    success = load_json(sp)
    if str(meta.get("task")) != task or int(meta.get("replicate", -1)) != int(rep):
        raise RuntimeError(f"{tag}: metadata identity mismatch")
    if int(meta.get("epochs", -1)) != int(expected_epochs):
        raise RuntimeError(f"{tag}: epochs={meta.get('epochs')} != {expected_epochs}")
    if str(meta.get("base_full_backbone_digest", "")) != expected_w0:
        raise RuntimeError(f"{tag}: metadata W0 digest mismatch")
    if str(meta.get("protocol_fingerprint", "")) != expected_fp:
        raise RuntimeError(f"{tag}: metadata protocol fingerprint mismatch")
    if expected_input_chans is not None:
        got = [int(x) for x in meta.get("input_chans", [])]
        if got != [int(x) for x in expected_input_chans]:
            raise RuntimeError(f"{tag}: metadata input_chans mismatch")
    expected_classes = sorted(int(x) for x in TASK_CLASS_NAMES[task])
    if [int(x) for x in meta.get("raw_classes", [])] != expected_classes:
        raise RuntimeError(f"{tag}: raw class IDs mismatch")
    if int(meta.get("num_classes", -1)) != len(expected_classes):
        raise RuntimeError(f"{tag}: num_classes mismatch")
    if int(meta.get("n_subjects", 95)) != 95:
        raise RuntimeError(f"{tag}: expected 95 subjects")
    opt = meta.get("optimizer", {}) or {}
    expected_opt = {
        "name": "AdamW",
        "backbone_lr": 1e-4,
        "head_lr": 1e-3,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.0,
        "gradient_clipping": False,
        "scheduler": None,
        "warmup": None,
        "early_stopping": False,
        "amp": False,
    }
    for field, expected_value in expected_opt.items():
        if field in opt and opt.get(field) != expected_value:
            raise RuntimeError(
                f"{tag}: optimizer field {field}={opt.get(field)!r} != {expected_value!r}"
            )
    if int(meta.get("batch_size", 32)) != 32 or int(meta.get("eval_batch_size", 64)) != 64:
        raise RuntimeError(f"{tag}: batch-size contract mismatch")

    actual = _artifact_hashes(run_dir)
    if str(success.get("metadata_sha256", "")) != actual["metadata_sha256"]:
        raise RuntimeError(f"{tag}: metadata hash mismatch")
    if str(success.get("delta_weights_sha256", "")) != actual["delta_weights_sha256"]:
        raise RuntimeError(f"{tag}: Delta-W hash mismatch")
    if str(success.get("adapted_checkpoint_sha256", "")) != actual["adapted_checkpoint_sha256"]:
        raise RuntimeError(f"{tag}: checkpoint hash mismatch")
    if str(success.get("protocol_fingerprint", "")) != expected_fp:
        raise RuntimeError(f"{tag}: success-marker fingerprint mismatch")
    if str(success.get("base_full_backbone_digest", "")) != expected_w0:
        raise RuntimeError(f"{tag}: success-marker W0 mismatch")

    meta_art = meta.get("artifacts", {}) or {}
    for k in ("delta_weights_sha256", "adapted_checkpoint_sha256"):
        if k in meta_art and str(meta_art[k]) != actual[k]:
            raise RuntimeError(f"{tag}: metadata artifact hash {k} mismatch")

    if expected_seed_base is not None:
        seed_expected = {
            "head_seed": core.stable_seed(expected_seed_base, task, rep, "head"),
            "runtime_seed": core.stable_seed(expected_seed_base, task, rep, "runtime", "final"),
            "loader_seed": core.stable_seed(expected_seed_base, task, rep, "loader", "final"),
        }
        for field, value in seed_expected.items():
            if field in meta and int(meta[field]) != int(value):
                raise RuntimeError(f"{tag}: {field} mismatch")

    verify_delta_archive(dp, tag)
    if verify_checkpoint_content:
        _verify_checkpoint_payload(
            cp, task, rep, expected_fp, expected_w0, expected_input_chans
        )
    return meta


def inspect_rejected_legacy(legacy_root: Path, legacy_fp: str) -> dict:
    """Collect diagnostics for old Motor/rep02 without allowing it into V5."""
    task, rep = REJECTED_LEGACY[0]
    run_dir = legacy_root / "runs" / task / f"rep{rep:02d}"
    out = {
        "run": run_key(task, rep),
        "origin_path": str(run_dir.resolve()) if run_dir.exists() else str(run_dir),
        "rejected_by_policy": True,
        "reason": (
            "Legacy Motor/rep02 is explicitly excluded from V5.0 after an "
            "artifact-integrity concern; V5.0 retrains the same replicate from W0."
        ),
        "legacy_protocol_fingerprint": legacy_fp,
        "files_present": {},
    }
    for name in ("metadata.json", "delta_weights.npz", "adapted_checkpoint.pth", "_SUCCESS.json"):
        p = run_dir / name
        out["files_present"][name] = p.is_file()
    try:
        success = load_json(run_dir / "_SUCCESS.json")
        out["recorded_hashes"] = {
            "metadata_sha256": success.get("metadata_sha256"),
            "delta_weights_sha256": success.get("delta_weights_sha256"),
            "adapted_checkpoint_sha256": success.get("adapted_checkpoint_sha256"),
        }
        actual = {}
        for name, key in (
            ("metadata.json", "metadata_sha256"),
            ("delta_weights.npz", "delta_weights_sha256"),
            ("adapted_checkpoint.pth", "adapted_checkpoint_sha256"),
        ):
            p = run_dir / name
            if p.is_file():
                actual[key] = sha256_file(p)
        out["actual_hashes"] = actual
        out["hash_matches"] = {
            k: (str(out["recorded_hashes"].get(k, "")) == str(actual.get(k, "")))
            for k in actual
        }
    except Exception as exc:
        out["diagnostic_error"] = repr(exc)
    return out


def compare_common_protocol(legacy: Mapping, new: Mapping) -> None:
    """Compare only scientific fields that must be invariant across old/new runs."""
    mismatches: List[Tuple[str, object, object]] = []

    def cmp(path: str, a, b):
        if a != b:
            mismatches.append((path, a, b))

    cmp("runner_sha256", legacy.get("runner_sha256"), new.get("runner_sha256"))
    cmp("replicates", legacy.get("replicates"), new.get("replicates"))
    cmp("input_chans", legacy.get("input_chans"), new.get("input_chans"))
    cmp("preprocessing_contract", legacy.get("preprocessing_contract"), new.get("preprocessing_contract"))
    cmp("optimization", legacy.get("optimization"), new.get("optimization"))
    cmp("convergence", legacy.get("convergence"), new.get("convergence"))
    cmp("tsv_contract", legacy.get("tsv_contract"), new.get("tsv_contract"))

    for key in ("seed_base", "model_init_seed", "determinism", "environment"):
        if key in legacy or key in new:
            cmp(key, legacy.get(key), new.get(key))

    lck, nck = legacy.get("checkpoint", {}) or {}, new.get("checkpoint", {}) or {}
    cmp("checkpoint.sha256", lck.get("sha256"), nck.get("sha256"))
    cmp("checkpoint.base_full_backbone_digest",
        lck.get("base_full_backbone_digest"), nck.get("base_full_backbone_digest"))
    lm, nm = legacy.get("modeling_file", {}) or {}, new.get("modeling_file", {}) or {}
    cmp("modeling_file.sha256", lm.get("sha256"), nm.get("sha256"))

    # The four legacy-task H5s must be byte-identical to the V5 preflight data.
    ldata, ndata = legacy.get("data", {}) or {}, new.get("data", {}) or {}
    for task in ("Motor", "P300", "SSS", "TS"):
        if task not in ldata or task not in ndata:
            mismatches.append((f"data.{task}", "missing", "missing"))
            continue
        for field in (
            "shape", "content_sha256", "labels_sha256", "subject_ids_sha256",
            "label_mapping", "label_mapping_verified",
        ):
            cmp(f"data.{task}.{field}", ldata[task].get(field), ndata[task].get(field))

    if mismatches:
        details = "\n".join(
            f"  {path}: legacy={a!r} | V5={b!r}" for path, a, b in mismatches
        )
        raise RuntimeError("Legacy/V5 scientific-contract mismatch:\n" + details)


def verify_legacy_final(legacy_root: Path, pf: dict):
    manifest, ok = verify_success_pair(
        legacy_root / "run_manifest.json",
        legacy_root / "_FINAL_SUCCESS.json",
        "run_manifest_sha256",
        "legacy v0.3.4 final manifest",
    )
    if [str(x) for x in manifest.get("tasks", [])] != ["Motor", "P300", "SSS", "TS"]:
        raise RuntimeError("Legacy final task grid is not the expected four-task grid")
    if int(manifest.get("replicates", -1)) != 2:
        raise RuntimeError("Legacy final must have exactly two replicates")
    if int(manifest.get("epochs", -1)) != EXPECTED_E_STAR:
        raise RuntimeError("Legacy final endpoint differs from E*=20")
    if int(ok.get("n_runs", -1)) != 8:
        raise RuntimeError("Legacy _FINAL_SUCCESS does not certify eight runs")

    new_protocol = pf["protocol"]
    expected_w0 = str(new_protocol["checkpoint"]["base_full_backbone_digest"])
    if str(manifest.get("base_full_backbone_digest", "")) != expected_w0:
        raise RuntimeError("Legacy final W0 differs from V5 preflight W0")
    legacy_fp = str(manifest.get("protocol_fingerprint", ""))
    if not legacy_fp:
        raise RuntimeError("Legacy final has no protocol fingerprint")

    compare_common_protocol(manifest.get("protocol", {}), new_protocol)

    input_chans = [int(x) for x in new_protocol["input_chans"]]
    seed_base = int(new_protocol.get("seed_base", 20260819))
    rows = []
    provenance = {}
    for task, rep in LEGACY_APPROVED:
        run_dir = legacy_root / "runs" / task / f"rep{rep:02d}"
        meta = verify_run_dir(
            run_dir, task, rep, EXPECTED_E_STAR, expected_w0, legacy_fp,
            expected_input_chans=input_chans,
            expected_seed_base=seed_base,
            verify_checkpoint_content=True,
        )
        hashes = _artifact_hashes(run_dir)
        key = run_key(task, rep)
        provenance[key] = {
            "source": "legacy_v0.3.4_reused",
            "origin_path": str(run_dir.resolve()),
            "protocol_fingerprint": legacy_fp,
            "base_full_backbone_digest": expected_w0,
            "epochs": EXPECTED_E_STAR,
            "artifacts": hashes,
        }
        rows.append({
            "task": task,
            "replicate": rep,
            "n_samples": int(meta.get("n_samples", -1)),
            "epochs": int(meta.get("epochs", -1)),
            "convergence_epoch": meta.get("convergence_epoch"),
            "final_bacc": float(meta["metrics_final"]["balanced_accuracy"]),
            "protocol_fingerprint": legacy_fp,
            "base_full_backbone_digest": expected_w0,
            "artifact_source": "legacy_v0.3.4_reused",
        })

    rejected = inspect_rejected_legacy(legacy_root, legacy_fp)
    return manifest, rows, provenance, rejected


def inspect(args):
    pf = verify_v5_preflight(args)
    pilot, _ = verify_legacy_pilot(Path(args.legacy_root))
    legacy_manifest, legacy_rows, legacy_provenance, rejected = verify_legacy_final(
        Path(args.legacy_root), pf
    )
    expected_w0 = str(pf["protocol"]["checkpoint"]["base_full_backbone_digest"])

    log("INSPECT PASSED")
    log(f"V5 protocol fingerprint = {pf['protocol_fingerprint']}")
    log(
        f"Legacy pilot E*={pilot['recommended_E_star']} "
        f"({pilot['E_star_source']}; n_uncertified_runs={pilot['n_uncertified_runs']})"
    )
    log("Seven approved legacy final runs are artifact-certified")
    status = rejected.get("hash_matches", {})
    if status:
        log(f"Rejected legacy Motor/rep02 diagnostics: {status}")
    log(f"Common W0 = {expected_w0[:16]}...")
    return pf, pilot, legacy_manifest, legacy_rows, legacy_provenance, rejected, expected_w0


# -----------------------------------------------------------------------------
# New-run training
# -----------------------------------------------------------------------------

def make_training_args(args, pf: dict) -> argparse.Namespace:
    cargs = make_core_args(args)
    p = pf["protocol"]
    opt = p["optimization"]
    conv = p["convergence"]
    cargs.mode = "final"
    cargs.tasks = list(ALL_TASKS)
    cargs.replicates = REPLICATES
    cargs.seed_base = int(p.get("seed_base", args.seed_base))
    cargs.model_init_seed = int(p.get("model_init_seed", args.model_init_seed))
    cargs.batch_size = int(opt["batch_size"])
    cargs.eval_batch_size = int(opt["eval_batch_size"])
    cargs.backbone_lr = float(opt["backbone_lr"])
    cargs.head_lr = float(opt["head_lr"])
    cargs.weight_decay = float(opt["weight_decay"])
    cargs.beta1 = float(opt["betas"][0])
    cargs.beta2 = float(opt["betas"][1])
    cargs.adam_eps = float(opt["eps"])
    cargs.amp = bool(opt["amp"])
    cargs.conv_window = int(conv["window"])
    cargs.conv_patience = int(conv["patience"])
    cargs.conv_loss_rel = float(conv["loss_relative_range"])
    cargs.conv_bacc_abs = float(conv["bacc_absolute_range"])
    cargs.conv_min_bacc_margin = float(conv["minimum_bacc_margin_over_chance"])
    cargs.energy_probes = [int(x) for x in p["tsv_contract"]["energy_probes"]]
    cargs.input_chans = ",".join(str(int(x)) for x in p["input_chans"])
    cargs.input_chans_file = ""
    cargs.allow_provisional_channels = False
    cargs.overwrite_run = False
    return cargs


def safely_remove_new_run_dir(root: Path, task: str, rep: int) -> None:
    if (task, rep) not in NEW_RUNS:
        raise RuntimeError(f"Refusing to remove non-new run {run_key(task, rep)}")
    run_dir = root / "runs" / task / f"rep{rep:02d}"
    task_dir = root / "runs" / task
    if task_dir.is_symlink():
        raise RuntimeError(f"Refusing to modify task-level symlink: {task_dir}")
    expected_parent = task_dir.resolve()
    if run_dir.parent.resolve() != expected_parent:
        raise RuntimeError(f"Unsafe run path: {run_dir}")
    if run_dir.is_symlink():
        run_dir.unlink()
        return
    if run_dir.exists():
        shutil.rmtree(run_dir)


def train_new_runs(args, pf: dict, expected_w0: str) -> None:
    fp = str(pf["protocol_fingerprint"])
    protocol = pf["protocol"]
    checkpoint = Path(args.checkpoint)
    modeling_file = Path(args.modeling_file)
    if sha256_file(checkpoint) != str(protocol["checkpoint"]["sha256"]):
        raise RuntimeError("Checkpoint bytes changed since V5 preflight")
    if sha256_file(modeling_file) != str(protocol["modeling_file"]["sha256"]):
        raise RuntimeError("modeling_finetune.py changed since V5 preflight")

    cargs = make_training_args(args, pf)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    input_chans = [int(x) for x in protocol["input_chans"]]
    seed_base = int(protocol.get("seed_base", args.seed_base))

    # Validate all existing new-run directories before touching any GPU work.
    to_train: Dict[str, List[int]] = {"Motor": [], "Rest": []}
    for task, rep in NEW_RUNS:
        run_dir = Path(args.root) / "runs" / task / f"rep{rep:02d}"
        if run_dir.exists() and any(run_dir.iterdir()):
            try:
                verify_run_dir(
                    run_dir, task, rep, EXPECTED_E_STAR, expected_w0, fp,
                    expected_input_chans=input_chans,
                    expected_seed_base=seed_base,
                    verify_checkpoint_content=True,
                )
                log(f"Resume: certified new run already exists, skip {run_key(task, rep)}")
                continue
            except Exception as exc:
                if not args.overwrite_new_run:
                    raise RuntimeError(
                        f"Non-empty but uncertified new-run directory exists: {run_dir}\n"
                        f"Reason: {exc}\n"
                        "Refusing to mix stale artifacts. Use --overwrite-new-run only "
                        "after confirming the directory may be replaced."
                    ) from exc
                log(f"Replacing invalid new run {run_key(task, rep)}: {exc}")
                safely_remove_new_run_dir(Path(args.root), task, rep)
        to_train[task].append(rep)

    for task in ("Motor", "Rest"):
        reps = to_train[task]
        if not reps:
            continue
        meta = core.load_task_meta(task, Path(args.data_root) / TASK_FILES[task])
        task_data = core.load_task_into_memory(
            meta, reserve_gb=float(args.ram_reserve_gb)
        )
        try:
            for rep in reps:
                log(
                    f"Training NEW {run_key(task, rep)} from W0 for exactly "
                    f"E*={EXPECTED_E_STAR} epochs"
                )
                core.run_task(
                    cargs, task, rep, meta, input_chans,
                    modeling_file, checkpoint, device,
                    epochs=EXPECTED_E_STAR,
                    phase="final",
                    protocol_fingerprint=fp,
                    task_data=task_data,
                )
                verify_run_dir(
                    Path(args.root) / "runs" / task / f"rep{rep:02d}",
                    task, rep, EXPECTED_E_STAR, expected_w0, fp,
                    expected_input_chans=input_chans,
                    expected_seed_base=seed_base,
                    verify_checkpoint_content=True,
                )
                log(f"Certified NEW {run_key(task, rep)}")
        finally:
            del task_data
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# -----------------------------------------------------------------------------
# Composite assembly
# -----------------------------------------------------------------------------

def link_one_legacy(root: Path, legacy_root: Path, task: str, rep: int) -> None:
    src = (legacy_root / "runs" / task / f"rep{rep:02d}").resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"Legacy run directory missing: {src}")
    task_dir = root / "runs" / task
    if task_dir.is_symlink():
        raise RuntimeError(
            f"{task_dir} is a task-level symlink. V5.0 requires replicate-level links "
            "so rejected Motor/rep02 cannot leak into the grid."
        )
    task_dir.mkdir(parents=True, exist_ok=True)
    dst = task_dir / f"rep{rep:02d}"
    if dst.is_symlink():
        if dst.resolve() == src:
            return
        raise RuntimeError(f"{dst} points to {dst.resolve()}, expected {src}")
    if dst.exists():
        raise RuntimeError(
            f"{dst} exists but is not the approved legacy symlink. Refusing overwrite."
        )
    os.symlink(src, dst, target_is_directory=True)
    log(f"Linked legacy {run_key(task, rep)} -> {src}")


def exact_run_grid_audit(root: Path) -> None:
    expected = {run_key(t, r) for t in ALL_TASKS for r in (1, 2)}
    observed = set()
    runs_root = root / "runs"
    if not runs_root.is_dir():
        raise RuntimeError("V5 runs/ directory is missing")
    for task_dir in runs_root.iterdir():
        if not task_dir.is_dir():
            continue
        task = task_dir.name
        for rep_dir in task_dir.iterdir():
            if not rep_dir.name.startswith("rep") or not rep_dir.is_dir():
                continue
            try:
                rep = int(rep_dir.name[3:])
            except Exception:
                observed.add(f"{task}/{rep_dir.name}")
                continue
            observed.add(run_key(task, rep))
    if observed != expected:
        raise RuntimeError(
            f"V5 exact run grid mismatch. missing={sorted(expected-observed)}, "
            f"extra={sorted(observed-expected)}"
        )


def assemble(args, pf, pilot, legacy_manifest, legacy_rows,
             legacy_provenance, rejected, expected_w0: str) -> dict:
    root = Path(args.root)
    legacy_root = Path(args.legacy_root)
    for task, rep in LEGACY_APPROVED:
        link_one_legacy(root, legacy_root, task, rep)

    new_fp = str(pf["protocol_fingerprint"])
    old_fp = str(legacy_manifest["protocol_fingerprint"])
    seed_base = int(pf["protocol"].get("seed_base", args.seed_base))
    input_chans = [int(x) for x in pf["protocol"]["input_chans"]]

    new_rows = []
    run_provenance = dict(legacy_provenance)
    for task, rep in NEW_RUNS:
        run_dir = root / "runs" / task / f"rep{rep:02d}"
        meta = verify_run_dir(
            run_dir, task, rep, EXPECTED_E_STAR, expected_w0, new_fp,
            expected_input_chans=input_chans,
            expected_seed_base=seed_base,
            verify_checkpoint_content=True,
        )
        hashes = _artifact_hashes(run_dir)
        key = run_key(task, rep)
        run_provenance[key] = {
            "source": "v5.0_new_training",
            "origin_path": str(run_dir.resolve()),
            "protocol_fingerprint": new_fp,
            "base_full_backbone_digest": expected_w0,
            "epochs": EXPECTED_E_STAR,
            "artifacts": hashes,
        }
        new_rows.append({
            "task": task,
            "replicate": rep,
            "n_samples": int(meta.get("n_samples", -1)),
            "epochs": int(meta.get("epochs", -1)),
            "convergence_epoch": meta.get("convergence_epoch"),
            "final_bacc": float(meta["metrics_final"]["balanced_accuracy"]),
            "protocol_fingerprint": new_fp,
            "base_full_backbone_digest": expected_w0,
            "artifact_source": "v5.0_new_training",
        })

    all_rows = list(legacy_rows) + new_rows
    order = {t: i for i, t in enumerate(ALL_TASKS)}
    all_rows.sort(key=lambda r: (order[str(r["task"])], int(r["replicate"])))
    expected_grid = {(t, r) for t in ALL_TASKS for r in (1, 2)}
    observed_grid = {(str(r["task"]), int(r["replicate"])) for r in all_rows}
    if observed_grid != expected_grid:
        raise RuntimeError("Composite 10-run row grid is incomplete")
    if set(run_provenance) != {run_key(t, r) for t, r in expected_grid}:
        raise RuntimeError("Composite per-run provenance grid is incomplete")
    if {str(r["base_full_backbone_digest"]) for r in all_rows} != {expected_w0}:
        raise RuntimeError("Composite runs do not share one W0")
    if {int(r["epochs"]) for r in all_rows} != {EXPECTED_E_STAR}:
        raise RuntimeError("Composite runs do not share E*=20")

    exact_run_grid_audit(root)

    provenance_policy = {
        "policy": "seven_legacy_plus_three_new_frozen_E_star",
        "legacy_root": str(legacy_root.resolve()),
        "legacy_pilot_recommended_E_star": int(pilot["recommended_E_star"]),
        "legacy_pilot_E_star_source": str(pilot["E_star_source"]),
        "legacy_pilot_n_uncertified_runs": int(pilot["n_uncertified_runs"]),
        "n_legacy_reused": 7,
        "n_new": 3,
        "n_rejected_legacy": 1,
        "rejected_legacy": rejected,
        "scientific_statement": (
            "E*=20 is frozen from the certified original four-task eight-run pilot. "
            "Seven artifact-certified v0.3.4 final runs are reused. Legacy Motor/rep02 "
            "is explicitly rejected and replaced. New Motor/rep02 and Rest rep01/rep02 "
            "start from the identical pretrained W0 and use the same optimization/seed "
            "contract for exactly 20 epochs. No new ten-run pilot is claimed."
        ),
    }

    composite_contract = {
        "version": VERSION,
        "V5_protocol_fingerprint": new_fp,
        "legacy_protocol_fingerprint": old_fp,
        "tasks": ALL_TASKS,
        "replicates": REPLICATES,
        "epochs": EXPECTED_E_STAR,
        "base_full_backbone_digest": expected_w0,
        "run_provenance": run_provenance,
        "provenance_policy": provenance_policy,
        "orchestrator_sha256": script_sha256(),
        "mature_core_sha256": sha256_file(Path(core.__file__).resolve()),
    }
    composite_fp = core.canonical_json_sha256(composite_contract)

    manifest = {
        "version": VERSION,
        "protocol_fingerprint": new_fp,
        "composite_protocol_fingerprint": composite_fp,
        "protocol": pf["protocol"],
        "tasks": ALL_TASKS,
        "replicates": REPLICATES,
        "expected_run_grid": [
            {"task": t, "replicate": r}
            for t in ALL_TASKS for r in (1, 2)
        ],
        "all_subjects_pooled": True,
        "epochs": EXPECTED_E_STAR,
        "epoch_source": "inherited_legacy_pilot_max_epoch_fallback",
        "base_full_backbone_digest": expected_w0,
        "run_provenance": run_provenance,
        "provenance_policy": provenance_policy,
        "code_provenance": {
            "orchestrator_path": str(Path(__file__).resolve()),
            "orchestrator_sha256": script_sha256(),
            "mature_core_path": str(Path(core.__file__).resolve()),
            "mature_core_sha256": sha256_file(Path(core.__file__).resolve()),
        },
        "runs": all_rows,
    }

    core.write_csv(root / "run_summary.csv", all_rows)
    core.dump_json(root / "run_manifest.json", manifest)
    core.dump_json(root / "_FINAL_SUCCESS.json", {
        "version": VERSION,
        "protocol_fingerprint": new_fp,
        "composite_protocol_fingerprint": composite_fp,
        "run_manifest_sha256": sha256_file(root / "run_manifest.json"),
        "n_runs": 10,
        "n_legacy_reused": 7,
        "n_new": 3,
        "n_rejected_legacy": 1,
        "epochs": EXPECTED_E_STAR,
        "epoch_source": "inherited_legacy_pilot_max_epoch_fallback",
    })
    core.plot_epoch_curves(root, pilot=False)
    log(
        "Composite final CERTIFIED: 7 legacy + 3 new; "
        f"E*=20, W0={expected_w0[:12]}..., composite={composite_fp[:12]}..."
    )
    return manifest


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="LaBraM x M3CV Task Adaptation Geometry V5.0 orchestrator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--mode",
        choices=["preflight", "inspect", "final", "assemble", "all"],
        default="preflight",
    )
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--legacy-root", type=Path, default=DEFAULT_LEGACY_ROOT)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--modeling-file", type=Path, default=DEFAULT_MODELING_FILE)
    p.add_argument("--seed-base", type=int, default=20260819)
    p.add_argument("--model-init-seed", type=int, default=314159)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--ram-reserve-gb", type=float, default=4.0)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--overwrite-new-run", action="store_true",
        help=(
            "Replace a non-empty but uncertified V5 Motor/rep02 or Rest run. "
            "Never modifies the legacy root."
        ),
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.seed_base != 20260819:
        raise RuntimeError("V5.0 scientific protocol fixes seed_base=20260819")
    if args.model_init_seed != 314159:
        raise RuntimeError("V5.0 scientific protocol fixes model_init_seed=314159")
    if args.ram_reserve_gb < 0:
        raise ValueError("--ram-reserve-gb must be >=0")

    if args.mode == "preflight":
        run_v5_preflight(args)
        return

    if args.mode == "all":
        run_v5_preflight(args)

    pf, pilot, legacy_manifest, legacy_rows, legacy_prov, rejected, expected_w0 = inspect(args)
    if args.mode == "inspect":
        return

    if args.mode in ("final", "all"):
        train_new_runs(args, pf, expected_w0)

    if args.mode in ("final", "assemble", "all"):
        assemble(
            args, pf, pilot, legacy_manifest, legacy_rows,
            legacy_prov, rejected, expected_w0,
        )


if __name__ == "__main__":
    main()
