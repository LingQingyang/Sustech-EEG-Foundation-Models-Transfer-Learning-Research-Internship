"""Artifact layout, atomic I/O, run discovery, and resume signatures.

This module knows the canonical ten-run layout but deliberately knows nothing
about historical legacy/new assembly.  A run is accepted because its artifacts
and scientific contract validate, not because of where it was first produced.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .config import EXPECTED_MODULE_SHAPES, TASK_ORDER


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{now()}] {message}", flush=True)


def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(x) for x in value]
    return value


def canonical_json_sha256(payload) -> str:
    raw = json.dumps(
        json_safe(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def stable_seed(base: int, *parts: object) -> int:
    raw = "|".join([str(int(base))] + [str(x) for x in parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**31 - 1)


def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def load_json(path: Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fieldnames: Optional[Sequence[str]] = None,
) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    rows = list(rows)
    if fieldnames is None:
        names: List[str] = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    names.append(str(key))
        fieldnames = names
    if not fieldnames:
        raise ValueError(f"Cannot write a headerless empty CSV: {path}")
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_safe(row.get(key, "")) for key in fieldnames})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def sanitize_module_key(module: str) -> str:
    return module.replace("/", "__")


def unsanitize_module_key(key: str) -> str:
    return key.replace("__", "/")


def module_layer_role(module: str) -> Tuple[int, str]:
    layer, role = module.split("/", 1)
    return int(layer[1:]), role


@dataclass(frozen=True)
class RunRecord:
    task: str
    replicate: int
    run_dir: Path
    metadata_path: Path
    checkpoint_path: Path
    delta_path: Path
    success_path: Path
    metadata: Mapping[str, object]

    @property
    def key(self) -> Tuple[str, int]:
        return self.task, self.replicate

    @property
    def tag(self) -> str:
        return f"{self.task}_rep{self.replicate:02d}"


def run_dir(root: Path, task: str, replicate: int) -> Path:
    return Path(root) / "runs" / task / f"rep{replicate:02d}"


def discover_runs(root: Path, validate_hashes: bool = True) -> List[RunRecord]:
    """Discover the exact canonical 5 x 2 run grid.

    Symlinks are neither required nor forbidden.  Historical provenance is not
    part of the active scientific interface.
    """

    root = Path(root)
    records: List[RunRecord] = []
    for task in TASK_ORDER:
        for replicate in (1, 2):
            directory = run_dir(root, task, replicate)
            required = {
                "metadata": directory / "metadata.json",
                "checkpoint": directory / "adapted_checkpoint.pth",
                "delta": directory / "delta_weights.npz",
                "success": directory / "_SUCCESS.json",
            }
            missing = [str(path) for path in required.values() if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    f"{task}/rep{replicate:02d} is incomplete; missing {missing}"
                )
            metadata = load_json(required["metadata"])
            success = load_json(required["success"])
            if str(metadata.get("task")) != task or int(metadata.get("replicate", -1)) != replicate:
                raise RuntimeError(f"Identity mismatch in {required['metadata']}")
            if str(success.get("task")) != task or int(success.get("replicate", -1)) != replicate:
                raise RuntimeError(f"Identity mismatch in {required['success']}")
            if int(metadata.get("epochs", -1)) != 20:
                raise RuntimeError(f"{task}/rep{replicate:02d}: expected exactly 20 epochs")
            if validate_hashes:
                artifacts = metadata.get("artifacts", {})
                declared_delta = str(artifacts.get("delta_weights_sha256", ""))
                declared_checkpoint = str(artifacts.get("adapted_checkpoint_sha256", ""))
                if declared_delta and sha256_file(required["delta"]) != declared_delta:
                    raise RuntimeError(f"Delta hash mismatch: {required['delta']}")
                if declared_checkpoint and sha256_file(required["checkpoint"]) != declared_checkpoint:
                    raise RuntimeError(f"Checkpoint hash mismatch: {required['checkpoint']}")
            records.append(
                RunRecord(
                    task=task,
                    replicate=replicate,
                    run_dir=directory,
                    metadata_path=required["metadata"],
                    checkpoint_path=required["checkpoint"],
                    delta_path=required["delta"],
                    success_path=required["success"],
                    metadata=metadata,
                )
            )
    return records


def load_delta_weights(path: Path) -> Dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as archive:
        values = {
            unsanitize_module_key(key): np.asarray(archive[key], dtype=np.float64)
            for key in archive.files
        }
    return values


def audit_delta_archives(runs: Sequence[RunRecord]) -> Dict[str, object]:
    expected = dict(EXPECTED_MODULE_SHAPES)
    max_abs = 0.0
    for record in runs:
        values = load_delta_weights(record.delta_path)
        shapes = {key: tuple(int(x) for x in value.shape) for key, value in values.items()}
        if shapes != expected:
            missing = sorted(set(expected) - set(shapes))
            extra = sorted(set(shapes) - set(expected))
            wrong = {
                key: (expected[key], shapes[key])
                for key in set(expected) & set(shapes)
                if expected[key] != shapes[key]
            }
            raise RuntimeError(
                f"{record.tag}: invalid 72-matrix archive; missing={missing}, "
                f"extra={extra}, wrong_shapes={wrong}"
            )
        for value in values.values():
            if not np.isfinite(value).all():
                raise FloatingPointError(f"{record.tag}: non-finite delta matrix")
            if value.size:
                max_abs = max(max_abs, float(np.max(np.abs(value))))
    return {
        "status": "passed",
        "n_runs": len(runs),
        "n_modules_per_run": len(expected),
        "max_absolute_delta_entry": max_abs,
    }


def load_decompositions(runs: Sequence[RunRecord]):
    """Load and SVD-decompose the complete run x matrix grid once."""
    from .geometry import compute_svd

    records = {}
    for run_index, run in enumerate(runs, start=1):
        for module, delta in load_delta_weights(run.delta_path).items():
            records[(run.task, run.replicate, module)] = compute_svd(delta)
        log(f"SVD cache {run_index}/{len(runs)}: {run.tag}")
    expected_count = len(runs) * len(EXPECTED_MODULE_SHAPES)
    if len(records) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} run-matrix decompositions, found {len(records)}"
        )
    return records


class ProgressStore:
    """Small fail-closed resume marker keyed by a scientific signature."""

    def __init__(self, path: Path, signature_payload, resume: bool = True):
        self.path = Path(path)
        self.signature = canonical_json_sha256(signature_payload)
        self.completed = set()
        if resume and self.path.is_file():
            payload = load_json(self.path)
            if str(payload.get("signature")) != self.signature:
                raise RuntimeError(
                    f"Stale resume marker {self.path}; rerun without resume instead of mixing settings."
                )
            self.completed = set(str(x) for x in payload.get("completed", []))

    def has(self, key: str) -> bool:
        return str(key) in self.completed

    def mark(self, key: str) -> None:
        self.completed.add(str(key))
        dump_json(
            self.path,
            {"signature": self.signature, "completed": sorted(self.completed)},
        )


def artifact_hashes(root: Path) -> Dict[str, str]:
    root = Path(root)
    out = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and ".tmp." not in path.name:
            out[str(path.relative_to(root))] = sha256_file(path)
    return out
