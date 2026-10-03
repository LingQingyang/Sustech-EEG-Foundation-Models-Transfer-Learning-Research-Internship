#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LaBraM x M3CV Task Adaptation Geometry
======================================
Training / convergence / delta-extraction runner.

Version: v0.3.4

Scientific contract
-------------------
1) No subject-heldout split. For each task, ALL available subjects and samples
   in the Session-1 H5 are pooled for full fine-tuning.
2) Every task/replicate starts from the identical pretrained LaBraM backbone W0.
3) Full fine-tuning uses one shared protocol:
      AdamW
      backbone LR = 1e-4
      head LR     = 1e-3
      weight decay = 0
      batch size   = 32
      class-weighted CE
      NO gradient clipping
      NO scheduler
      NO warm-up
      NO early stopping
4) A pilot run determines one common E* by convergence audit. Final runs restart
   from W0 and train exactly E* epochs.
5) Complete adapted checkpoints and all 72 2-D TSV-module Delta W matrices are
   saved for post-hoc functional-rank, STI, coverage and novelty analysis.

Data assumptions
----------------
Each H5 contains:
    data        [N,64,800], float32
    labels      [N]
    subject_ids [N]

H5 data are already GLOBAL Z-scored:
    NO /100
    NO second z-score
Input is reshaped:
    [64,800] -> [64,4,200]

The exact 64-channel positional mapping remains a hard dependency for final
scientific runs. The provisional 1..64 mapping requires an explicit flag.

Typical workflow
----------------
# 1) Preflight / smoke
python3 run_tsv_labram_m3cv_v0_3_4.py --mode preflight ...

# 2) Convergence pilot
python3 run_tsv_labram_m3cv_v0_3_4.py --mode pilot ...

# 3) Final runs; --epochs 0 reads E* from pilot/pilot_summary.json
python3 run_tsv_labram_m3cv_v0_3_4.py --mode final ...

# Or pilot + final in one invocation
python3 run_tsv_labram_m3cv_v0_3_4.py --mode all ...
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

VERSION = "v0.3.4"
EPS = 1e-12

TASK_FILES = {
    "Motor": "m3cv_Motor_Session1_4s.h5",
    "P300": "m3cv_P300_Session1_4s.h5",
    "SSS": "m3cv_SSS_Session1_4s.h5",
    "TS": "m3cv_TS_Session1_4s.h5",
}
TASK_CLASS_NAMES = {
    "Motor": {0: "FT", 1: "RH", 2: "LH"},
    "P300": {0: "Non-target", 1: "Target"},
    "SSS": {0: "SSVEP", 1: "SSAEP", 2: "SSSEP"},
    "TS": {0: "VEP", 1: "AEP", 2: "SEP"},
}
EXPECTED_CLASSES = {k: len(v) for k, v in TASK_CLASS_NAMES.items()}
MODULE_ROLES = ("Q", "K", "V", "O", "fc1", "fc2")
N_BLOCKS = 12
N_TSV_MODULES = 72

ALLOWED_CHECKPOINT_ONLY_PREFIXES = ("lm_head.", "projection_head.", "norm.")
ALLOWED_CHECKPOINT_ONLY_EXACT = {
    "logit_scale", "mask_token", "head.weight", "head.bias",
}
ALLOWED_MODEL_MISSING_PREFIXES = ("fc_norm.",)
_MODELING_IMPORTED = False


# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------

def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def parse_csv_strs(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_csv_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def stable_seed(base_seed: int, *parts: object) -> int:
    text = f"{int(base_seed)}|" + "|".join(str(x) for x in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def json_safe(x):
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def _atomic_tmp_path(path: Path | str, suffix: str = "") -> Path:
    p = Path(path)
    token = f".tmp.{os.getpid()}.{time.time_ns()}"
    return p.with_name(p.name + token + suffix)


def dump_json(path: Path | str, payload) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    tmp = _atomic_tmp_path(path)
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(json_safe(payload), f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def load_json(path: Path | str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path | str, rows: Sequence[Mapping[str, object]],
              fieldnames: Optional[Sequence[str]] = None) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    rows = list(rows)
    if fieldnames is None:
        fields = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    fields.append(key)
        fieldnames = fields
    tmp = _atomic_tmp_path(path)
    try:
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
            w.writeheader()
            for row in rows:
                w.writerow({k: json_safe(row.get(k, "")) for k in fieldnames})
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def canonical_json_sha256(payload) -> str:
    text = json.dumps(
        json_safe(payload), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_torch_save(payload, path: Path | str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    tmp = _atomic_tmp_path(path)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def torch_state_digest(state: Mapping[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        t = state[key]
        if not torch.is_tensor(t):
            continue
        arr = t.detach().cpu().contiguous().numpy()
        h.update(key.encode("utf-8"))
        h.update(str(arr.dtype).encode("ascii"))
        h.update(str(tuple(arr.shape)).encode("ascii"))
        h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def sanitize_module_key(module: str) -> str:
    return module.replace("/", "__")


def unsanitize_module_key(key: str) -> str:
    return key.replace("__", "/")


def module_layer_role(module: str) -> Tuple[int, str]:
    left, role = module.split("/")
    return int(left[1:]), role


def median_or_nan(xs: Sequence[float]) -> float:
    a = np.asarray(xs, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if a.size else float("nan")


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------

@dataclass
class TaskMeta:
    task: str
    path: Path
    n: int
    shape: Tuple[int, int, int]
    labels: np.ndarray
    subjects: np.ndarray
    classes: np.ndarray


def load_task_meta(task: str, path: Path) -> TaskMeta:
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as h5:
        for key in ("data", "labels", "subject_ids"):
            if key not in h5:
                raise KeyError(f"{path}: missing {key}; found {list(h5.keys())}")
        data_ds = h5["data"]
        labels_ds = h5["labels"]
        subjects_ds = h5["subject_ids"]
        shape = tuple(int(x) for x in data_ds.shape)
        if len(shape) != 3 or shape[1:] != (64, 800):
            raise ValueError(f"{task}: expected [N,64,800], found {shape}")
        if np.dtype(data_ds.dtype) != np.dtype(np.float32):
            raise TypeError(f"{task}: data must be float32, found {data_ds.dtype}")
        if not np.issubdtype(labels_ds.dtype, np.integer):
            raise TypeError(f"{task}: labels must be integer, found {labels_ds.dtype}")
        if not np.issubdtype(subjects_ds.dtype, np.integer):
            raise TypeError(f"{task}: subject_ids must be integer, found {subjects_ds.dtype}")
        labels = np.asarray(labels_ds[:], dtype=np.int64)
        subjects = np.asarray(subjects_ds[:], dtype=np.int64)
    if shape[0] < 1:
        raise ValueError(f"{task}: empty dataset")
    if len(labels) != shape[0] or len(subjects) != shape[0]:
        raise ValueError(f"{task}: data/labels/subject_ids length mismatch")
    classes = np.sort(np.unique(labels))
    expected = np.asarray(sorted(TASK_CLASS_NAMES[task]), dtype=np.int64)
    if not np.array_equal(classes, expected):
        raise ValueError(
            f"{task}: expected exact raw labels {expected.tolist()}, found {classes.tolist()}"
        )
    if np.any(subjects < 0):
        raise ValueError(f"{task}: negative subject_ids found")
    return TaskMeta(task, path, int(shape[0]), shape, labels, subjects, classes)


def _sha256_ndarray(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode("ascii"))
    h.update(str(tuple(a.shape)).encode("ascii"))
    h.update(a.tobytes(order="C"))
    return h.hexdigest()


def _norm_label_name(x: object) -> str:
    if isinstance(x, bytes):
        x = x.decode("utf-8", errors="replace")
    return re.sub(r"[^a-z0-9]+", "", str(x).lower())


def _extract_label_mapping_value(h5):
    for name in ("label_mapping", "label_map", "class_mapping", "class_names"):
        if name in h5.attrs:
            return h5.attrs[name], f"attr:{name}"
        if name in h5:
            try:
                return h5[name][()], f"dataset:{name}"
            except Exception:
                pass
    return None, None


def _parse_label_mapping(value, n_classes: int) -> Dict[int, str]:
    if value is None:
        return {}
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            value = value.item()
        else:
            value = value.tolist()
    if isinstance(value, (list, tuple)) and len(value) == n_classes:
        # Some preprocessing scripts store ["FT=0", "RH=1", ...] while
        # others store positional ["FT", "RH", ...]. Handle both explicitly.
        structured = {}
        structured_ok = True
        for item in value:
            text_item = item.decode("utf-8", errors="replace") if isinstance(item, bytes) else str(item)
            pair = re.split(r"\s*(?:=|:|->)\s*", text_item.strip(), maxsplit=1)
            if len(pair) != 2:
                structured_ok = False
                break
            left, right = pair[0].strip(" '\""), pair[1].strip(" '\"")
            try:
                structured[int(left)] = _norm_label_name(right)
            except Exception:
                try:
                    structured[int(right)] = _norm_label_name(left)
                except Exception:
                    structured_ok = False
                    break
        if structured_ok and len(structured) == n_classes:
            return structured
        return {i: _norm_label_name(v) for i, v in enumerate(value)}
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, Mapping):
        out = {}
        for k, v in value.items():
            try:
                code = int(k)
                out[code] = _norm_label_name(v)
                continue
            except Exception:
                pass
            try:
                code = int(v)
                out[code] = _norm_label_name(k)
            except Exception:
                pass
        return out

    text = str(value).strip()
    # JSON first when available. Recurse only into structured JSON; a JSON
    # string would otherwise recurse forever on itself.
    try:
        obj = json.loads(text)
        if isinstance(obj, (dict, list, tuple)):
            parsed = _parse_label_mapping(obj, n_classes)
            if parsed:
                return parsed
    except Exception:
        pass

    out = {}
    text = text.strip("{}[]()")
    for token in re.split(r"[,;|]", text):
        token = token.strip()
        if not token:
            continue
        m = re.split(r"\s*(?:=|:|->)\s*", token, maxsplit=1)
        if len(m) != 2:
            continue
        left, right = m[0].strip(" '\""), m[1].strip(" '\"")
        try:
            out[int(left)] = _norm_label_name(right)
            continue
        except Exception:
            pass
        try:
            out[int(right)] = _norm_label_name(left)
        except Exception:
            pass
    return out


def audit_h5_data(meta: TaskMeta, chunk_rows: int = 64,
                  mean_abs_max: float = 0.25,
                  std_min: float = 0.50,
                  std_max: float = 2.00,
                  allow_missing_label_mapping: bool = False) -> Dict[str, object]:
    """Full sequential integrity scan. This is intentionally strict and fail-closed."""
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be >=1")
    h = hashlib.sha256()
    h.update(meta.task.encode("utf-8"))
    h.update(str(meta.shape).encode("ascii"))
    h.update(_sha256_ndarray(meta.labels).encode("ascii"))
    h.update(_sha256_ndarray(meta.subjects).encode("ascii"))

    count = 0
    s1 = 0.0
    s2 = 0.0
    xmin = float("inf")
    xmax = float("-inf")
    nonfinite = 0
    flat_probe_count = 0
    probe_stride = max(1, meta.n // 32)
    channel_sum = np.zeros(64, dtype=np.float64)
    channel_sumsq = np.zeros(64, dtype=np.float64)
    channel_count = np.zeros(64, dtype=np.int64)

    with h5py.File(meta.path, "r") as h5:
        mapping_value, mapping_source = _extract_label_mapping_value(h5)
        parsed_mapping = _parse_label_mapping(mapping_value, len(meta.classes))
        if not parsed_mapping and not allow_missing_label_mapping:
            raise RuntimeError(
                f"{meta.task}: H5 has no parseable label_mapping metadata. "
                "Refusing to assume class semantics; use the explicitly unsafe override only after manual verification."
            )
        if parsed_mapping:
            expected_mapping = {
                int(code): _norm_label_name(name)
                for code, name in TASK_CLASS_NAMES[meta.task].items()
            }
            if parsed_mapping != expected_mapping:
                raise RuntimeError(
                    f"{meta.task}: label mapping metadata {parsed_mapping} != expected {expected_mapping}"
                )
        h.update(str(mapping_source or "MISSING_LABEL_MAPPING").encode("utf-8"))
        h.update(json.dumps(parsed_mapping, sort_keys=True).encode("utf-8"))

        ds = h5["data"]
        for start in range(0, meta.n, int(chunk_rows)):
            stop = min(meta.n, start + int(chunk_rows))
            x = np.asarray(ds[start:stop], dtype=np.float32)
            h.update(memoryview(np.ascontiguousarray(x)).cast("B"))
            finite = np.isfinite(x)
            if not finite.all():
                nonfinite += int(x.size - int(finite.sum()))
                # Rare failure path: keep diagnostics valid without NaN poisoning.
                xf = x[finite].astype(np.float64, copy=False)
                if xf.size:
                    count += int(xf.size)
                    s1 += float(np.sum(xf, dtype=np.float64))
                    s2 += float(np.sum(xf * xf, dtype=np.float64))
                    xmin = min(xmin, float(np.min(xf)))
                    xmax = max(xmax, float(np.max(xf)))
            else:
                # Normal path avoids a full float64 copy of every chunk. EEG is
                # already float32 z-score data; accumulate sums in float64.
                count += int(x.size)
                s1 += float(np.sum(x, dtype=np.float64))
                s2 += float(np.sum(x * x, dtype=np.float64))
                xmin = min(xmin, float(np.min(x)))
                xmax = max(xmax, float(np.max(x)))

            # Per-channel diagnostics catch dead/corrupted electrodes that can hide
            # behind plausible global mean/std statistics.
            finite64 = finite
            if finite64.all():
                channel_sum += np.sum(x, axis=(0, 2), dtype=np.float64)
                channel_sumsq += np.sum(x * x, axis=(0, 2), dtype=np.float64)
                channel_count += int(x.shape[0] * x.shape[2])
            else:
                # The scan will fail below, but keep diagnostics numerically valid.
                for ch in range(64):
                    vals = x[:, ch, :][finite64[:, ch, :]].astype(np.float64, copy=False)
                    if vals.size:
                        channel_sum[ch] += float(np.sum(vals, dtype=np.float64))
                        channel_sumsq[ch] += float(np.sum(vals * vals, dtype=np.float64))
                        channel_count[ch] += int(vals.size)

            # Sparse per-sample sanity probe catches all-zero/constant corrupted rows.
            for local_i in range(x.shape[0]):
                global_i = start + local_i
                if global_i % probe_stride == 0:
                    if float(np.std(x[local_i], dtype=np.float64)) <= 1e-8:
                        flat_probe_count += 1

    if nonfinite:
        raise FloatingPointError(f"{meta.task}: found {nonfinite} non-finite EEG values")
    if count != int(np.prod(meta.shape)):
        raise RuntimeError(f"{meta.task}: scan element count mismatch {count} vs {np.prod(meta.shape)}")
    mean = s1 / max(count, 1)
    var = max(s2 / max(count, 1) - mean * mean, 0.0)
    std = math.sqrt(var)
    if abs(mean) > float(mean_abs_max):
        raise RuntimeError(
            f"{meta.task}: global mean {mean:.6g} violates assumed global Z-score "
            f"(|mean| <= {mean_abs_max})"
        )
    if not (float(std_min) <= std <= float(std_max)):
        raise RuntimeError(
            f"{meta.task}: global std {std:.6g} violates assumed global Z-score "
            f"({std_min} <= std <= {std_max})"
        )
    if flat_probe_count:
        raise RuntimeError(
            f"{meta.task}: {flat_probe_count} sampled rows were effectively constant; refusing GIGO"
        )

    expected_per_channel = int(meta.n * meta.shape[2])
    if not np.all(channel_count == expected_per_channel):
        raise RuntimeError(
            f"{meta.task}: per-channel finite element counts are inconsistent with the declared layout"
        )
    channel_mean = channel_sum / np.maximum(channel_count, 1)
    channel_var = np.maximum(channel_sumsq / np.maximum(channel_count, 1) - channel_mean**2, 0.0)
    channel_std = np.sqrt(channel_var)
    dead = np.flatnonzero(channel_std <= 1e-6).tolist()
    if dead:
        raise RuntimeError(
            f"{meta.task}: effectively dead/constant EEG channels at zero-based indices {dead}; refusing GIGO"
        )
    positive_std = channel_std[channel_std > 1e-12]
    std_ratio = float(np.max(positive_std) / max(np.min(positive_std), 1e-12))
    if std_ratio > 50.0:
        raise RuntimeError(
            f"{meta.task}: max/min channel std ratio={std_ratio:.2f} is implausibly large; "
            "possible channel corruption or unit mismatch"
        )
    if max(abs(xmin), abs(xmax)) > 1e4:
        raise RuntimeError(
            f"{meta.task}: |EEG z-score| exceeds 1e4; likely unit/corruption failure"
        )

    st = meta.path.stat()
    return {
        "task": meta.task,
        "path": str(meta.path.resolve()),
        "file_size_bytes": int(st.st_size),
        "file_mtime_ns": int(st.st_mtime_ns),
        "shape": list(meta.shape),
        "dtype": "float32",
        "n_samples": meta.n,
        "n_subjects": int(len(np.unique(meta.subjects))),
        "subject_ids_sha256": _sha256_ndarray(meta.subjects),
        "labels_sha256": _sha256_ndarray(meta.labels),
        "class_counts": {
            str(int(c)): int(np.sum(meta.labels == c)) for c in meta.classes
        },
        "label_mapping_source": mapping_source,
        "label_mapping": {str(k): v for k, v in parsed_mapping.items()},
        "label_mapping_verified": bool(parsed_mapping),
        "global_mean": float(mean),
        "global_std": float(std),
        "global_min": float(xmin),
        "global_max": float(xmax),
        "channel_mean": [float(x) for x in channel_mean],
        "channel_std": [float(x) for x in channel_std],
        "channel_std_ratio_max_min": std_ratio,
        "content_sha256": h.hexdigest(),
        "full_scan": True,
    }


class M3CV4sDataset(Dataset):
    """Compatibility lazy dataset. Scientific runners use H5BatchLoader below."""

    def __init__(self, meta: TaskMeta, indices: Optional[np.ndarray] = None):
        self.h5_path = str(meta.path)
        if indices is None:
            indices = np.arange(meta.n, dtype=np.int64)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.raw_classes = np.asarray(meta.classes, dtype=np.int64)
        self.label_map = {int(c): i for i, c in enumerate(self.raw_classes)}
        self._h5 = None

    def __len__(self) -> int:
        return int(len(self.indices))

    def _file(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __getitem__(self, item: int):
        h5 = self._file()
        idx = int(self.indices[item])
        x = np.asarray(h5["data"][idx], dtype=np.float32)
        raw_y = int(h5["labels"][idx])
        x = x.reshape(64, 4, 200)
        y = self.label_map[raw_y]
        return torch.from_numpy(np.ascontiguousarray(x)), torch.tensor(y, dtype=torch.long)

    def __del__(self):
        try:
            if self._h5 is not None:
                self._h5.close()
        except Exception:
            pass


class H5BatchLoader:
    """
    Batch-native HDF5 loader.

    Random training indices are sorted only for the HDF5 read, then restored to the
    requested random order. This avoids one HDF5 read per sample without loading a
    multi-GB task into RAM.
    """

    def __init__(self, meta: TaskMeta, batch_size: int, shuffle: bool,
                 seed: int = 0, indices: Optional[np.ndarray] = None,
                 pin_memory: bool = False):
        self.meta = meta
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.indices = (
            np.arange(meta.n, dtype=np.int64)
            if indices is None else np.asarray(indices, dtype=np.int64)
        )
        if self.batch_size < 1:
            raise ValueError("batch_size must be >=1")
        if self.indices.ndim != 1 or np.any(self.indices < 0) or np.any(self.indices >= meta.n):
            raise ValueError("invalid H5BatchLoader indices")
        if len(np.unique(self.indices)) != len(self.indices):
            raise ValueError("duplicate loader indices are not allowed")
        self.pin_memory = bool(pin_memory)
        self.rng = np.random.default_rng(int(seed))
        self.raw_classes = np.asarray(meta.classes, dtype=np.int64)

    def __len__(self) -> int:
        return int(math.ceil(len(self.indices) / self.batch_size))

    def _read_batch(self, h5, ids: np.ndarray):
        ids = np.asarray(ids, dtype=np.int64)
        if len(ids) == 0:
            raise RuntimeError("empty batch")
        if len(ids) == 1:
            x = np.asarray(h5["data"][int(ids[0]):int(ids[0]) + 1], dtype=np.float32)
            raw_y = np.asarray(h5["labels"][int(ids[0]):int(ids[0]) + 1], dtype=np.int64)
        elif np.all(np.diff(ids) == 1):
            a, b = int(ids[0]), int(ids[-1]) + 1
            x = np.asarray(h5["data"][a:b], dtype=np.float32)
            raw_y = np.asarray(h5["labels"][a:b], dtype=np.int64)
        else:
            order = np.argsort(ids)
            sorted_ids = ids[order]
            x_sorted = np.asarray(h5["data"][sorted_ids], dtype=np.float32)
            y_sorted = np.asarray(h5["labels"][sorted_ids], dtype=np.int64)
            inv = np.argsort(order)
            x = x_sorted[inv]
            raw_y = y_sorted[inv]
        y = np.searchsorted(self.raw_classes, raw_y)
        if np.any(y >= len(self.raw_classes)) or not np.array_equal(self.raw_classes[y], raw_y):
            raise RuntimeError(f"{self.meta.task}: unknown label encountered during batch read")
        x = np.ascontiguousarray(x.reshape(len(ids), 64, 4, 200))
        xt = torch.from_numpy(x)
        yt = torch.from_numpy(np.asarray(y, dtype=np.int64))
        if self.pin_memory:
            xt = xt.pin_memory()
            yt = yt.pin_memory()
        return xt, yt

    def __iter__(self):
        order = self.indices.copy()
        if self.shuffle:
            order = order[self.rng.permutation(len(order))]
        with h5py.File(self.meta.path, "r") as h5:
            for start in range(0, len(order), self.batch_size):
                ids = order[start:start + self.batch_size]
                yield self._read_batch(h5, ids)



@dataclass
class InMemoryTaskData:
    """One task staged once into host RAM and reused across epochs/replicates."""
    x: np.ndarray          # float32 [N,64,4,200]
    y: np.ndarray          # int64 [N] remapped to 0..C-1
    load_seconds: float
    nbytes: int


def _mem_available_bytes() -> Optional[int]:
    """Linux MemAvailable, if available. Used only to fail early instead of OOM."""
    try:
        with open('/proc/meminfo', 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return None


def load_task_into_memory(meta: TaskMeta, reserve_gb: float = 4.0) -> InMemoryTaskData:
    """
    Sequentially stage one H5 task into RAM.

    This is the deliberate v0.3.4 performance contract: the network-mounted H5 is
    touched once per task, never once per random minibatch. Only one task is held
    in RAM at a time, so the largest M3CV task is ~4 GB rather than all tasks at once.
    """
    required = int(np.prod(meta.shape)) * np.dtype(np.float32).itemsize
    avail = _mem_available_bytes()
    reserve = int(float(reserve_gb) * (1024**3))
    if avail is not None and avail < required + reserve:
        raise MemoryError(
            f"{meta.task}: in-memory training needs about {required/1024**3:.2f} GiB "
            f"plus {reserve_gb:.1f} GiB reserve, but MemAvailable is {avail/1024**3:.2f} GiB. "
            "Do not fall back to random HDF5 reads silently; free RAM or use a larger host."
        )
    t0 = time.time()
    st_before = meta.path.stat()
    with h5py.File(meta.path, 'r') as h5:
        # One contiguous read is vastly faster than random HDF5 fancy indexing over a mount.
        x = np.asarray(h5['data'][:], dtype=np.float32)
    st_after = meta.path.stat()
    if (st_before.st_size, st_before.st_mtime_ns) != (st_after.st_size, st_after.st_mtime_ns):
        raise RuntimeError(f"{meta.task}: H5 changed while being staged into RAM")
    if tuple(x.shape) != meta.shape:
        raise RuntimeError(f"{meta.task}: RAM staging shape changed {x.shape} != {meta.shape}")
    if not np.isfinite(x).all():
        raise FloatingPointError(f"{meta.task}: non-finite EEG appeared during RAM staging")
    x = np.ascontiguousarray(x.reshape(meta.n, 64, 4, 200))
    y = np.searchsorted(meta.classes, meta.labels).astype(np.int64, copy=False)
    if np.any(y >= len(meta.classes)) or not np.array_equal(meta.classes[y], meta.labels):
        raise RuntimeError(f"{meta.task}: label remapping failed during RAM staging")
    seconds = float(time.time() - t0)
    log(
        f"RAM staged {meta.task}: {x.nbytes/1024**3:.2f} GiB in {seconds:.1f}s "
        f"({x.nbytes/max(seconds,1e-9)/1024**2:.1f} MiB/s)"
    )
    return InMemoryTaskData(x=x, y=y, load_seconds=seconds, nbytes=int(x.nbytes))


class InMemoryBatchLoader:
    """Lightweight minibatch iterator over a task already resident in host RAM."""

    def __init__(self, task_data: InMemoryTaskData, batch_size: int, shuffle: bool,
                 seed: int = 0, indices: Optional[np.ndarray] = None):
        self.data = task_data
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.indices = (
            np.arange(len(task_data.y), dtype=np.int64)
            if indices is None else np.asarray(indices, dtype=np.int64)
        )
        if self.batch_size < 1:
            raise ValueError('batch_size must be >=1')
        if self.indices.ndim != 1 or np.any(self.indices < 0) or np.any(self.indices >= len(task_data.y)):
            raise ValueError('invalid in-memory loader indices')
        self.rng = np.random.default_rng(int(seed))

    def __len__(self) -> int:
        return int(math.ceil(len(self.indices) / self.batch_size))

    def __iter__(self):
        order = self.indices.copy()
        if self.shuffle:
            order = order[self.rng.permutation(len(order))]
        for start in range(0, len(order), self.batch_size):
            ids = order[start:start + self.batch_size]
            # Advanced indexing copies one compact minibatch from RAM; no disk I/O.
            x = torch.from_numpy(np.ascontiguousarray(self.data.x[ids]))
            y = torch.from_numpy(np.ascontiguousarray(self.data.y[ids]))
            yield x, y

def preflight_stratified_indices(meta: TaskMeta, n: int = 32) -> np.ndarray:
    n = max(int(n), len(meta.classes))
    chosen: List[int] = []
    for c in meta.classes:
        idx = np.flatnonzero(meta.labels == c)
        if len(idx) == 0:
            raise RuntimeError(f"{meta.task}: empty class {c}")
        chosen.append(int(idx[0]))
    remaining = np.setdiff1d(np.arange(meta.n, dtype=np.int64), np.asarray(chosen), assume_unique=False)
    rng = np.random.default_rng(stable_seed(20260820, meta.task, "preflight_rows"))
    if len(chosen) < n and len(remaining):
        take = min(n - len(chosen), len(remaining))
        chosen.extend(int(x) for x in rng.choice(remaining, size=take, replace=False))
    return np.asarray(sorted(set(chosen)), dtype=np.int64)

def class_weights_for_meta(meta: TaskMeta, device: torch.device) -> torch.Tensor:
    counts = np.asarray([np.sum(meta.labels == c) for c in meta.classes], dtype=np.float64)
    if np.any(counts <= 0):
        raise ValueError(f"{meta.task}: empty class in pooled data")
    weights = float(meta.n) / (len(meta.classes) * counts)
    # Mean sample weight is exactly 1: sum_c N_c * w_c / N = 1.
    return torch.tensor(weights, dtype=torch.float32, device=device)


def subject_metadata(metas: Mapping[str, TaskMeta]) -> Dict[str, object]:
    return {
        task: {
            "n_samples": meta.n,
            "n_subjects": int(len(np.unique(meta.subjects))),
            "subject_ids": [int(x) for x in np.unique(meta.subjects)],
            "raw_classes": [int(x) for x in meta.classes],
            "class_counts": {
                str(int(c)): int(np.sum(meta.labels == c)) for c in meta.classes
            },
        }
        for task, meta in metas.items()
    }


# -----------------------------------------------------------------------------
# LaBraM model loading
# -----------------------------------------------------------------------------

def find_modeling_file(explicit: str) -> Path:
    candidates: List[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env = os.environ.get("LABRAM_MODELING_FILE", "").strip()
    if env:
        candidates.append(Path(env).expanduser())
    candidates += [
        Path("/omni-eeg-01/task calibration/LaBraM/modeling_finetune.py"),
        Path.cwd() / "modeling_finetune.py",
        Path.cwd() / "src" / "modeling_finetune.py",
        Path.home() / "modeling_finetune.py",
        Path.home() / "LaBraM" / "modeling_finetune.py",
        Path.home() / "LaBraM" / "src" / "modeling_finetune.py",
    ]
    for p in candidates:
        if p.is_file():
            return p.resolve()
    raise FileNotFoundError(
        "Cannot find modeling_finetune.py. Pass --modeling-file or set LABRAM_MODELING_FILE."
    )


def import_modeling_file(path: Path) -> None:
    global _MODELING_IMPORTED
    if _MODELING_IMPORTED:
        return
    parent = str(path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    name = "_tsv_labram_modeling_finetune_v033"
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    _MODELING_IMPORTED = True


def is_allowed_checkpoint_only(key: str) -> bool:
    return key in ALLOWED_CHECKPOINT_ONLY_EXACT or any(
        key.startswith(p) for p in ALLOWED_CHECKPOINT_ONLY_PREFIXES
    )


def is_allowed_model_missing(key: str) -> bool:
    return "relative_position_index" in key or any(
        key.startswith(p) for p in ALLOWED_MODEL_MISSING_PREFIXES
    )


def strict_load_labram_checkpoint(model: nn.Module, checkpoint_path: Path) -> Dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint
    payload_name = "<root>"
    if isinstance(checkpoint, dict):
        for key in ("model", "module", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], Mapping):
                state = checkpoint[key]
                payload_name = key
                break
    if not isinstance(state, Mapping):
        raise TypeError("Checkpoint does not contain a state mapping")

    clean: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        k = str(key)
        if k.startswith("student."):
            k = k[8:]
        if k.startswith("module."):
            k = k[7:]
        clean[k] = value

    model_state = model.state_dict()
    time_embed_transform = None
    if "time_embed" in clean and "time_embed" in model_state:
        if tuple(clean["time_embed"].shape) != tuple(model_state["time_embed"].shape):
            old_shape = tuple(clean["time_embed"].shape)
            target_t = model_state["time_embed"].shape[1]
            if clean["time_embed"].shape[0] != model_state["time_embed"].shape[0]:
                raise RuntimeError("time_embed batch dimension mismatch")
            if clean["time_embed"].shape[-1] != model_state["time_embed"].shape[-1]:
                raise RuntimeError("time_embed embedding dimension mismatch")
            if clean["time_embed"].shape[1] < target_t:
                raise RuntimeError("checkpoint time_embed shorter than target")
            clean["time_embed"] = clean["time_embed"][:, :target_t]
            time_embed_transform = {"from": old_shape, "to": tuple(clean["time_embed"].shape)}

    checkpoint_only = sorted(k for k in clean if k not in model_state)
    unknown_checkpoint_only = [k for k in checkpoint_only if not is_allowed_checkpoint_only(k)]
    if unknown_checkpoint_only:
        raise RuntimeError(f"Unknown checkpoint-only keys: {unknown_checkpoint_only[:20]}")

    shape_mismatch = [
        (k, tuple(clean[k].shape), tuple(model_state[k].shape))
        for k in clean
        if k in model_state and tuple(clean[k].shape) != tuple(model_state[k].shape)
    ]
    if shape_mismatch:
        raise RuntimeError(f"Checkpoint/model shape mismatches: {shape_mismatch[:20]}")

    loadable = {
        k: v for k, v in clean.items()
        if k in model_state and tuple(v.shape) == tuple(model_state[k].shape)
    }
    model_missing = sorted(k for k in model_state if k not in loadable)
    unknown_model_missing = [k for k in model_missing if not is_allowed_model_missing(k)]
    if unknown_model_missing:
        raise RuntimeError(
            f"Unexpected model tensors not loaded from checkpoint: {unknown_model_missing[:20]}"
        )
    missing, unexpected = model.load_state_dict(loadable, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected load_state_dict keys: {list(unexpected)[:20]}")
    bad_missing = [k for k in missing if not is_allowed_model_missing(k)]
    if bad_missing:
        raise RuntimeError(f"Unexpected missing keys: {bad_missing[:20]}")
    return {
        "payload": payload_name,
        "checkpoint_tensor_count_after_prefix_cleanup": len(clean),
        "model_state_tensor_count": len(model_state),
        "loadable_tensor_count": len(loadable),
        "checkpoint_only_allowed": checkpoint_only,
        "model_missing_allowed": model_missing,
        "time_embed_transform": time_embed_transform,
    }


def build_labram_backbone(modeling_file: Path, checkpoint: Path,
                          device: torch.device, model_init_seed: int):
    import_modeling_file(modeling_file)
    from timm.models import create_model
    # Same model-side random tensors across ALL tasks/replicates.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(model_init_seed))
        model = create_model(
            "labram_base_patch200_200",
            pretrained=False,
            num_classes=0,
            num_patches_per_channel_input=4,
            init_values=0.1,
        )
    audit = strict_load_labram_checkpoint(model, checkpoint)
    digest = torch_state_digest(model.state_dict())
    model = model.to(device)
    for p in model.parameters():
        if p.is_floating_point():
            p.requires_grad_(True)
    return model, audit, digest


class TaskClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int,
                 input_chans: Sequence[int], head_seed: int):
        super().__init__()
        self.backbone = backbone
        self.input_chans = [int(x) for x in input_chans]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(head_seed))
            self.head = nn.Linear(200, int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone.forward_features(x, input_chans=self.input_chans)
        if z.ndim != 2 or z.shape[-1] != 200:
            raise ValueError(f"Expected LaBraM pooled features [B,200], found {tuple(z.shape)}")
        return self.head(z)


def resolve_input_chans(args) -> Tuple[List[int], str, bool]:
    if args.input_chans_file:
        text = Path(args.input_chans_file).read_text(encoding="utf-8")
        vals = [int(x) for x in text.replace("\n", ",").replace(" ", ",").split(",") if x.strip()]
        source = f"file:{args.input_chans_file}"
        provisional = False
    elif args.input_chans:
        vals = parse_csv_ints(args.input_chans)
        source = "cli"
        provisional = False
    else:
        vals = list(range(1, 65))
        source = "PROVISIONAL_1_to_64"
        provisional = True

    if len(vals) == 64:
        full = [0] + vals
    elif len(vals) == 65 and vals[0] == 0:
        full = vals
    else:
        raise ValueError("Channel mapping must contain 64 values or 65 including CLS=0")
    if full[0] != 0 or len(set(full)) != len(full) or min(full) < 0 or max(full) > 128:
        raise ValueError("Invalid LaBraM channel positional mapping")
    if provisional and not args.allow_provisional_channels:
        raise RuntimeError(
            "Exact 64-channel order is unconfirmed. Scientific run refused. "
            "Pass --input-chans-file / --input-chans, or explicitly use "
            "--allow-provisional-channels for smoke/pilot only."
        )
    return full, source, provisional


# -----------------------------------------------------------------------------
# TSV module extraction
# -----------------------------------------------------------------------------

def extract_tsv_weights(model: nn.Module) -> Dict[str, torch.Tensor]:
    if not hasattr(model, "blocks") or len(model.blocks) != N_BLOCKS:
        raise ValueError("Expected LaBraM base with 12 Transformer blocks")
    out: Dict[str, torch.Tensor] = {}
    for l, block in enumerate(model.blocks):
        qkv = block.attn.qkv.weight.detach()
        if tuple(qkv.shape) != (600, 200):
            raise ValueError(f"Block {l}: qkv expected (600,200), found {tuple(qkv.shape)}")
        o = block.attn.proj.weight.detach()
        fc1 = block.mlp.fc1.weight.detach()
        fc2 = block.mlp.fc2.weight.detach()
        expected_shapes = {
            "O": (200, 200),
            "fc1": (800, 200),
            "fc2": (200, 800),
        }
        actual_shapes = {"O": tuple(o.shape), "fc1": tuple(fc1.shape), "fc2": tuple(fc2.shape)}
        if actual_shapes != expected_shapes:
            raise ValueError(
                f"Block {l}: unexpected LaBraM-base TSV module shapes {actual_shapes}; "
                f"expected {expected_shapes}"
            )
        q, k, v = torch.split(qkv, 200, dim=0)
        out[f"L{l:02d}/Q"] = q.cpu().clone()
        out[f"L{l:02d}/K"] = k.cpu().clone()
        out[f"L{l:02d}/V"] = v.cpu().clone()
        out[f"L{l:02d}/O"] = o.cpu().clone()
        out[f"L{l:02d}/fc1"] = fc1.cpu().clone()
        out[f"L{l:02d}/fc2"] = fc2.cpu().clone()
    if len(out) != N_TSV_MODULES:
        raise AssertionError(f"Expected 72 TSV modules, got {len(out)}")
    return out


def save_delta_npz(path: Path, base: Mapping[str, torch.Tensor],
                   adapted: Mapping[str, torch.Tensor]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    payload = {
        sanitize_module_key(m):
            (adapted[m] - base[m]).numpy().astype(np.float32, copy=False)
        for m in sorted(base)
    }
    tmp = _atomic_tmp_path(path, suffix=".npz")
    try:
        np.savez_compressed(tmp, **payload)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def spectrum_stats_from_s(s: np.ndarray, probes: Sequence[int]):
    s = np.asarray(s, dtype=np.float64)
    e = s * s
    total = float(e.sum())
    cum = np.cumsum(e / max(total, EPS)) if e.size else np.zeros(0, dtype=float)

    def rank_at(thr):
        if total <= EPS or not len(cum):
            return 0
        return min(int(np.searchsorted(cum, thr, side="left")) + 1, len(cum))

    out = {
        "delta_norm": float(np.linalg.norm(s)),
        "rank80": rank_at(0.80),
        "rank90": rank_at(0.90),
        "rank95": rank_at(0.95),
        "rank99": rank_at(0.99),
    }
    for r in probes:
        rr = min(int(r), len(cum))
        out[f"E{int(r)}"] = float(cum[rr - 1]) if rr else 0.0
    return out


def spectrum_stats(delta: np.ndarray, probes: Sequence[int]):
    s = np.linalg.svd(np.asarray(delta, dtype=np.float64),
                      full_matrices=False, compute_uv=False)
    return s, spectrum_stats_from_s(s, probes)


def _basis_projection_drift(A: np.ndarray, B: np.ndarray) -> float:
    """Normalized Frobenius distance between orthogonal projectors, rank-aware."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    ra, rb = A.shape[1], B.shape[1]
    if ra == 0 and rb == 0:
        return 0.0
    overlap = float(np.sum((A.T @ B) ** 2)) if ra and rb else 0.0
    d2 = max(float(ra + rb) - 2.0 * overlap, 0.0)
    return math.sqrt(d2 / max(float(ra + rb), 1.0))


def tsv_epoch_audit(backbone: nn.Module, base_tsv: Mapping[str, torch.Tensor],
                    probes: Sequence[int]):
    adapted = extract_tsv_weights(backbone)
    rows = []
    rels, r90s, e5s = [], [], []
    bases: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for module in sorted(base_tsv):
        delta = (adapted[module] - base_tsv[module]).numpy().astype(np.float64, copy=False)
        U, s, Vh = np.linalg.svd(delta, full_matrices=False)
        st = spectrum_stats_from_s(s, probes)
        r90 = int(st["rank90"])
        bases[module] = (U[:, :r90], Vh.T[:, :r90])
        base_norm = float(torch.linalg.vector_norm(base_tsv[module].double()).item())
        rel = float(st["delta_norm"]) / max(base_norm, EPS)
        layer, role = module_layer_role(module)
        row = {
            "module": module, "layer": layer, "role": role,
            "relative_delta_norm": rel,
            **st,
        }
        rows.append(row)
        rels.append(rel)
        r90s.append(float(st["rank90"]))
        e5s.append(float(st.get("E5", float("nan"))))
    summary = {
        "median_module_relative_delta_norm": median_or_nan(rels),
        "median_r90": median_or_nan(r90s),
        "median_E5": median_or_nan(e5s),
    }
    return summary, rows, bases


def tsv_subspace_drift(prev_bases, cur_bases) -> Tuple[float, float]:
    if prev_bases is None:
        return float("nan"), float("nan")
    du, dv = [], []
    for module in sorted(cur_bases):
        pu, pv = prev_bases[module]
        cu, cv = cur_bases[module]
        du.append(_basis_projection_drift(pu, cu))
        dv.append(_basis_projection_drift(pv, cv))
    return median_or_nan(du), median_or_nan(dv)


# -----------------------------------------------------------------------------
# Full-backbone state audit
# -----------------------------------------------------------------------------

def snapshot_float_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
        if torch.is_tensor(v) and v.is_floating_point()
    }


@torch.no_grad()
def relative_state_update_norm(model: nn.Module,
                               base_state: Mapping[str, torch.Tensor]) -> float:
    num = 0.0
    den = 0.0
    state = model.state_dict()
    for k, b in base_state.items():
        a = state[k].detach().cpu()
        d = (a.double() - b.double())
        num += float(torch.sum(d * d).item())
        den += float(torch.sum(b.double() * b.double()).item())
    return math.sqrt(max(num, 0.0)) / max(math.sqrt(max(den, 0.0)), EPS)


@torch.no_grad()
def relative_state_step_norm(model: nn.Module,
                             prev_state: Mapping[str, torch.Tensor],
                             base_state: Mapping[str, torch.Tensor]) -> float:
    num = 0.0
    den = 0.0
    state = model.state_dict()
    for k, b in base_state.items():
        a = state[k].detach().cpu().double()
        p = prev_state[k].double()
        d = a - p
        num += float(torch.sum(d * d).item())
        bd = b.double()
        den += float(torch.sum(bd * bd).item())
    return math.sqrt(max(num, 0.0)) / max(math.sqrt(max(den, 0.0)), EPS)


def global_grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    sq = 0.0
    found = False
    for p in parameters:
        if p.grad is None:
            continue
        g = p.grad.detach()
        if not torch.isfinite(g).all():
            return float("nan")
        sq += float(torch.sum(g.float() * g.float()).item())
        found = True
    return math.sqrt(max(sq, 0.0)) if found else 0.0


# -----------------------------------------------------------------------------
# Metrics / training
# -----------------------------------------------------------------------------

def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, c: int) -> np.ndarray:
    cm = np.zeros((c, c), dtype=np.int64)
    for a, b in zip(y_true.astype(int), y_pred.astype(int)):
        cm[a, b] += 1
    return cm


def metrics_from_cm(cm: np.ndarray) -> Dict[str, object]:
    n = int(cm.sum())
    acc = float(np.trace(cm) / n) if n else float('nan')
    recalls, f1s = [], []
    for c in range(cm.shape[0]):
        tp = float(cm[c, c])
        fn = float(cm[c, :].sum() - cm[c, c])
        fp = float(cm[:, c].sum() - cm[c, c])
        # All task classes have support by contract. If a model predicts none of
        # a class, precision/F1 are 0 rather than NaN; excluding that class would
        # spuriously inflate macro-F1.
        rec = tp / (tp + fn) if tp + fn else 0.0
        pre = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * rec * pre / (rec + pre) if rec + pre else 0.0
        recalls.append(rec)
        f1s.append(f1)
    return {
        'n': n,
        'accuracy': acc,
        'balanced_accuracy': float(np.mean(recalls)) if recalls else float('nan'),
        'macro_f1': float(np.mean(f1s)) if f1s else float('nan'),
        'recall_per_class': recalls,
        'confusion_matrix': cm,
    }

@torch.no_grad()
def evaluate(classifier: TaskClassifier, loader: DataLoader, device: torch.device,
             num_classes: int, class_weights: Optional[torch.Tensor] = None):
    classifier.eval()
    weighted_loss_sum = 0.0
    weight_sum = 0.0
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = classifier(x)
        per_sample = F.cross_entropy(
            logits, y, weight=class_weights, reduction="none"
        )
        if class_weights is None:
            denom_weights = torch.ones_like(y, dtype=per_sample.dtype)
        else:
            denom_weights = class_weights[y]
        weighted_loss_sum += float(per_sample.sum().item())
        weight_sum += float(denom_weights.sum().item())
        ys.append(y.detach().cpu().numpy())
        ps.append(logits.argmax(dim=1).detach().cpu().numpy())
    if not ys:
        raise RuntimeError("Empty loader")
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    out = metrics_from_cm(confusion_matrix_np(y_true, y_pred, num_classes))
    out["objective_loss"] = weighted_loss_sum / max(weight_sum, EPS)
    return out


def make_optimizer(classifier: TaskClassifier, args) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        [
            {
                "params": [p for p in classifier.backbone.parameters() if p.requires_grad],
                "lr": float(args.backbone_lr),
                "weight_decay": float(args.weight_decay),
            },
            {
                "params": [p for p in classifier.head.parameters() if p.requires_grad],
                "lr": float(args.head_lr),
                "weight_decay": 0.0,
            },
        ],
        betas=(float(args.beta1), float(args.beta2)),
        eps=float(args.adam_eps),
    )


def make_loaders(meta: TaskMeta, args, loader_seed: int, device: torch.device,
                 task_data: Optional[InMemoryTaskData] = None):
    """Training/eval loaders. Scientific runs default to host-RAM staging."""
    if task_data is None:
        # Kept only for tiny preflight probes / compatibility. Never use this for
        # long shuffled training because random HDF5 fancy indexing is too slow.
        train_loader = H5BatchLoader(
            meta, batch_size=args.batch_size, shuffle=True, seed=loader_seed,
            pin_memory=False,
        )
        eval_loader = H5BatchLoader(
            meta, batch_size=args.eval_batch_size, shuffle=False, seed=0,
            pin_memory=False,
        )
    else:
        train_loader = InMemoryBatchLoader(
            task_data, batch_size=args.batch_size, shuffle=True, seed=loader_seed
        )
        eval_loader = InMemoryBatchLoader(
            task_data, batch_size=args.eval_batch_size, shuffle=False, seed=0
        )
    return train_loader, eval_loader

def make_eval_loader(meta: TaskMeta, batch_size: int, device: torch.device,
                     indices: Optional[np.ndarray] = None,
                     task_data: Optional[InMemoryTaskData] = None):
    if task_data is not None:
        return InMemoryBatchLoader(
            task_data, batch_size=int(batch_size), shuffle=False, seed=0,
            indices=indices,
        )
    return H5BatchLoader(
        meta, batch_size=int(batch_size), shuffle=False, seed=0,
        indices=indices, pin_memory=False,
    )

def train_one_epoch(classifier: TaskClassifier, loader, device: torch.device,
                    optimizer: torch.optim.Optimizer, class_weights: torch.Tensor,
                    scaler, amp: bool, epoch: int) -> Dict[str, object]:
    """One training epoch with lightweight online metrics and no per-batch full-model audits."""
    classifier.train()
    c = int(classifier.head.out_features)
    cm_gpu = torch.zeros((c, c), dtype=torch.int64, device=device)
    loss_num = torch.zeros((), dtype=torch.float64, device=device)
    loss_den = torch.zeros((), dtype=torch.float64, device=device)
    n_steps = 0

    for bi, (x, y) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=False)
        y = y.to(device, non_blocking=False)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=bool(amp and device.type == 'cuda')):
            logits = classifier(x)
            loss = F.cross_entropy(logits, y, weight=class_weights)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at epoch={epoch} batch={bi}")
        scaler.scale(loss).backward()
        # Scientific protocol: no clipping. Also no O(P) global-grad sweep every batch.
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            cm_gpu += torch.bincount(
                y.to(torch.int64) * c + pred.to(torch.int64), minlength=c * c
            ).reshape(c, c)
            denom = class_weights[y].sum().to(torch.float64)
            loss_num += loss.detach().to(torch.float64) * denom
            loss_den += denom
        n_steps += 1

    cm = cm_gpu.detach().cpu().numpy()
    met = metrics_from_cm(cm)
    met['objective_loss'] = float((loss_num / torch.clamp(loss_den, min=EPS)).item())
    met['optimizer_steps'] = int(n_steps)
    met['amp_scale'] = float(scaler.get_scale())
    return met

def _relative_range(values: Sequence[float]) -> float:
    a = np.asarray(values, dtype=np.float64)
    if not np.isfinite(a).all() or a.size == 0:
        return float("inf")
    return float((np.max(a) - np.min(a)) / max(abs(float(np.mean(a))), EPS))


def _absolute_range(values: Sequence[float]) -> float:
    a = np.asarray(values, dtype=np.float64)
    if not np.isfinite(a).all() or a.size == 0:
        return float("inf")
    return float(np.max(a) - np.min(a))


def _stable_at(rows: Sequence[Mapping[str, object]], i: int, args) -> bool:
    """Lean rolling convergence audit based on optimization behavior only."""
    w = int(args.conv_window)
    if i + 1 < w:
        return False
    win = rows[i - w + 1:i + 1]
    if not all(bool(r.get('finite_ok', False)) for r in win):
        return False
    loss_rel = _relative_range([float(r['objective_loss']) for r in win])
    bacc_range = _absolute_range([float(r['balanced_accuracy']) for r in win])
    learning_ok = all(
        float(r['balanced_accuracy']) >= float(r['chance_bacc']) + float(args.conv_min_bacc_margin)
        for r in win
    )
    return (
        loss_rel <= float(args.conv_loss_rel)
        and bacc_range <= float(args.conv_bacc_abs)
        and learning_ok
    )

def find_convergence_epoch(rows: Sequence[Mapping[str, object]], args) -> Optional[int]:
    """Return the epoch at which convergence is actually *certified*.

    This is the end of the required stable-window streak, not the first stable
    window. Using the certification epoch keeps E* correct for any patience >= 1.
    """
    patience = int(args.conv_patience)
    streak = 0
    for i in range(len(rows)):
        if _stable_at(rows, i, args):
            streak += 1
            if streak >= patience:
                return int(rows[i]["epoch"])
        else:
            streak = 0
    return None


# -----------------------------------------------------------------------------
# Run one task/replicate
# -----------------------------------------------------------------------------

def _prepare_run_dir(run_dir: Path, overwrite: bool) -> Path:
    if run_dir.exists() and any(run_dir.iterdir()):
        if not overwrite:
            raise RuntimeError(
                f"Non-empty run dir exists: {run_dir}. Refusing to mix stale and new artifacts; "
                "use --overwrite-run to replace it explicitly."
            )
        shutil.rmtree(run_dir)
    return ensure_dir(run_dir)


def run_task(args, task: str, replicate: int, meta: TaskMeta,
             input_chans: Sequence[int], modeling_file: Path, checkpoint: Path,
             device: torch.device, epochs: int, phase: str,
             protocol_fingerprint: str, task_data: InMemoryTaskData):
    """Run one task/replicate with lean audits and RAM-resident minibatching."""
    if phase not in ('pilot', 'final'):
        raise ValueError(phase)

    if phase == 'pilot':
        run_dir = Path(args.outdir) / 'pilot' / 'runs' / task / f'rep{replicate:02d}'
    else:
        run_dir = Path(args.outdir) / 'runs' / task / f'rep{replicate:02d}'
    run_dir = _prepare_run_dir(run_dir, bool(args.overwrite_run))

    log('=' * 96)
    log(
        f"{phase.upper()} task={task} replicate={replicate} n={meta.n} epochs={epochs} "
        f"io=RAM({task_data.nbytes/1024**3:.2f}GiB)"
    )

    head_seed = stable_seed(args.seed_base, task, replicate, 'head')
    runtime_seed = stable_seed(args.seed_base, task, replicate, 'runtime', phase)
    loader_seed = stable_seed(args.seed_base, task, replicate, 'loader', phase)

    backbone, checkpoint_audit, base_digest = build_labram_backbone(
        modeling_file, checkpoint, device, args.model_init_seed
    )
    base_tsv = extract_tsv_weights(backbone)
    classifier = TaskClassifier(
        backbone, len(meta.classes), input_chans, head_seed=head_seed
    ).to(device)
    seed_everything(runtime_seed)

    train_loader, eval_loader = make_loaders(
        meta, args, loader_seed, device, task_data=task_data
    )
    weights = class_weights_for_meta(meta, device)
    optimizer = make_optimizer(classifier, args)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp and device.type == 'cuda'))

    epoch_rows: List[Dict[str, object]] = []
    for epoch in range(1, int(epochs) + 1):
        t0 = time.time()
        met = train_one_epoch(
            classifier, train_loader, device, optimizer, weights,
            scaler, args.amp, epoch
        )
        finite_ok = all(
            math.isfinite(float(met[k]))
            for k in ('objective_loss', 'accuracy', 'balanced_accuracy', 'macro_f1')
        )
        row = {
            'task': task,
            'replicate': replicate,
            'phase': phase,
            'epoch': epoch,
            'n_samples': meta.n,
            'optimizer_steps_this_epoch': int(met['optimizer_steps']),
            'objective_loss': float(met['objective_loss']),
            'accuracy': float(met['accuracy']),
            'balanced_accuracy': float(met['balanced_accuracy']),
            'chance_bacc': float(1.0 / len(meta.classes)),
            'macro_f1': float(met['macro_f1']),
            'amp_scale': float(met['amp_scale']),
            'finite_ok': bool(finite_ok),
            'seconds': float(time.time() - t0),
            'metric_scope': 'online_train_pass',
        }
        if not finite_ok:
            raise FloatingPointError(f"{task}/rep{replicate}: non-finite epoch metrics at epoch {epoch}")
        epoch_rows.append(row)
        log(
            f"  epoch {epoch:02d}/{epochs}: loss={row['objective_loss']:.4f}, "
            f"bACC={row['balanced_accuracy']:.4f}, {row['seconds']:.1f}s"
        )

    write_csv(run_dir / 'epoch_metrics.csv', epoch_rows)
    conv_epoch = find_convergence_epoch(epoch_rows, args)

    # One clean eval-mode pass at the end of the run, not after every epoch.
    final_metrics = evaluate(classifier, eval_loader, device, len(meta.classes), weights)
    if not all(
        math.isfinite(float(final_metrics[k]))
        for k in ('objective_loss', 'accuracy', 'balanced_accuracy', 'macro_f1')
    ):
        raise FloatingPointError(f"{task}/rep{replicate}: non-finite final eval metrics")
    if float(final_metrics['balanced_accuracy']) < float(1.0 / len(meta.classes)) + float(args.conv_min_bacc_margin):
        raise RuntimeError(
            f"{task}/rep{replicate}: final bACC={final_metrics['balanced_accuracy']:.4f} is too close "
            "to chance; refusing a likely failed adaptation run."
        )

    # Geometry is audited once at the trained endpoint. Repeating 72 SVDs every
    # epoch was expensive and does not materially improve E* selection.
    tsv_summary, _, _ = tsv_epoch_audit(backbone, base_tsv, args.energy_probes)

    if phase == 'pilot':
        summary = {
            'version': VERSION,
            'task': task,
            'replicate': replicate,
            'n_samples': meta.n,
            'protocol_fingerprint': protocol_fingerprint,
            'base_full_backbone_digest': base_digest,
            'convergence_epoch': conv_epoch,
            'epochs_run': int(epochs),
            'online_metrics_final': epoch_rows[-1],
            'eval_metrics_final': final_metrics,
            'endpoint_tsv_summary': tsv_summary,
            'ram_staging': {
                'nbytes': int(task_data.nbytes),
                'load_seconds': float(task_data.load_seconds),
            },
        }
        dump_json(run_dir / 'pilot_run_summary.json', summary)
        dump_json(run_dir / '_SUCCESS.json', {
            'phase': 'pilot',
            'task': task,
            'replicate': replicate,
            'protocol_fingerprint': protocol_fingerprint,
            'convergence_certified': conv_epoch is not None,
            'summary_sha256': sha256_file(run_dir / 'pilot_run_summary.json'),
        })
        result = {
            'task': task,
            'replicate': replicate,
            'convergence_epoch': conv_epoch,
            'epochs_run': int(epochs),
            'n_samples': meta.n,
            'final_bacc': float(final_metrics['balanced_accuracy']),
            'protocol_fingerprint': protocol_fingerprint,
            'base_full_backbone_digest': base_digest,
        }
    else:
        dump_json(run_dir / 'final_task_metrics.json', final_metrics)
        adapted_tsv = extract_tsv_weights(backbone)
        save_delta_npz(run_dir / 'delta_weights.npz', base_tsv, adapted_tsv)

        spectrum_rows_final = []
        spectrum_summary = []
        for module in sorted(base_tsv):
            delta = (adapted_tsv[module] - base_tsv[module]).numpy().astype(np.float64, copy=False)
            svals, st = spectrum_stats(delta, args.energy_probes)
            e = svals * svals
            total = max(float(e.sum()), EPS)
            cum = np.cumsum(e / total)
            base_norm = float(torch.linalg.vector_norm(base_tsv[module].double()).item())
            layer, role = module_layer_role(module)
            for a, sigma in enumerate(svals, start=1):
                spectrum_rows_final.append({
                    'task': task, 'replicate': replicate,
                    'module': module, 'layer': layer, 'role': role,
                    'mode': a, 'sigma': float(sigma),
                    'energy_fraction': float((sigma * sigma) / total),
                    'cumulative_energy': float(cum[a - 1]),
                })
            spectrum_summary.append({
                'task': task, 'replicate': replicate,
                'module': module, 'layer': layer, 'role': role,
                'd_out': int(delta.shape[0]), 'd_in': int(delta.shape[1]),
                'base_norm': base_norm,
                'relative_delta_norm': float(st['delta_norm']) / max(base_norm, EPS),
                **st,
            })
        write_csv(run_dir / 'spectrum_full.csv', spectrum_rows_final)
        write_csv(run_dir / 'spectrum_summary.csv', spectrum_summary)

        metadata = {
            'version': VERSION,
            'task': task,
            'replicate': replicate,
            'num_classes': len(meta.classes),
            'raw_classes': [int(x) for x in meta.classes],
            'n_samples': meta.n,
            'n_subjects': int(len(np.unique(meta.subjects))),
            'all_subjects_pooled': True,
            'epochs': int(epochs),
            'pilot_convergence_epoch_for_this_trajectory': conv_epoch,
            'protocol_fingerprint': protocol_fingerprint,
            'batch_size': int(args.batch_size),
            'eval_batch_size': int(args.eval_batch_size),
            'optimizer_steps': int(sum(int(r['optimizer_steps_this_epoch']) for r in epoch_rows)),
            'optimizer': {
                'name': 'AdamW',
                'backbone_lr': args.backbone_lr,
                'head_lr': args.head_lr,
                'betas': [args.beta1, args.beta2],
                'eps': args.adam_eps,
                'weight_decay': args.weight_decay,
                'gradient_clipping': False,
                'scheduler': None,
                'warmup': None,
                'early_stopping': False,
                'amp': bool(args.amp),
            },
            'head_initialization': 'PyTorch nn.Linear default under fixed head_seed',
            'loss': {
                'name': 'class_weighted_cross_entropy',
                'weights': weights.detach().cpu().numpy().tolist(),
            },
            'input_chans': [int(x) for x in input_chans],
            'head_seed': int(head_seed),
            'runtime_seed': int(runtime_seed),
            'loader_seed': int(loader_seed),
            'base_full_backbone_digest': base_digest,
            'checkpoint_audit': checkpoint_audit,
            'metrics_final': final_metrics,
            'endpoint_tsv_summary': tsv_summary,
            'ram_staging': {
                'nbytes': int(task_data.nbytes),
                'load_seconds': float(task_data.load_seconds),
            },
        }

        checkpoint_out = run_dir / 'adapted_checkpoint.pth'
        atomic_torch_save({
            'task': task,
            'replicate': replicate,
            'backbone': backbone.state_dict(),
            'head': classifier.head.state_dict(),
            'input_chans': list(input_chans),
            'protocol_fingerprint': protocol_fingerprint,
            'base_full_backbone_digest': base_digest,
        }, checkpoint_out)

        metadata['artifacts'] = {
            'delta_weights_sha256': sha256_file(run_dir / 'delta_weights.npz'),
            'adapted_checkpoint_sha256': sha256_file(checkpoint_out),
        }
        dump_json(run_dir / 'metadata.json', metadata)
        dump_json(run_dir / '_SUCCESS.json', {
            'phase': 'final',
            'task': task,
            'replicate': replicate,
            'protocol_fingerprint': protocol_fingerprint,
            'base_full_backbone_digest': base_digest,
            'metadata_sha256': sha256_file(run_dir / 'metadata.json'),
            **metadata['artifacts'],
        })
        result = {
            'task': task, 'replicate': replicate,
            'n_samples': meta.n,
            'epochs': int(epochs),
            'convergence_epoch': conv_epoch,
            'final_bacc': float(final_metrics['balanced_accuracy']),
            'protocol_fingerprint': protocol_fingerprint,
            'base_full_backbone_digest': base_digest,
        }

    del classifier, backbone, train_loader, eval_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result

def _import_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_epoch_curves(root: Path, pilot: bool) -> None:
    try:
        plt = _import_plt()
    except Exception as e:
        log(f"WARNING: matplotlib unavailable, skipping plots: {e}")
        return
    base = root / "pilot" / "runs" if pilot else root / "runs"
    out = ensure_dir(root / "figures" / "training")
    rows = []
    for p in sorted(base.glob("*/rep*/epoch_metrics.csv")):
        with open(p, "r", encoding="utf-8", newline="") as f:
            rows.extend(list(csv.DictReader(f)))
    if not rows:
        return

    metrics = [
        ("objective_loss", "Online training objective loss"),
        ("balanced_accuracy", "Online training balanced accuracy"),
        ("seconds", "Epoch wall-clock seconds"),
    ]
    tasks = sorted(set(r["task"] for r in rows))
    for key, ylabel in metrics:
        fig, ax = plt.subplots(figsize=(8, 5))
        for task in tasks:
            for rep in sorted(set(int(r["replicate"]) for r in rows if r["task"] == task)):
                rr = [r for r in rows if r["task"] == task and int(r["replicate"]) == rep]
                rr.sort(key=lambda x: int(x["epoch"]))
                vals = [
                    float(x[key]) if str(x.get(key, "")).strip() not in ("", "None")
                    else float("nan")
                    for x in rr
                ]
                ax.plot(
                    [int(x["epoch"]) for x in rr],
                    vals,
                    marker="o", label=f"{task} rep{rep}",
                )
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / f"{'pilot' if pilot else 'final'}_{key}.png", dpi=180)
        plt.close(fig)


# -----------------------------------------------------------------------------
# Modes
# -----------------------------------------------------------------------------

def load_all_metas(args) -> Dict[str, TaskMeta]:
    root = Path(args.data_root)
    out = {}
    for task in args.tasks:
        if task not in TASK_FILES:
            raise ValueError(f"Unknown task {task}")
        out[task] = load_task_meta(task, root / TASK_FILES[task])
    return out


def audit_subject_composition(metas: Mapping[str, TaskMeta], tasks: Sequence[str],
                              allow_subject_set_mismatch: bool = False,
                              allow_subject_class_gaps: bool = False) -> Dict[str, object]:
    """Guard against confounding task geometry with subject-population differences.

    For the intended M3CV pooled-task experiment every selected task should expose
    the same subject population, and every subject should contribute every class of
    the corresponding task. Deviations are scientific-contract violations unless an
    explicitly unsafe override is supplied after manual inspection.
    """
    if not tasks:
        raise ValueError("Need at least one task for subject-composition audit")

    subject_sets = {
        t: tuple(int(x) for x in np.unique(metas[t].subjects))
        for t in tasks
    }
    ref_task = str(tasks[0])
    ref_set = subject_sets[ref_task]
    mismatches = {}
    for t in tasks[1:]:
        if subject_sets[t] != ref_set:
            a, b = set(ref_set), set(subject_sets[t])
            mismatches[t] = {
                "missing_vs_reference": sorted(int(x) for x in a - b),
                "extra_vs_reference": sorted(int(x) for x in b - a),
            }
    if mismatches and not allow_subject_set_mismatch:
        raise RuntimeError(
            "Selected tasks do not contain the same subject-ID set. This would confound "
            f"task geometry with subject composition: {mismatches}. Refusing GIGO."
        )

    class_gaps = {}
    per_subject_counts = {}
    for t in tasks:
        meta = metas[t]
        task_counts = {}
        gaps = []
        for subject in subject_sets[t]:
            mask = meta.subjects == int(subject)
            counts = {
                int(c): int(np.sum(meta.labels[mask] == int(c)))
                for c in meta.classes
            }
            task_counts[str(int(subject))] = {str(k): v for k, v in counts.items()}
            for c, n in counts.items():
                if n <= 0:
                    gaps.append({"subject": int(subject), "class": int(c)})
        per_subject_counts[t] = task_counts
        if gaps:
            class_gaps[t] = gaps
    if class_gaps and not allow_subject_class_gaps:
        preview = {k: v[:20] for k, v in class_gaps.items()}
        raise RuntimeError(
            "Some subject x class cells are empty. The pooled task objective would then "
            f"mix task adaptation with missing subject/class support: {preview}. Refusing GIGO."
        )

    return {
        "reference_task": ref_task,
        "same_subject_set": not bool(mismatches),
        "subject_set_mismatches": mismatches,
        "subject_ids": {t: list(subject_sets[t]) for t in tasks},
        "n_subjects": {t: len(subject_sets[t]) for t in tasks},
        "subject_class_gaps": class_gaps,
        "per_subject_class_counts": per_subject_counts,
        "unsafe_allow_subject_set_mismatch": bool(allow_subject_set_mismatch),
        "unsafe_allow_subject_class_gaps": bool(allow_subject_class_gaps),
    }


def build_protocol_payload(args, data_audits, subject_composition_audit,
                           input_chans, input_source, provisional, checkpoint,
                           modeling_file, base_digest) -> Dict[str, object]:
    try:
        import timm
        timm_version = getattr(timm, "__version__", "unknown")
    except Exception:
        timm_version = "unknown"
    return {
        "protocol_schema": 3,
        "runner_version": VERSION,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "tasks": list(args.tasks),
        "replicates": int(args.replicates),
        "all_subjects_pooled": True,
        "data": {
            t: {
                "path": a["path"],
                "shape": a["shape"],
                "file_size_bytes": int(a["file_size_bytes"]),
                "file_mtime_ns": int(a["file_mtime_ns"]),
                "content_sha256": a["content_sha256"],
                "labels_sha256": a["labels_sha256"],
                "subject_ids_sha256": a["subject_ids_sha256"],
                "global_mean": a["global_mean"],
                "global_std": a["global_std"],
                "label_mapping_source": a["label_mapping_source"],
                "label_mapping": a["label_mapping"],
                "label_mapping_verified": bool(a["label_mapping_verified"]),
            }
            for t, a in data_audits.items()
        },
        "data_contract": {
            "exact_label_ids_required": True,
            "label_semantics_metadata_required": not bool(args.unsafe_allow_missing_label_mapping),
            "unsafe_allow_missing_label_mapping": bool(args.unsafe_allow_missing_label_mapping),
            "same_subject_set_required": not bool(args.unsafe_allow_subject_set_mismatch),
            "unsafe_allow_subject_set_mismatch": bool(args.unsafe_allow_subject_set_mismatch),
            "complete_subject_x_class_support_required": not bool(args.unsafe_allow_subject_class_gaps),
            "unsafe_allow_subject_class_gaps": bool(args.unsafe_allow_subject_class_gaps),
            "subject_composition_audit": subject_composition_audit,
        },
        "preprocessing_contract": {
            "h5_already_global_zscored": True,
            "divide_by_100": False,
            "second_zscore": False,
            "reshape": "[64,800]->[64,4,200]",
        },
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "sha256": sha256_file(checkpoint),
            "base_full_backbone_digest": base_digest,
        },
        "modeling_file": {
            "path": str(modeling_file.resolve()),
            "sha256": sha256_file(modeling_file),
        },
        "input_chans": [int(x) for x in input_chans],
        "input_chans_source": str(input_source),
        "provisional_channels": bool(provisional),
        "model_init_seed": int(args.model_init_seed),
        "seed_base": int(args.seed_base),
        "head_initialization": "PyTorch nn.Linear default under fixed head_seed",
        "optimization": {
            "optimizer": "AdamW",
            "backbone_lr": float(args.backbone_lr),
            "head_lr": float(args.head_lr),
            "weight_decay": float(args.weight_decay),
            "betas": [float(args.beta1), float(args.beta2)],
            "eps": float(args.adam_eps),
            "batch_size": int(args.batch_size),
            "eval_batch_size": int(args.eval_batch_size),
            "class_weighted_ce": True,
            "gradient_clipping": False,
            "scheduler": None,
            "warmup": None,
            "early_stopping": False,
            "amp": bool(args.amp),
        },
        "final_epoch_contract": {
            "pilot_E_star_required": not bool(args.unsafe_allow_manual_final_epochs),
            "unsafe_allow_manual_final_epochs": bool(args.unsafe_allow_manual_final_epochs),
        },
        "convergence": {
            "window": int(args.conv_window),
            "patience": int(args.conv_patience),
            "loss_relative_range": float(args.conv_loss_rel),
            "bacc_absolute_range": float(args.conv_bacc_abs),
            "minimum_bacc_margin_over_chance": float(args.conv_min_bacc_margin),
            "metric_scope": "online_train_pass",
            "fallback_if_uncertified": "pilot_max_epochs",
        },
        "io_contract": {
            "mode": "host_ram_preload_one_task_at_a_time",
            "random_hdf5_training_reads": False,
            "full_eval_every_epoch": False,
            "endpoint_tsv_svd_only": True,
        },
        "tsv_contract": {
            "blocks": N_BLOCKS,
            "roles": list(MODULE_ROLES),
            "n_modules": N_TSV_MODULES,
            "energy_probes": [int(x) for x in args.energy_probes],
        },
        "environment": {
            "python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "timm": timm_version,
            "numpy": np.__version__,
            "h5py": h5py.__version__,
            "device_type": str(torch.device(args.device).type),
        },
    }


def _preflight_model_task_audits(args, metas, input_chans, model, device):
    audits = {}
    for task in args.tasks:
        meta = metas[task]
        idx = preflight_stratified_indices(meta, args.preflight_probe_samples)
        loader = make_eval_loader(meta, len(idx), device, indices=idx)
        x, y = next(iter(loader))
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        classifier = TaskClassifier(
            model, len(meta.classes), input_chans,
            head_seed=stable_seed(args.seed_base, task, 0, "preflight_head"),
        ).to(device)
        classifier.zero_grad(set_to_none=True)
        logits = classifier(x)
        if tuple(logits.shape) != (len(idx), len(meta.classes)):
            raise RuntimeError(f"{task}: preflight logits shape {tuple(logits.shape)}")
        if not torch.isfinite(logits).all():
            raise FloatingPointError(f"{task}: non-finite preflight logits")
        weights = class_weights_for_meta(meta, device)
        loss = F.cross_entropy(logits, y, weight=weights)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{task}: non-finite preflight loss")
        loss.backward()
        backbone_grad = global_grad_norm(model.parameters())
        head_grad = global_grad_norm(classifier.head.parameters())
        if not math.isfinite(backbone_grad) or backbone_grad <= 0:
            raise RuntimeError(f"{task}: backbone gradient is zero/non-finite in preflight")
        if not math.isfinite(head_grad) or head_grad <= 0:
            raise RuntimeError(f"{task}: head gradient is zero/non-finite in preflight")

        module_grad_norms = []
        for l, block in enumerate(model.blocks):
            for role, param in [
                ("qkv", block.attn.qkv.weight),
                ("O", block.attn.proj.weight),
                ("fc1", block.mlp.fc1.weight),
                ("fc2", block.mlp.fc2.weight),
            ]:
                if param.grad is None or not torch.isfinite(param.grad).all():
                    raise RuntimeError(f"{task}: missing/non-finite grad L{l:02d}/{role}")
                gn = float(torch.linalg.vector_norm(param.grad.detach().float()).item())
                if gn <= 0:
                    raise RuntimeError(f"{task}: zero grad L{l:02d}/{role}")
                module_grad_norms.append(gn)

        audits[task] = {
            "probe_indices": [int(i) for i in idx],
            "probe_input_shape": list(x.shape),
            "probe_input_mean": float(x.mean().item()),
            "probe_input_std": float(x.std().item()),
            "probe_labels": sorted(set(int(v) for v in y.detach().cpu().tolist())),
            "logits_shape": list(logits.shape),
            "loss": float(loss.item()),
            "backbone_grad_norm": float(backbone_grad),
            "head_grad_norm": float(head_grad),
            "min_tsv_module_grad_norm": float(min(module_grad_norms)),
            "max_tsv_module_grad_norm": float(max(module_grad_norms)),
        }
        classifier.zero_grad(set_to_none=True)
        del classifier, loader, x, y, logits, loss

    # Exercise the actual optimizer/scaler step once after all task forward/backward
    # checks. This preflight model is discarded immediately afterwards.
    task = args.tasks[0]
    meta = metas[task]
    idx = preflight_stratified_indices(meta, args.preflight_probe_samples)
    loader = make_eval_loader(meta, len(idx), device, indices=idx)
    x, y = next(iter(loader))
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    classifier = TaskClassifier(
        model, len(meta.classes), input_chans,
        head_seed=stable_seed(args.seed_base, task, 0, "preflight_optimizer_head"),
    ).to(device)
    weights = class_weights_for_meta(meta, device)
    optimizer = make_optimizer(classifier, args)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp and device.type == "cuda"))
    before = snapshot_float_state(model)
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(enabled=bool(args.amp and device.type == "cuda")):
        loss = F.cross_entropy(classifier(x), y, weight=weights)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gn = global_grad_norm(classifier.parameters())
    if not math.isfinite(gn) or gn <= 0:
        raise RuntimeError("Preflight optimizer-step gradient is zero/non-finite")
    scaler.step(optimizer)
    scaler.update()
    step = relative_state_step_norm(model, before, before)
    if not math.isfinite(step) or step <= 0:
        raise RuntimeError("Preflight optimizer step did not change the backbone")
    audits["_optimizer_step"] = {
        "task": task,
        "relative_backbone_step": float(step),
        "global_grad_norm": float(gn),
        "optimizer_param_groups": [len(g["params"]) for g in optimizer.param_groups],
        "amp_scale_after_step": float(scaler.get_scale()),
    }
    del classifier, loader, x, y, loss, optimizer, scaler
    return audits


def run_preflight(args, metas, input_chans, input_source, provisional,
                  modeling_file, checkpoint, device):
    out = ensure_dir(Path(args.outdir))
    for stale in (out / "preflight.json", out / "_PREFLIGHT_SUCCESS.json"):
        stale.unlink(missing_ok=True)
    dump_json(out / "subject_metadata.json", subject_metadata(metas))

    subject_composition_audit = audit_subject_composition(
        metas, args.tasks,
        allow_subject_set_mismatch=bool(args.unsafe_allow_subject_set_mismatch),
        allow_subject_class_gaps=bool(args.unsafe_allow_subject_class_gaps),
    )
    dump_json(out / "subject_composition_audit.json", subject_composition_audit)
    log(
        "Preflight subject-composition audit PASSED: "
        + ", ".join(f"{t}={subject_composition_audit['n_subjects'][t]} subjects" for t in args.tasks)
    )

    data_audits = {}
    for task in args.tasks:
        log(f"Preflight full data integrity scan: {task} ({metas[task].n} samples)")
        data_audits[task] = audit_h5_data(
            metas[task],
            chunk_rows=args.preflight_chunk_rows,
            mean_abs_max=args.preflight_mean_abs_max,
            std_min=args.preflight_std_min,
            std_max=args.preflight_std_max,
            allow_missing_label_mapping=bool(args.unsafe_allow_missing_label_mapping),
        )
        log(
            f"  {task}: mean={data_audits[task]['global_mean']:.4g}, "
            f"std={data_audits[task]['global_std']:.4g}, "
            f"sha={data_audits[task]['content_sha256'][:12]}..."
        )

    model, checkpoint_audit, base_digest = build_labram_backbone(
        modeling_file, checkpoint, device, args.model_init_seed
    )
    task_model_audits = _preflight_model_task_audits(
        args, metas, input_chans, model, device
    )

    protocol = build_protocol_payload(
        args, data_audits, subject_composition_audit, input_chans,
        input_source, provisional, checkpoint, modeling_file, base_digest,
    )
    fingerprint = canonical_json_sha256(protocol)
    payload = {
        "version": VERSION,
        "timestamp": now(),
        "status": "PASSED",
        "protocol_fingerprint": fingerprint,
        "protocol": protocol,
        "checkpoint_audit": checkpoint_audit,
        "data_audits": data_audits,
        "model_task_audits": task_model_audits,
        "environment_verbose": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "h5py": h5py.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
    }
    dump_json(out / "preflight.json", payload)
    dump_json(out / "_PREFLIGHT_SUCCESS.json", {
        "version": VERSION,
        "protocol_fingerprint": fingerprint,
        "preflight_sha256": sha256_file(out / "preflight.json"),
    })
    log(f"Preflight PASSED. Protocol fingerprint={fingerprint}")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload



def try_reuse_preflight(args, metas, input_chans, input_source, provisional,
                        modeling_file, checkpoint) -> Optional[Dict[str, object]]:
    """Cheaply reuse a certified preflight when nothing material has changed."""
    if bool(getattr(args, 'force_preflight', False)):
        return None
    root = Path(args.outdir)
    pp = root / 'preflight.json'
    sp = root / '_PREFLIGHT_SUCCESS.json'
    if not pp.is_file() or not sp.is_file():
        return None
    try:
        payload = load_json(pp)
        success = load_json(sp)
        if str(payload.get('status')) != 'PASSED' or str(payload.get('version')) != VERSION:
            return None
        if success.get('preflight_sha256') != sha256_file(pp):
            return None
        protocol = payload.get('protocol', {})
        if str(protocol.get('runner_sha256', '')) != sha256_file(Path(__file__).resolve()):
            return None
        if [str(x) for x in protocol.get('tasks', [])] != [str(x) for x in args.tasks]:
            return None
        if int(protocol.get('replicates', -1)) != int(args.replicates):
            return None
        if [int(x) for x in protocol.get('input_chans', [])] != [int(x) for x in input_chans]:
            return None
        if bool(protocol.get('provisional_channels', True)) != bool(provisional):
            return None
        if str(protocol.get('input_chans_source', '')) != str(input_source):
            return None
        if sha256_file(checkpoint) != str(protocol.get('checkpoint', {}).get('sha256', '')):
            return None
        if sha256_file(modeling_file) != str(protocol.get('modeling_file', {}).get('sha256', '')):
            return None

        opt = protocol.get('optimization', {})
        expected_opt = {
            'backbone_lr': float(args.backbone_lr),
            'head_lr': float(args.head_lr),
            'weight_decay': float(args.weight_decay),
            'betas': [float(args.beta1), float(args.beta2)],
            'eps': float(args.adam_eps),
            'batch_size': int(args.batch_size),
            'eval_batch_size': int(args.eval_batch_size),
            'amp': bool(args.amp),
        }
        for k, v in expected_opt.items():
            if opt.get(k) != v:
                return None

        conv = protocol.get('convergence', {})
        expected_conv = {
            'window': int(args.conv_window),
            'patience': int(args.conv_patience),
            'loss_relative_range': float(args.conv_loss_rel),
            'bacc_absolute_range': float(args.conv_bacc_abs),
            'minimum_bacc_margin_over_chance': float(args.conv_min_bacc_margin),
        }
        for k, v in expected_conv.items():
            if conv.get(k) != v:
                return None

        # Labels/subjects are cheap to hash; EEG bytes are trusted from the previous
        # full scan only if file size and mtime are unchanged.
        audits = payload.get('data_audits', {})
        for t in args.tasks:
            a = audits.get(t, {})
            st = metas[t].path.stat()
            if int(a.get('file_size_bytes', -1)) != int(st.st_size):
                return None
            if int(a.get('file_mtime_ns', -1)) != int(st.st_mtime_ns):
                return None
            if str(a.get('labels_sha256', '')) != _sha256_ndarray(metas[t].labels):
                return None
            if str(a.get('subject_ids_sha256', '')) != _sha256_ndarray(metas[t].subjects):
                return None

        fp = str(payload.get('protocol_fingerprint', ''))
        if canonical_json_sha256(protocol) != fp:
            return None
        if str(success.get('protocol_fingerprint', '')) != fp:
            return None
        log(f"Reusing certified preflight: fingerprint={fp}")
        return payload
    except Exception as e:
        log(f"Preflight reuse check failed ({e}); running a fresh full preflight.")
        return None


def get_or_run_preflight(args, metas, input_chans, input_source, provisional,
                         modeling_file, checkpoint, device):
    reused = try_reuse_preflight(
        args, metas, input_chans, input_source, provisional, modeling_file, checkpoint
    )
    if reused is not None:
        return reused
    return run_preflight(
        args, metas, input_chans, input_source, provisional,
        modeling_file, checkpoint, device,
    )

def run_pilot(args, metas, input_chans, modeling_file, checkpoint, device,
              preflight_payload):
    pilot_root = ensure_dir(Path(args.outdir) / 'pilot')
    for stale in (
        pilot_root / 'pilot_summary.json',
        pilot_root / 'convergence_summary.csv',
        pilot_root / '_SUCCESS.json',
    ):
        stale.unlink(missing_ok=True)
    fp = str(preflight_payload['protocol_fingerprint'])
    rows = []

    # Stage one task once, reuse it for all replicates, then release it.
    for task in args.tasks:
        task_data = load_task_into_memory(metas[task], reserve_gb=args.ram_reserve_gb)
        try:
            for rep in range(1, args.replicates + 1):
                rows.append(run_task(
                    args, task, rep, metas[task], input_chans,
                    modeling_file, checkpoint, device,
                    epochs=args.pilot_max_epochs, phase='pilot',
                    protocol_fingerprint=fp, task_data=task_data,
                ))
        finally:
            del task_data
            gc.collect()

    digests = sorted(set(r['base_full_backbone_digest'] for r in rows))
    if len(digests) != 1:
        raise RuntimeError(f"Pilot runs do not share one W0 digest: {digests}")
    if digests[0] != preflight_payload['protocol']['checkpoint']['base_full_backbone_digest']:
        raise RuntimeError('Pilot W0 digest disagrees with preflight W0 digest')
    if any(str(r['protocol_fingerprint']) != fp for r in rows):
        raise RuntimeError('Pilot protocol fingerprint mismatch across runs')

    missing = [r for r in rows if r['convergence_epoch'] is None]
    certified = [int(r['convergence_epoch']) for r in rows if r['convergence_epoch'] is not None]
    if missing:
        # Conservative, transparent fallback: if the lightweight plateau test did
        # not trigger for every run, use the full pilot budget rather than aborting
        # days of otherwise healthy training. This is not hidden; it is recorded.
        e_star = int(args.pilot_max_epochs)
        e_star_source = 'pilot_max_epoch_fallback'
        log(
            'WARNING: lightweight convergence not certified for: '
            + ', '.join(f"{r['task']}/rep{r['replicate']}" for r in missing)
            + f". Using conservative common E*={e_star}."
        )
    else:
        e_star = min(max(certified) + 1, int(args.pilot_max_epochs))
        e_star_source = 'rolling_plateau_max_plus_one'

    write_csv(pilot_root / 'convergence_summary.csv', rows)
    summary = {
        'version': VERSION,
        'protocol_fingerprint': fp,
        'protocol': preflight_payload['protocol'],
        'recommended_E_star': int(e_star),
        'E_star_source': e_star_source,
        'pilot_max_epochs': int(args.pilot_max_epochs),
        'all_subjects_pooled': True,
        'convergence_criteria': preflight_payload['protocol']['convergence'],
        'n_uncertified_runs': len(missing),
        'runs': rows,
    }
    dump_json(pilot_root / 'pilot_summary.json', summary)
    plot_epoch_curves(Path(args.outdir), pilot=True)
    dump_json(pilot_root / '_SUCCESS.json', {
        'version': VERSION,
        'protocol_fingerprint': fp,
        'recommended_E_star': int(e_star),
        'E_star_source': e_star_source,
        'n_uncertified_runs': len(missing),
        'pilot_summary_sha256': sha256_file(pilot_root / 'pilot_summary.json'),
    })
    log(f"Pilot complete. Recommended common E* = {e_star} ({e_star_source})")
    return e_star

def resolve_final_epochs(args, protocol_fingerprint: str) -> Tuple[int, str]:
    if args.epochs > 0:
        if not bool(args.unsafe_allow_manual_final_epochs):
            raise RuntimeError(
                "Manual final epochs bypass the common-E* pilot contract. "
                "Use --epochs 0 for scientific runs. If this is a deliberate diagnostic, "
                "add --unsafe-allow-manual-final-epochs explicitly."
            )
        return int(args.epochs), "UNSAFE_manual_cli"
    pilot_root = Path(args.outdir) / "pilot"
    p = pilot_root / "pilot_summary.json"
    success = pilot_root / "_SUCCESS.json"
    if not p.is_file() or not success.is_file():
        raise FileNotFoundError(
            f"--epochs=0 requires a completed matching pilot: {p} and {success}"
        )
    summary = load_json(p)
    ssuccess = load_json(success)
    for src, obj in (("pilot_summary", summary), ("pilot_success", ssuccess)):
        old = str(obj.get("protocol_fingerprint", ""))
        if old != str(protocol_fingerprint):
            raise RuntimeError(
                f"{src} protocol fingerprint mismatch. Old pilot cannot be reused after any "
                "change to data, channel mapping, checkpoint, code, optimizer or convergence contract."
            )
    if ssuccess.get("pilot_summary_sha256") != sha256_file(p):
        raise RuntimeError("Pilot summary hash mismatch; pilot artifacts may be stale/corrupted")
    e = summary.get("recommended_E_star")
    if e is None or int(e) < 1:
        raise RuntimeError(f"Pilot summary has no certified E*: {p}")
    return int(e), "matching_pilot"


def run_final(args, metas, input_chans, modeling_file, checkpoint, device,
              epochs, epoch_source, preflight_payload):
    root = Path(args.outdir)
    for stale in (root / 'run_manifest.json', root / 'run_summary.csv', root / '_FINAL_SUCCESS.json'):
        stale.unlink(missing_ok=True)
    fp = str(preflight_payload['protocol_fingerprint'])
    rows = []
    for task in args.tasks:
        task_data = load_task_into_memory(metas[task], reserve_gb=args.ram_reserve_gb)
        try:
            for rep in range(1, args.replicates + 1):
                rows.append(run_task(
                    args, task, rep, metas[task], input_chans,
                    modeling_file, checkpoint, device, epochs=epochs, phase='final',
                    protocol_fingerprint=fp, task_data=task_data,
                ))
        finally:
            del task_data
            gc.collect()

    expected = {(t, r) for t in args.tasks for r in range(1, args.replicates + 1)}
    observed = {(str(r['task']), int(r['replicate'])) for r in rows}
    if observed != expected:
        raise RuntimeError(f"Final run grid incomplete: expected={expected}, observed={observed}")
    digests = sorted(set(r['base_full_backbone_digest'] for r in rows))
    if len(digests) != 1:
        raise RuntimeError(f"Final runs do not share one W0 digest: {digests}")
    if any(str(r['protocol_fingerprint']) != fp for r in rows):
        raise RuntimeError('Final protocol fingerprint mismatch across runs')

    write_csv(root / 'run_summary.csv', rows)
    manifest = {
        'version': VERSION,
        'protocol_fingerprint': fp,
        'protocol': preflight_payload['protocol'],
        'tasks': args.tasks,
        'replicates': args.replicates,
        'expected_run_grid': [
            {'task': t, 'replicate': r}
            for t in args.tasks for r in range(1, args.replicates + 1)
        ],
        'all_subjects_pooled': True,
        'epochs': int(epochs),
        'epoch_source': str(epoch_source),
        'base_full_backbone_digest': digests[0],
        'runs': rows,
    }
    dump_json(root / 'run_manifest.json', manifest)
    dump_json(root / '_FINAL_SUCCESS.json', {
        'version': VERSION,
        'protocol_fingerprint': fp,
        'run_manifest_sha256': sha256_file(root / 'run_manifest.json'),
        'n_runs': len(rows),
    })
    plot_epoch_curves(root, pilot=False)
    log('Final full fine-tuning complete and artifact-certified.')

def build_argparser():
    p = argparse.ArgumentParser(
        description="LaBraM x M3CV Task Adaptation Geometry v0.3.4 fail-closed trainer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", choices=["preflight", "pilot", "final", "all"], default="preflight")
    p.add_argument(
        "--data-root", default="/omni-eeg-01/task calibration/dataset/m3cv"
    )
    p.add_argument(
        "--checkpoint",
        default=os.environ.get(
            "LABRAM_CKPT",
            "/omni-eeg-01/task calibration/LaBraM/checkpoints/labram-base.pth",
        ),
    )
    p.add_argument("--modeling-file", default="/omni-eeg-01/task calibration/LaBraM/modeling_finetune.py")
    p.add_argument(
        "--outdir",
        default=str(Path.home() / "tsv_labram_m3cv_v0_3_4"),
    )
    p.add_argument("--tasks", default="Motor,P300,SSS,TS")
    p.add_argument("--replicates", type=int, default=2)
    p.add_argument("--seed-base", type=int, default=20260819)
    p.add_argument("--model-init-seed", type=int, default=314159)

    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=0)

    p.add_argument("--backbone-lr", type=float, default=1e-4)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--adam-eps", type=float, default=1e-8)
    p.add_argument("--amp", action="store_true")

    p.add_argument("--pilot-max-epochs", type=int, default=20)
    p.add_argument(
        "--epochs", type=int, default=0,
        help="Final epochs. 0 means read recommended_E_star from pilot summary.",
    )
    p.add_argument(
        "--unsafe-allow-manual-final-epochs", action="store_true",
        help="UNSAFE diagnostic escape hatch: permit --epochs > 0 without certified pilot E*.",
    )
    p.add_argument("--conv-window", type=int, default=3)
    p.add_argument("--conv-patience", type=int, default=2)
    p.add_argument("--conv-loss-rel", type=float, default=0.03)
    p.add_argument("--conv-bacc-abs", type=float, default=0.01)
    p.add_argument(
        "--conv-min-bacc-margin", type=float, default=0.05,
        help="Convergence is not certified if training bACC merely plateaus at chance.",
    )

    p.add_argument("--energy-probes", default="1,2,5,10,20,40,80,120,160")

    p.add_argument("--preflight-chunk-rows", type=int, default=512)
    p.add_argument("--preflight-probe-samples", type=int, default=32)
    p.add_argument("--preflight-mean-abs-max", type=float, default=0.25)
    p.add_argument("--preflight-std-min", type=float, default=0.50)
    p.add_argument("--preflight-std-max", type=float, default=2.00)
    p.add_argument("--ram-reserve-gb", type=float, default=4.0,
                   help="Host RAM kept free while staging one task in memory.")
    p.add_argument("--force-preflight", action="store_true",
                   help="Ignore a matching certified preflight and rerun the full H5 audit.")
    p.add_argument(
        "--unsafe-allow-missing-label-mapping", action="store_true",
        help="UNSAFE: allow H5 files without parseable label_mapping metadata after manual verification.",
    )
    p.add_argument(
        "--unsafe-allow-subject-set-mismatch", action="store_true",
        help="UNSAFE: allow selected tasks to contain different subject-ID sets.",
    )
    p.add_argument(
        "--unsafe-allow-subject-class-gaps", action="store_true",
        help="UNSAFE: allow a subject to have zero samples for one or more task classes.",
    )

    p.add_argument("--input-chans", default="")
    p.add_argument("--input-chans-file", default="")
    p.add_argument("--allow-provisional-channels", action="store_true")
    p.add_argument("--overwrite-run", action="store_true")
    p.add_argument("--device", default="cuda")
    return p


def validate_args(args):
    if args.replicates < 1:
        raise ValueError("--replicates must be >=1")
    if args.batch_size < 1 or args.eval_batch_size < 1:
        raise ValueError("batch sizes must be >=1")
    if args.weight_decay != 0.0:
        raise RuntimeError(
            "v0.3.4 scientific protocol fixes weight_decay=0. "
            "Do not silently introduce task-independent drift."
        )
    if args.conv_window < 2:
        raise ValueError("--conv-window must be >=2")
    if args.conv_patience < 1:
        raise ValueError("--conv-patience must be >=1")
    min_pilot = args.conv_window + args.conv_patience
    if args.pilot_max_epochs < min_pilot:
        raise ValueError(
            f"--pilot-max-epochs must be >= {min_pilot} for the rolling convergence audit"
        )
    if args.backbone_lr <= 0 or args.head_lr <= 0:
        raise ValueError("learning rates must be positive")
    if args.preflight_chunk_rows < 1 or args.preflight_probe_samples < 1:
        raise ValueError("preflight chunk/probe sizes must be positive")
    if not (0 < args.preflight_std_min < args.preflight_std_max):
        raise ValueError("invalid preflight std bounds")
    if args.preflight_mean_abs_max <= 0:
        raise ValueError("--preflight-mean-abs-max must be positive")
    if args.ram_reserve_gb < 0:
        raise ValueError("--ram-reserve-gb must be >=0")


def main():
    args = build_argparser().parse_args()
    args.tasks = parse_csv_strs(args.tasks)
    args.energy_probes = parse_csv_ints(args.energy_probes)
    validate_args(args)

    outdir = ensure_dir(args.outdir)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    checkpoint = Path(args.checkpoint).expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    modeling_file = find_modeling_file(args.modeling_file)
    input_chans, source, provisional = resolve_input_chans(args)
    if provisional and args.mode in ("final", "all"):
        raise RuntimeError(
            "Final scientific runs are forbidden with the provisional 1..64 channel mapping. "
            "Provide --input-chans-file or --input-chans with the confirmed mapping."
        )
    metas = load_all_metas(args)

    dump_json(outdir / "subject_metadata.json", subject_metadata(metas))
    dump_json(outdir / "final_config.json", {
        **vars(args),
        "tasks": args.tasks,
        "energy_probes": args.energy_probes,
        "input_chans": input_chans,
        "input_chans_source": source,
        "provisional_channels": provisional,
        "checkpoint": str(checkpoint),
        "modeling_file": str(modeling_file),
        "all_subjects_pooled": True,
        "io_mode": "host_ram_preload_one_task_at_a_time",
    })

    log(f"Starting {VERSION}, mode={args.mode}, device={device}")
    if args.mode == "preflight":
        run_preflight(
            args, metas, input_chans, source, provisional,
            modeling_file, checkpoint, device,
        )
    elif args.mode == "pilot":
        pf = get_or_run_preflight(
            args, metas, input_chans, source, provisional,
            modeling_file, checkpoint, device,
        )
        run_pilot(args, metas, input_chans, modeling_file, checkpoint, device, pf)
    elif args.mode == "final":
        pf = get_or_run_preflight(
            args, metas, input_chans, source, provisional,
            modeling_file, checkpoint, device,
        )
        epochs, epoch_source = resolve_final_epochs(args, pf["protocol_fingerprint"])
        run_final(
            args, metas, input_chans, modeling_file, checkpoint, device,
            epochs, epoch_source, pf,
        )
    elif args.mode == "all":
        pf = get_or_run_preflight(
            args, metas, input_chans, source, provisional,
            modeling_file, checkpoint, device,
        )
        e_star = run_pilot(
            args, metas, input_chans, modeling_file, checkpoint, device, pf
        )
        run_final(
            args, metas, input_chans, modeling_file, checkpoint, device,
            e_star, "pilot_same_invocation", pf,
        )


if __name__ == "__main__":
    main()
