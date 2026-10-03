"""M3CV data contract and RAM-resident minibatch loading.

The Session-1 4 s files are treated as already globally z-scored.  No division
by 100 and no second normalization are applied.  Each sample is reshaped from
``[64, 800]`` to ``[64, 4, 200]`` for LaBraM.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .config import TASKS, TASK_ORDER, ProjectConfig
from .io import log, stable_seed


def _h5py():
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required for M3CV data access") from exc
    return h5py


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for training or functional evaluation") from exc
    return torch


@dataclass(frozen=True)
class TaskMeta:
    task: str
    path: Path
    n: int
    shape: Tuple[int, int, int]
    labels: np.ndarray
    subjects: np.ndarray
    classes: np.ndarray

    @property
    def n_classes(self) -> int:
        return int(len(self.classes))


def load_task_meta(task: str, path: Path) -> TaskMeta:
    h5py = _h5py()
    path = Path(path)
    if task not in TASKS:
        raise KeyError(f"Unknown task: {task}")
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        for key in ("data", "labels", "subject_ids"):
            if key not in handle:
                raise KeyError(f"{path}: missing {key}; found {list(handle.keys())}")
        data = handle["data"]
        labels_ds = handle["labels"]
        subjects_ds = handle["subject_ids"]
        shape = tuple(int(x) for x in data.shape)
        if len(shape) != 3 or shape[1:] != (64, 800):
            raise ValueError(f"{task}: expected data [N,64,800], found {shape}")
        if np.dtype(data.dtype) != np.dtype(np.float32):
            raise TypeError(f"{task}: data must be float32, found {data.dtype}")
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
    expected = np.asarray(TASKS[task].raw_labels, dtype=np.int64)
    if not np.array_equal(classes, expected):
        raise ValueError(
            f"{task}: expected raw labels {expected.tolist()}, found {classes.tolist()}"
        )
    if np.any(subjects < 0):
        raise ValueError(f"{task}: negative subject id")
    return TaskMeta(task, path, int(shape[0]), shape, labels, subjects, classes)


def load_all_task_meta(project: ProjectConfig) -> Dict[str, TaskMeta]:
    return {
        task: load_task_meta(task, project.task_path(task))
        for task in TASK_ORDER
    }


def sha256_ndarray(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(array.dtype).encode("ascii"))
    h.update(str(tuple(array.shape)).encode("ascii"))
    h.update(array.tobytes(order="C"))
    return h.hexdigest()


def _normalise_label_name(value: object) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _label_mapping_value(handle):
    for name in ("label_mapping", "label_map", "class_mapping", "class_names"):
        if name in handle.attrs:
            return handle.attrs[name], f"attr:{name}"
        if name in handle:
            return handle[name][()], f"dataset:{name}"
    return None, None


def _parse_label_mapping(value, raw_labels: Sequence[int]) -> Dict[int, str]:
    if value is None:
        return {}
    if isinstance(value, np.ndarray):
        value = value.item() if value.ndim == 0 else value.tolist()
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, Mapping):
        parsed = {}
        for key, item in value.items():
            try:
                parsed[int(key)] = _normalise_label_name(item)
            except (TypeError, ValueError):
                try:
                    parsed[int(item)] = _normalise_label_name(key)
                except (TypeError, ValueError):
                    continue
        return parsed
    if isinstance(value, (list, tuple)):
        if len(value) != len(raw_labels):
            return {}
        structured: Dict[int, str] = {}
        for item in value:
            text = item.decode("utf-8", errors="replace") if isinstance(item, bytes) else str(item)
            pair = re.split(r"\s*(?:=|:|->)\s*", text.strip(), maxsplit=1)
            if len(pair) != 2:
                structured = {}
                break
            left, right = (x.strip(" '\"") for x in pair)
            try:
                structured[int(left)] = _normalise_label_name(right)
            except ValueError:
                try:
                    structured[int(right)] = _normalise_label_name(left)
                except ValueError:
                    structured = {}
                    break
        if structured:
            return structured
        return {
            int(code): _normalise_label_name(name)
            for code, name in zip(raw_labels, value)
        }
    text = str(value).strip()
    try:
        decoded = json.loads(text)
    except Exception:
        decoded = None
    if isinstance(decoded, (dict, list, tuple)):
        parsed = _parse_label_mapping(decoded, raw_labels)
        if parsed:
            return parsed
    parsed = {}
    for token in re.split(r"[,;|]", text.strip("{}[]()")):
        pair = re.split(r"\s*(?:=|:|->)\s*", token.strip(), maxsplit=1)
        if len(pair) != 2:
            continue
        left, right = (x.strip(" '\"") for x in pair)
        try:
            parsed[int(left)] = _normalise_label_name(right)
        except ValueError:
            try:
                parsed[int(right)] = _normalise_label_name(left)
            except ValueError:
                pass
    return parsed


def audit_h5_data(
    meta: TaskMeta,
    chunk_rows: int = 512,
    mean_abs_max: float = 0.25,
    std_min: float = 0.50,
    std_max: float = 2.00,
) -> Dict[str, object]:
    """Sequential full-file audit of shape, labels, finiteness, and scale."""

    h5py = _h5py()
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    count = 0
    total = 0.0
    total2 = 0.0
    minimum = float("inf")
    maximum = float("-inf")
    channel_sum = np.zeros(64, dtype=np.float64)
    channel_sum2 = np.zeros(64, dtype=np.float64)
    content_hash = hashlib.sha256()
    with h5py.File(meta.path, "r") as handle:
        mapping_value, mapping_source = _label_mapping_value(handle)
        parsed = _parse_label_mapping(mapping_value, meta.classes)
        expected = {
            int(code): _normalise_label_name(name)
            for code, name in TASKS[meta.task].class_names.items()
        }
        if parsed != expected:
            raise RuntimeError(
                f"{meta.task}: label mapping {parsed} does not match expected {expected}"
            )
        dataset = handle["data"]
        for start in range(0, meta.n, int(chunk_rows)):
            stop = min(meta.n, start + int(chunk_rows))
            x = np.asarray(dataset[start:stop], dtype=np.float32)
            if not np.isfinite(x).all():
                raise FloatingPointError(f"{meta.task}: non-finite EEG values")
            content_hash.update(memoryview(np.ascontiguousarray(x)).cast("B"))
            count += int(x.size)
            total += float(np.sum(x, dtype=np.float64))
            total2 += float(np.sum(x * x, dtype=np.float64))
            minimum = min(minimum, float(np.min(x)))
            maximum = max(maximum, float(np.max(x)))
            channel_sum += np.sum(x, axis=(0, 2), dtype=np.float64)
            channel_sum2 += np.sum(x * x, axis=(0, 2), dtype=np.float64)
    expected_count = int(np.prod(meta.shape))
    if count != expected_count:
        raise RuntimeError(f"{meta.task}: scanned {count}, expected {expected_count}")
    mean = total / count
    std = math.sqrt(max(total2 / count - mean * mean, 0.0))
    if abs(mean) > mean_abs_max or not std_min <= std <= std_max:
        raise RuntimeError(
            f"{meta.task}: expected global z-score scale, observed mean={mean:.6g}, std={std:.6g}"
        )
    per_channel_count = meta.n * meta.shape[2]
    channel_mean = channel_sum / per_channel_count
    channel_std = np.sqrt(
        np.maximum(channel_sum2 / per_channel_count - channel_mean**2, 0.0)
    )
    if np.any(channel_std <= 1e-6):
        dead = np.flatnonzero(channel_std <= 1e-6).tolist()
        raise RuntimeError(f"{meta.task}: dead channels at zero-based indices {dead}")
    stat = meta.path.stat()
    return {
        "task": meta.task,
        "path": str(meta.path.resolve()),
        "file_size_bytes": int(stat.st_size),
        "file_mtime_ns": int(stat.st_mtime_ns),
        "shape": list(meta.shape),
        "dtype": "float32",
        "n_samples": meta.n,
        "n_subjects": int(len(np.unique(meta.subjects))),
        "raw_classes": [int(x) for x in meta.classes],
        "class_counts": {
            str(int(code)): int(np.sum(meta.labels == code)) for code in meta.classes
        },
        "label_mapping_source": mapping_source,
        "label_mapping": {str(k): v for k, v in parsed.items()},
        "labels_sha256": sha256_ndarray(meta.labels),
        "subject_ids_sha256": sha256_ndarray(meta.subjects),
        "data_content_sha256": content_hash.hexdigest(),
        "global_mean": float(mean),
        "global_std": float(std),
        "global_min": minimum,
        "global_max": maximum,
        "channel_mean": channel_mean.tolist(),
        "channel_std": channel_std.tolist(),
        "full_scan": True,
        "normalisation": "already globally z-scored; no further scaling",
    }


def audit_subject_composition(metas: Mapping[str, TaskMeta]) -> Dict[str, object]:
    subject_sets = {task: set(int(x) for x in meta.subjects) for task, meta in metas.items()}
    reference = subject_sets[TASK_ORDER[0]]
    mismatch = {
        task: sorted(reference.symmetric_difference(subjects))
        for task, subjects in subject_sets.items()
        if subjects != reference
    }
    if mismatch:
        raise RuntimeError(f"Task subject sets differ: {mismatch}")
    class_gaps = []
    for task, meta in metas.items():
        for subject in sorted(subject_sets[task]):
            observed = set(int(x) for x in meta.labels[meta.subjects == subject])
            missing = sorted(set(int(x) for x in meta.classes) - observed)
            if missing:
                class_gaps.append({"task": task, "subject": subject, "missing": missing})
    if class_gaps:
        raise RuntimeError(f"Subject-by-class support gaps: {class_gaps[:20]}")
    return {
        "status": "passed",
        "same_subject_set": True,
        "n_subjects": len(reference),
        "subject_ids": sorted(reference),
        "complete_subject_by_class_support": True,
    }


@dataclass
class InMemoryTaskData:
    x: np.ndarray
    y: np.ndarray
    load_seconds: float
    nbytes: int


def _mem_available_bytes() -> Optional[int]:
    try:
        with Path("/proc/meminfo").open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        return None
    return None


def load_task_into_memory(meta: TaskMeta, reserve_gb: float = 4.0) -> InMemoryTaskData:
    h5py = _h5py()
    required = int(np.prod(meta.shape)) * np.dtype(np.float32).itemsize
    available = _mem_available_bytes()
    reserve = int(float(reserve_gb) * 1024**3)
    if available is not None and available < required + reserve:
        raise MemoryError(
            f"{meta.task}: needs {required / 1024**3:.2f} GiB plus "
            f"{reserve_gb:.1f} GiB reserve; MemAvailable={available / 1024**3:.2f} GiB"
        )
    before = meta.path.stat()
    start = time.time()
    with h5py.File(meta.path, "r") as handle:
        x = np.asarray(handle["data"][:], dtype=np.float32)
    after = meta.path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"{meta.task}: H5 changed during RAM staging")
    if tuple(x.shape) != meta.shape or not np.isfinite(x).all():
        raise RuntimeError(f"{meta.task}: invalid data encountered during RAM staging")
    x = np.ascontiguousarray(x.reshape(meta.n, 64, 4, 200))
    y = np.searchsorted(meta.classes, meta.labels).astype(np.int64, copy=False)
    if not np.array_equal(meta.classes[y], meta.labels):
        raise RuntimeError(f"{meta.task}: label remapping failed")
    seconds = time.time() - start
    log(f"RAM staged {meta.task}: {x.nbytes / 1024**3:.2f} GiB in {seconds:.1f}s")
    return InMemoryTaskData(x=x, y=y, load_seconds=float(seconds), nbytes=int(x.nbytes))


class InMemoryBatchLoader:
    def __init__(
        self,
        data: InMemoryTaskData,
        batch_size: int,
        shuffle: bool,
        seed: int = 0,
        indices: Optional[np.ndarray] = None,
    ):
        self.data = data
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.indices = (
            np.arange(len(data.y), dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.indices.ndim != 1 or np.any(self.indices < 0) or np.any(self.indices >= len(data.y)):
            raise ValueError("invalid loader indices")
        self.rng = np.random.default_rng(int(seed))

    def __len__(self) -> int:
        return int(math.ceil(len(self.indices) / self.batch_size))

    def __iter__(self):
        torch = _torch()
        order = self.indices.copy()
        if self.shuffle:
            order = order[self.rng.permutation(len(order))]
        for start in range(0, len(order), self.batch_size):
            indices = order[start : start + self.batch_size]
            x = torch.from_numpy(np.ascontiguousarray(self.data.x[indices]))
            y = torch.from_numpy(np.ascontiguousarray(self.data.y[indices]))
            yield x, y


def make_loaders(
    data: InMemoryTaskData,
    batch_size: int,
    eval_batch_size: int,
    loader_seed: int,
):
    return (
        InMemoryBatchLoader(data, batch_size, True, loader_seed),
        InMemoryBatchLoader(data, eval_batch_size, False, 0),
    )


def make_eval_loader(
    data: InMemoryTaskData,
    batch_size: int,
    indices: Optional[np.ndarray] = None,
):
    return InMemoryBatchLoader(data, batch_size, False, 0, indices=indices)


def class_weights(meta: TaskMeta, device):
    torch = _torch()
    counts = np.asarray([np.sum(meta.labels == code) for code in meta.classes], dtype=np.float64)
    if np.any(counts <= 0):
        raise ValueError(f"{meta.task}: empty class")
    weights = float(meta.n) / (len(meta.classes) * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def preflight_probe_indices(meta: TaskMeta, n: int = 32) -> np.ndarray:
    n = max(int(n), meta.n_classes)
    selected: List[int] = []
    for code in meta.classes:
        selected.append(int(np.flatnonzero(meta.labels == code)[0]))
    remaining = np.setdiff1d(np.arange(meta.n), np.asarray(selected), assume_unique=False)
    rng = np.random.default_rng(stable_seed(20260820, meta.task, "preflight"))
    if len(selected) < n:
        selected.extend(
            int(x)
            for x in rng.choice(remaining, size=min(n - len(selected), len(remaining)), replace=False)
        )
    return np.asarray(sorted(set(selected)), dtype=np.int64)


def load_probe_batch(meta: TaskMeta, indices: Sequence[int]):
    """Load a small deterministic H5 batch for the model preflight only."""
    h5py = _h5py()
    torch = _torch()
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or len(indices) == 0:
        raise ValueError("probe indices must be a non-empty vector")
    order = np.argsort(indices)
    sorted_indices = indices[order]
    with h5py.File(meta.path, "r") as handle:
        x_sorted = np.asarray(handle["data"][sorted_indices], dtype=np.float32)
        y_sorted = np.asarray(handle["labels"][sorted_indices], dtype=np.int64)
    inverse = np.argsort(order)
    x = np.ascontiguousarray(x_sorted[inverse].reshape(len(indices), 64, 4, 200))
    raw_y = y_sorted[inverse]
    y = np.searchsorted(meta.classes, raw_y).astype(np.int64)
    if not np.array_equal(meta.classes[y], raw_y):
        raise RuntimeError(f"{meta.task}: probe label remapping failed")
    return torch.from_numpy(x), torch.from_numpy(y)
