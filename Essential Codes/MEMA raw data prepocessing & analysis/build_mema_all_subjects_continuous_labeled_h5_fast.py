#!/usr/bin/env python3
"""
Fast MEMA continuous HDF5 builder for multiple subjects.

Output format matches the previous Subject 1 HDF5 structure:

data/
└── subject_XX/
    ├── preprocessed_eeg      float32 [32, T]
    ├── event                 int8    [1, T]
    ├── electrodes            string  [32]
    ├── label                 int8    [4, T]
    ├── task_id               int16   [1, T]
    ├── task_labels           int8    [12, 4]
    ├── task_bounds           int64   [12, 2]
    └── sample_index          int64   [T]

Label order: [attention, valence, arousal, dominance]
During event=0, label is filled with -1 and task_id with -1.

Speed note
----------
Default backend is scipy-iir, which is much faster than the earlier MNE FIR
pipeline. It keeps the same data layout and label logic, but the numerical
filtered waveform will not be bit-identical to the earlier MNE FIR output.
Use --filter-backend mne-fir if you need the conservative old-style filtering.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import h5py
import numpy as np
from scipy import signal
from scipy.io import loadmat


CHANNEL_NAMES = (
    "FP1", "FP2", "Fz", "F3", "F4", "F7", "F8",
    "FCz", "FC3", "FC4", "FT7", "FT8",
    "Cz", "C3", "C4", "T3", "T4",
    "CPz", "CP3", "CP4", "TP7", "TP8",
    "Pz", "P3", "P4", "T5", "T6",
    "Oz", "O1", "O2", "HEOL", "HEOR",
)
CHANNEL_TYPES = ("eeg",) * 30 + ("eog",) * 2
EEG_INDICES = np.arange(30, dtype=np.int64)
EOG_INDICES = np.arange(30, 32, dtype=np.int64)

LABEL_NAMES = ("attention", "valence", "arousal", "dominance")
LABEL_FILL_VALUE = -1
TASK_ID_FILL_VALUE = -1

NormalizationMode = Literal["none", "all_global", "eeg_global", "channelwise"]
FilterBackend = Literal["scipy-iir", "mne-fir"]


@dataclass(frozen=True)
class PreprocessingConfig:
    raw_sfreq: float = 500.0
    notch_freq: float | None = 50.0
    notch_width: float = 4.0
    highpass_freq: float | None = 0.1
    lowpass_freq: float | None = 40.0
    target_sfreq: float | None = 200.0
    normalization: NormalizationMode = "eeg_global"
    normalize_eog_separately: bool = True
    filter_backend: FilterBackend = "scipy-iir"
    mne_n_jobs: int = 16
    eps: float = 1e-8

    def validate(self) -> None:
        if self.raw_sfreq <= 0:
            raise ValueError("raw_sfreq must be positive")
        if self.target_sfreq is not None and self.target_sfreq <= 0:
            raise ValueError("target_sfreq must be positive or None")
        nyquist = self.raw_sfreq / 2.0
        if self.notch_freq is not None and not (0 < self.notch_freq < nyquist):
            raise ValueError("notch_freq must be between 0 and Nyquist")
        if self.highpass_freq is not None and self.highpass_freq < 0:
            raise ValueError("highpass_freq must be non-negative or None")
        if self.lowpass_freq is not None and not (0 < self.lowpass_freq < nyquist):
            raise ValueError("lowpass_freq must be between 0 and Nyquist")
        if (
            self.highpass_freq is not None
            and self.lowpass_freq is not None
            and self.highpass_freq >= self.lowpass_freq
        ):
            raise ValueError("highpass_freq must be lower than lowpass_freq")
        if self.notch_width <= 0:
            raise ValueError("notch_width must be positive")
        if self.filter_backend not in ("scipy-iir", "mne-fir"):
            raise ValueError("filter_backend must be scipy-iir or mne-fir")


def parse_subjects(spec: str) -> list[int]:
    subjects: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start, stop = int(left), int(right)
            if start > stop:
                raise ValueError(f"Bad subject range: {part}")
            subjects.update(range(start, stop + 1))
        else:
            subjects.add(int(part))
    result = sorted(subjects)
    bad = [s for s in result if s < 1 or s > 20]
    if bad:
        raise ValueError(f"Subject IDs must be in 1..20, got {bad}")
    if not result:
        raise ValueError("No subjects selected")
    return result


def infer_delimiter_and_skiprows(path: Path) -> tuple[str | None, int]:
    candidate_delimiters: tuple[str | None, ...] = (None, ",", "\t", ";")
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_number, line in enumerate(handle):
            stripped = line.strip()
            if not stripped:
                continue
            for delimiter in candidate_delimiters:
                tokens = stripped.split() if delimiter is None else stripped.split(delimiter)
                if len(tokens) < 34:
                    continue
                try:
                    [float(token.strip()) for token in tokens[:34]]
                except ValueError:
                    continue
                return delimiter, line_number
    raise ValueError(f"Could not find a numeric row with 34 columns in {path}")


def load_data2(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return signals [32,T] float32, sample_index [T] int64, event [T] int8."""
    if not path.is_file():
        raise FileNotFoundError(path)
    delimiter, skiprows = infer_delimiter_and_skiprows(path)
    matrix = np.loadtxt(
        path,
        delimiter=delimiter,
        skiprows=skiprows,
        dtype=np.float32,
        ndmin=2,
    )
    if matrix.shape[1] != 34:
        raise ValueError(f"Expected exactly 34 columns, got {matrix.shape} in {path}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"NaN/Inf detected in {path}")

    signals = np.asarray(matrix[:, :32].T, dtype=np.float32)
    sample_index_float = matrix[:, 32]
    event_float = matrix[:, 33]

    if not np.allclose(sample_index_float, np.rint(sample_index_float), atol=1e-4):
        raise ValueError(f"Non-integer sample index in {path}")
    sample_index = np.rint(sample_index_float).astype(np.int64)
    if np.any(np.diff(sample_index) < 0):
        # Some MEMA files restart their sample counter inside one Data2 file.
        # For continuous HDF5 construction, row order is the reliable time axis.
        # Keep the HDF5 field name unchanged, but store a monotonic row index.
        sample_index = np.arange(sample_index.shape[0], dtype=np.int64)

    if not np.allclose(event_float, np.rint(event_float), atol=1e-4):
        raise ValueError(f"Non-integer event marker in {path}")
    event = np.rint(event_float).astype(np.int8)
    if not set(np.unique(event).tolist()).issubset({0, 1}):
        raise ValueError(f"Event marker contains values outside 0/1 in {path}")

    return signals, sample_index, event


def nearest_resample_indices(
    n_source: int,
    source_sfreq: float,
    n_target: int,
    target_sfreq: float,
) -> np.ndarray:
    target_times = np.arange(n_target, dtype=np.float64) / float(target_sfreq)
    source_indices = np.rint(target_times * float(source_sfreq)).astype(np.int64)
    return np.clip(source_indices, 0, n_source - 1)


def preprocess_scipy_iir(
    signals: np.ndarray,
    sample_index: np.ndarray,
    event: np.ndarray,
    config: PreprocessingConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Fast zero-phase IIR filtering + polyphase resampling + normalization."""
    x = np.asarray(signals, dtype=np.float64)

    if config.notch_freq is not None:
        q = float(config.notch_freq) / float(config.notch_width)
        b, a = signal.iirnotch(
            w0=float(config.notch_freq),
            Q=q,
            fs=float(config.raw_sfreq),
        )
        x = signal.filtfilt(b, a, x, axis=1)

    if config.highpass_freq is not None or config.lowpass_freq is not None:
        if config.highpass_freq is not None and config.lowpass_freq is not None:
            wp: float | list[float] = [
                float(config.highpass_freq),
                float(config.lowpass_freq),
            ]
            btype = "bandpass"
        elif config.highpass_freq is not None:
            wp = float(config.highpass_freq)
            btype = "highpass"
        else:
            wp = float(config.lowpass_freq)  # type: ignore[arg-type]
            btype = "lowpass"
        sos = signal.butter(
            N=4,
            Wn=wp,
            btype=btype,
            fs=float(config.raw_sfreq),
            output="sos",
        )
        x = signal.sosfiltfilt(sos, x, axis=1)

    sfreq_out = float(config.raw_sfreq)
    if (
        config.target_sfreq is not None
        and not math.isclose(config.target_sfreq, config.raw_sfreq, rel_tol=0, abs_tol=1e-9)
    ):
        # For MEMA default 500 Hz -> 200 Hz, this is up=2, down=5.
        ratio = float(config.target_sfreq) / float(config.raw_sfreq)
        if math.isclose(ratio, 0.4, rel_tol=0, abs_tol=1e-12):
            up, down = 2, 5
        else:
            from fractions import Fraction
            frac = Fraction(ratio).limit_denominator(1000)
            up, down = frac.numerator, frac.denominator
        x = signal.resample_poly(x, up=up, down=down, axis=1)
        sfreq_out = float(config.target_sfreq)

    x = x.astype(np.float32, copy=False)
    x, norm_stats = normalize_signals(
        x,
        mode=config.normalization,
        normalize_eog_separately=config.normalize_eog_separately,
        eps=config.eps,
    )

    idx = nearest_resample_indices(
        n_source=signals.shape[1],
        source_sfreq=float(config.raw_sfreq),
        n_target=x.shape[1],
        target_sfreq=sfreq_out,
    )
    return (
        x,
        sample_index[idx].astype(np.int64, copy=False),
        event[idx].astype(np.int8, copy=False),
        norm_stats,
    )


def preprocess_mne_fir(
    signals: np.ndarray,
    sample_index: np.ndarray,
    event: np.ndarray,
    config: PreprocessingConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Conservative MNE FIR pipeline matching the earlier approach, with n_jobs."""
    import mne

    info = mne.create_info(
        ch_names=list(CHANNEL_NAMES),
        sfreq=float(config.raw_sfreq),
        ch_types=list(CHANNEL_TYPES),
        verbose=False,
    )
    raw = mne.io.RawArray(np.asarray(signals, dtype=np.float64), info, verbose=False)

    if config.notch_freq is not None:
        raw.notch_filter(
            freqs=[float(config.notch_freq)],
            notch_widths=float(config.notch_width),
            method="fir",
            n_jobs=int(config.mne_n_jobs),
            verbose=False,
        )
    if config.highpass_freq is not None or config.lowpass_freq is not None:
        raw.filter(
            l_freq=config.highpass_freq,
            h_freq=config.lowpass_freq,
            method="fir",
            n_jobs=int(config.mne_n_jobs),
            verbose=False,
        )

    sfreq_out = float(config.raw_sfreq)
    if (
        config.target_sfreq is not None
        and not math.isclose(config.target_sfreq, config.raw_sfreq, rel_tol=0, abs_tol=1e-9)
    ):
        raw.resample(float(config.target_sfreq), npad="auto", verbose=False)
        sfreq_out = float(config.target_sfreq)

    x = raw.get_data().astype(np.float32, copy=False)
    x, norm_stats = normalize_signals(
        x,
        mode=config.normalization,
        normalize_eog_separately=config.normalize_eog_separately,
        eps=config.eps,
    )
    idx = nearest_resample_indices(
        n_source=signals.shape[1],
        source_sfreq=float(config.raw_sfreq),
        n_target=x.shape[1],
        target_sfreq=sfreq_out,
    )
    return (
        x,
        sample_index[idx].astype(np.int64, copy=False),
        event[idx].astype(np.int8, copy=False),
        norm_stats,
    )


def safe_zscore(values: np.ndarray, *, eps: float) -> tuple[np.ndarray, float, float]:
    mean = float(np.mean(values, dtype=np.float64))
    std = float(np.std(values, dtype=np.float64))
    denominator = std if std >= eps else 1.0
    return ((values - mean) / denominator).astype(np.float32, copy=False), mean, std


def normalize_signals(
    signals: np.ndarray,
    *,
    mode: NormalizationMode,
    normalize_eog_separately: bool,
    eps: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    x = np.asarray(signals, dtype=np.float32)
    out = x.copy()
    stats: dict[str, Any] = {"mode": mode}

    if mode == "none":
        return out, stats

    if mode == "all_global":
        out, mean, std = safe_zscore(out, eps=eps)
        stats.update({"mean": mean, "std": std})
        return out, stats

    if mode == "eeg_global":
        eeg_values, eeg_mean, eeg_std = safe_zscore(out[EEG_INDICES], eps=eps)
        out[EEG_INDICES] = eeg_values
        stats["eeg"] = {"mean": eeg_mean, "std": eeg_std}

        if normalize_eog_separately:
            eog_values, eog_mean, eog_std = safe_zscore(out[EOG_INDICES], eps=eps)
            out[EOG_INDICES] = eog_values
            stats["eog"] = {"mean": eog_mean, "std": eog_std}
        else:
            stats["eog"] = {"normalized": False}
        return out.astype(np.float32, copy=False), stats

    if mode == "channelwise":
        means = np.mean(out, axis=1, dtype=np.float64)
        stds = np.std(out, axis=1, dtype=np.float64)
        denominators = np.where(stds >= eps, stds, 1.0)
        out = (out - means[:, None]) / denominators[:, None]
        stats.update({"means": means.tolist(), "stds": stds.tolist()})
        return out.astype(np.float32, copy=False), stats

    raise ValueError(f"Unsupported normalization mode: {mode}")


def load_subject_task_labels(labels_dir: Path, subject_id: int) -> np.ndarray:
    paths = (
        labels_dir / "label_attention.mat",
        labels_dir / "label_valence.mat",
        labels_dir / "label_arousal.mat",
        labels_dir / "label_dominance.mat",
    )
    row = subject_id - 1
    columns = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        content = loadmat(path)
        if "label" not in content:
            raise KeyError(f"{path} does not contain variable 'label'")
        matrix = np.asarray(content["label"])
        if matrix.shape != (20, 12):
            raise ValueError(f"Expected label shape (20,12), got {matrix.shape} in {path}")
        columns.append(matrix[row].astype(np.int8, copy=False))
    labels = np.stack(columns, axis=1).astype(np.int8, copy=False)  # [12,4]
    if not np.isin(labels, [0, 1, 2]).all():
        raise ValueError(f"Subject {subject_id}: labels contain values outside 0/1/2")
    return labels


def find_task_runs(event: np.ndarray) -> np.ndarray:
    marker = np.asarray(event).reshape(-1)
    if marker.size == 0:
        raise ValueError("event is empty")
    if not np.isin(marker, [0, 1]).all():
        raise ValueError("event must contain only 0 and 1")
    padded = np.pad((marker == 1).astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    if starts.size != stops.size:
        raise RuntimeError("Task starts and stops do not match")
    return np.column_stack((starts, stops)).astype(np.int64)


def make_time_aligned_labels(
    event: np.ndarray,
    task_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return label, task_id, selected task bounds, cleaned event, and dropped event=1 bounds.

    MEMA label files contain exactly 12 task labels per subject. A few raw Data2 files
    contain extra short event=1 runs, likely marker glitches or repeated counters.
    When there are more than 12 event=1 runs, this function keeps the longest 12 runs,
    restores chronological order, and marks the dropped runs as event=0 so the HDF5
    invariant remains true: event=1 has a valid task label, event=0 has label=-1.
    """
    original_event = np.asarray(event, dtype=np.int8).reshape(-1)
    bounds_all = find_task_runs(original_event)
    n_labels = int(task_labels.shape[0])

    if bounds_all.shape[0] == n_labels:
        bounds = bounds_all
        dropped_bounds = np.empty((0, 2), dtype=np.int64)
        cleaned_event = original_event.copy()
    elif bounds_all.shape[0] > n_labels:
        lengths = bounds_all[:, 1] - bounds_all[:, 0]
        keep_indices = np.sort(np.argsort(lengths)[-n_labels:])
        drop_indices = np.setdiff1d(np.arange(bounds_all.shape[0]), keep_indices)
        bounds = bounds_all[keep_indices].astype(np.int64, copy=False)
        dropped_bounds = bounds_all[drop_indices].astype(np.int64, copy=False)
        cleaned_event = original_event.copy()
        for start, stop in dropped_bounds:
            cleaned_event[start:stop] = 0
    else:
        raise ValueError(
            f"Found {bounds_all.shape[0]} event=1 intervals but have {n_labels} task labels"
        )

    n_time = cleaned_event.size
    label = np.full((len(LABEL_NAMES), n_time), LABEL_FILL_VALUE, dtype=np.int8)
    task_id = np.full((1, n_time), TASK_ID_FILL_VALUE, dtype=np.int16)
    for task_idx, (start, stop) in enumerate(bounds):
        label[:, start:stop] = task_labels[task_idx, :, None]
        task_id[:, start:stop] = task_idx

    return label, task_id, bounds, cleaned_event, dropped_bounds


def compression_kwargs(mode: str) -> dict[str, Any]:
    if mode == "lzf":
        return {"compression": "lzf", "shuffle": True}
    if mode == "gzip":
        return {"compression": "gzip", "compression_opts": 4, "shuffle": True}
    if mode == "none":
        return {}
    raise ValueError(f"Unknown compression mode: {mode}")


def write_subject_h5(
    output_h5: Path,
    subject_id: int,
    source_file: Path,
    eeg: np.ndarray,
    event: np.ndarray,
    sample_index: np.ndarray,
    task_labels: np.ndarray,
    label: np.ndarray,
    task_id: np.ndarray,
    task_bounds: np.ndarray,
    config: PreprocessingConfig,
    normalization_stats: dict[str, Any],
    compression: str,
) -> None:
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_h5.with_suffix(output_h5.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    comp = compression_kwargs(compression)
    string_dtype = h5py.string_dtype(encoding="utf-8")

    metadata = {
        "source_file": str(source_file),
        "channel_names": list(CHANNEL_NAMES),
        "channel_types": list(CHANNEL_TYPES),
        "sfreq_raw": config.raw_sfreq,
        "sfreq_preprocessed": config.target_sfreq or config.raw_sfreq,
        "preprocessing_config": asdict(config),
        "normalization_stats": normalization_stats,
    }

    with h5py.File(tmp_path, "w") as h5:
        data_group = h5.create_group("data")
        subject_group = data_group.create_group(f"subject_{subject_id:02d}")

        subject_group.create_dataset(
            "preprocessed_eeg",
            data=eeg,
            chunks=(eeg.shape[0], min(20_000, eeg.shape[1])),
            **comp,
        )
        subject_group.create_dataset(
            "event",
            data=event.reshape(1, -1),
            chunks=(1, min(100_000, event.size)),
            **comp,
        )
        subject_group.create_dataset(
            "electrodes",
            data=np.asarray(CHANNEL_NAMES, dtype=object),
            dtype=string_dtype,
        )
        subject_group.create_dataset(
            "label",
            data=label,
            chunks=(label.shape[0], min(100_000, label.shape[1])),
            **comp,
        )
        subject_group.create_dataset(
            "task_id",
            data=task_id,
            chunks=(1, min(100_000, task_id.shape[1])),
            **comp,
        )
        subject_group.create_dataset("task_labels", data=task_labels, dtype=np.int8)
        subject_group.create_dataset("task_bounds", data=task_bounds, dtype=np.int64)
        subject_group.create_dataset(
            "sample_index",
            data=sample_index,
            chunks=(min(100_000, sample_index.size),),
            **comp,
        )

        subject_group.attrs["subject_id"] = subject_id
        subject_group.attrs["sfreq"] = float(config.target_sfreq or config.raw_sfreq)
        subject_group.attrs["eeg_layout"] = "channels_x_time"
        subject_group.attrs["event_layout"] = "1_x_time"
        subject_group.attrs["label_layout"] = "4_x_time"
        subject_group.attrs["label_order"] = np.asarray(LABEL_NAMES, dtype=object)
        subject_group.attrs["label_fill_value"] = LABEL_FILL_VALUE
        subject_group.attrs["label_fill_meaning"] = "No four-dimensional task label applies when event=0"
        subject_group.attrs["task_id_fill_value"] = TASK_ID_FILL_VALUE
        subject_group.attrs["source_file"] = str(source_file)
        subject_group.attrs["metadata_json"] = json.dumps(metadata, ensure_ascii=False)
        subject_group.attrs["preprocessing_config_json"] = json.dumps(asdict(config), ensure_ascii=False)

        if subject_id == 1:
            subject_group.attrs["warning"] = (
                "Subject1 main file is preserved unchanged. If using Subject1_2 as replacement, "
                "that replacement must be handled explicitly outside this script."
            )

        h5.attrs["format_version"] = "1.0"
        h5.attrs["hierarchy"] = "data/subject/(preprocessed_eeg, event, electrodes, label)"
        h5.attrs["description"] = (
            "Continuous preprocessed MEMA EEG including both event=0 and event=1 periods, "
            "with time-aligned four-dimensional labels."
        )

    os.replace(tmp_path, output_h5)


def process_one_subject(
    subject_id: int,
    raw_dir: str,
    labels_dir: str,
    output_dir: str,
    compression: str,
    overwrite: bool,
    config_dict: dict[str, Any],
) -> dict[str, Any]:
    start_time = time.time()
    raw_dir_path = Path(raw_dir)
    labels_dir_path = Path(labels_dir)
    output_dir_path = Path(output_dir)
    config = PreprocessingConfig(**config_dict)
    config.validate()

    source_file = raw_dir_path / f"Subject{subject_id} Data2.txt"
    output_h5 = output_dir_path / f"mema_subject{subject_id}_continuous_labeled.h5"

    if output_h5.exists() and not overwrite:
        return {
            "subject_id": subject_id,
            "status": "skipped_existing",
            "output_h5": str(output_h5),
            "elapsed_sec": round(time.time() - start_time, 3),
        }

    signals, sample_index_raw, event_raw = load_data2(source_file)

    if config.filter_backend == "scipy-iir":
        eeg, sample_index, event, norm_stats = preprocess_scipy_iir(
            signals, sample_index_raw, event_raw, config
        )
    else:
        eeg, sample_index, event, norm_stats = preprocess_mne_fir(
            signals, sample_index_raw, event_raw, config
        )

    task_labels = load_subject_task_labels(labels_dir_path, subject_id)
    label, task_id, task_bounds, event, dropped_event1_bounds = make_time_aligned_labels(event, task_labels)

    outside = event == 0
    inside = event == 1
    if not np.all(label[:, outside] == LABEL_FILL_VALUE):
        raise RuntimeError(f"Subject {subject_id}: outside-task labels are not -1")
    if np.any(label[:, inside] == LABEL_FILL_VALUE):
        raise RuntimeError(f"Subject {subject_id}: inside-task labels contain -1")

    write_subject_h5(
        output_h5=output_h5,
        subject_id=subject_id,
        source_file=source_file,
        eeg=eeg,
        event=event,
        sample_index=sample_index,
        task_labels=task_labels,
        label=label,
        task_id=task_id,
        task_bounds=task_bounds,
        config=config,
        normalization_stats=norm_stats,
        compression=compression,
    )

    elapsed = time.time() - start_time
    return {
        "subject_id": subject_id,
        "status": "ok",
        "source_file": str(source_file),
        "output_h5": str(output_h5),
        "raw_shape": [int(signals.shape[0]), int(signals.shape[1])],
        "preprocessed_shape": [int(eeg.shape[0]), int(eeg.shape[1])],
        "event_values": np.unique(event).astype(int).tolist(),
        "task_count": int(task_bounds.shape[0]),
        "task_bounds_first_last": [
            task_bounds[0].astype(int).tolist(),
            task_bounds[-1].astype(int).tolist(),
        ],
        "dropped_event1_interval_count": int(dropped_event1_bounds.shape[0]),
        "dropped_event1_bounds": dropped_event1_bounds.astype(int).tolist(),
        "task_labels": task_labels.astype(int).tolist(),
        "elapsed_sec": round(elapsed, 3),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast builder for MEMA continuous labeled HDF5 files.")
    parser.add_argument("--raw-dir", type=Path, default=Path("/mnt/dataset4/DATASETS/online_learning/data/raw_data"))
    parser.add_argument("--labels-dir", type=Path, default=Path("/mnt/dataset4/DATASETS/online_learning/data/For_DL"))
    parser.add_argument("--output-dir", type=Path, default=Path("/mnt/dataset4/lingqy/mema/output"))
    parser.add_argument("--subjects", type=str, default="2-20", help="Examples: 2-20, 1,3,5, 1-20")
    parser.add_argument("--compression", choices=("lzf", "gzip", "none"), default="lzf")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--filter-backend", choices=("scipy-iir", "mne-fir"), default="scipy-iir")
    parser.add_argument("--mne-n-jobs", type=int, default=16)
    parser.add_argument("--max-workers", type=int, default=1, help="Subject-level parallel workers. Start with 1 or 2.")
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--normalization", choices=("none", "all_global", "eeg_global", "channelwise"), default="eeg_global")
    parser.add_argument("--target-sfreq", type=float, default=200.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subjects = parse_subjects(args.subjects)
    config = PreprocessingConfig(
        filter_backend=args.filter_backend,
        mne_n_jobs=args.mne_n_jobs,
        normalization=args.normalization,
        target_sfreq=args.target_sfreq,
    )
    config.validate()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_json = args.summary_json or (args.output_dir / "mema_build_summary.json")

    print("Selected subjects:", subjects, flush=True)
    print("Output dir:", args.output_dir, flush=True)
    print("Filter backend:", config.filter_backend, flush=True)
    print("Subject-level workers:", args.max_workers, flush=True)

    config_dict = asdict(config)
    all_results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    if args.max_workers <= 1:
        for subject_id in subjects:
            print(f"\n[Subject {subject_id:02d}] start", flush=True)
            try:
                result = process_one_subject(
                    subject_id,
                    str(args.raw_dir),
                    str(args.labels_dir),
                    str(args.output_dir),
                    args.compression,
                    args.overwrite,
                    config_dict,
                )
                all_results.append(result)
                print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
            except Exception as exc:  # keep partial successes visible
                err = {"subject_id": subject_id, "status": "error", "error": repr(exc)}
                errors.append(err)
                print(json.dumps(err, ensure_ascii=False, indent=2), flush=True)
                continue
    else:
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            future_map = {
                executor.submit(
                    process_one_subject,
                    subject_id,
                    str(args.raw_dir),
                    str(args.labels_dir),
                    str(args.output_dir),
                    args.compression,
                    args.overwrite,
                    config_dict,
                ): subject_id
                for subject_id in subjects
            }
            for future in as_completed(future_map):
                subject_id = future_map[future]
                try:
                    result = future.result()
                    all_results.append(result)
                    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
                except Exception as exc:
                    err = {"subject_id": subject_id, "status": "error", "error": repr(exc)}
                    errors.append(err)
                    print(json.dumps(err, ensure_ascii=False, indent=2), flush=True)
                    continue

    payload = {
        "subjects": subjects,
        "config": config_dict,
        "results": sorted(all_results, key=lambda x: x["subject_id"]),
        "errors": errors,
    }
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\nSummary saved: {summary_json}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
