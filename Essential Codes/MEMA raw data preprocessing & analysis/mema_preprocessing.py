"""
MEMA EEG preprocessing utilities.

Expected raw Data2 TXT layout
-----------------------------
Columns 1-32 : signal channels
Column 33    : sample/data index
Column 34    : task/rest marker (1=task, 0=rest)

Signal channels
---------------
1-30 : EEG
31-32: horizontal EOG (HEOL, HEOR)

Reference preprocessing adapted from Xinyu's CogBCI pipeline:
- 50 Hz FIR notch filter, notch width 4 Hz
- 0.1 Hz high-pass and 40 Hz low-pass FIR filtering
- optional resampling from 500 Hz to 200 Hz
- global Z-score normalization

The default normalization is ``eeg_global``:
all EEG values share one mean/std, while EOG is normalized separately.
Use ``all_global`` to reproduce Xinyu's global normalization over every
signal channel.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import mne
import numpy as np


MEMA_CHANNEL_NAMES: tuple[str, ...] = (
    "FP1", "FP2", "Fz", "F3", "F4", "F7", "F8",
    "FCz", "FC3", "FC4", "FT7", "FT8",
    "Cz", "C3", "C4", "T3", "T4",
    "CPz", "CP3", "CP4", "TP7", "TP8",
    "Pz", "P3", "P4", "T5", "T6",
    "Oz", "O1", "O2",
    "HEOL", "HEOR",
)

MEMA_CHANNEL_TYPES: tuple[str, ...] = ("eeg",) * 30 + ("eog",) * 2
MEMA_EEG_INDICES = np.arange(30, dtype=np.int64)
MEMA_EOG_INDICES = np.arange(30, 32, dtype=np.int64)

NormalizationMode = Literal["none", "all_global", "eeg_global", "channelwise"]


@dataclass(frozen=True)
class PreprocessingConfig:
    """Configuration for MEMA preprocessing."""

    raw_sfreq: float = 500.0
    notch_freq: float | None = 50.0
    notch_width: float = 4.0
    highpass_freq: float | None = 0.1
    lowpass_freq: float | None = 40.0
    target_sfreq: float | None = 200.0
    normalization: NormalizationMode = "eeg_global"
    normalize_eog_separately: bool = True
    eps: float = 1e-8

    def validate(self) -> None:
        if self.raw_sfreq <= 0:
            raise ValueError("raw_sfreq must be positive.")
        if self.target_sfreq is not None and self.target_sfreq <= 0:
            raise ValueError("target_sfreq must be positive or None.")
        if self.notch_freq is not None and self.notch_freq <= 0:
            raise ValueError("notch_freq must be positive or None.")
        if self.notch_width <= 0:
            raise ValueError("notch_width must be positive.")
        if self.highpass_freq is not None and self.highpass_freq < 0:
            raise ValueError("highpass_freq must be non-negative or None.")
        if self.lowpass_freq is not None and self.lowpass_freq <= 0:
            raise ValueError("lowpass_freq must be positive or None.")
        if (
            self.highpass_freq is not None
            and self.lowpass_freq is not None
            and self.highpass_freq >= self.lowpass_freq
        ):
            raise ValueError("highpass_freq must be lower than lowpass_freq.")
        nyquist = self.raw_sfreq / 2.0
        if self.lowpass_freq is not None and self.lowpass_freq >= nyquist:
            raise ValueError(
                f"lowpass_freq={self.lowpass_freq} must be below Nyquist={nyquist}."
            )
        if self.notch_freq is not None and self.notch_freq >= nyquist:
            raise ValueError(
                f"notch_freq={self.notch_freq} must be below Nyquist={nyquist}."
            )


@dataclass
class MemaRawRecording:
    """Raw MEMA Data2 content after layout parsing."""

    signals: np.ndarray            # [32, T]
    sample_index: np.ndarray       # [T]
    event: np.ndarray              # [T], values 0/1
    channel_names: tuple[str, ...] = MEMA_CHANNEL_NAMES
    channel_types: tuple[str, ...] = MEMA_CHANNEL_TYPES
    source_file: str | None = None


@dataclass
class MemaPreprocessedRecording:
    """Raw and preprocessed MEMA signals with aligned event streams."""

    raw_signals: np.ndarray                 # [32, T_raw]
    preprocessed_signals: np.ndarray        # [32, T_preprocessed]
    sample_index_raw: np.ndarray            # [T_raw]
    sample_index_preprocessed: np.ndarray   # [T_preprocessed]
    event_raw: np.ndarray                   # [T_raw]
    event_preprocessed: np.ndarray          # [T_preprocessed]
    channel_names: tuple[str, ...]
    channel_types: tuple[str, ...]
    sfreq_raw: float
    sfreq_preprocessed: float
    preprocessing_config: dict[str, Any]
    normalization_stats: dict[str, Any]
    source_file: str | None = None


def _infer_delimiter_and_skiprows(path: Path) -> tuple[str | None, int]:
    """
    Find the first row containing at least 34 numeric values.

    Returns
    -------
    delimiter:
        None for whitespace-delimited data, otherwise a literal delimiter.
    skiprows:
        Number of leading rows to skip.
    """
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

    raise ValueError(
        f"Could not find a numeric row with at least 34 columns in {path}."
    )


def load_mema_data2_txt(
    file_path: str | Path,
    *,
    delimiter: str | None = None,
    skiprows: int | None = None,
    dtype: np.dtype = np.float64,
) -> MemaRawRecording:
    """
    Load and validate a MEMA Data2 TXT file.

    Parameters
    ----------
    file_path:
        Path to the raw Data2 TXT file.
    delimiter:
        Explicit delimiter. If omitted, it is inferred from the first numeric row.
    skiprows:
        Number of header rows. If omitted, it is inferred.
    dtype:
        Numeric dtype used while loading.

    Returns
    -------
    MemaRawRecording
        Signals are transposed to [channels, time].
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"MEMA file does not exist: {path}")

    inferred_delimiter, inferred_skiprows = _infer_delimiter_and_skiprows(path)
    actual_delimiter = inferred_delimiter if delimiter is None else delimiter
    actual_skiprows = inferred_skiprows if skiprows is None else skiprows

    matrix = np.loadtxt(
        path,
        delimiter=actual_delimiter,
        skiprows=actual_skiprows,
        dtype=dtype,
        ndmin=2,
    )

    if matrix.shape[1] < 34:
        raise ValueError(
            f"Expected at least 34 columns, got shape {matrix.shape} in {path}."
        )
    if matrix.shape[1] > 34:
        raise ValueError(
            f"Expected exactly 34 columns, got {matrix.shape[1]} in {path}. "
            "Refusing to guess which extra columns should be ignored."
        )
    if not np.isfinite(matrix[:, :32]).all():
        raise ValueError("Signal columns contain NaN or infinite values.")

    signals = np.asarray(matrix[:, :32].T, dtype=np.float32)
    sample_index_float = matrix[:, 32]
    event_float = matrix[:, 33]

    if not np.isfinite(sample_index_float).all():
        raise ValueError("Sample-index column contains NaN or infinite values.")
    if not np.isfinite(event_float).all():
        raise ValueError("Event column contains NaN or infinite values.")

    if not np.allclose(sample_index_float, np.rint(sample_index_float), atol=1e-6):
        raise ValueError("Sample-index column contains non-integer values.")
    sample_index = np.rint(sample_index_float).astype(np.int64)

    if np.any(np.diff(sample_index) < 0):
        raise ValueError("Sample-index column is not monotonically non-decreasing.")

    if not np.allclose(event_float, np.rint(event_float), atol=1e-6):
        raise ValueError("Event column contains non-integer values.")
    event = np.rint(event_float).astype(np.int8)

    unique_events = set(np.unique(event).tolist())
    if not unique_events.issubset({0, 1}):
        raise ValueError(
            f"Event marker must contain only 0/1, found {sorted(unique_events)}."
        )

    return MemaRawRecording(
        signals=signals,
        sample_index=sample_index,
        event=event,
        source_file=str(path.resolve()),
    )


def _make_mne_raw(
    signals: np.ndarray,
    sfreq: float,
    channel_names: Sequence[str] = MEMA_CHANNEL_NAMES,
    channel_types: Sequence[str] = MEMA_CHANNEL_TYPES,
) -> mne.io.RawArray:
    signals = np.asarray(signals, dtype=np.float64)
    if signals.ndim != 2:
        raise ValueError(f"signals must be 2-D [channels, time], got {signals.shape}.")
    if signals.shape[0] != len(channel_names):
        raise ValueError(
            f"signals has {signals.shape[0]} channels, "
            f"but {len(channel_names)} channel names were supplied."
        )
    if len(channel_names) != len(channel_types):
        raise ValueError("channel_names and channel_types must have equal length.")

    info = mne.create_info(
        ch_names=list(channel_names),
        sfreq=float(sfreq),
        ch_types=list(channel_types),
        verbose=False,
    )
    return mne.io.RawArray(signals, info, verbose=False)


def notch_filter(
    signals: np.ndarray,
    sfreq: float,
    *,
    freq: float = 50.0,
    notch_width: float = 4.0,
) -> np.ndarray:
    """Apply Xinyu-style FIR power-line notch filtering."""
    raw = _make_mne_raw(signals, sfreq)
    raw.notch_filter(
        freqs=[float(freq)],
        notch_widths=float(notch_width),
        method="fir",
        verbose=False,
    )
    return raw.get_data().astype(np.float32, copy=False)


def highpass_filter(
    signals: np.ndarray,
    sfreq: float,
    *,
    cutoff: float = 0.1,
) -> np.ndarray:
    """Apply an FIR high-pass filter."""
    raw = _make_mne_raw(signals, sfreq)
    raw.filter(l_freq=float(cutoff), h_freq=None, method="fir", verbose=False)
    return raw.get_data().astype(np.float32, copy=False)


def lowpass_filter(
    signals: np.ndarray,
    sfreq: float,
    *,
    cutoff: float = 40.0,
) -> np.ndarray:
    """Apply an FIR low-pass filter."""
    raw = _make_mne_raw(signals, sfreq)
    raw.filter(l_freq=None, h_freq=float(cutoff), method="fir", verbose=False)
    return raw.get_data().astype(np.float32, copy=False)


def _safe_zscore(
    values: np.ndarray,
    *,
    eps: float,
) -> tuple[np.ndarray, float, float]:
    mean = float(np.mean(values, dtype=np.float64))
    std = float(np.std(values, dtype=np.float64))
    denominator = std if std >= eps else 1.0
    normalized = (values - mean) / denominator
    return normalized, mean, std


def normalize_signals(
    signals: np.ndarray,
    *,
    mode: NormalizationMode = "eeg_global",
    eeg_indices: np.ndarray = MEMA_EEG_INDICES,
    eog_indices: np.ndarray = MEMA_EOG_INDICES,
    normalize_eog_separately: bool = True,
    eps: float = 1e-8,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Normalize [channels, time] signals.

    Modes
    -----
    none:
        Return values unchanged.
    all_global:
        One mean/std for all 32 signal channels. This reproduces Xinyu's
        global normalization most directly.
    eeg_global:
        One mean/std for the 30 EEG channels. EOG channels are optionally
        normalized using their own joint mean/std.
    channelwise:
        Each channel receives its own mean/std.
    """
    x = np.asarray(signals, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"signals must be [channels, time], got {x.shape}.")

    output = x.copy()
    stats: dict[str, Any] = {"mode": mode}

    if mode == "none":
        return output, stats

    if mode == "all_global":
        output, mean, std = _safe_zscore(output, eps=eps)
        stats.update({"mean": mean, "std": std})
        return output.astype(np.float32, copy=False), stats

    if mode == "eeg_global":
        eeg_values, eeg_mean, eeg_std = _safe_zscore(output[eeg_indices], eps=eps)
        output[eeg_indices] = eeg_values
        stats["eeg"] = {"mean": eeg_mean, "std": eeg_std}

        if len(eog_indices) > 0:
            if normalize_eog_separately:
                eog_values, eog_mean, eog_std = _safe_zscore(
                    output[eog_indices], eps=eps
                )
                output[eog_indices] = eog_values
                stats["eog"] = {"mean": eog_mean, "std": eog_std}
            else:
                stats["eog"] = {"normalized": False}
        return output.astype(np.float32, copy=False), stats

    if mode == "channelwise":
        means = np.mean(output, axis=1, dtype=np.float64)
        stds = np.std(output, axis=1, dtype=np.float64)
        denominators = np.where(stds >= eps, stds, 1.0)
        output = (output - means[:, None]) / denominators[:, None]
        stats.update(
            {
                "means": means.tolist(),
                "stds": stds.tolist(),
            }
        )
        return output.astype(np.float32, copy=False), stats

    raise ValueError(f"Unsupported normalization mode: {mode}")


def _nearest_resample_indices(
    n_source: int,
    source_sfreq: float,
    n_target: int,
    target_sfreq: float,
) -> np.ndarray:
    """
    Map target samples to nearest source samples.

    This is suitable for discrete event markers and sample indices.
    """
    target_times = np.arange(n_target, dtype=np.float64) / target_sfreq
    source_indices = np.rint(target_times * source_sfreq).astype(np.int64)
    return np.clip(source_indices, 0, n_source - 1)


def preprocess_mema_recording(
    recording: MemaRawRecording,
    config: PreprocessingConfig = PreprocessingConfig(),
) -> MemaPreprocessedRecording:
    """
    Execute the MEMA preprocessing pipeline.

    Order
    -----
    1. FIR notch filtering.
    2. FIR high/low-pass filtering.
    3. Optional resampling.
    4. Signal normalization.
    5. Nearest-neighbour alignment of event/sample-index streams.
    """
    config.validate()

    if recording.signals.shape[0] != 32:
        raise ValueError(
            f"MEMA recording must have 32 signal channels, got "
            f"{recording.signals.shape[0]}."
        )

    raw = _make_mne_raw(
        recording.signals,
        config.raw_sfreq,
        recording.channel_names,
        recording.channel_types,
    )

    if config.notch_freq is not None:
        raw.notch_filter(
            freqs=[float(config.notch_freq)],
            notch_widths=float(config.notch_width),
            method="fir",
            verbose=False,
        )

    if config.highpass_freq is not None or config.lowpass_freq is not None:
        raw.filter(
            l_freq=config.highpass_freq,
            h_freq=config.lowpass_freq,
            method="fir",
            verbose=False,
        )

    sfreq_preprocessed = config.raw_sfreq
    if (
        config.target_sfreq is not None
        and not math.isclose(
            config.target_sfreq,
            config.raw_sfreq,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        raw.resample(float(config.target_sfreq), npad="auto", verbose=False)
        sfreq_preprocessed = float(config.target_sfreq)

    filtered = raw.get_data().astype(np.float32, copy=False)
    normalized, normalization_stats = normalize_signals(
        filtered,
        mode=config.normalization,
        normalize_eog_separately=config.normalize_eog_separately,
        eps=config.eps,
    )

    target_indices = _nearest_resample_indices(
        n_source=recording.signals.shape[1],
        source_sfreq=config.raw_sfreq,
        n_target=normalized.shape[1],
        target_sfreq=sfreq_preprocessed,
    )
    event_preprocessed = recording.event[target_indices].astype(np.int8, copy=False)
    sample_index_preprocessed = recording.sample_index[target_indices].astype(
        np.int64, copy=False
    )

    return MemaPreprocessedRecording(
        raw_signals=recording.signals.astype(np.float32, copy=True),
        preprocessed_signals=normalized,
        sample_index_raw=recording.sample_index.astype(np.int64, copy=True),
        sample_index_preprocessed=sample_index_preprocessed,
        event_raw=recording.event.astype(np.int8, copy=True),
        event_preprocessed=event_preprocessed,
        channel_names=recording.channel_names,
        channel_types=recording.channel_types,
        sfreq_raw=float(config.raw_sfreq),
        sfreq_preprocessed=sfreq_preprocessed,
        preprocessing_config=asdict(config),
        normalization_stats=normalization_stats,
        source_file=recording.source_file,
    )


def load_and_preprocess_mema(
    file_path: str | Path,
    config: PreprocessingConfig = PreprocessingConfig(),
    *,
    delimiter: str | None = None,
    skiprows: int | None = None,
) -> MemaPreprocessedRecording:
    """Convenience interface: load a Data2 TXT file and preprocess it."""
    recording = load_mema_data2_txt(
        file_path,
        delimiter=delimiter,
        skiprows=skiprows,
    )
    return preprocess_mema_recording(recording, config)


def event_transition_indices(event: np.ndarray) -> np.ndarray:
    """Return indices where a 0/1 event stream changes value."""
    marker = np.asarray(event).reshape(-1)
    if marker.size < 2:
        return np.empty(0, dtype=np.int64)
    return np.flatnonzero(marker[1:] != marker[:-1]) + 1


def summarize_recording(
    recording: MemaRawRecording | MemaPreprocessedRecording,
) -> dict[str, Any]:
    """Return a JSON-serializable summary for smoke testing."""
    if isinstance(recording, MemaRawRecording):
        transitions = event_transition_indices(recording.event)
        return {
            "kind": "raw",
            "source_file": recording.source_file,
            "signal_shape": list(recording.signals.shape),
            "sample_index_start": int(recording.sample_index[0]),
            "sample_index_end": int(recording.sample_index[-1]),
            "event_values": np.unique(recording.event).astype(int).tolist(),
            "event_transition_count": int(len(transitions)),
            "event_transition_indices_first_20": transitions[:20].astype(int).tolist(),
            "channel_names": list(recording.channel_names),
            "channel_types": list(recording.channel_types),
        }

    transitions_raw = event_transition_indices(recording.event_raw)
    transitions_processed = event_transition_indices(recording.event_preprocessed)
    return {
        "kind": "preprocessed",
        "source_file": recording.source_file,
        "raw_signal_shape": list(recording.raw_signals.shape),
        "preprocessed_signal_shape": list(recording.preprocessed_signals.shape),
        "sfreq_raw": recording.sfreq_raw,
        "sfreq_preprocessed": recording.sfreq_preprocessed,
        "raw_event_transition_count": int(len(transitions_raw)),
        "processed_event_transition_count": int(len(transitions_processed)),
        "raw_event_values": np.unique(recording.event_raw).astype(int).tolist(),
        "processed_event_values": (
            np.unique(recording.event_preprocessed).astype(int).tolist()
        ),
        "preprocessing_config": recording.preprocessing_config,
        "normalization_stats": recording.normalization_stats,
        "channel_names": list(recording.channel_names),
        "channel_types": list(recording.channel_types),
    }


def save_npz_preview(
    output_path: str | Path,
    recording: MemaPreprocessedRecording,
) -> Path:
    """
    Save one preprocessed recording as NPZ for validation.

    This is a smoke-test format only. The subject/trial HDF5 writer will be
    implemented separately.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "source_file": recording.source_file,
        "channel_names": list(recording.channel_names),
        "channel_types": list(recording.channel_types),
        "sfreq_raw": recording.sfreq_raw,
        "sfreq_preprocessed": recording.sfreq_preprocessed,
        "preprocessing_config": recording.preprocessing_config,
        "normalization_stats": recording.normalization_stats,
    }

    np.savez_compressed(
        path,
        raw_signals=recording.raw_signals,
        preprocessed_signals=recording.preprocessed_signals,
        sample_index_raw=recording.sample_index_raw,
        sample_index_preprocessed=recording.sample_index_preprocessed,
        event_raw=recording.event_raw,
        event_preprocessed=recording.event_preprocessed,
        metadata_json=np.array(json.dumps(metadata, ensure_ascii=False)),
    )
    return path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and preprocess one MEMA Data2 TXT file."
    )
    parser.add_argument("input_txt", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--raw-sfreq", type=float, default=500.0)
    parser.add_argument("--target-sfreq", type=float, default=200.0)
    parser.add_argument(
        "--no-resample",
        action="store_true",
        help="Keep the preprocessed signal at the original sampling rate.",
    )
    parser.add_argument(
        "--normalization",
        choices=("none", "all_global", "eeg_global", "channelwise"),
        default="eeg_global",
    )
    parser.add_argument("--highpass", type=float, default=0.1)
    parser.add_argument("--lowpass", type=float, default=40.0)
    parser.add_argument("--notch", type=float, default=50.0)
    parser.add_argument("--notch-width", type=float, default=4.0)
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only parse and validate the raw TXT; do not filter it.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    raw_recording = load_mema_data2_txt(args.input_txt)
    if args.summary_only:
        print(json.dumps(summarize_recording(raw_recording), indent=2, ensure_ascii=False))
        return

    config = PreprocessingConfig(
        raw_sfreq=args.raw_sfreq,
        notch_freq=args.notch,
        notch_width=args.notch_width,
        highpass_freq=args.highpass,
        lowpass_freq=args.lowpass,
        target_sfreq=None if args.no_resample else args.target_sfreq,
        normalization=args.normalization,
    )
    processed = preprocess_mema_recording(raw_recording, config)
    print(json.dumps(summarize_recording(processed), indent=2, ensure_ascii=False))

    if args.output is not None:
        saved_path = save_npz_preview(args.output, processed)
        print(f"Saved validation NPZ to: {saved_path}")


if __name__ == "__main__":
    main()
