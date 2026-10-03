#!/usr/bin/env bash
set -euo pipefail

# LaBraM x M3CV revised four-spectrum analysis launcher (analysis only)
# -------------------------------------------------------------------
# Usage:
#   bash run_labram_m3cv_four_spectra_v6_0.sh preflight
#   bash run_labram_m3cv_four_spectra_v6_0.sh heatmap
#   bash run_labram_m3cv_four_spectra_v6_0.sh individual
#   bash run_labram_m3cv_four_spectra_v6_0.sh shared-energy
#   bash run_labram_m3cv_four_spectra_v6_0.sh random
#   bash run_labram_m3cv_four_spectra_v6_0.sh shared-functional
#   bash run_labram_m3cv_four_spectra_v6_0.sh sti
#   bash run_labram_m3cv_four_spectra_v6_0.sh plot
#   bash run_labram_m3cv_four_spectra_v6_0.sh main   # recommended fast complete run (includes random baseline)
#   bash run_labram_m3cv_four_spectra_v6_0.sh all
#
# This launcher NEVER starts fine-tuning. It reads the existing v5.0 final
# checkpoints/deltas and performs decomposition, reconstruction/evaluation,
# matched-random baselines, STI, tables, and figures only.

MODE="${1:-preflight}"
PROFILE="${PROFILE:-fast}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYZE_PY="${ANALYZE_PY:-${SCRIPT_DIR}/analyze_labram_m3cv_four_spectra_v6_0.py}"

# Known project paths. All are overridable from the environment.
ROOT="${ROOT:-/omni-eeg-01/task calibration/results/tsv_labram_m3cv/tsv_labram_m3cv_v5_0}"
OUTDIR="${OUTDIR:-${ROOT}/analysis_four_spectra_v6_0}"
CORE_SCRIPT="${CORE_SCRIPT:-}"
TASKS="${TASKS:-Rest,Motor,P300,SSS,TS}"
DEVICE="${DEVICE:-cuda}"
RAM_RESERVE_GB="${RAM_RESERVE_GB:-4}"
if [[ "${PROFILE}" == "full" ]]; then
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
  SHARED_RANDOM_N="${SHARED_RANDOM_N:-1000}"
  STI_RANDOM_N="${STI_RANDOM_N:-1000}"
else
  # Overnight-oriented defaults. These preserve the core four-spectrum logic
  # but use aligned replicate tracks, sparser functional probes and small nulls.
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
  SHARED_RANDOM_N="${SHARED_RANDOM_N:-128}"
  STI_RANDOM_N="${STI_RANDOM_N:-128}"
fi
RANDOM_SEED="${RANDOM_SEED:-20260907}"
ORTHO_RTOL="${ORTHO_RTOL:-0}"
METRIC_TOL="${METRIC_TOL:-2e-5}"
RESUME="${RESUME:-1}"
SAVE_RANDOM_DRAWS="${SAVE_RANDOM_DRAWS:-0}"

# Standard LaBraM environment on the Volcengine server.
if [[ -d "$HOME/venvs/labram/bin" ]]; then
  export PATH="$HOME/venvs/labram/bin:$PATH"
fi

mkdir -p "${OUTDIR}/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="${OUTDIR}/logs/${MODE}_${STAMP}.log"

ARGS=(
  --mode "${MODE}"
  --root "${ROOT}"
  --outdir "${OUTDIR}"
  --tasks "${TASKS}"
  --device "${DEVICE}"
  --profile "${PROFILE}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --ram-reserve-gb "${RAM_RESERVE_GB}"
  --shared-random-n "${SHARED_RANDOM_N}"
  --sti-random-n "${STI_RANDOM_N}"
  --random-seed "${RANDOM_SEED}"
  --ortho-rtol "${ORTHO_RTOL}"
  --metric-tol "${METRIC_TOL}"
)

if [[ -n "${CORE_SCRIPT}" ]]; then
  ARGS+=(--core-script "${CORE_SCRIPT}")
fi
if [[ "${RESUME}" == "0" ]]; then
  ARGS+=(--no-resume)
fi
if [[ "${SAVE_RANDOM_DRAWS}" == "1" ]]; then
  ARGS+=(--save-random-draws)
fi

printf 'Analysis mode: %s\n' "${MODE}"
printf 'Analysis profile: %s\n' "${PROFILE}"
printf 'Final-run root: %s\n' "${ROOT}"
printf 'Output dir: %s\n' "${OUTDIR}"
printf 'Log: %s\n' "${LOG}"
printf 'No fine-tuning will be launched.\n\n'

python3 "${ANALYZE_PY}" "${ARGS[@]}" 2>&1 | tee "${LOG}"
