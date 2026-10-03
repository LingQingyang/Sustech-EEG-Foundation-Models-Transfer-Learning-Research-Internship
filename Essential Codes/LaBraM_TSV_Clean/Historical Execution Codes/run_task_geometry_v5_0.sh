#!/usr/bin/env bash
set -euo pipefail

# LaBraM x M3CV Task Adaptation Geometry V5.0
# Active source set: two .py + this .sh + README.
# The already-existing run_tsv_labram_m3cv_v0_3_4.py is a frozen numerical core dependency.

MODE="${1:-preflight}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -d "$HOME/venvs/labram/bin" ]]; then
  export PATH="$HOME/venvs/labram/bin:$PATH"
fi

TRAIN_PY="${TRAIN_PY:-${SCRIPT_DIR}/run_task_geometry_v5_0.py}"
ANALYZE_PY="${ANALYZE_PY:-${SCRIPT_DIR}/analyze_task_geometry_v5_0.py}"

ROOT="${ROOT:-/omni-eeg-01/task calibration/results/tsv_labram_m3cv/tsv_labram_m3cv_v5_0}"
LEGACY_ROOT="${LEGACY_ROOT:-/omni-eeg-01/task calibration/results/tsv_labram_m3cv/tsv_labram_m3cv_v0_3_4}"
DATA_ROOT="${DATA_ROOT:-/omni-eeg-01/task calibration/dataset/m3cv}"
LABRAM_CKPT="${LABRAM_CKPT:-/omni-eeg-01/task calibration/LaBraM/checkpoints/labram-base.pth}"
MODELING_FILE="${MODELING_FILE:-/omni-eeg-01/task calibration/LaBraM/modeling_finetune.py}"

DEVICE="${DEVICE:-cuda}"
WORKERS="${WORKERS:-0}"
RAM_RESERVE_GB="${RAM_RESERVE_GB:-4}"
SEED_BASE="${SEED_BASE:-20260819}"
MODEL_INIT_SEED="${MODEL_INIT_SEED:-314159}"

ANALYSIS_MODE="${ANALYSIS_MODE:-all}"
RANDOM_N="${RANDOM_N:-1000}"
RANDOM_SEED="${RANDOM_SEED:-20260828}"
FULL_CONTEXT_PAIRINGS="${FULL_CONTEXT_PAIRINGS:-all}"
RESUME="${RESUME:-1}"

mkdir -p "${ROOT}/logs"

COMMON_TRAIN=(
  --root "${ROOT}"
  --legacy-root "${LEGACY_ROOT}"
  --data-root "${DATA_ROOT}"
  --checkpoint "${LABRAM_CKPT}"
  --modeling-file "${MODELING_FILE}"
  --seed-base "${SEED_BASE}"
  --model-init-seed "${MODEL_INIT_SEED}"
  --workers "${WORKERS}"
  --ram-reserve-gb "${RAM_RESERVE_GB}"
  --device "${DEVICE}"
)

run_preflight() {
  python3 "${TRAIN_PY}" --mode preflight "${COMMON_TRAIN[@]}" \
    2>&1 | tee "${ROOT}/logs/v5_0_preflight.log"
}

run_inspect() {
  python3 "${TRAIN_PY}" --mode inspect "${COMMON_TRAIN[@]}" \
    2>&1 | tee "${ROOT}/logs/v5_0_inspect.log"
}

run_final() {
  local extra=()
  if [[ "${OVERWRITE_NEW_RUN:-0}" == "1" ]]; then
    extra+=(--overwrite-new-run)
  fi
  python3 "${TRAIN_PY}" --mode final "${COMMON_TRAIN[@]}" "${extra[@]}" \
    2>&1 | tee "${ROOT}/logs/v5_0_final.log"
}

run_assemble() {
  python3 "${TRAIN_PY}" --mode assemble "${COMMON_TRAIN[@]}" \
    2>&1 | tee "${ROOT}/logs/v5_0_assemble.log"
}

run_analyze() {
  local extra=()
  if [[ "${RESUME}" != "1" ]]; then
    extra+=(--no-resume)
  fi
  python3 "${ANALYZE_PY}" \
    --mode "${ANALYSIS_MODE}" \
    --root "${ROOT}" \
    --data-root "${DATA_ROOT}" \
    --checkpoint "${LABRAM_CKPT}" \
    --modeling-file "${MODELING_FILE}" \
    --tasks "Rest,Motor,P300,SSS,TS" \
    --eval-batch-size 64 \
    --workers "${WORKERS}" \
    --ram-reserve-gb "${RAM_RESERVE_GB}" \
    --device "${DEVICE}" \
    --model-init-seed "${MODEL_INIT_SEED}" \
    --coarse-step 0.05 \
    --refine-step 0.01 \
    --target-functional-retention 0.99 \
    --random-n "${RANDOM_N}" \
    --random-seed "${RANDOM_SEED}" \
    --full-context-pairings "${FULL_CONTEXT_PAIRINGS}" \
    "${extra[@]}" \
    2>&1 | tee "${ROOT}/logs/v5_0_analysis.log"
}

case "${MODE}" in
  preflight)
    run_preflight
    ;;
  inspect)
    run_inspect
    ;;
  final)
    run_final
    ;;
  assemble)
    run_assemble
    ;;
  analyze|analysis)
    run_analyze
    ;;
  selftest|analysis-selftest)
    python3 "${ANALYZE_PY}" --mode selftest
    ;;
  pilot)
    echo "V5.0 deliberately has no new pilot." >&2
    echo "E*=20 is frozen from the certified v0.3.4 eight-run pilot fallback." >&2
    exit 3
    ;;
  all)
    run_preflight
    # final performs the fail-closed inspect internally before any training.
    run_final
    run_analyze
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    echo "Use: preflight | inspect | final | assemble | analyze | selftest | all" >&2
    exit 2
    ;;
esac
