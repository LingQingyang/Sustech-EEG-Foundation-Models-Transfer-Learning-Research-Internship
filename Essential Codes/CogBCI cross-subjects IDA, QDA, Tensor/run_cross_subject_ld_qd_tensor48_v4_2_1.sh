#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT="${SCRIPT:-/home/lqy/run_cross_subject_ld_qd_tensor48_v4_2_1.py}"

# SMOKE=1 runs one real-H5 BIOT fold at M=64.
# PREFLIGHT_ONLY=1 performs metadata, dimension, and duplicate-subject audits only.
if [[ "${SMOKE:-0}" == "1" ]]; then
  OUTDIR="${OUTDIR:-/mnt/dataset4/yinuo/FM_flow/dataset/exp_crosssubject_ld_qd_tensor_multitask_EO_v4_2_1_smoke}"
  MODELS="${MODELS:-BIOT}"
  MODEL_DIMS="${MODEL_DIMS:-BIOT:64}"
  FOLDS="${FOLDS:-1}"
  ALPHA_GRID="${ALPHA_GRID:-0.05,0.5,1}"
  RIDGE_GRID="${RIDGE_GRID:-0.01,0.1,1}"
  TRAIN_CAP="${TRAIN_CAP:-48}"
  BASIS_CAP="${BASIS_CAP:-12}"
  INNER_SCORE_CAP="${INNER_SCORE_CAP:-48}"
  RANDOMIZED_N_ITER="${RANDOMIZED_N_ITER:-2}"
  RANDOMIZED_OVERSAMPLES="${RANDOMIZED_OVERSAMPLES:-20}"
  TRAIN_BLOCKS_PER_SESSION="${TRAIN_BLOCKS_PER_SESSION:-2}"
  BASIS_BLOCKS_PER_SESSION="${BASIS_BLOCKS_PER_SESSION:-2}"
  DUPLICATE_AUDIT_ROWS="${DUPLICATE_AUDIT_ROWS:-8}"
  RUN_SELF_TEST="${RUN_SELF_TEST:-1}"
else
  OUTDIR="${OUTDIR:-/mnt/dataset4/yinuo/FM_flow/dataset/exp_crosssubject_ld_qd_tensor_multitask_EO_v4_2_1}"
  MODELS="${MODELS:-BIOT,CBraMod,LaBraM,EEGPT,EEGMamba}"
  MODEL_DIMS="${MODEL_DIMS:-CBraMod:500,LaBraM:500,EEGPT:500,EEGMamba:500,BIOT:256}"
  FOLDS="${FOLDS:-}"
  ALPHA_GRID="${ALPHA_GRID:-0.001,0.01,0.05,0.1,0.25,0.5,0.9,0.99,1}"
  RIDGE_GRID="${RIDGE_GRID:-0.0001,0.001,0.01,0.1,1,10,100}"
  TRAIN_CAP="${TRAIN_CAP:-96}"
  BASIS_CAP="${BASIS_CAP:-24}"
  INNER_SCORE_CAP="${INNER_SCORE_CAP:-96}"
  RANDOMIZED_N_ITER="${RANDOMIZED_N_ITER:-4}"
  RANDOMIZED_OVERSAMPLES="${RANDOMIZED_OVERSAMPLES:-32}"
  TRAIN_BLOCKS_PER_SESSION="${TRAIN_BLOCKS_PER_SESSION:-4}"
  BASIS_BLOCKS_PER_SESSION="${BASIS_BLOCKS_PER_SESSION:-2}"
  DUPLICATE_AUDIT_ROWS="${DUPLICATE_AUDIT_ROWS:-16}"
  RUN_SELF_TEST="${RUN_SELF_TEST:-0}"
fi

SAMPLING_MODE="${SAMPLING_MODE:-block_spread}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"

mkdir -p "$OUTDIR/logs"
RUN_LOG="$OUTDIR/logs/run_$(date +%Y%m%d_%H%M%S).log"

{
  echo "[$(date '+%F %T')] Strict cross-subject LD/QD/Tensor v4.2.1"
  echo "smoke:          ${SMOKE:-0}"
  echo "preflight only: $PREFLIGHT_ONLY"
  echo "script:         $SCRIPT"
  echo "outdir:         $OUTDIR"
  echo "models:         $MODELS"
  echo "model dims:     $MODEL_DIMS"
  echo "outer folds:    ${FOLDS:-all}"
  echo "covariance:     arithmetic pooled shrinkage"
  echo "alpha grid:     $ALPHA_GRID"
  echo "ridge grid:     $RIDGE_GRID"
  echo "train cap:      $TRAIN_CAP per subject/class"
  echo "basis cap:      $BASIS_CAP per subject/class"
  echo "inner cap:      $INNER_SCORE_CAP per subject/class"
  echo "sampling:       $SAMPLING_MODE"
  echo "train blocks:   $TRAIN_BLOCKS_PER_SESSION per session/class"
  echo "basis blocks:   $BASIS_BLOCKS_PER_SESSION per session/class"
  echo "SVD n_iter:     $RANDOMIZED_N_ITER"
  echo "SVD oversample: $RANDOMIZED_OVERSAMPLES"
  echo "duplicate rows: $DUPLICATE_AUDIT_ROWS per subject"
  echo "threads:        ${OMP_NUM_THREADS}"
} | tee "$RUN_LOG"

if [[ "$RUN_SELF_TEST" == "1" ]]; then
  echo "[$(date '+%F %T')] running algebraic self-test" | tee -a "$RUN_LOG"
  "$PYTHON_BIN" -u "$SCRIPT" --self-test --seed 0 --outdir "$OUTDIR" \
    2>&1 | tee -a "$RUN_LOG"
fi

EXTRA_ARGS=()
if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
  EXTRA_ARGS+=(--preflight-only)
fi

set +e
"$PYTHON_BIN" -u "$SCRIPT" \
  --models "$MODELS" \
  --model-dims "$MODEL_DIMS" \
  --rest-raw-label 1 \
  --expected-n-subjects 27 \
  --outer-folds 4 \
  --cross-inner-folds 3 \
  --folds "$FOLDS" \
  --alpha-grid "$ALPHA_GRID" \
  --ridge-grid "$RIDGE_GRID" \
  --cov-interpolation arithmetic \
  --train-cap-per-subject-class "$TRAIN_CAP" \
  --basis-cap-per-subject-class "$BASIS_CAP" \
  --sampling-mode "$SAMPLING_MODE" \
  --train-blocks-per-session "$TRAIN_BLOCKS_PER_SESSION" \
  --basis-blocks-per-session "$BASIS_BLOCKS_PER_SESSION" \
  --basis-max-gap-rows 16 \
  --basis-max-span-rows 256 \
  --project-max-gap-rows 8 \
  --project-max-span-rows 256 \
  --randomized-n-iter "$RANDOMIZED_N_ITER" \
  --randomized-oversamples "$RANDOMIZED_OVERSAMPLES" \
  --require-requested-dim \
  --inner-max-per-subject-class "$INNER_SCORE_CAP" \
  --duplicate-audit-rows "$DUPLICATE_AUDIT_ROWS" \
  --duplicate-audit-max-gap-rows 0 \
  --duplicate-audit-max-span-rows 1 \
  --fail-on-duplicate-subjects \
  --seed 0 \
  --resume \
  --outdir "$OUTDIR" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee -a "$RUN_LOG"
status=${PIPESTATUS[0]}
set -e

echo "[$(date '+%F %T')] finished status=$status" | tee -a "$RUN_LOG"
exit "$status"
