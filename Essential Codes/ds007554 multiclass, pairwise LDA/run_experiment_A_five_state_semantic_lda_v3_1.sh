#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT="${SCRIPT:-/home/lqy/run_experiment_A_five_state_semantic_lda_v3_1.py}"
OUTDIR="${OUTDIR:-/mnt/dataset4/yinuo/FM_flow/dataset/expA_five_state_semantic_lda_v3_1}"
SUBJECTS="${SUBJECTS:-9,25,26}"
MODELS_STR="${MODELS:-BIOT LaBraM CBraMod EEGMamba EEGPT}"
read -r -a MODELS_ARR <<< "$MODELS_STR"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"

mkdir -p "$OUTDIR/logs"
MASTER_LOG="$OUTDIR/logs/launcher_$(date +%Y%m%d_%H%M%S).log"

{
  echo "[$(date '+%F %T')] Experiment A v3.1: five-state full-context semantic LDA"
  echo "script: $SCRIPT"
  echo "outdir: $OUTDIR"
  echo "models: ${MODELS_ARR[*]}"
  echo "subjects: $SUBJECTS"
  echo "SVD ranks: 100,200,300,500"
} | tee -a "$MASTER_LOG"

failures=0
for MODEL in "${MODELS_ARR[@]}"; do
  MODEL_LOG="$OUTDIR/logs/${MODEL}_$(date +%Y%m%d_%H%M%S).log"
  echo "[$(date '+%F %T')] starting $MODEL" | tee -a "$MASTER_LOG"
  set +e
  "$PYTHON_BIN" -u "$SCRIPT" \
    --models "$MODEL" \
    --subjects "$SUBJECTS" \
    --task-ids MA=0,NB=1,NBMA=5,Full=6 \
    --vectorization flatten \
    --run-start-guard-sec 5 \
    --baseline-guard-sec 2 \
    --task-guard-sec 10 \
    --task-end-guard-sec 5 \
    --min-windows-per-segment 10 \
    --svd-ranks 100,200,300,500 \
    --main-svd-rank 200 \
    --main-shrinkage 0.9 \
    --sensitivity-shrinkage 0.7,0.99 \
    --n-permutations "${N_PERMUTATIONS:-200}" \
    --degeneracy-gap 0.10 \
    --dimension-alpha 0.05 \
    --outdir "$OUTDIR" \
    --fail-fast \
    2>&1 | tee "$MODEL_LOG"
  status=${PIPESTATUS[0]}
  set -e
  if [[ $status -ne 0 ]]; then
    failures=$((failures + 1))
    echo "[$(date '+%F %T')] $MODEL FAILED status=$status" | tee -a "$MASTER_LOG"
  else
    echo "[$(date '+%F %T')] $MODEL complete" | tee -a "$MASTER_LOG"
  fi
done

echo "[$(date '+%F %T')] finished; failures=$failures" | tee -a "$MASTER_LOG"
exit "$failures"
