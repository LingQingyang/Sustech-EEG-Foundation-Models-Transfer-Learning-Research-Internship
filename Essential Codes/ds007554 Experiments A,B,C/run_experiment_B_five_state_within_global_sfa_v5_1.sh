#!/usr/bin/env bash
set -euo pipefail

# Final launcher for Experiment B v5.1.

PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT="${SCRIPT:-/home/lqy/run_experiment_B_five_state_within_global_sfa_v5_1.py}"
SMOKE="${SMOKE:-0}"

if [[ "$SMOKE" == "1" ]]; then
  ABC_ROOT="${ABC_ROOT:-/mnt/dataset4/yinuo/FM_flow/dataset/expABC_five_state_integrated_v5_1_1_smoke}"
  OUTDIR="${OUTDIR:-$ABC_ROOT/B}"
  TEMPDIR="${TEMPDIR:-$ABC_ROOT/.tmp_B}"
  MODELS="${MODELS:-BIOT}"
  SUBJECTS="${SUBJECTS:-9}"
  SVD_RANKS="${SVD_RANKS:-10}"
  BIOT_SVD_RANKS="${BIOT_SVD_RANKS:-$SVD_RANKS}"
  OTHER_SVD_RANKS="${OTHER_SVD_RANKS:-$SVD_RANKS}"
  MAIN_RANK="${MAIN_RANK:-10}"
  HELDOUT_SESSIONS="${HELDOUT_SESSIONS:-1}"
  N_TRAIN_SHUFFLE="${N_TRAIN_SHUFFLE:-24}"
  N_HELDOUT_SHUFFLE="${N_HELDOUT_SHUFFLE:-24}"
  TOPK="${TOPK:-1,2,5}"
  C_HANDOFF_TOPK="${C_HANDOFF_TOPK:-5}"
  RUN_SELF_TEST="${RUN_SELF_TEST:-1}"
else
  ABC_ROOT="${ABC_ROOT:-/mnt/dataset4/yinuo/FM_flow/dataset/expABC_five_state_integrated_v5_1_1}"
  OUTDIR="${OUTDIR:-$ABC_ROOT/B}"
  TEMPDIR="${TEMPDIR:-$ABC_ROOT/.tmp_B}"
  MODELS="${MODELS:-BIOT LaBraM CBraMod EEGMamba EEGPT}"
  SUBJECTS="${SUBJECTS:-9,25,26}"
  SVD_RANKS="${SVD_RANKS:-100,200,300,500}"
  BIOT_SVD_RANKS="${BIOT_SVD_RANKS:-100,200,256}"
  OTHER_SVD_RANKS="${OTHER_SVD_RANKS:-100,200,300,500}"
  MAIN_RANK="${MAIN_RANK:-200}"
  HELDOUT_SESSIONS="${HELDOUT_SESSIONS:-}"
  N_TRAIN_SHUFFLE="${N_TRAIN_SHUFFLE:-100}"
  N_HELDOUT_SHUFFLE="${N_HELDOUT_SHUFFLE:-200}"
  TOPK="${TOPK:-1,2,3,5,10,20,50,100}"
  C_HANDOFF_TOPK="${C_HANDOFF_TOPK:-20}"
  RUN_SELF_TEST="${RUN_SELF_TEST:-0}"
fi

GRIDS="${GRIDS:-raw}"
MAX_COEFFICIENT_GLOBAL_AXES="${MAX_COEFFICIENT_GLOBAL_AXES:-20}"
SKIP_PLOTS="${SKIP_PLOTS:-1}"
RESUME="${RESUME:-1}"
FAIL_FAST="${FAIL_FAST:-1}"
RUN_COMPILE_CHECK="${RUN_COMPILE_CHECK:-1}"

[[ -f "$SCRIPT" ]] || { echo "Missing required file: $SCRIPT" >&2; exit 2; }

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"

mkdir -p "$OUTDIR/logs" "$TEMPDIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="$OUTDIR/logs/launcher_${STAMP}.log"
RUN_LOG="$OUTDIR/logs/run_${STAMP}.log"
read -r -a MODELS_ARR <<< "$MODELS"

{
  echo "[$(date '+%F %T')] Experiment B v5.1"
  echo "script: $SCRIPT"
  echo "smoke: $SMOKE"
  echo "models: ${MODELS_ARR[*]}"
  echo "subjects: $SUBJECTS"
  echo "BIOT ranks: $BIOT_SVD_RANKS"
  echo "other ranks: $OTHER_SVD_RANKS"
  echo "main rank: $MAIN_RANK"
  echo "heldout: ${HELDOUT_SESSIONS:-all}"
  echo "grids: $GRIDS"
  echo "surrogates train/heldout: $N_TRAIN_SHUFFLE/$N_HELDOUT_SHUFFLE"
  echo "C handoff top-k: $C_HANDOFF_TOPK"
  echo "resume/fail-fast: $RESUME/$FAIL_FAST"
  echo "outdir: $OUTDIR"
  echo "tempdir: $TEMPDIR"
} | tee -a "$MASTER_LOG"

if [[ "$RUN_COMPILE_CHECK" == "1" ]]; then
  "$PYTHON_BIN" -m py_compile "$SCRIPT"
fi
if [[ "$RUN_SELF_TEST" == "1" ]]; then
  "$PYTHON_BIN" "$SCRIPT" --self-test 2>&1 | tee -a "$MASTER_LOG"
fi

MODEL_RANK_ARGS=(
  --model-svd-ranks "BIOT=$BIOT_SVD_RANKS"
  --model-svd-ranks "LaBraM=$OTHER_SVD_RANKS"
  --model-svd-ranks "CBraMod=$OTHER_SVD_RANKS"
  --model-svd-ranks "EEGMamba=$OTHER_SVD_RANKS"
  --model-svd-ranks "EEGPT=$OTHER_SVD_RANKS"
)

CMD=(
  "$PYTHON_BIN" -u "$SCRIPT"
  --models "${MODELS_ARR[@]}"
  --subjects "$SUBJECTS"
  --expected-sessions 3
  --task-ids MA=0,NB=1,NBMA=5,Full=6
  --vectorization flatten
  --grids "$GRIDS"
  --min-phase-windows-raw 10
  --min-adjacencies-per-run 5
  --svd-ranks "$SVD_RANKS"
  "${MODEL_RANK_ARGS[@]}"
  --main-svd-rank "$MAIN_RANK"
  --svd-oversamples 24
  --svd-power-iterations 2
  --cov-shrinkage 0
  --slow-advantage-coverage 0.99
  --training-axis-alpha 0.05
  --max-retained-axes 500
  --degeneracy-gap 0.10
  --c-handoff-top-k "$C_HANDOFF_TOPK"
  --top-k "$TOPK"
  --n-train-shuffle "$N_TRAIN_SHUFFLE"
  --n-heldout-shuffle "$N_HELDOUT_SHUFFLE"
  --max-coefficient-global-axes "$MAX_COEFFICIENT_GLOBAL_AXES"
  --read-batch-size 512
  --tempdir "$TEMPDIR"
  --seed 0
  --outdir "$OUTDIR"
)
[[ -n "$HELDOUT_SESSIONS" ]] && CMD+=(--heldout-sessions "$HELDOUT_SESSIONS")
[[ "$SKIP_PLOTS" == "1" ]] && CMD+=(--skip-plots)
[[ "$RESUME" == "1" ]] && CMD+=(--resume)
[[ "$FAIL_FAST" == "1" ]] && CMD+=(--fail-fast)

set +e
"${CMD[@]}" 2>&1 | tee "$RUN_LOG"
status=${PIPESTATUS[0]}
set -e

echo "[$(date '+%F %T')] Experiment B finished status=$status" | tee -a "$MASTER_LOG"
exit "$status"
