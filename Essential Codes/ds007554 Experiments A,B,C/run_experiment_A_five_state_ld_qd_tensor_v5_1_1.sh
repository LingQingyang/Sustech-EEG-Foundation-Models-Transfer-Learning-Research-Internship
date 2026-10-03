#!/usr/bin/env bash
set -euo pipefail

# Final launcher for Experiment A v5.1.1.
# Uses the patched shared core whose filename remains ld_qd_feature_core_v5_0.py
# for import compatibility.

PYTHON_BIN="${PYTHON_BIN:-python3}"
CORE="${CORE:-/home/lqy/ld_qd_feature_core_v5_0.py}"
SCRIPT="${SCRIPT:-/home/lqy/run_experiment_A_five_state_ld_qd_tensor_v5_1_1.py}"
SMOKE="${SMOKE:-0}"

if [[ "$SMOKE" == "1" ]]; then
  ABC_ROOT="${ABC_ROOT:-/mnt/dataset4/yinuo/FM_flow/dataset/expABC_five_state_integrated_v5_1_1_smoke}"
  OUTDIR="${OUTDIR:-$ABC_ROOT/A}"
  MODELS="${MODELS:-BIOT}"
  SUBJECTS="${SUBJECTS:-9}"
  SVD_RANKS="${SVD_RANKS:-10}"
  BIOT_SVD_RANKS="${BIOT_SVD_RANKS:-$SVD_RANKS}"
  OTHER_SVD_RANKS="${OTHER_SVD_RANKS:-$SVD_RANKS}"
  HELDOUT_SESSIONS="${HELDOUT_SESSIONS:-1}"
  ALPHA_GRID="${ALPHA_GRID:-0.05,0.5,1}"
  RIDGE_GRID="${RIDGE_GRID:-0.01,0.1,1}"
  N_PERMUTATIONS="${N_PERMUTATIONS:-10}"
  RUN_SELF_TEST="${RUN_SELF_TEST:-1}"
else
  ABC_ROOT="${ABC_ROOT:-/mnt/dataset4/yinuo/FM_flow/dataset/expABC_five_state_integrated_v5_1_1}"
  OUTDIR="${OUTDIR:-$ABC_ROOT/A}"
  MODELS="${MODELS:-BIOT LaBraM CBraMod EEGMamba EEGPT}"
  SUBJECTS="${SUBJECTS:-9,25,26}"
  SVD_RANKS="${SVD_RANKS:-100,200,300,500}"
  BIOT_SVD_RANKS="${BIOT_SVD_RANKS:-100,200,256}"
  OTHER_SVD_RANKS="${OTHER_SVD_RANKS:-100,200,300,500}"
  HELDOUT_SESSIONS="${HELDOUT_SESSIONS:-}"
  ALPHA_GRID="${ALPHA_GRID:-0.001,0.01,0.05,0.1,0.25,0.5,0.9,1}"
  RIDGE_GRID="${RIDGE_GRID:-0.0001,0.001,0.01,0.1,1,10,100}"
  N_PERMUTATIONS="${N_PERMUTATIONS:-200}"
  RUN_SELF_TEST="${RUN_SELF_TEST:-0}"
fi

RESUME="${RESUME:-1}"
FAIL_FAST="${FAIL_FAST:-1}"
RUN_COMPILE_CHECK="${RUN_COMPILE_CHECK:-1}"

for required in "$CORE" "$SCRIPT"; do
  [[ -f "$required" ]] || { echo "Missing required file: $required" >&2; exit 2; }
done

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
export PYTHONPATH="$(dirname "$CORE"):${PYTHONPATH:-}"

mkdir -p "$OUTDIR/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="$OUTDIR/logs/launcher_${STAMP}.log"
RUN_LOG="$OUTDIR/logs/run_${STAMP}.log"

{
  echo "[$(date '+%F %T')] Experiment A v5.1.1"
  echo "script: $SCRIPT"
  echo "core: $CORE"
  echo "smoke: $SMOKE"
  echo "models: $MODELS"
  echo "subjects: $SUBJECTS"
  echo "BIOT ranks: $BIOT_SVD_RANKS"
  echo "other ranks: $OTHER_SVD_RANKS"
  echo "heldout: ${HELDOUT_SESSIONS:-all}"
  echo "permutations: $N_PERMUTATIONS"
  echo "resume/fail-fast: $RESUME/$FAIL_FAST"
  echo "outdir: $OUTDIR"
} | tee -a "$MASTER_LOG"

if [[ "$RUN_COMPILE_CHECK" == "1" ]]; then
  "$PYTHON_BIN" -m py_compile "$CORE" "$SCRIPT"
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
  --models "$MODELS"
  --subjects "$SUBJECTS"
  --task-ids MA=0,NB=1,NBMA=5,Full=6
  --expected-sessions 3
  --min-phase-windows 10
  --svd-ranks "$SVD_RANKS"
  "${MODEL_RANK_ARGS[@]}"
  --alpha-grid "$ALPHA_GRID"
  --ridge-grid "$RIDGE_GRID"
  --cov-interpolation arithmetic
  --read-batch-size 512
  --randomized-n-iter 2
  --randomized-oversamples 20
  --n-permutations "$N_PERMUTATIONS"
  --seed 0
  --outdir "$OUTDIR"
)
[[ -n "$HELDOUT_SESSIONS" ]] && CMD+=(--heldout-sessions "$HELDOUT_SESSIONS")
[[ "$RESUME" == "1" ]] && CMD+=(--resume)
[[ "$FAIL_FAST" == "1" ]] && CMD+=(--fail-fast)

set +e
"${CMD[@]}" 2>&1 | tee "$RUN_LOG"
status=${PIPESTATUS[0]}
set -e

echo "[$(date '+%F %T')] Experiment A finished status=$status" | tee -a "$MASTER_LOG"
exit "$status"
