#!/usr/bin/env bash
set -euo pipefail

# Final integrated launcher for:
#   A v5.1.1 + patched shared core v5.1.1
#   B v5.1
#   C v5.1
#   integrator v5.0
#
# The integrated launcher delegates A/B/C execution to the three final
# standalone launchers, keeping one parameter definition per experiment.

PYTHON_BIN="${PYTHON_BIN:-python3}"
CORE="${CORE:-/home/lqy/ld_qd_feature_core_v5_0.py}"
A_SCRIPT="${A_SCRIPT:-/home/lqy/run_experiment_A_five_state_ld_qd_tensor_v5_1_1.py}"
B_SCRIPT="${B_SCRIPT:-/home/lqy/run_experiment_B_five_state_within_global_sfa_v5_1.py}"
C_SCRIPT="${C_SCRIPT:-/home/lqy/run_experiment_C_slowspace_ld_qd_tensor_v5_1.py}"
INTEGRATOR="${INTEGRATOR:-/home/lqy/integrate_experiments_ABC_five_state_v5_0.py}"
A_LAUNCHER="${A_LAUNCHER:-/home/lqy/run_experiment_A_five_state_ld_qd_tensor_v5_1_1.sh}"
B_LAUNCHER="${B_LAUNCHER:-/home/lqy/run_experiment_B_five_state_within_global_sfa_v5_1.sh}"
C_LAUNCHER="${C_LAUNCHER:-/home/lqy/run_experiment_C_slowspace_ld_qd_tensor_v5_1.sh}"
SMOKE="${SMOKE:-0}"

if [[ "$SMOKE" == "1" ]]; then
  ABC_ROOT="${ABC_ROOT:-/mnt/dataset4/yinuo/FM_flow/dataset/expABC_five_state_integrated_v5_1_1_smoke}"
  MODELS_STR="${MODELS:-BIOT}"
  SUBJECTS="${SUBJECTS:-9}"
  SVD_RANKS="${SVD_RANKS:-10}"
  BIOT_SVD_RANKS="${BIOT_SVD_RANKS:-$SVD_RANKS}"
  OTHER_SVD_RANKS="${OTHER_SVD_RANKS:-$SVD_RANKS}"
  MAIN_RANK="${MAIN_RANK:-10}"
  HELDOUT_SESSIONS="${HELDOUT_SESSIONS:-1}"
  ALPHA_GRID="${ALPHA_GRID:-0.05,0.5,1}"
  RIDGE_GRID="${RIDGE_GRID:-0.01,0.1,1}"
  A_PERMUTATIONS="${A_PERMUTATIONS:-10}"
  B_TRAIN_SHUFFLES="${B_TRAIN_SHUFFLES:-24}"
  B_HELDOUT_SHUFFLES="${B_HELDOUT_SHUFFLES:-24}"
  C_PERMUTATIONS="${C_PERMUTATIONS:-10}"
  TOPK="${TOPK:-1,2,5}"
  C_HANDOFF_TOPK="${C_HANDOFF_TOPK:-5}"
  RUN_SELF_TEST="${RUN_SELF_TEST:-1}"
else
  ABC_ROOT="${ABC_ROOT:-/mnt/dataset4/yinuo/FM_flow/dataset/expABC_five_state_integrated_v5_1_1}"
  MODELS_STR="${MODELS:-BIOT LaBraM CBraMod EEGMamba EEGPT}"
  SUBJECTS="${SUBJECTS:-9,25,26}"
  SVD_RANKS="${SVD_RANKS:-100,200,300,500}"
  BIOT_SVD_RANKS="${BIOT_SVD_RANKS:-100,200,256}"
  OTHER_SVD_RANKS="${OTHER_SVD_RANKS:-100,200,300,500}"
  MAIN_RANK="${MAIN_RANK:-200}"
  HELDOUT_SESSIONS="${HELDOUT_SESSIONS:-}"
  ALPHA_GRID="${ALPHA_GRID:-0.001,0.01,0.05,0.1,0.25,0.5,0.9,1}"
  RIDGE_GRID="${RIDGE_GRID:-0.0001,0.001,0.01,0.1,1,10,100}"
  A_PERMUTATIONS="${A_PERMUTATIONS:-200}"
  B_TRAIN_SHUFFLES="${B_TRAIN_SHUFFLES:-100}"
  B_HELDOUT_SHUFFLES="${B_HELDOUT_SHUFFLES:-200}"
  C_PERMUTATIONS="${C_PERMUTATIONS:-200}"
  TOPK="${TOPK:-1,2,3,5,10,20,50,100}"
  C_HANDOFF_TOPK="${C_HANDOFF_TOPK:-20}"
  RUN_SELF_TEST="${RUN_SELF_TEST:-0}"
fi

RUN_A="${RUN_A:-1}"
RUN_B="${RUN_B:-1}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
RUN_C="${RUN_C:-1}"
RUN_INTEGRATE="${RUN_INTEGRATE:-1}"
AXIS_SOURCES="${AXIS_SOURCES:-slow_prefix,fast_suffix,pca_prefix,ambient_svd}"
N_RANDOM_SUBSPACES="${N_RANDOM_SUBSPACES:-0}"
MAX_CANDIDATE_RANK="${MAX_CANDIDATE_RANK:-0}"

A_ROOT="${A_ROOT:-$ABC_ROOT/A}"
B_ROOT="${B_ROOT:-$ABC_ROOT/B}"
C_ROOT="${C_ROOT:-$ABC_ROOT/C}"
PREFLIGHT_ROOT="${PREFLIGHT_ROOT:-$ABC_ROOT/preflight}"
INTEGRATED_ROOT="${INTEGRATED_ROOT:-$ABC_ROOT/integrated}"
TEMP_ROOT="${TEMP_ROOT:-$ABC_ROOT/.tmp_B}"
LOG_ROOT="$ABC_ROOT/logs"

for required in \
  "$CORE" "$A_SCRIPT" "$B_SCRIPT" "$C_SCRIPT" "$INTEGRATOR" \
  "$A_LAUNCHER" "$B_LAUNCHER" "$C_LAUNCHER"; do
  [[ -f "$required" ]] || { echo "Missing required file: $required" >&2; exit 2; }
done

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
export PYTHONPATH="$(dirname "$CORE"):${PYTHONPATH:-}"

mkdir -p "$LOG_ROOT" "$TEMP_ROOT"
MASTER_LOG="$LOG_ROOT/pipeline_$(date +%Y%m%d_%H%M%S).log"

run_logged() {
  local label="$1"
  shift
  local logfile="$LOG_ROOT/${label}_$(date +%Y%m%d_%H%M%S).log"
  echo "[$(date '+%F %T')] START $label" | tee -a "$MASTER_LOG"
  set +e
  "$@" 2>&1 | tee "$logfile"
  local status=${PIPESTATUS[0]}
  set -e
  echo "[$(date '+%F %T')] END $label status=$status" | tee -a "$MASTER_LOG"
  [[ $status -eq 0 ]] || exit "$status"
}

{
  echo "[$(date '+%F %T')] Integrated five-state ABC v5.1.1 final"
  echo "root: $ABC_ROOT"
  echo "core: $CORE"
  echo "A: $A_SCRIPT"
  echo "B: $B_SCRIPT"
  echo "C: $C_SCRIPT"
  echo "integrator: $INTEGRATOR"
  echo "models: $MODELS_STR"
  echo "subjects: $SUBJECTS"
  echo "BIOT ranks: $BIOT_SVD_RANKS"
  echo "other ranks: $OTHER_SVD_RANKS"
  echo "heldout: ${HELDOUT_SESSIONS:-all}"
  echo "axis sources: $AXIS_SOURCES"
  echo "B covariance: exact empirical support, no loading"
  echo "B temporal null: vector-wise within-state-stratified permutation"
  echo "B->C fixed top-k: $C_HANDOFF_TOPK"
  echo "run A/B/P/C/I: $RUN_A/$RUN_B/$RUN_PREFLIGHT/$RUN_C/$RUN_INTEGRATE"
} | tee -a "$MASTER_LOG"

if [[ "$RUN_SELF_TEST" == "1" ]]; then
  run_logged integrator_self_test "$PYTHON_BIN" "$INTEGRATOR" --self-test
fi

if [[ "$RUN_A" == "1" ]]; then
  run_logged A env \
    PYTHON_BIN="$PYTHON_BIN" CORE="$CORE" SCRIPT="$A_SCRIPT" \
    SMOKE="$SMOKE" ABC_ROOT="$ABC_ROOT" OUTDIR="$A_ROOT" \
    MODELS="$MODELS_STR" SUBJECTS="$SUBJECTS" SVD_RANKS="$SVD_RANKS" \
    BIOT_SVD_RANKS="$BIOT_SVD_RANKS" OTHER_SVD_RANKS="$OTHER_SVD_RANKS" \
    HELDOUT_SESSIONS="$HELDOUT_SESSIONS" ALPHA_GRID="$ALPHA_GRID" \
    RIDGE_GRID="$RIDGE_GRID" N_PERMUTATIONS="$A_PERMUTATIONS" \
    RUN_SELF_TEST="$RUN_SELF_TEST" RESUME=1 FAIL_FAST=1 \
    bash "$A_LAUNCHER"
fi

if [[ "$RUN_B" == "1" ]]; then
  run_logged B env \
    PYTHON_BIN="$PYTHON_BIN" SCRIPT="$B_SCRIPT" \
    SMOKE="$SMOKE" ABC_ROOT="$ABC_ROOT" OUTDIR="$B_ROOT" TEMPDIR="$TEMP_ROOT" \
    MODELS="$MODELS_STR" SUBJECTS="$SUBJECTS" SVD_RANKS="$SVD_RANKS" \
    BIOT_SVD_RANKS="$BIOT_SVD_RANKS" OTHER_SVD_RANKS="$OTHER_SVD_RANKS" \
    MAIN_RANK="$MAIN_RANK" HELDOUT_SESSIONS="$HELDOUT_SESSIONS" \
    N_TRAIN_SHUFFLE="$B_TRAIN_SHUFFLES" N_HELDOUT_SHUFFLE="$B_HELDOUT_SHUFFLES" \
    TOPK="$TOPK" C_HANDOFF_TOPK="$C_HANDOFF_TOPK" \
    RUN_SELF_TEST="$RUN_SELF_TEST" RESUME=1 FAIL_FAST=1 SKIP_PLOTS=1 \
    bash "$B_LAUNCHER"
fi

if [[ "$RUN_PREFLIGHT" == "1" ]]; then
  run_logged preflight "$PYTHON_BIN" "$INTEGRATOR" \
    --a-root "$A_ROOT" --b-root "$B_ROOT" \
    --outdir "$PREFLIGHT_ROOT" --preflight
fi

if [[ "$RUN_C" == "1" ]]; then
  run_logged C env \
    PYTHON_BIN="$PYTHON_BIN" CORE="$CORE" SCRIPT="$C_SCRIPT" \
    SMOKE="$SMOKE" ABC_ROOT="$ABC_ROOT" A_ROOT="$A_ROOT" B_ROOT="$B_ROOT" OUTDIR="$C_ROOT" \
    MODELS="$MODELS_STR" SUBJECTS="$SUBJECTS" HELDOUT_SESSIONS="$HELDOUT_SESSIONS" \
    ALPHA_GRID="$ALPHA_GRID" RIDGE_GRID="$RIDGE_GRID" N_PERMUTATIONS="$C_PERMUTATIONS" \
    AXIS_SOURCES="$AXIS_SOURCES" N_RANDOM_SUBSPACES="$N_RANDOM_SUBSPACES" \
    MAX_CANDIDATE_RANK="$MAX_CANDIDATE_RANK" \
    RUN_SELF_TEST="$RUN_SELF_TEST" RESUME=1 FAIL_FAST=1 \
    bash "$C_LAUNCHER"
fi

if [[ "$RUN_INTEGRATE" == "1" ]]; then
  run_logged integrate "$PYTHON_BIN" "$INTEGRATOR" \
    --a-root "$A_ROOT" --b-root "$B_ROOT" --c-root "$C_ROOT" \
    --outdir "$INTEGRATED_ROOT" --strict
fi

echo "[$(date '+%F %T')] pipeline complete" | tee -a "$MASTER_LOG"
