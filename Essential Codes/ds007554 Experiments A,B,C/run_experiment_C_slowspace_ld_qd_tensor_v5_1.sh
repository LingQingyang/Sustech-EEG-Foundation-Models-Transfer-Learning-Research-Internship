#!/usr/bin/env bash
set -euo pipefail

# Final launcher for Experiment C v5.1.

PYTHON_BIN="${PYTHON_BIN:-python3}"
CORE="${CORE:-/home/lqy/ld_qd_feature_core_v5_0.py}"
SCRIPT="${SCRIPT:-/home/lqy/run_experiment_C_slowspace_ld_qd_tensor_v5_1.py}"
SMOKE="${SMOKE:-0}"

if [[ "$SMOKE" == "1" ]]; then
  ABC_ROOT="${ABC_ROOT:-/mnt/dataset4/yinuo/FM_flow/dataset/expABC_five_state_integrated_v5_1_1_smoke}"
  A_ROOT="${A_ROOT:-$ABC_ROOT/A}"
  B_ROOT="${B_ROOT:-$ABC_ROOT/B}"
  OUTDIR="${OUTDIR:-$ABC_ROOT/C}"
  MODELS="${MODELS:-BIOT}"
  SUBJECTS="${SUBJECTS:-9}"
  SVD_RANKS="${SVD_RANKS:-10}"
  HELDOUT_SESSIONS="${HELDOUT_SESSIONS:-1}"
  ALPHA_GRID="${ALPHA_GRID:-0.05,0.5,1}"
  RIDGE_GRID="${RIDGE_GRID:-0.01,0.1,1}"
  N_PERMUTATIONS="${N_PERMUTATIONS:-10}"
  RUN_SELF_TEST="${RUN_SELF_TEST:-1}"
else
  ABC_ROOT="${ABC_ROOT:-/mnt/dataset4/yinuo/FM_flow/dataset/expABC_five_state_integrated_v5_1_1}"
  A_ROOT="${A_ROOT:-$ABC_ROOT/A}"
  B_ROOT="${B_ROOT:-$ABC_ROOT/B}"
  OUTDIR="${OUTDIR:-$ABC_ROOT/C}"
  MODELS="${MODELS:-BIOT LaBraM CBraMod EEGMamba EEGPT}"
  SUBJECTS="${SUBJECTS:-9,25,26}"
  SVD_RANKS="${SVD_RANKS:-}"
  HELDOUT_SESSIONS="${HELDOUT_SESSIONS:-}"
  ALPHA_GRID="${ALPHA_GRID:-0.001,0.01,0.05,0.1,0.25,0.5,0.9,1}"
  RIDGE_GRID="${RIDGE_GRID:-0.0001,0.001,0.01,0.1,1,10,100}"
  N_PERMUTATIONS="${N_PERMUTATIONS:-200}"
  RUN_SELF_TEST="${RUN_SELF_TEST:-0}"
fi

AXIS_SOURCES="${AXIS_SOURCES:-slow_prefix,fast_suffix,pca_prefix,ambient_svd}"
N_RANDOM_SUBSPACES="${N_RANDOM_SUBSPACES:-0}"
MAX_CANDIDATE_RANK="${MAX_CANDIDATE_RANK:-0}"
REQUIRE_A_ALIGNMENT="${REQUIRE_A_ALIGNMENT:-0}"
REQUIRE_AT_LEAST_ONE_COMPLETED="${REQUIRE_AT_LEAST_ONE_COMPLETED:-1}"
RESUME="${RESUME:-1}"
FAIL_FAST="${FAIL_FAST:-1}"
RUN_COMPILE_CHECK="${RUN_COMPILE_CHECK:-1}"

for required in "$CORE" "$SCRIPT"; do
  [[ -f "$required" ]] || { echo "Missing required file: $required" >&2; exit 2; }
done
[[ -d "$A_ROOT" ]] || { echo "Missing A root: $A_ROOT" >&2; exit 2; }
[[ -d "$B_ROOT" ]] || { echo "Missing B root: $B_ROOT" >&2; exit 2; }

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
  echo "[$(date '+%F %T')] Experiment C v5.1"
  echo "script: $SCRIPT"
  echo "core: $CORE"
  echo "smoke: $SMOKE"
  echo "A root: $A_ROOT"
  echo "B root: $B_ROOT"
  echo "models: $MODELS"
  echo "subjects: $SUBJECTS"
  echo "ranks: ${SVD_RANKS:-all B handoffs}"
  echo "heldout: ${HELDOUT_SESSIONS:-all}"
  echo "axis sources: $AXIS_SOURCES"
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

CMD=(
  "$PYTHON_BIN" -u "$SCRIPT"
  --a-root "$A_ROOT"
  --b-root "$B_ROOT"
  --models "$MODELS"
  --subjects "$SUBJECTS"
  --grids raw
  --axis-sources "$AXIS_SOURCES"
  --n-random-subspaces "$N_RANDOM_SUBSPACES"
  --alpha-grid "$ALPHA_GRID"
  --ridge-grid "$RIDGE_GRID"
  --cov-interpolation arithmetic
  --max-candidate-rank "$MAX_CANDIDATE_RANK"
  --n-permutations "$N_PERMUTATIONS"
  --seed 0
  --outdir "$OUTDIR"
)
[[ -n "$SVD_RANKS" ]] && CMD+=(--svd-ranks "$SVD_RANKS")
[[ -n "$HELDOUT_SESSIONS" ]] && CMD+=(--heldout-sessions "$HELDOUT_SESSIONS")
[[ "$REQUIRE_A_ALIGNMENT" == "1" ]] && CMD+=(--require-a-alignment)
[[ "$REQUIRE_AT_LEAST_ONE_COMPLETED" == "1" ]] && CMD+=(--require-at-least-one-completed)
[[ "$RESUME" == "1" ]] && CMD+=(--resume)
[[ "$FAIL_FAST" == "1" ]] && CMD+=(--fail-fast)

set +e
"${CMD[@]}" 2>&1 | tee "$RUN_LOG"
status=${PIPESTATUS[0]}
set -e

echo "[$(date '+%F %T')] Experiment C finished status=$status" | tee -a "$MASTER_LOG"
exit "$status"
