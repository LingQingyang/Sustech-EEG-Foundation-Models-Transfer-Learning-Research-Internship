#!/usr/bin/env bash
set -euo pipefail

if [[ -d "$HOME/venvs/labram/bin" ]]; then
  export PATH="$HOME/venvs/labram/bin:$PATH"
fi

ANALYSIS_DIR="${ANALYSIS_DIR:-/omni-eeg-01/task calibration/results/tsv_labram_m3cv/tsv_labram_m3cv_v5_0/analysis_four_spectra_v6_0}"
OUTDIR="${OUTDIR:-${ANALYSIS_DIR}/redraw_v6_1_png_only}"
PYFILE="${PYFILE:-$HOME/redraw_labram_m3cv_four_spectra_v6_1_nopandas.py}"

python3 "$PYFILE" \
  --analysis-dir "$ANALYSIS_DIR" \
  --outdir "$OUTDIR" \
  --shared-q 0.95 \
  --overwrite
