#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash run.sh <selftest|preflight|train|audit|spectra|heatmap|sti|plot|all> [options]" >&2
  exit 2
fi

python3 -m labram_tsv.run "$@"
