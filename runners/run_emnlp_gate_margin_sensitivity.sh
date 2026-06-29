#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
cd "${REPO_ROOT}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-1}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

exec "${PYTHON_BIN}" runners/run_emnlp_gate_margin_sensitivity.py \
  --datasets "${DATASETS:-minerva,chemlex}" \
  --seeds "${SEEDS:-0-9}" \
  --max-rounds "${MAX_ROUNDS:-10}" \
  --margins="${MARGINS:--0.10,0.10,0.20}" \
  --output-root "${OUTPUT_ROOT:-results/care_gate_margin_sensitivity_10x10}" \
  --api-workers "${API_WORKERS:-8}" \
  --executor "${API_EXECUTOR:-thread}" \
  --default-care-root "${DEFAULT_CARE_ROOT:-${REPO_ROOT}/results/care_main_replay_30x10}" \
  "$@"
