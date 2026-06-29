#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-1}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${REPO_ROOT}/../.venv/bin/python" ]]; then
    PYTHON_BIN="${REPO_ROOT}/../.venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

exec "${PYTHON_BIN}" runners/run_emnlp_full_experiments.py \
  --datasets "${DATASETS:-minerva,chemlex}" \
  --seeds "${SEEDS:-0-29}" \
  --max-rounds "${MAX_ROUNDS:-10}" \
  --output-root "${OUTPUT_ROOT:-results/care_main_replay_30x10}" \
  --baseline-workers "${BASELINE_WORKERS:-28}" \
  --api-workers "${API_WORKERS:-20}" \
  --baseline-executor "${BASELINE_EXECUTOR:-process}" \
  --api-executor "${API_EXECUTOR:-thread}" \
  "$@"
