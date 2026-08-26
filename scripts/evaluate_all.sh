#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 TRIAL_ROOT SCORE_ROOT EVALUATION_ROOT" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
TRIAL_ROOT="$1"
SCORE_ROOT="$2"
EVALUATION_ROOT="$3"

for setting in 50_L5 50_L20 200_L5 200_L20; do
  for task in speaker gender age accent; do
    echo "Evaluating ${setting}/${task}"
    "${PYTHON}" "${ROOT}/experiments/common/evaluate_trials.py" \
      --scored_dir "${SCORE_ROOT}/${setting}/${task}_${setting}" \
      --trial_dir "${TRIAL_ROOT}/${setting}/${task}" \
      --task "${task}" \
      --output_dir "${EVALUATION_ROOT}/${setting}/${task}"
  done
done
