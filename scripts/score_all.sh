#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 DATASET TRIAL_ROOT SCORE_ROOT MODEL [MODEL ...]" >&2
  echo "DATASET must be naturalvoices or sporc." >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
DATASET="$1"
TRIAL_ROOT="$2"
SCORE_ROOT="$3"
shift 3
MODELS=("$@")

case "${DATASET}" in
  naturalvoices|sporc) ;;
  *) echo "Unknown dataset: ${DATASET}" >&2; exit 2 ;;
esac

for setting in 50_L5 50_L20 200_L5 200_L20; do
  for task in speaker gender age accent; do
    echo "Scoring ${DATASET}/${setting}/${task}: ${MODELS[*]}"
    "${PYTHON}" "${ROOT}/experiments/common/score_trials.py" \
      --trial_dir "${TRIAL_ROOT}/${setting}/${task}" \
      --output_dir "${SCORE_ROOT}/${setting}" \
      --models "${MODELS[@]}" \
      --task "${task}" \
      --llm_mode score \
      --pause_s "${PAUSE_S:-0}"
  done
done
