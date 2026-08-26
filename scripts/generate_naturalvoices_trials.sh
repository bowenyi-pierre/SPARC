#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
OUTPUT_ROOT="${1:-${ROOT}/outputs/trials/naturalvoices}"
GENERATOR="${ROOT}/experiments/naturalvoices/generate_trials.py"

settings=(
  "50 5"
  "50 20"
  "200 5"
  "200 20"
)

for item in "${settings[@]}"; do
  read -r probe_words enrollment_length <<<"${item}"
  setting="${probe_words}_L${enrollment_length}"
  echo "Generating NaturalVoices ${setting}"
  "${PYTHON}" "${GENERATOR}" \
    --output_dir "${OUTPUT_ROOT}/${setting}" \
    --probe_words "${probe_words}" \
    --L "${enrollment_length}" \
    --max_speakers 2000 \
    --seed 42
done
