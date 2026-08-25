#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 SPORC_METADATA_JSONL [OUTPUT_ROOT]" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
RAW_JSONL="$1"
OUTPUT_ROOT="${2:-${ROOT}/outputs/trials/sporc}"

"${PYTHON}" "${ROOT}/experiments/sporc/generate_topic_controlled_trials.py" \
  --raw_jsonl "${RAW_JSONL}" \
  --output_root "${OUTPUT_ROOT}" \
  --settings 50_L5 50_L20 200_L5 200_L20 \
  --topic_threshold 0.10 \
  --speaker_positive_rss_policy any \
  --speaker_negatives_per_positive 1 \
  --demographic_trials_per_class 2000 \
  --seed 42
