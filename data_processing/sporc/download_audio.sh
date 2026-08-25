#!/usr/bin/env bash
# Redownload only missing audio files
set -uo pipefail

URLS_FILE="${1:?Usage: $0 URLS_FILE [OUTPUT_DIR] [PAUSE_SECONDS]}"
OUT_DIR="${2:-audio_files}"
PAUSE_S="${3:-1.5}"
mkdir -p "$OUT_DIR"

i=0
skipped=0
downloaded=0
failed=0

while IFS= read -r url; do
  [[ -z "${url}" || "${url}" =~ ^# ]] && continue

  i=$((i+1))

  # Determine extension
  ext="mp3"
  if [[ "${url%%\?*}" =~ \.(m4a|mp3|wav|ogg|flac) ]]; then
    ext="${BASH_REMATCH[1]}"
  fi

  out="${OUT_DIR}/$(printf "%03d" "$i")_audio.${ext}"

  # Skip if file already exists and is non-empty
  if [[ -s "$out" ]]; then
    skipped=$((skipped+1))
    continue
  fi

  echo "[$i] Downloading -> $out"
  if curl -L -A "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36" \
       --silent --show-error --fail --connect-timeout 15 --max-time 600 \
       -o "$out" "$url"; then
    downloaded=$((downloaded+1))
  else
    echo " ---> WARNING: Failed to download [$i]. Skipping."
    failed=$((failed+1))
    # Remove empty/partial file from failed download
    rm -f "$out"
  fi

  sleep "$PAUSE_S"

done < "$URLS_FILE"

echo ""
echo "=== Done ==="
echo "Skipped (already existed): $skipped"
echo "Downloaded:                $downloaded"
echo "Failed:                    $failed"
echo "Total URLs processed:      $i"
