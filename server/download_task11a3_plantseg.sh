#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:?usage: download_task11a3_plantseg.sh OUTPUT_ROOT [USE_TURBO=1] [RESUME=0]}
USE_TURBO=${2:-1}
RESUME=${3:-0}
URL='https://zenodo.org/records/17719108/files/plantseg.zip?download=1'
EXPECTED_SIZE=1057281724
EXPECTED_MD5='9358a66dff88cdd15c4fe009763c40a3'
PARTIAL="$ROOT/plantseg.zip.partial"
FINAL="$ROOT/plantseg.zip"
STATUS="$ROOT/status.json"
LOG="$ROOT/download.log"

if [[ -e "$ROOT" ]]; then
  if [[ "$RESUME" != 1 || ! -f "$PARTIAL" || -e "$FINAL" ]]; then
    echo "BLOCK: unsafe existing output root: $ROOT" >&2
    exit 2
  fi
  if [[ -f "$ROOT/failure.json" ]]; then
    mv "$ROOT/failure.json" "$ROOT/failure.pre_resume.$(date +%Y%m%dT%H%M%S).json"
  fi
else
  mkdir -p "$ROOT"
fi
exec > >(tee -a "$LOG") 2>&1

failed() {
  code=$?
  if [[ $code -ne 0 ]]; then
    printf '{"state":"failed","exit_code":%d}\n' "$code" > "$ROOT/failure.json"
    printf '{"state":"failed","stage":"download_or_verify"}\n' > "$STATUS.tmp"
    mv "$STATUS.tmp" "$STATUS"
  fi
  exit "$code"
}
trap failed EXIT

printf '{"state":"running","stage":"download"}\n' > "$STATUS.tmp"
mv "$STATUS.tmp" "$STATUS"
if [[ "$USE_TURBO" == 1 ]]; then
  source /etc/network_turbo
fi

curl \
  --fail \
  --location \
  --continue-at - \
  --retry 12 \
  --retry-all-errors \
  --retry-delay 10 \
  --connect-timeout 30 \
  --output "$PARTIAL" \
  "$URL"

actual_size=$(stat -c '%s' "$PARTIAL")
if [[ "$actual_size" != "$EXPECTED_SIZE" ]]; then
  echo "size mismatch: expected=$EXPECTED_SIZE actual=$actual_size" >&2
  exit 3
fi

actual_md5=$(md5sum "$PARTIAL" | awk '{print $1}')
if [[ "$actual_md5" != "$EXPECTED_MD5" ]]; then
  echo "MD5 mismatch: expected=$EXPECTED_MD5 actual=$actual_md5" >&2
  exit 4
fi

printf '{"state":"running","stage":"sha256"}\n' > "$STATUS.tmp"
mv "$STATUS.tmp" "$STATUS"
actual_sha256=$(sha256sum "$PARTIAL" | awk '{print $1}')
mv "$PARTIAL" "$FINAL"

cat > "$ROOT/download_report.json" <<EOF
{
  "version": "task11a3-plantseg-download-report-1",
  "state": "completed",
  "record_id": "17719108",
  "doi": "10.5281/zenodo.17719108",
  "url": "$URL",
  "license": "CC-BY-NC-4.0",
  "file": "plantseg.zip",
  "size": $actual_size,
  "md5": "$actual_md5",
  "sha256": "$actual_sha256"
}
EOF
printf '%s  plantseg.zip\n' "$actual_sha256" > "$ROOT/completion.sha256"
printf '{"state":"completed","stage":"verified"}\n' > "$STATUS.tmp"
mv "$STATUS.tmp" "$STATUS"
trap - EXIT
echo "PLANTSEG_DOWNLOAD_VERIFIED sha256=$actual_sha256"
