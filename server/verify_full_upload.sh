#!/usr/bin/env bash
set -euo pipefail

if (( $# < 3 || $# > 5 )); then
  echo "usage: $0 INCOMING_DIR MANIFEST_TSV REPORT_DIR [POLL_SECONDS] [MAX_WAIT_SECONDS]" >&2
  exit 64
fi

incoming_dir=$1
manifest_path=$2
report_dir=$3
poll_seconds=${4:-60}
max_wait_seconds=${5:-64800}

if [[ ! -d "$incoming_dir" || ! -f "$manifest_path" ]]; then
  echo 'incoming directory or manifest is missing' >&2
  exit 66
fi
if [[ ! "$poll_seconds" =~ ^[0-9]+$ || ! "$max_wait_seconds" =~ ^[0-9]+$ ]]; then
  echo 'poll and timeout values must be non-negative integers' >&2
  exit 64
fi

mkdir -p "$report_dir"
verified_marker="$report_dir/full_upload.verified"
success_report="$report_dir/full_upload.sha256.tsv"
failure_report="$report_dir/full_upload.failure.tsv"
rm -f "$verified_marker" "$failure_report"

validate_manifest() {
  local name expected_size expected_hash row_count=0
  while IFS=$'\t' read -r name expected_size expected_hash; do
    [[ -z "$name" ]] && continue
    ((row_count += 1))
    if [[ "$name" == */* || "$name" == '.' || "$name" == '..' ]]; then
      echo "unsafe manifest filename: $name" >&2
      return 1
    fi
    if [[ ! "$expected_size" =~ ^[0-9]+$ || ! "$expected_hash" =~ ^[0-9a-fA-F]{64}$ ]]; then
      echo "invalid manifest row for: $name" >&2
      return 1
    fi
  done < "$manifest_path"
  (( row_count > 0 ))
}

validate_manifest
start_epoch=$(date +%s)

while true; do
  all_complete=1
  while IFS=$'\t' read -r name expected_size expected_hash; do
    [[ -z "$name" ]] && continue
    path="$incoming_dir/$name"
    if [[ ! -f "$path" ]]; then
      all_complete=0
      continue
    fi
    actual_size=$(stat -c %s "$path")
    if (( actual_size > expected_size )); then
      printf '%s\tSIZE_EXCEEDED\t%s\t%s\n' "$name" "$expected_size" "$actual_size" > "$failure_report"
      exit 1
    fi
    if (( actual_size != expected_size )); then
      all_complete=0
    fi
  done < "$manifest_path"

  if (( all_complete == 1 )); then
    break
  fi

  now_epoch=$(date +%s)
  if (( now_epoch - start_epoch >= max_wait_seconds )); then
    : > "$failure_report"
    while IFS=$'\t' read -r name expected_size expected_hash; do
      [[ -z "$name" ]] && continue
      if [[ -f "$incoming_dir/$name" ]]; then
        actual_size=$(stat -c %s "$incoming_dir/$name")
      else
        actual_size=MISSING
      fi
      printf '%s\tTIMEOUT\t%s\t%s\n' "$name" "$expected_size" "$actual_size" >> "$failure_report"
    done < "$manifest_path"
    exit 2
  fi
  sleep "$poll_seconds"
done

tmp_report="$success_report.tmp.$$"
: > "$tmp_report"
hash_failed=0
while IFS=$'\t' read -r name expected_size expected_hash; do
  [[ -z "$name" ]] && continue
  path="$incoming_dir/$name"
  actual_size=$(stat -c %s "$path")
  actual_hash=$(sha256sum "$path" | awk '{print $1}')
  if [[ "$actual_hash" != "${expected_hash,,}" ]]; then
    printf '%s\tHASH_MISMATCH\t%s\t%s\n' "$name" "${expected_hash,,}" "$actual_hash" >> "$failure_report"
    hash_failed=1
  else
    printf '%s\t%s\t%s\n' "$name" "$actual_size" "$actual_hash" >> "$tmp_report"
  fi
done < "$manifest_path"

if (( hash_failed == 1 )); then
  rm -f "$tmp_report"
  exit 1
fi

mv -f "$tmp_report" "$success_report"
date --iso-8601=seconds > "$verified_marker"
echo "VERIFIED: $verified_marker"
