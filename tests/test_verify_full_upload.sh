#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VERIFY_SCRIPT="$SCRIPT_DIR/../server/verify_full_upload.sh"
TMP_ROOT=$(mktemp -d)
trap 'rm -rf "$TMP_ROOT"' EXIT

make_manifest() {
  local incoming=$1
  local manifest=$2
  shift 2
  : > "$manifest"
  local name
  for name in "$@"; do
    printf '%s\t%s\t%s\n' \
      "$name" \
      "$(stat -c %s "$incoming/$name")" \
      "$(sha256sum "$incoming/$name" | awk '{print $1}')" \
      >> "$manifest"
  done
}

test_success_creates_verified_marker() {
  local case_dir="$TMP_ROOT/success"
  mkdir -p "$case_dir/incoming" "$case_dir/report"
  printf 'alpha' > "$case_dir/incoming/a.bin"
  printf 'beta' > "$case_dir/incoming/b.bin"
  printf 'gamma' > "$case_dir/incoming/c.bin"
  make_manifest "$case_dir/incoming" "$case_dir/manifest.tsv" a.bin b.bin c.bin

  "$VERIFY_SCRIPT" "$case_dir/incoming" "$case_dir/manifest.tsv" "$case_dir/report" 1 5

  test -f "$case_dir/report/full_upload.verified"
  grep -Fq $'a.bin\t5\t' "$case_dir/report/full_upload.sha256.tsv"
}

test_hash_mismatch_fails_without_marker() {
  local case_dir="$TMP_ROOT/hash_mismatch"
  mkdir -p "$case_dir/incoming" "$case_dir/report"
  printf 'alpha' > "$case_dir/incoming/a.bin"
  make_manifest "$case_dir/incoming" "$case_dir/manifest.tsv" a.bin
  sed -i 's/[0-9a-f]\{64\}$/0000000000000000000000000000000000000000000000000000000000000000/' "$case_dir/manifest.tsv"

  if "$VERIFY_SCRIPT" "$case_dir/incoming" "$case_dir/manifest.tsv" "$case_dir/report" 1 5; then
    echo 'expected hash mismatch to fail' >&2
    return 1
  fi

  test ! -e "$case_dir/report/full_upload.verified"
  grep -Fq $'a.bin\tHASH_MISMATCH' "$case_dir/report/full_upload.failure.tsv"
}

test_timeout_fails_without_marker() {
  local case_dir="$TMP_ROOT/timeout"
  mkdir -p "$case_dir/incoming" "$case_dir/report"
  printf '%s\t%s\t%s\n' missing.bin 1 0000000000000000000000000000000000000000000000000000000000000000 > "$case_dir/manifest.tsv"

  if "$VERIFY_SCRIPT" "$case_dir/incoming" "$case_dir/manifest.tsv" "$case_dir/report" 1 0; then
    echo 'expected missing file timeout to fail' >&2
    return 1
  fi

  test ! -e "$case_dir/report/full_upload.verified"
  grep -Fq $'missing.bin\tTIMEOUT' "$case_dir/report/full_upload.failure.tsv"
}

test_success_creates_verified_marker
test_hash_mismatch_fails_without_marker
test_timeout_fails_without_marker
echo 'test_verify_full_upload: PASS'
