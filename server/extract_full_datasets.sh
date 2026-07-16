#!/usr/bin/env bash
set -euo pipefail

dataset_root=/root/autodl-tmp/EviAgriDiag/datasets
incoming=$dataset_root/_incoming
raw_root=$dataset_root/raw
log_dir=$dataset_root/extraction_2026-07-13
mkdir -p "$raw_root" "$log_dir"
exec > >(tee -a "$log_dir/extract.log") 2>&1

extract_tar_once() {
  local archive=$1
  local target=$2
  if [[ -f "$target/.extract_complete" ]]; then
    echo "SKIP complete: $target"
    return
  fi
  mkdir -p "$target"
  echo "START tar: $archive -> $target"
  tar -xf "$archive" -C "$target"
  touch "$target/.extract_complete"
  echo "DONE tar: $target"
}

extract_tar_once \
  "$incoming/ages_raw_2026-07-12.tar" \
  "$raw_root/ages_2026-07-12"

extract_tar_once \
  "$incoming/ip102_classification_raw_2026-07-12.tar" \
  "$raw_root/ip102_classification_2026-07-12"

detection_target=$raw_root/ip102_detection_2026-07-12
if [[ ! -f "$detection_target/.extract_complete" ]]; then
  mkdir -p "$detection_target/Detection/VOC2007"
  echo "START zip metadata: IP102 Detection"
  unzip -q \
    "$incoming/IP102_Detection-20260712T104348Z-2-001.zip" \
    'Detection/VOC2007/ImageSets/*' \
    -d "$detection_target"
  echo "START nested tar: JPEGImages"
  unzip -p \
    "$incoming/IP102_Detection-20260712T104348Z-2-001.zip" \
    Detection/VOC2007/JPEGImages.tar \
    | tar -xf - -C "$detection_target/Detection/VOC2007"
  echo "START nested tar: Annotations"
  unzip -p \
    "$incoming/IP102_Detection-20260712T104348Z-2-001.zip" \
    Detection/VOC2007/Annotations.tar \
    | tar -xf - -C "$detection_target/Detection/VOC2007"
  touch "$detection_target/.extract_complete"
  echo "DONE detection: $detection_target"
else
  echo "SKIP complete: $detection_target"
fi

for target in \
  "$raw_root/ages_2026-07-12" \
  "$raw_root/ip102_classification_2026-07-12" \
  "$raw_root/ip102_detection_2026-07-12"; do
  chmod -R a-w "$target"
done

echo "EXTRACTION_COMPLETE $(date --iso-8601=seconds)"
du -sh \
  "$raw_root/ages_2026-07-12" \
  "$raw_root/ip102_classification_2026-07-12" \
  "$raw_root/ip102_detection_2026-07-12"
