#!/usr/bin/env bash
set -euo pipefail

FEATURE_PY=/root/miniconda3/envs/eviagri/bin/python
EVAL_PY=/root/miniconda3/bin/python
PROJECT=/root/EviAgri-VL/server
TASK10B=/root/autodl-tmp/EviAgriDiag/experiments/task10_micro_first/2026-07-17/task10b_v2
ROOT=/root/autodl-tmp/EviAgriDiag/experiments/task11_confidence_router/2026-07-23/task11a
STRESS="$ROOT/stress_features"
EVALUATION="$ROOT/evaluation"
MODEL=/root/autodl-tmp/EviAgriDiag/models/Qwen/Qwen2___5-VL-3B-Instruct
EXPECTED_BASE_SHA=5c730bab8d37d125f430d6b2fae1721359c04818f1dd86682e2f33a1ebbcaccc

test -d "$TASK10B/protocol"
test -d "$TASK10B/formal_features"
test ! -e "$ROOT"
mkdir -p "$ROOT"

(
  cd "$TASK10B/protocol"
  sha256sum -c completion.sha256
)
(
  cd "$TASK10B/formal_features"
  sha256sum -c completion.sha256
  observed=$(sha256sum features.npy | cut -d' ' -f1)
  test "$observed" = "$EXPECTED_BASE_SHA"
)

"$FEATURE_PY" "$PROJECT/extract_task11a_stress_features.py" \
  --source-manifest "$TASK10B/protocol/manifest.jsonl" \
  --model-path "$MODEL" \
  --output-root "$STRESS"
(
  cd "$STRESS"
  sha256sum -c completion.sha256
)

"$EVAL_PY" "$PROJECT/evaluate_task11a_confidence_router.py" \
  --base-feature-root "$TASK10B/formal_features" \
  --stress-feature-root "$STRESS" \
  --output-root "$EVALUATION" \
  --repetitions 1000
(
  cd "$EVALUATION"
  sha256sum -c completion.sha256
)
