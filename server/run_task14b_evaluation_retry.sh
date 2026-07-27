#!/usr/bin/env bash
set -euo pipefail

PY=/root/miniconda3/bin/python
PROJECT=/root/EviAgri-VL/task14b_code_20260727_protocol_v1_attempt_02
SOURCE=/root/autodl-tmp/EviAgriDiag/experiments/task14b_independent_token_oracle/2026-07-27/protocol_v1/attempt_01
ROOT=/root/autodl-tmp/EviAgriDiag/experiments/task14b_independent_token_oracle/2026-07-27/protocol_v1/attempt_02

test ! -e "$ROOT"
test -x "$PY"
test -f "$PROJECT/evaluate_task14a_token_oracle.py"
(cd "$PROJECT" && sha256sum -c code_manifest.sha256)
"$PY" -c 'import numpy, sklearn'

test "$(sha256sum "$SOURCE/protocol/completion.sha256" | cut -d' ' -f1)" = 0d6d384aeab9537e511e61e5d85341b921882019a7308046baa7afe2e6540a6a
test "$(sha256sum "$SOURCE/features/completion.sha256" | cut -d' ' -f1)" = da29ab13f95018281ece36705765f3d00de44d4530786768a43b9082a0810029
test "$(sha256sum "$SOURCE/protocol/manifest.jsonl" | cut -d' ' -f1)" = 34c538aff663ba9ba40c6a9ec4eabb1538cf9a0a42cfa215de0ea88670a61a89
test "$(sha256sum "$SOURCE/features/global_features.npy" | cut -d' ' -f1)" = 44992daf451c533d62f62c98903625a0a3e6d6d6d769458bbca9b53938995bde
test "$(sha256sum "$SOURCE/features/region_features.npy" | cut -d' ' -f1)" = 013857b881bc25cf8cd98fa56bf790101a5e5d270d5516517f8eaf3673335da9
test "$(sha256sum "$SOURCE/features/feature_rows.jsonl" | cut -d' ' -f1)" = 06d9f43d564aa3fb2b26b929750af6113cec4ee7937bfec145a41ebff32ec364
(cd "$SOURCE/protocol" && sha256sum -c completion.sha256)
(cd "$SOURCE/features" && sha256sum -c completion.sha256)

mkdir -p "$ROOT"
"$PY" - "$ROOT/retry_manifest.json" <<'PY'
import json, sys
payload={
    "version":"task14b-evaluation-retry-1",
    "source_attempt":"attempt_01",
    "retry_attempt":"attempt_02",
    "reason":"fixed hard-coded 144-row index assertion for frozen 128-row replication",
    "scientific_protocol_changed":False,
    "protocol_manifest_sha256":"34c538aff663ba9ba40c6a9ec4eabb1538cf9a0a42cfa215de0ea88670a61a89",
    "feature_rows_sha256":"06d9f43d564aa3fb2b26b929750af6113cec4ee7937bfec145a41ebff32ec364",
}
with open(sys.argv[1],"x",encoding="utf-8",newline="\n") as handle:
    json.dump(payload,handle,indent=2,sort_keys=True); handle.write("\n")
PY

"$PY" "$PROJECT/evaluate_task14a_token_oracle.py" \
  --protocol-root "$SOURCE/protocol" --feature-root "$SOURCE/features" \
  --output-root "$ROOT/evaluation" --repetitions 1000 \
  --train-per-class 4 --val-per-class 0 --test-per-class 2 --null-count 32 \
  --decision-mode replication --report-version task14b-token-oracle-replication-1
(cd "$ROOT/evaluation" && sha256sum -c completion.sha256)
sha256sum "$ROOT/retry_manifest.json" "$ROOT/evaluation/completion.sha256" > "$ROOT/completion.sha256"
(cd "$ROOT" && sha256sum -c completion.sha256)
