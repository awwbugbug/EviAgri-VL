# Repository Guidelines

## Project Structure & Module Organization

Read `docs/research/CURRENT_STATE.md` before changing code or launching experiments. Read `README.md` only for first-time orientation or public project context. Experiment, protocol, training, inference, and evaluation code lives in `server/`; local utilities live in `scripts/`; tests live in `tests/`. Frozen specifications are under `docs/superpowers/`, active hypotheses under `docs/research/`, and compact decisions in dated `关键记忆/` files. Datasets, weights, PDFs, staging files, and generated results stay outside Git in ignored directories such as `本地数据集/` and `artifacts/`.

## Build, Test, and Development Commands

There is no packaging build step. Use the matching dependency profile:

```powershell
pip install -r server/requirements-evaluation.txt
python -m pytest -q
python -m pytest tests/test_task12a_complementarity.py -q
python -m py_compile server/evaluate_task12a_complementarity.py
git diff --check
```

Run targeted tests during development and the full suite before committing. `git diff --check` catches whitespace errors; `.gitattributes` enforces line endings.

## Coding Style & Naming Conventions

Use four-space Python indentation, type hints for public helpers, `pathlib.Path`, deterministic seeds, and explicit JSON contracts. Prefer fail-closed behavior. Modules use `snake_case`; tests use `test_<module>.py`. Experiment paths follow `task<id>_<name>/<date>/protocol_vN/attempt_NN`. Increment the protocol only for scientific changes and the attempt for engineering retries. Start shell entrypoints with `set -euo pipefail`; launch them using `bash script.sh`.

## Testing Guidelines

Pytest is primary. Give every protocol, parser, gate, and failure path a focused regression test. Scientific decisions require hashes, finite-value checks, and immutable output directories; no numeric coverage target is imposed.

## Commit & Pull Request Guidelines

Use concise imperative commits such as `Add Task12A complementarity hypothesis test`. Keep changes logically scoped. Pull requests state the goal, protocol/attempt, tests, artifact pointers or hashes, decision, and limitations. Never describe a launch failure as a model result.

## Security & Agent Workflow

Never commit credentials, SSH endpoints, keys, datasets, checkpoints, or logs. At session start, verify Git state, read `CURRENT_STATE.md`, inspect the frozen spec, and use `docs/remote-deployment-runbook.md`. Do not open locked data or scale training without an explicit recorded gate.

## End-of-Session Handoff

Update `CURRENT_STATE.md` only when conclusions, next task, blockers, or authorization changed. Add a dated `关键记忆/` note only for consequential experiments or decisions; waiting without new evidence needs none. Record artifact paths, hashes, and remote run state, never credentials or endpoints. Before handoff, run relevant tests and `git diff --check`, commit and push tracked work, then report worktree cleanliness, remote jobs, and shutdown safety. `CURRENT_STATE.md` is the replaceable present snapshot; dated memories are the append-only history.
