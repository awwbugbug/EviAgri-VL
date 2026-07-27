# Repository Guidelines

## Project Structure & Module Organization

Start with `README.md`, then read `docs/research/CURRENT_STATE.md` before changing code or launching experiments. Python experiment, data-protocol, training, inference, and evaluation code lives in `server/`. Local preparation utilities live in `scripts/`; tests live in `tests/` and follow the corresponding server module. Frozen plans and specifications are under `docs/superpowers/`, while active hypotheses are under `docs/research/`. Compact scientific decisions belong in dated `关键记忆/` files. Raw datasets, model weights, PDFs, transfer staging, and generated results remain outside Git in ignored directories such as `本地数据集/` and `artifacts/`.

## Build, Test, and Development Commands

This repository has no packaging build step. Use the matching dependency profile when needed:

```powershell
pip install -r server/requirements-evaluation.txt
python -m pytest -q
python -m pytest tests/test_task12a_complementarity.py -q
python -m py_compile server/evaluate_task12a_complementarity.py
git diff --check
```

Run targeted tests while developing and the full suite before committing. `git diff --check` catches whitespace errors; `.gitattributes` enforces LF for Python, shell, Markdown, and JSON files.

## Coding Style & Naming Conventions

Use four-space Python indentation, type hints for public helpers, `pathlib.Path`, deterministic seeds, and explicit JSON contracts. Prefer small fail-closed functions over implicit fallback behavior. Python modules and tests use `snake_case`; test files are named `test_<module>.py`. Experiment paths use `task<id>_<name>/<date>/protocol_vN/attempt_NN`. Increment `protocol_vN` only for scientific changes; use a new `attempt_NN` for engineering retries. Shell entrypoints must begin with `set -euo pipefail` and be launched with `bash script.sh`.

## Testing Guidelines

Pytest is the primary framework. Every protocol, parser, gate, and failure path should have a focused regression test. No numeric coverage target is imposed, but new scientific decisions must be exercised by tests and verified with hashes, finite-value checks, and immutable output directories.

## Commit & Pull Request Guidelines

Use concise imperative commits consistent with history: `Add Task12A complementarity hypothesis test` or `Record Task11C paired crop probe failure`. Keep code, protocol, and result-record commits logically scoped. Pull requests should state the hypothesis or engineering goal, changed protocol/attempt, tests run, output hashes or artifact pointers, scientific decision, and any limitations. Never describe an engineering launch failure as a model result.

## Security & Agent Workflow

Never commit credentials, SSH endpoints, local keys, model/data bodies, checkpoints, or logs. On each work session, verify Git state, read `CURRENT_STATE.md`, inspect the applicable frozen spec, and perform the complete remote preflight in `docs/remote-deployment-runbook.md`. Do not open locked confirmatory data or start larger training unless the recorded gate explicitly authorizes it.
