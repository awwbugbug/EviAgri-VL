# Current Research State

> Last consolidated: 2026-07-27. This file is the fast handoff entry, not a replacement for frozen specs or result artifacts.

## Start Here

1. Run `git status --short`, `git pull --ff-only`, and compare `HEAD` with `origin/main`.
2. Read this file, then `2026-07-27-research-question-hypothesis-tree-v1.md` and `2026-07-27-context-preserved-evidence-mechanism-v0.md`.
3. For evidence details, read dated memories `14_Task11C1...`, `15_Task12A...`, and `16_Task12B...` under `关键记忆/对话信息_2026_7_27/`.
4. Before remote work, read `docs/remote-deployment-runbook.md` and obtain the current endpoint from the user; credentials never enter Git.

## Current Scientific Position

- Locked question: improve fine-grained pest diagnosis without concrete diagnoses on disease, damage, empty, or unsupported images.
- Task11C1 falsified crop replacement: removing global context increased PlantSeg false positives.
- Task12A and independent Task12B found directionally consistent conditional value from local features when global context was preserved (`GL` versus same-width `GG`). Task12B effects were smaller and several confidence intervals crossed zero.
- Status: H1 has two-batch exploratory support and may enter mechanism design, but it is not confirmed. H2 token selection and H3 presence/taxonomy separation remain competing explanations.
- Task13A tested H3 with a separate frozen-feature presence head. It preserved 100% positive coverage and reduced PlantDoc FPR from 90% to 0% and PlantSeg FPR from 81.25% to 18.75%, but failed the preregistered PlantSeg FPR <10% gate. Decision: `H3_BLOCK_H2_PRIORITY`; no learned gate is authorized.
- No QLoRA, large training, dynamic gating, or Task8 confirmatory access is currently authorized.

## Next Task

Perform the hypothesis-tree literature/red-team reset required after the mechanism-study sequence, then freeze one Task14A H2 micro study: full-frame token selection or weighting without crop replacement or a second encoder. It must use fresh exploratory samples, preserve global context, and run before any larger training.

## Storage Boundaries

- Tracked source of truth: code, specs, compact reports, and `关键记忆/`.
- Local ignored evidence: `artifacts/`, `本地数据集/`, `transfer_staging/`.
- Remote outputs: immutable `<experiment>/<date>/protocol_vN/attempt_NN` directories.
- Always verify live `HEAD` and `origin/main`; do not rely on an embedded commit as current state.
