# Current Research State

> Last consolidated: 2026-07-27. This file is the fast handoff entry, not a replacement for frozen specs or result artifacts.

## Start Here

1. Run `git status --short`, `git pull --ff-only`, and compare `HEAD` with `origin/main`.
2. Read this file, then `2026-07-27-research-question-hypothesis-tree-v1.md` and `2026-07-27-context-preserved-evidence-mechanism-v0.md`.
3. For recent evidence, read dated memories `20_Task13A...`, `22_Task14A...`, and `23_Task14B...` under `关键记忆/对话信息_2026_7_27/`.
4. Before remote work, read `docs/remote-deployment-runbook.md` and obtain the current endpoint from the user; credentials never enter Git.

## Current Scientific Position

- Locked question: improve fine-grained pest diagnosis without concrete diagnoses on disease, damage, empty, or unsupported images.
- Task11C1 falsified crop replacement: removing global context increased PlantSeg false positives.
- Task12A and independent Task12B found directionally consistent conditional value from local features when global context was preserved (`GL` versus same-width `GG`). Task12B effects were smaller and several confidence intervals crossed zero.
- Status: H1 has two-batch exploratory support and may enter mechanism design, but it is not confirmed. H3 presence/taxonomy separation remains informative but below its safety gate. Simple H2 mean region pooling is retired after two independent failures; more complex selectors are not thereby disproven, but have no current upper-bound justification.
- Task13A tested H3 with a separate frozen-feature presence head. It preserved 100% positive coverage and reduced PlantDoc FPR from 90% to 0% and PlantSeg FPR from 81.25% to 18.75%, but failed the preregistered PlantSeg FPR <10% gate. Decision: `H3_BLOCK_H2_PRIORITY`; no learned gate is authorized.
- Task14A tested H2 with one full-frame encoding and annotation-oracle region pooling. `GR-GG` changed Accuracy by 0 and increased true-class probability by 0.00657, but also significantly increased PlantSeg pest confidence by 0.00416 (95% CI 0.00242 to 0.00570) and reduced confidence AUROC by 0.01074. Decision: `H2_ORACLE_NO_GAIN`; no learned selector is authorized.
- Task14B independently replicated the same oracle. `GR-GG` reduced Accuracy by 3.125 pp and Macro-F1 by 2.083 pp while significantly increasing PlantSeg pest confidence by 0.00269 (95% CI 0.00133 to 0.00426). Decision: `H2_MEAN_REGION_RETIRED`.
- No QLoRA, large training, dynamic gating, or Task8 confirmatory access is currently authorized.

## Next Task

Freeze a Task15A paired resolution forensic before implementation. On the already diagnostic Task14A/14B images, compare the existing one-pass region pool `R` with a separately re-encoded local crop `L`, always preserving the same global anchor and same-width controls. This is a post-hoc mechanism decomposition, not independent confirmation: it asks whether Task12's local benefit came from magnification/re-encoding rather than within-frame selection. It must not tune thresholds, authorize a learned module, or read Task8.

## Storage Boundaries

- Tracked source of truth: code, specs, compact reports, and `关键记忆/`.
- Local ignored evidence: `artifacts/`, `本地数据集/`, `transfer_staging/`.
- Remote outputs: immutable `<experiment>/<date>/protocol_vN/attempt_NN` directories.
- Always verify live `HEAD` and `origin/main`; do not rely on an embedded commit as current state.
