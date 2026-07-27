# Task 13A frozen-feature evidence-presence head

## Question

Can a separate evidence-presence head reject healthy and damage-only images
without erasing fine-grained pest diagnosis? This is the smallest H3
discriminator before any learned fusion or gate.

## Frozen protocol

- Use existing Qwen2.5-VL-3B global pooled features; Qwen remains frozen.
- Exclude every Task11C1, Task12A, and Task12B ID.
- Remaining IP102 positives: per class, two train, one positive-only threshold
  validation, and two fresh test images (32/16/32 total).
- PlantDoc healthy null: two train and two held-out test images per plant class
  (20/20). PlantSeg is never fitted or used for threshold selection; select 32
  mechanism-untouched damage-null images stratified over mask ratio.
- Compare `T0`, a positive-only 16-way taxonomy confidence score, with `P1`, a
  binary presence head trained on IP102-positive plus PlantDoc-null features.
- Both use `C=1`, balanced logistic regression, seeds 17/29/43. Thresholds are
  independently fixed from positive validation scores at at least 90% recall;
  no null test score may tune a threshold.

## Metrics and decision

Report positive coverage, supported diagnosis, PlantDoc and PlantSeg FPR,
balanced accuracy, source-specific and combined AUROC, three-seed mean/std/worst,
and 1,000 paired image bootstraps against `T0`.

`PASS_H3_FEASIBILITY` requires every seed to satisfy: P1 positive coverage
>=87.5%, PlantDoc FPR <10%, PlantSeg FPR <10%, combined AUROC >=0.90, and
PlantSeg FPR no worse than T0. Mean supported-diagnosis loss must be no more
than 1/32. Passing authorizes only a tiny gated-fusion prototype. Failure is
`H3_BLOCK_H2_PRIORITY` and redirects the next micro study toward H2 full-frame
token selection. Neither outcome authorizes QLoRA, Task8, or large training.

## Interpretation limit

Cross-source separation can still contain dataset-domain signal. PlantSeg OOD
testing reduces but does not eliminate this confound; Task13A establishes
feasibility, not a semantic proof of pest presence.
