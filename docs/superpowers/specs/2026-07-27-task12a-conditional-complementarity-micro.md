# Task 12A conditional global/local complementarity micro

## Data

- Positive train: 4 lexicographically fixed Task10B train images per class
  (64 total).
- Positive validation: 1 fixed Task10B validation image per class (16 total),
  descriptive only; no model or threshold selection.
- Positive test: 2 fixed Task10B dev images per class (32 total), excluding all
  Task11C.0/11C.1 IDs.
- Real-null test: 32 PlantSeg images stratified over mask ratio after excluding
  all Task11C IDs. No null image participates in fitting.
- Task8 locked families are never read.

Official IP102 boxes and audited PlantSeg masks define local evidence. The
Task11C0 v2 rule is frozen: expanded crop area below 0.95 uses a crop; otherwise
the original image is an identity fallback.

## Representation tournament

Qwen remains completely frozen. Existing global features are reused; only 144
local inputs are extracted.

- `G`: global feature (2048 dimensions).
- `L`: local feature (2048 dimensions).
- `GG`: `[G,G]/sqrt(2)` dimension/regularization control (4096 dimensions).
- `GL`: `[G,L]/sqrt(2)` candidate complementary representation (4096 dimensions).

Every condition uses the same balanced logistic classifier (`C=1`, `lbfgs`),
seeds 17/29/43, train rows, and untouched test rows. There is no hyperparameter,
temperature, or threshold selection.

## Primary evidence and branching

Report positive Accuracy, Macro-F1, mean true-class probability, null maximum
class confidence, and positive-vs-null confidence AUROC. Report paired 1000
bootstrap intervals and per-image complementarity.

The primary comparison is `GL-GG`, not `GL-G`, so extra dimensionality alone
cannot count as evidence.

- H1 conditional gain: Accuracy improves by at least 1/32 and mean true-class
  probability increases.
- Reliability safe: mean null confidence does not increase and confidence AUROC
  does not decrease.
- Gain + safe → `H1_PRIORITY`.
- Gain + unsafe → `H3_PRIORITY`.
- No conditional gain → `H2_PRIORITY`.

This is exploratory branching evidence, not a confirmatory claim. It never
authorizes training or Task8 access.

Path: `task12a_conditional_complementarity/2026-07-27/protocol_v1/attempt_01`.

