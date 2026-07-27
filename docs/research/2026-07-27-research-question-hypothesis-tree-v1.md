# EviAgri-VL research question and rolling hypothesis tree v1

## Locked research question

How can an agricultural vision-language system improve fine-grained pest
diagnosis while avoiding concrete pest predictions on disease, damage, empty,
or otherwise unsupported images?

The answer is not locked. Evidence-First ordering, family-safe splits, separate
positive/real-null/synthetic-null metrics, fresh exploratory batches, and a
single-use confirmatory set are locked.

## Competing mechanisms

- **H1 — complementary local evidence:** local evidence contains class
  information not recoverable from the global pooled representation, but must
  be combined without removing context.
- **H2 — within-frame token selection:** cropping is the wrong intervention;
  evidence should be selected or weighted inside the complete image token set.
- **H3 — presence/taxonomy separation:** the main failure comes from asking one
  positive-only taxonomy head to perform both evidence-presence rejection and
  pest classification.

Task 11C.1 rejected crop replacement. It did not establish H1, H2, or H3.

Task 12A supplied the first independent support for H1: against the same-width
`GG` control, `GL` improved fresh positive accuracy and Macro-F1 while reducing
fresh PlantSeg confidence and improving confidence AUROC. This is one
exploratory batch only. H1 is prioritized, not accepted; H2 and H3 remain live
until an unchanged second-batch replication succeeds or fails.

## Exploration and confirmation boundary

- Exploratory mechanisms may be changed only between versioned micro studies.
- A batch that informed a mechanism becomes locked diagnostic evidence and
  cannot tune the next study.
- A mechanism needs directionally consistent evidence on two independent
  exploratory batches before any scale-up.
- Only after architecture, losses, and decision rules are frozen may the Task8
  confirmatory families be opened once.

## Exit rules

- Retire a mechanism family after two independent falsifying micro studies.
- After three mechanism studies, perform a literature/red-team reset before
  adding another module.
- Never rescue a failed mechanism by simultaneously changing data, head,
  threshold, and loss.
- `STRUCTURAL_FAILURE` blocks larger training; negative evidence updates this
  tree instead of being relabeled as engineering success.

## Current discriminator

Task 12A asks whether local features add class information conditional on the
global representation. It compares `G`, `L`, dimension-control `G+G`, and
`G+L` using fresh exploration samples and fixed linear probes. A positive gain
with null harm prioritizes H3; a safe gain prioritizes H1; no conditional gain
prioritizes H2. No branch authorizes QLoRA or full training.

Task 12A outcome: `H1_PRIORITY`, `reliability_safe=true`. Required next action:
repeat the identical tournament on a disjoint second exploration batch before
designing any learned fusion module.

Task 12B outcome: the disjoint replication preserved the direction with smaller
effects (`GL-GG` Accuracy +3.125 pp; Macro-F1 +5.417 pp). Point estimates for
true-class probability, null confidence, and confidence AUROC remained safe,
but their individual 95% intervals crossed zero. H1 therefore has two-batch
directional exploratory support and may enter mechanism design; it is not a
confirmed effect. H2 and H3 remain active controls.
