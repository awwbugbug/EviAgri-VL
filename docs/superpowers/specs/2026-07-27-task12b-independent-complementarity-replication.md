# Task 12B independent complementarity replication

Task 12B repeats Task 12A without changing representations, classifier,
metrics, branch rules, or sample counts.

The only change is the exploration batch. Every Task11C and Task12A ID is
excluded before deterministic selection. The next four remaining train images,
one validation image, and two dev images per class form the positive batch; 32
remaining PlantSeg rows stratified over mask ratio form the real-null batch.

The frozen conditions remain `G`, `L`, same-width `GG`, and `GL`. The primary
comparison remains `GL-GG`. Qwen is frozen, null is never fitted, and Task8 is
never read.

- If conditional gain and reliability safety both reproduce, H1 has two-batch
  exploratory support and may enter mechanism design, but not large training.
- If gain fails, H1 is not locked and H2 is prioritized.
- If gain reproduces with null harm, H3 is prioritized before fusion design.

Path: `task12b_independent_replication/2026-07-27/protocol_v1/attempt_01`.

