# Task 11C.1 paired full-image vs evidence-crop feature probe

## Scientific question

Does an oracle evidence crop improve the existing frozen visual-to-classifier
bridge without increasing pest false positives on disease/damage real-null
images?

## Frozen comparison

- Samples: the exact 16 IP102 positives and 16 PlantSeg real-null images from
  Task 11C.0 protocol v2.
- Conditions: `full` uses the original image; `local` uses the protocol-v2
  `effective_crop`, or the original image for `identity_full_frame`.
- Qwen2.5-VL-3B-Instruct visual tower, resolution bounds, post-merge mean pooling,
  L2 normalization, Task10B train split, logistic head, seeds 17/29/43,
  temperature 0.18887372662036642, and threshold 0.63 remain frozen.
- Only image pixels enter the model. No optimizer, QLoRA, Task8, new backbone, or
  threshold fitting is allowed.

## Outputs and uncertainty

Report full/local and paired deltas for positive forced accuracy, supported
diagnosis rate, true-class probability and margin; null FPR and confidence;
effective-crop and identity-fallback subgroups; 1000 paired image bootstraps.

Identity fallback is a pipeline invariant: full/local features must have cosine
similarity at least 0.99999 and predictions must agree for every seed. Effective
crops must produce a median cosine below 0.999.

## Frozen micro-feasibility gates

- positive accuracy and supported-diagnosis deltas each at least -0.0625;
- positive mean true-class probability delta greater than zero;
- local PlantSeg FPR below 0.10 and no higher than full;
- PlantSeg mean confidence delta below zero;
- all identity and feature-change invariants pass.

Passing authorizes only a somewhat larger paired crop validation. Confidence
intervals excluding zero are reported as `strong_signal`; they are not required
for this 32-image feasibility gate. Failure or inconclusive evidence blocks
training and scaling.

Path: `task11c1_paired_crop_probe/2026-07-27/protocol_v1/attempt_01`.

