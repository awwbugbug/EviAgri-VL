# Task14A Full-Frame Token Oracle

## Question and scope

Task14A tests whether spatial token selection has a credible upper bound before
any selector is learned. It is an annotation-only mechanism probe, not a
deployable method. Qwen2.5-VL-3B remains frozen and each image is encoded once.
No language-model inference, QLoRA, dynamic gate, second encoder, crop
replacement, Task8 access, or larger backbone is allowed.

## Frozen data protocol

- Positive source: the same 16 IP102 classes frozen by Task10B, but entirely
  fresh source images and near-duplicate components.
- Per class: 4 `probe_train`, 1 `probe_val`, and 2 `probe_test` examples
  (64/16/32 total positives).
- Real-null source: 32 PlantSeg disease/damage images selected across mask-ratio
  quantiles after excluding every ID used by Tasks11C, 12A, 12B, and 13A.
- Task10B source SHA256 values and near-duplicate components are excluded.
- Task8 is used only through the existing hashed exclusion boundary; its rows
  and contents are never opened.
- Protocol creation is metadata-only and must block on any quota, hash,
  cardinality, collision, or family-isolation failure.

## Frozen representations

For post-merge row-major visual tokens from one full-frame encoding:

- `G`: L2-normalized mean of all tokens.
- `R`: L2-normalized mean of tokens whose centers fall in the IP102 GT bbox;
  PlantSeg uses cells with at least 5% mask occupancy.
- `GG`: `concat(G,G) / sqrt(2)`, the same-width global control.
- `GR`: `concat(G,R) / sqrt(2)`, global anchor plus oracle region.

If a positive bbox selects no center, use the single cell with maximum bbox
overlap. An empty PlantSeg selection is a protocol failure. Record token grid,
region-token count, and area fraction for every image.

## Evaluation and decision

Fit the same balanced linear taxonomy probe for seeds 17, 29, and 43 using only
positive training examples. Validation and null data cannot select thresholds,
models, or hyperparameters. Report Accuracy, Macro-F1, mean true-class
probability, PlantSeg mean confidence, confidence AUROC, region fractions, and
1,000 paired image bootstraps.

The preregistered primary contrast is `GR-GG`:

- gain: mean Accuracy delta >= 1/32 and true-probability delta > 0;
- safe: PlantSeg confidence delta <= 0 and confidence-AUROC delta >= 0.

Gain plus safety yields `H2_ORACLE_SUPPORTED` and authorizes only a tiny learned
selector prototype. Gain without safety yields `H2_ORACLE_UNSAFE`. Otherwise
the result is `H2_ORACLE_NO_GAIN`. No branch authorizes Task8 or large training.
