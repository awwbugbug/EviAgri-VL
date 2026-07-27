# Task14B Independent Token-Oracle Replication

## Purpose

Task14B is the unchanged independent replication of Task14A. It asks whether
the first batch's `H2_ORACLE_NO_GAIN` result reproduces on disjoint IP102
near-duplicate families and disjoint PlantSeg images. It does not tune the
region rule, model, classifier, gates, or metrics.

## Frozen protocol

- Use the same 16 Task10B classes and Qwen2.5-VL-3B checkpoint.
- Exclude all Task10B and Task14A source SHA256 values and near-duplicate
  components; exclude every PlantSeg ID used through Task14A.
- Select exactly 4 positive training and 2 positive test images per class
  (64/32). Task14A validation was report-only and selected no setting, so no
  validation split is allocated in this replication.
- Select 32 fresh PlantSeg disease/damage nulls across mask-ratio quantiles.
- Block rather than approximate if any class has fewer than six fresh
  components, any cross-split family overlap exists, or any input hash changes.
- The Task8 exclusion file remains a hashed boundary only; locked contents are
  not read.

## Unchanged mechanism and evaluation

Each image receives one full-frame encoding. Preserve Task14A's row-major
post-merge pooling exactly: global mean `G`, bbox/mask region mean `R`,
same-width control `GG`, and context-preserved `GR`. IP102 uses bbox-center
cells with max-overlap fallback; PlantSeg uses mask occupancy >=5%.

Fit the same balanced positive-only linear probes for seeds 17, 29, and 43.
Report the same positive metrics, PlantSeg confidence, confidence AUROC, token
fractions, and 1,000 paired fresh-image bootstraps. The primary comparison and
gates remain `GR-GG`: Accuracy delta >=1/32 plus true-probability delta >0;
reliability additionally requires null-confidence delta <=0 and AUROC delta
>=0.

## Preregistered branch

- If gain and safety both pass, report `H2_REPLICATION_INCONSISTENT`; do not
  implement a selector because Task14A and Task14B disagree.
- Otherwise report `H2_MEAN_REGION_RETIRED`; two disjoint batches have failed
  to justify simple oracle mean region pooling.

Neither branch authorizes Task8, learned gating, QLoRA, a new backbone, or any
larger experiment.
