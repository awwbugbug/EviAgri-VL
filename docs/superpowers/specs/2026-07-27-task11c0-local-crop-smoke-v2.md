# Task 11C.0 local-evidence crop smoke protocol v2

## Why v2 exists

Protocol v1 required every evidence crop to be strictly smaller than the full
image. The frozen 32-sample smoke showed that this is infeasible for legitimate
target-dominant images: some official boxes or mask components already span
nearly the whole frame. No model inference or training was run.

## Single protocol change

The sample selection, annotations, expansion fractions, and all other gates stay
unchanged. Only model-input assignment changes:

- expanded evidence area below 0.95: `effective_crop`;
- expanded evidence area at or above 0.95: `identity_full_frame` using the
  original source image, not a re-encoded crop.

Identity fallbacks remain in the same split and are reported separately. They
must never be described as a localization improvement.

## Frozen gates

- exactly 16 IP102 positives and 16 PlantSeg real-null samples;
- every evidence box remains contained and every review crop is nonempty;
- every sample receives exactly one mode;
- at least 75% of samples are effective crops;
- every effective crop is below 0.95 and every identity fallback is at or above
  0.95;
- identity fallbacks point to the original source image;
- median crop area remains below 0.75.

Failure blocks Task 11C.1. Passing only approves consideration of the next
micro experiment; it does not approve training or larger-scale evaluation.
