# H2 full-frame token-selection literature and red-team reset

## Primary-source lessons

- [TokenLearner](https://arxiv.org/abs/2106.11297) shows that adaptive learned
  tokens can outperform plain pooling, but its gains rely on learned modules and
  substantial vision training. It is a later candidate, not the next probe.
- [TokenPacker](https://arxiv.org/abs/2407.02392) uses a coarse global foundation
  enriched by region cues. This supports our global-anchor/local-residual idea,
  but replacing the projector now would confound mechanism and training.
- [DeCo](https://arxiv.org/abs/2405.20985) argues for parameter-free spatial
  pooling that preserves dense locality and warns that aggressive semantic
  abstraction can erase fine detail. It motivates a spatially explicit control.
- [OC-VTP](https://openaccess.thecvf.com/content/CVPR2026F/html/Li_Object-Centric_Vision_Token_Pruning_for_Vision_Language_Models_CVPRF_2026_paper.html)
  selects tokens by reconstruction coverage. Its objective is efficiency and
  representativeness, not pest-vs-lesion evidence, so it is not adopted yet.

## Local architecture check

The installed Qwen2.5-VL vision tower returns post-merge tokens before mean
pooling. A read-only probe produced grid `[1,28,38]`, merge size 2, and exactly
`266×2048` output tokens (`14×19`). The implementation reverses window order
before return, preserving row-major spatial correspondence. H2 can therefore be
tested without a crop, second encoder, language model, or model modification.

## Frozen next discriminator

Task14A is an annotation-only oracle upper bound on a fresh exploratory batch:

- encode each full image once;
- compare global mean `G`, oracle in-frame region mean `R`, same-width `GG`, and
  context-preserved `GR`; primary comparison is `GR-GG`;
- use IP102 GT boxes for positives and PlantSeg masks only to measure whether
  disease/damage tokens become pest-like;
- fit fixed positive-only linear probes; null never fits or selects a threshold;
- preserve family-safe splits, three seeds, paired bootstrap, and Task8 lock.

If oracle `GR` cannot improve positive taxonomy safely, a learned selector has
no credible upper-bound rationale. If it succeeds, only a tiny learned-selector
prototype is authorized. Oracle annotations are diagnostic and cannot be
presented as a deployable method.
