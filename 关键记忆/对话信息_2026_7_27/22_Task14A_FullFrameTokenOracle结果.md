# Task14A Full-Frame Token Oracle 结果

- 状态：完整完成；所有 completion SHA256 通过，非工程失败。
- 数据：全新 family-safe IP102 64 train / 16 val / 32 test；全新 PlantSeg 32 null；Task8 内容未读取。
- 方法：Qwen2.5-VL-3B 冻结；每图仅一次全图编码；比较 G/R/GG/GR，主比较 GR-GG。
- 结果：Accuracy Δ=0；Macro-F1 Δ=+0.00417；true-prob Δ=+0.00657，CI [0.00455,0.00880]。
- 可靠性：PlantSeg confidence Δ=+0.00416，CI [0.00242,0.00570]；AUROC Δ=-0.01074。
- 决策：`H2_ORACLE_NO_GAIN`；区域平均池化同时放大真实害虫和病斑伪证据，不授权 learned selector/大训练/Task8。
- 证据：`artifacts/2026-07-27_task14/14A_full_frame_token_oracle_result`；feature rows SHA `1b673141...`。
- 下一步：只允许独立、未调参的 Task14B 复现；先严格检查 4 train + 2 test/类是否可行，不足则 BLOCK。
