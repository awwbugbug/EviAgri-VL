# Task13A 冻结特征 Presence Head

- 协议：排除Task11C/12A/12B；IP102正样本32/16/32，PlantDoc null 20训练/20测试，PlantSeg 32仅测试；Qwen冻结，3 seeds，阈值仅由正样本val确定。
- P1保持正样本coverage 1.0、Supported Diagnosis 0.84375；PlantDoc FPR `0.90→0`，PlantSeg FPR `0.8125→0.1875`，combined AUROC `0.97476→0.98438`。
- paired bootstrap：combined FPR delta -0.73077，95% CI `[-0.86538,-0.57692]`；PlantSeg FPR delta -0.625，CI `[-0.84375,-0.40625]`。
- 决策：`H3_BLOCK_H2_PRIORITY`。虽显著改善，但PlantSeg FPR未满足预注册 `<10%`；禁止事后放宽，禁止gated fusion、大训练和Task8。
- 6个PlantSeg误接收覆盖细菌斑、褐腐、白粉病、叶斑、脐腐、scorch，说明病斑/损伤仍会触发presence。
- 证据：`artifacts/2026-07-27_task13/13A_frozen_presence_head_result`；服务器 `task13a_frozen_presence_head/2026-07-27/protocol_v1/attempt_01`，全部SHA256通过。
- 下一步：按假设树先做文献/红队reset，再冻结H2全帧token selection微实验。
