# Task 11A.2 PlantDoc real-null（2026-07-23）

- 目的：不用 synthetic 扰动，检验冻结 Task11A router 是否在外部健康叶片上凭空输出 IP102 害虫。
- 数据：PlantDoc 官方 test，commit=`5467f6012d78d1c446145d5f582da6096f852ae8`；10 个 healthy folders×4=40。首轮 raw URL 在25张后因特殊文件名 404，失败目录保留；r2 改为 pinned Git blob，40 SHA 唯一、与 Task10B 零重叠、completion 全通过。
- 视觉门：放大逐张检查 40/40 未见昆虫。局限：第25张作物目录疑似错标；第36张有叶孔但无虫；部分为白底图库图。只能作为小型外部 real-null，不能用于训练/调阈值。
- 特征：同 Qwen2.5-VL-3B 冻结视觉塔，40×2048，21.41s，peak VRAM=7.61GB；features SHA=`412815de...9907a`。
- 固定协议：3 seeds=`17/29/43`；temperature=`0.18887372662036642`，tau=`0.63`；PlantDoc 未参与拟合或调参。
- 结果：三个 seed 均 `0/40` 接受，real-null FPR=`0%`，refusal=`100%`，max confidence=`0.36957`，JSON contract=`100%`。
- 统计修正：普通 bootstrap 在全零事件时退化为 `[0,0]`，故 evaluation_v2 另报 Clopper-Pearson 95% CI=`[0,8.81%]`；预注册 FPR<10%、CI upper<25% 均 PASS。
- 科学边界：这是积极的可行性证据，但不推翻原 Task11A prereg FAIL，也不证明所有真实场景可靠；只授权规划更贴近 IP102 拍摄域的真实 no-pest 小集，仍不授权大型训练、Task11B、动态 LoRA/Gating、SAM2、7B 或 Task8 confirmatory。
- 权威结果：server `.../task11a2_plantdoc_real_null/2026-07-23/evaluation_v3`（补齐 per-class FPR 与 accepted diagnosis distribution）；local `artifacts/2026-07-23_task11/11A2_plantdoc_real_null/`。
