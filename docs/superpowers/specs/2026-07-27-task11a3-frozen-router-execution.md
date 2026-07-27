# Task 11A.3：290张人工通过 PlantSeg 的一次性冻结 Router

## 授权边界

- 输入只允许最终人工审计 `KEEP` 的290张官方 Validation 原图；48张 EXCLUDE 与旧 smoke 不进入模型。
- 最终人工审计 SHA256=`56d8256949315a420cc57fd71bfeb72105020eb5900de500f2c520cb6ae9efb2`；状态必须为 `COMPLETE/READY_FOR_FROZEN_ROUTER`。
- 模型只读取 pixels；文件名、路径、病名、宿主、mask 与人工结论不得进入特征提取输入。
- PlantSeg 不参与训练、温度拟合、阈值选择、早停或方案选择；router 只运行一次。

## 冻结模型与 Router

- Qwen2.5-VL-3B-Instruct 冻结视觉塔；post-merge token mean + L2，2048维。
- Task10B同一16类 LogisticRegression；Task10B base feature SHA256=`5c730bab8d37d125f430d6b2fae1721359c04818f1dd86682e2f33a1ebbcaccc`。
- seeds=`17/29/43`；temperature=`0.18887372662036642`；tau=`0.63`；不改超参数。

## 数据与泄漏门

- 从已验证官方 `plantseg.zip` 重新提取原始字节，不使用审查页的JPEG衍生图。
- 原图/mask SHA必须与盲审索引逐张一致；290张唯一；与Task10B和Task11A.2图像SHA零重叠。
- 输出目录必须不存在；dataset/features/evaluation分别生成completion SHA256。

## 指标与冻结分流

- 三seed分别报告 FPR、refusal、置信度、接受类别分布、per-plant/per-disease、JSON contract。
- 1,000次 image bootstrap；任一seed接受计为该图命中的 Clopper-Pearson 95% CI。
- 探索性报告 mask-ratio/confidence Spearman，禁止用于调参。
- PASS需同时满足：平均FPR `<10%`、any-seed exact 95% upper `<25%`、JSON contract=`100%`。
- PASS只授权规划Task11B极小 evidence/localization head；FAIL进入法医分析。无论结果如何，均不授权QLoRA、动态LoRA/Gating、SAM2、7B或Task8 confirmatory。
