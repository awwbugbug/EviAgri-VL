# Task 9D 正式评估结果（2026-07-16）

## 完成性
- 9 个 Static QLoRA + Base 共 10 组；每组 2560 条，共 25,600 条。
- 十组 manifest、解码契约一致；推理与正式评估 `completion.sha256` 全通过。
- 1000 次 family paired bootstrap + McNemar 已完成；未读取 Task8 locked set，未启动 9E。
- 评估前修复两处口径漏洞：`native_0/1/2` 模板差距曾错误恒为 0；640 条提示审计样本曾重复混入主诊断指标。修复后全套 166 tests passed。

## 决策
- A/B/C **全部淘汰**；`selected_protocol=null`。
- `protocol_repair_passed=false`，`scientific_passed=false`，**禁止进入 9E/动态 LoRA**。
- 三组所有 seed 的共同淘汰原因：semantic consistency 或 task compliance <99%。

## 核心指标（3-seed mean）
- Base：canonical Accuracy=0，Macro-F1=0，schema validity=0；严格 JSON 下不是有效性能基线。
- A：Accuracy 1.11%，Macro-F1 0.66%，overall Null FPR 22.23%。
- B：Accuracy 0.85%，Macro-F1 0.50%，overall Null FPR 9.26%。
- C：Accuracy 0.78%，Macro-F1 0.61%，overall Null FPR 7.53%。
- A/B/C 的 blank、blur Null FPR 均为 0，模板差距均 <1.7pp；但 task compliance 仅约 64.8%–71.2%。
- 定位出现信号（adapter Mean IoU 约 0.13–0.29），但 Supported Diagnosis 几乎为 0，不能视为方法成功。

## 客观解释与下一步
- v2 小样本训练学会了 JSON/拒答/部分定位，却没有学会可靠类别诊断；分类性能接近坍缩。
- C 的整体 Null FPR 和 seed 稳定性最好，但仍不满足协议修复门槛，不能据此选 C 全量训练。
- 下一步应先做 **Task9D 输出法医审计**：分解 Base JSON 失败、adapter 过度拒答、错误 pest_id/名称、query-label 对齐及 source_visual/semantic-null 失败来源；审计完成前不再训练。

## 产物
- 本地：`artifacts/2026-07-16_task9/9D_formal_evaluation/task9d_decision_report.json`
- 服务器：`.../evaluation/formal_metrics_v1/`
