# Task11A.4：冻结 Router 的 24 例误接收法医

## 边界

- 只读取 Task11A.3 `protocol_v1/attempt_03`、Task10B 冻结特征和既有人工审查图。
- 不重新推理、不训练新模型、不改变温度/阈值、不读取 Task8、不启动 Task11B。
- Task11A.3 的 PASS 结论保持不变；法医指标只解释 24 个错误，不参与事后改门槛。

## 核对与分析

- 三 seed 预测必须逐字节一致，每组 290 行，误接收集合严格为 24 张。
- 重新拟合既定 Task10B LogisticRegression 仅用于复现冻结 Router，并验证预测、置信度和接收决策逐项一致。
- 报告 top-1/top-2 margin、熵、预测类别集中度、与 IP102 train 正样本的最近余弦相似度。
- 分析 PlantSeg plant/disease、病种关键词、mask ratio；这些字段只用于输出后法医，不进入模型。
- 输出 24 例原图和病斑 overlay 审查页、逐例表、JSON/Markdown 报告与 SHA256。

## 下一实验选择边界

- 法医只允许冻结一个单变量、极小规模的表示验证假设。
- 优先保持既定 Task11B-RepProbe：仅比较当前 vision-pooled 表示与精确定位的 Qwen `3L/4` query-token 表示；family/source split、分类器和全部评测保持一致。
- PlantSeg 已参与本次病因分析，后续只能作为开发/描述性外部集，不能再冒充一次性确认集。
- 仍禁止大型 QLoRA、动态 LoRA/Gating、SAM2、7B、新 backbone 与 Task8 confirmatory。
