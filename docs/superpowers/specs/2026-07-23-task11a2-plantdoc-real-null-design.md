# Task 11A.2：PlantDoc 外部 Real-Null 微评估

## 动机与边界

IP102 background-only crop 连续两版因文字泄漏和触角越框而 BLOCK；Task9 real_null 只是 semantic negative。Task 11A.2 引入一个最小、真实、无 synthetic 扰动的外部 no-pest 集，只检验现有 confidence router 是否会在健康叶片图上输出具体 IP102 害虫类别。

这是 external real-null/OOD 微评估，不代表 IP102 域内无虫性能，也不进入训练。

## 数据冻结

- 官方源：`pratikkayal/PlantDoc-Dataset`，CC BY 4.0，固定 Git commit。
- 只读官方 `test` split。
- 健康类目录：`Apple leaf`、`Bell_pepper leaf`、`Blueberry leaf`、`Cherry leaf`、`Peach leaf`、`Raspberry leaf`、`Soyabean leaf`、`Strawberry leaf`、`Tomato leaf`、`grape leaf`。
- 每类按 `SHA256(commit|class|filename)` 排序取前4张，共40张；不得按模型输出或视觉难度换样本。
- 保存原始 URL、Git blob SHA、文件 SHA256、许可证和 commit；与 Task10B train/val/dev source SHA 必须零重叠。

## 构造有效性 gate

- 40/40 可解码、非重复、无可读类别文字。
- 人工视觉复核 40/40 不得含昆虫、虫卵、幼虫、蛹、螨或疑似害虫局部。
- 任一失败则整轮 `BLOCK_INVALID_EXTERNAL_NULL`；不得删图后直接评估，只能先修订预定义类/来源。

## 冻结评估

- Qwen2.5-VL-3B 视觉塔完全冻结；沿用 Task10B 的 16类 LogisticRegression。
- 温度=`0.18887372662036642`、拒答阈值=`0.63`，不得用 PlantDoc 调参。
- 报告 overall/per-healthy-class Null FPR、具体诊断分布、最大置信度、1,000次按图片 bootstrap 95% CI。
- gate：overall FPR `<10%` 且 bootstrap 上界 `<25%`。

## 分流

- PASS：证明 confidence router 对该外部健康叶片 null 有基本拒答能力；只授权规划更贴近 IP102 域的真实 no-pest 数据收集。
- FAIL：softmax confidence 明确不足，下一步设计冻结特征上的极小 evidence/OOD head；仍不启动 QLoRA、7B、动态模块或大型训练。

