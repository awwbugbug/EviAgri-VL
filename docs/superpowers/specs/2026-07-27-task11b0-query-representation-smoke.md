# Task11B.0：Qwen query-token 表示提取 smoke

## 目标与边界

- 仅验证 Qwen2.5-VL-3B 的中间层 query-token 表示能否被精确、稳定、图像依赖地提取。
- 4张IP102正样本（class 9/22/82/87）+4张PlantSeg误接收；每张原图与RGB=127 blank配对，共16个表示。
- 全模型冻结、`torch.inference_mode()`、无优化器、无分类器、无阈值选择，不读取Task8，不启动Task11B.1。
- 图像以外的唯一模型输入是所有样本完全相同的中性问题；class/disease/path只作输出后审计字段。

## 精确表示合同

- 以本地模型配置为准：36个text blocks、hidden size 2048。
- 目标为第27个block输出，即`hidden_states[ceil(3L/4)]`；hidden_states[0]是embedding输出。
- 使用实际processor展开图像token后的`input_ids`定位完整19-token问题子序列；必须且只能命中一次。
- 对该问题token span做mean pool与L2归一化；固定`use_fast=False`、min/max pixels沿用Task10B。

## Smoke门禁

- 架构、层号、维度、query token数量精确匹配；16个表示全部有限且单位范数。
- 同图重复前向最大绝对差≤1e-6。
- 原图/blank的cosine distance中位数≥1e-4，且每对L2距离≥1e-3。
- 通过只授权Task11B.1 RepProbe设计与小规模执行；失败则停止并检查query span/层位/视觉融合，不能扩大训练。
