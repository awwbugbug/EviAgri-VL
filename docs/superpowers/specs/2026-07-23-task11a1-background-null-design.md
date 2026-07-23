# Task 11A.1：成对 Background-Only Null 协议

## 目的

Task 11A 的 blur gate 失败后，视觉法医确认部分强模糊图仍保留可辨虫体，因此 blur 不能无条件视为零证据。Task 11A.1 不训练模型，只验证一个更严格的 null construct：从同一 IP102 图像中裁出与全部 GT bbox 保持安全边距的纯背景区域。

## 冻结输入

- 只读取 Task10B v2 的 train/val/dev manifest、IP102 VOC XML 和对应像素。
- 不读取 Task8 locked set，不新增 backbone，不更新 Qwen 参数。
- 原图 split、source SHA 和 near-duplicate component 保持不变。
- Router、温度 `0.1888737266` 与阈值 `0.63` 均沿用 Task11A，不重新调参。

## Val/Dev 分离规则

- val：`64x64` crop；bbox safety margin=`0.05 * min(width,height)`；`17x17` 候选网格。
- dev：`72x72` crop；bbox safety margin=`0.08 * min(width,height)`；`19x19` 候选网格。
- 候选必须与所有 expanded bbox 严格零交集。
- 每个 source 的候选仅按固定 namespace + source ID 的 SHA256 选择；不读取类别预测或模型结果。
- 几何不可行的 source 记录为 ineligible，不复制、移动或跨 split 替换。

## 两级有效性 gate

### Smoke gate

- val/dev 各 12 张，按 head/medium/tail 尽量平衡，选择规则确定性。
- 24/24 必须满足：XML 尺寸与像素一致、crop 尺寸正确、expanded bbox 交集为 0、输出 SHA256 完整。
- 人工视觉复核 24/24 不得出现完整虫体、局部虫体或可疑未标注目标；任一失败即 `BLOCK_INVALID_NULL`。

### Full protocol gate

- 几何可用量至少 val>=24、dev>=32，且两边均覆盖 head/medium/tail。
- 全量 derived crops 与 manifest 均签名；不覆盖 smoke 或历史产物。
- 全量视觉复核通过后，才允许一次 frozen-feature 提取与固定阈值评估。

## 评估与分流

- 主要指标：background-only Null FPR、95% source bootstrap CI、三 seed 一致性。
- `FPR < 10%`：支持 confidence router 在真实无目标区域的可行性，只授权规划轻量 evidence/localization head。
- `FPR >= 10%`：confidence 不足，下一步才允许设计极小的显式 evidence/OOD head。
- 任一 construct-validity gate 失败：停止在协议层，不得用模型结果补救或事后删样本。

