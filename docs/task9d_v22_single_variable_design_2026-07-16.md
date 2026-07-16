# Task 9D v2.2 单变量极小规模验证设计（冻结版）

## 1. 法医结论与待证假设

Task 9D 未证明“研究方向失败”，但证明当前任务定义不能支持开放诊断结论：

- 三份训练日程的标签映射一致：96 个名称对应 96 个唯一 ID，无一对多或多对一。
- 九个 adapter 的严格 JSON schema 有效率为 99.69%–100%，格式不是主因。
- 正样本平均 `evidence_present` 率为 56.88%，平均查询名复述率为 56.68%；模型一旦接受查询，约 99.7% 会复述 prompt 中已有的类别名。
- 平均精确 ID 准确率仅 0.91%，即使只看已接受的正样本也仅约 0.8%–2.6%；名称与 ID 的内部一致率平均仅 44.38%。
- 忽略冗余 ID、只评估 evidence 判定时，各 adapter 的 Balanced Accuracy 为 63.72%–77.94%，并非约 1%。

明确病因是：**训练与评测把 query-conditioned evidence verification、查询名复述、96 类任意整数 ID 记忆和定位塞进同一自回归目标；正式 Accuracy 又把 ID 精确生成当作诊断正确的必要条件。** 这既掩盖了已有的核验能力，也可能让 taxonomy token 的监督干扰 evidence 学习。

待证因果假设 H1：**正样本 taxonomy value token（`pest_id`、`pest_name`）的生成损失，是 evidence 学习不稳定和正样本过度拒答的重要原因。**

本实验只验证 H1，不宣称验证开放类别诊断能力。

## 2. 唯一实验变量

唯一变量：`taxonomy_value_loss_weight`。

| 项目 | v2.2-Control | v2.2-TaxMask |
|---|---:|---:|
| `diagnosis.pest_id` 与 `diagnosis.pest_name` 的 value token loss 权重 | 1.0 | 0.0 |

以下 token 在两组均正常监督：JSON 结构、字段名、标点、`evidence_present`、`evidence_region`、`reliability`，以及 null 样本的全部语义一致性字段。

两组使用完全相同的 target 文本、字段顺序、prompt、图像和解码协议；TaxMask 仅在 loss mask 中屏蔽正样本 taxonomy 的**值 token**。不得改 schema、删字段、外部回填 ID、改变模板或调整其他 loss 权重。

## 3. 极小规模数据与训练冻结

- 数据范围：仅 Task 9 的 `train`、`val`、`task9_dev_audit`；Task 8 的 186 families 保持锁定且禁止读取。
- 协议底座：Task 9D 的 B 协议（正样本 + 成对 semantic wrong-query + 轮换视觉反事实）。
- 类别：从可行类别中按固定哈希选择 32 类，并保持 head/medium/tail 分层。
- 训练：每类 2 个 train family，共 64 families；每 family 三个角色，共 192 条唯一训练样本。
- 开发审计：每类 1 个独立且具备完整条件的 dev family，共 32 families；每 family 11 行，共 352 行。
- family 不能跨 split、复制或移动；不足时 BLOCK，不得近似补齐。
- 两组 family、样本顺序、反事实图、模板分配完全一致。
- Base：同一 Qwen2.5-VL-3B；同一 q_proj/v_proj Static QLoRA 配置。
- 优化器、rank、学习率、batch/累积、图像分辨率、max tokens、解码、parser 均沿用 Task 9D。
- 最大 64 optimizer steps（约为 Task 9D 的三分之一），固定 final checkpoint，不早停、不按 dev 选 checkpoint。
- 配对 seeds：17、29、43；共 2 arms × 3 seeds = 6 个微型 run。
- 每个 run 保存 adapter、SHA256、loss mask 审计、训练日志、实际样本数、时长、peak VRAM、predictions 和 metrics。

## 3.1 训练前 loss reduction 硬门槛

- Control 与 TaxMask 均采用：先对每个样本的 active token 交叉熵求均值，再对 batch 内样本求均值；禁止使用跨样本 token-global mean。
- 当前 batch size=1、gradient accumulation=8，因此每个 micro-example 在 optimizer window 内的名义梯度权重均为 1/8，与 active-token 数无关。
- 训练前必须对共享的 512-row schedule 真实运行生产 collator，分别输出 positive、semantic-negative、visual-counterfactual 的 active-token min/max/mean/sum、平均样本 loss 权重、累计梯度权重及归一化梯度权重。
- Control/TaxMask 的同角色样本数、平均样本 loss 权重及归一化总梯度权重必须完全一致；null labels 必须逐 token 相同；正样本只能少 `pest_id/pest_name` value token。
- fast/slow tokenizer 的 assistant target token IDs 必须逐样本完全一致。任一条件不满足则 BLOCK，禁止创建训练输出。

## 4. 预注册指标

主指标（直接检验 H1）：

- Evidence Verification Balanced Accuracy：canonical positive 的 `evidence_present=true` TPR 与全部 null 的 `evidence_present=false` TNR 的均值。
- Positive Evidence TPR。
- Overall / semantic / visual Null FPR。
- 三个配对 seed 的 TaxMask − Control 差值。

守门指标：

- blank、blur、shuffle 分条件 Null FPR。
- syntax validity、schema validity、evidence semantic consistency、evidence task compliance。
- evidence_region 在正样本上的存在率与在 null 上的 False Localization Rate。
- 中性提示与训练提示 gap。
- taxonomy 输出只作法医观察，不作为 v2.2 通过依据；查询名命中仍标记为 `query_name_echo`，不得称为视觉诊断准确率。

统计：以 family 为重采样单位做 1,000 次 paired bootstrap；同时报告三 seed 均值、标准差和最差 seed。微实验只接受大效应，不以单个 p 值替代效应门槛。

## 5. 冻结通过/失败规则

H1 通过必须同时满足：

1. TaxMask 相对 Control 的 Evidence Verification Balanced Accuracy 三 seed 平均提升至少 8pp；
2. 1,000 次 paired bootstrap 的 95% CI 下界大于 0；
3. 至少 2/3 seeds 提升，且最差 seed 不低于 Control 超过 3pp；
4. Positive Evidence TPR 平均提升至少 10pp；
5. Overall、semantic、visual Null FPR 均不得恶化超过 3pp；blank/blur Null FPR 必须低于 10%；
6. syntax/schema validity 不低于 99%，evidence semantic consistency 与 task compliance 不低于 99%；
7. prompt gap 低于 5pp。

任一条件不满足，H1 失败。不得训练后修改门槛。

## 6. 结果分流

- **H1 通过：**只授权设计“evidence verifier 与 taxonomy predictor 解耦”的后续 Static QLoRA 小消融；仍不授权 Task 9E、动态 LoRA/Gating 或 Task 8 confirmatory audit。
- **H1 失败但 evidence 指标与 Control 相近：**说明约 1% Accuracy 主要是任务/评测错配，而不是 taxonomy loss 拖累；停止继续优化这个复合 schema，下一步应单独定义开放诊断任务。
- **H1 失败且 evidence 指标更差：**保留 taxonomy 监督可能对核验有辅助作用；返回输出/梯度法医，不扩大训练。

## 7. 当前状态

本文件仅冻结实验设计。v2.2 未启动；服务器保持开机；Task 8 locked set 未读取。
