# Task 11A：置信度感知 Taxonomy Router 极小验证

## 目标与病因假设

Task 10B 已证明冻结 Qwen2.5-VL-3B 视觉特征在严格去重的 16 类 IP102 子集上可分（Dev Macro-F1=0.8094）；Task 10C 则证明 `q_proj/v_proj` 生成式 JSON 桥接没有学会 canonical pest-ID。Task 11A 只检验一个问题：

> 显式线性 taxonomy router 加上验证集冻结的置信度拒答阈值，能否在保持原图诊断的同时，对不可用视觉输入可靠 abstain？

本实验不是完整方法，也不使用 Task 8 locked set。它只决定下一步应进入轻量 evidence/OOD 头，还是继续显式 router 路线。

## 冻结资产

- 模型：Qwen2.5-VL-3B-Instruct，视觉塔完全冻结。
- 数据：Task 10B v2 的同一 16 类 family/source 去重 split，train/val/dev=192/48/80。
- 原图特征：复用签名的 `320 x 2048` float32 特征，SHA256=`5c730bab8d37d125f430d6b2fae1721359c04818f1dd86682e2f33a1ebbcaccc`。
- Router：Task 10B 的 `LogisticRegression(C=1, class_weight=balanced, lbfgs)`；不调超参数。
- Seeds：17/29/43；只用于既有 router 与确定性 shuffle。
- 输出：router 后由确定性 renderer 生成固定 schema JSON；不让语言模型再次生成 pest ID。

## 单变量与反事实分离

- Control：强制输出 top-1 类别，不拒答。
- Confidence Router：只增加一个阈值 `tau`；若最大校准概率 `< tau`，输出 abstain。
- `tau` 只由 val 原图和 val synthetic-null 冻结；dev 标签与 dev 结果不得参与选择。
- val 扰动：Gaussian blur radius=6、6x6 patch permutation、RGB blank=(127,127,127)。
- dev 扰动：Gaussian blur radius=10、8x8 patch permutation、RGB blank=(114,114,114)。
- 原图仍使用 Task 10B 已签名特征；仅为 val/dev 扰动重新提取冻结视觉特征。

## 阈值规则

对 `tau` 的固定网格 `[0.00, 0.01, ..., 1.00]`：

1. 原图“应接受”仅指 router top-1 正确的 val 样本；错误原图不奖励接受。
2. 所有 val synthetic-null 应拒答。
3. 最大化上述二分类 Balanced Accuracy；并列时选择更小阈值，避免过度拒答。
4. 温度缩放只允许在 val 原图 NLL 上拟合单一正温度；不得读取 dev。

## 预注册指标与 gates

按 seed 报告并做 1,000 次 source-level paired bootstrap：

- 原图：Accuracy、Macro-F1、coverage、selective accuracy、head/medium/tail。
- null：overall/blank/blur/shuffle Null FPR、abstention rate、concrete diagnosis under null。
- 输出：syntax validity、schema validity、semantic consistency、task compliance。

Task 11A 通过需同时满足：

1. 原图 Macro-F1 相对强制 router 不低于 `-3pp`；
2. 原图 coverage `>= 0.70`；
3. blank Null FPR `< 0.10`；
4. blur Null FPR `< 0.10`；
5. shuffle Null FPR `< 0.25`；
6. 四级 JSON 指标均为 `1.0`；
7. 三个 seed 均通过，且所有输入/输出 SHA256 完整。

## 分流

- PASS：只授权规划 Task 11B 小规模 evidence/localization head；仍不授权大型训练、动态 LoRA、SAM2、7B 或 Task 8 confirmatory audit。
- FAIL：证明 softmax confidence 不足以承担 evidence presence/null 判别；下一步设计显式二分类 evidence/OOD head，继续使用冻结特征与极小验证。

