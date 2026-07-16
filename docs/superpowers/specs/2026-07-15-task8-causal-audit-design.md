# Task 8 同协议反事实审计设计

## 1. 目标与结论边界

Task 8 不证明新模块有效，而是判断 Static QLoRA v1 是否通过进入复杂模块前的科学门槛。它回答三个问题：adapter 是否在公平协议下优于 Base Qwen；满分 null 指标是否来自提示模板捷径；模型是否真正依赖图像证据。

通过 Task 8 之前，不训练动态 LoRA、不下载 7B、不扩展 CDDM。Task 7 仅作为可复现工程底座和待审计候选模型。

## 2. 四组模型与受控变量

| 组 | 权重 | 提示 | 用途 |
|---|---|---|---|
| B0 | Base Qwen2.5-VL-3B | 普通诊断提示 | 通用 zero-shot 参照 |
| B1 | Base Qwen2.5-VL-3B | 中性 Evidence-First 提示 | adapter 核心对照 |
| B2 | Static QLoRA v1 | 与 B1 字节一致 | adapter 核心实验组 |
| B3 | Static QLoRA v1 | 原训练模板 | 模板捷径检测 |

B1 与 B2 除权重外完全一致。B0 与 B1、B2 与 B3 只允许提示这一预注册变量不同。四组共享同一图像字节、处理器、`min_pixels=200704`、`max_pixels=401408`、4-bit NF4、BF16、`max_new_tokens=128`、`do_sample=false`、JSON schema、解析器和评测器。

中性提示在正负样本中使用同一模板，并显式给出“queried pest”。正样本查询真实类别；错误类别和无目标条件查询原始类别但图像不支持该类别。模板不得包含 positive、null、task type、split、文件名或目录。

## 3. 成对反事实 family

正式筛查固定种子 `20260715`，从 IP102 official test 的 102 类中每类选择 2 个 GT bbox 正样本，共 204 个 family。smoke 仅选 4 个 family。每个 family 生成六个条件：

1. `original_correct`：原图 + 中性正确类别问题，预期正常诊断与定位。
2. `original_wrong_query`：原图 + 错误类别问题，预期不跟随诱导类别。
3. `shuffled_image`：另一 family 图像 + 原问题，预期原类别诊断下降。
4. `strong_blur`：原图强高斯模糊 + 原问题，预期拒答增加。
5. `blank_image`：同尺寸中性灰图 + 原问题，预期拒答且不生成 bbox。
6. `no_target_image`：显式选择不同 GT 类图像、缩放到原图尺寸 + 原问题，预期拒答且不生成原类别。

所有派生图像保存为不含类别的 opaque 文件名。一个 `audit_id` 在 B0-B3 中引用同一 `image_sha256`。清单记录 `family_id`、条件、真实类、查询类、GT bbox、原始图像哈希和派生图像哈希，但这些元数据不进入模型提示。

## 4. 隐性泄漏审计

正式推理前必须生成并通过 `leakage_report.json`：

- 路径/文件名：审计运行时只使用 opaque 路径；提示文本不得出现路径、文件名、split、positive/null 或 task type。
- 模板：统计 positive/null 的 prompt hash、长度和固定短语；B1/B2 必须同模板，B3 的原始模板差异必须被报告而不是掩盖。
- 答案：报告目标 JSON 长度、固定短语和类别可预测性，不能把答案差异误当模型视觉能力。
- split：以 SHA256 检查 train/val/test 精确重复；以 64-bit dHash 和 Hamming distance `<=4` 筛查跨 split 增强/近重复；同一 source image ID 不得跨 split。
- 推理边界：提示构造器只接收类别显示名和 schema，不接收源路径；图像路径仅传给视觉处理器。

发现跨 split 精确重复、B1/B2 协议哈希不一致、opaque 路径含类别、审计 ID 重复或派生图像哈希不一致时，硬失败并禁止 GPU 正式运行。近重复只生成需人工复核的候选清单，不静默删除。

## 5. 指标定义

### Positive：仅 `original_correct` 且存在 GT bbox

- Diagnosis Accuracy、102 类 Macro-F1。
- Mean IoU：无有效预测框按 0 计入，不能排除失败样本。
- Pointing Game：预测框中心落入任一 GT bbox 为 1，否则为 0。
- Supported Diagnosis Rate：类别正确、`evidence_present=true`、预测框有效且 IoU `>=0.5`。

### Null/counterfactual：其余五个条件

- Null FPR：输出具体 diagnosis 对象的比例。
- Refusal Accuracy：`diagnosis="uncertain"`、`evidence_present=false` 且 bbox 为 null。
- False Localization Rate：`evidence_present=true` 的比例。
- Predicted-Box-on-Null Rate：bbox 非 null 的比例。
- EBHR：在不支持查询类别时仍输出该具体类别，或具体诊断没有有效视觉证据。
- Prompt Compliance Error：schema 无效、字段顺序错误、或错误类别条件中跟随诱导类别。

null 样本不计算 IoU/Pointing Game，更不能以空框记为满分。

### Overall

- Presence Balanced Accuracy、Presence Macro-F1、Presence F1。
- 诊断 Macro-F1 只在有真实目标的原始正样本上报告。
- 每个条件单独成表，不以总平均替代条件结果。

## 6. 统计协议

- 固定 1,000 次 bootstrap，种子 `20260715`。
- 以 `family_id` 为重采样单位，六个条件共同进入/退出，避免伪独立。
- 每组每条件报告点估计与 percentile 95% CI。
- B1-B2 使用 paired family bootstrap，报告差值及 95% CI。
- 二分类指标使用精确 McNemar：报告 B1 错/B2 对与 B1 对/B2 错的 discordant counts 和双侧 p 值。
- 多类别 Macro-F1 使用 paired bootstrap，不用普通未配对检验。

## 7. 文件与运行边界

- `server/task8_protocol.py`：提示、协议常量、schema 和 protocol hash。
- `server/build_task8_audit.py`：family 采样、六条件图像和不可变 JSONL 清单。
- `server/audit_task8_leakage.py`：路径、模板、答案、跨 split 哈希/近重复审计。
- `server/run_task8_inference.py`：一次加载 Base/Adapter，按 B0-B3 可恢复生成预测。
- `server/evaluate_task8.py`：分条件指标、bootstrap、paired bootstrap、McNemar 和 A/B gate。
- `server/run_task8_smoke.sh`：无覆盖 smoke 启动器。
- `tests/test_task8_*.py`：所有协议和统计的单元测试。

所有 JSON/JSONL 使用临时文件原子发布或逐行 flush；已存在且非空的输出目录默认拒绝覆盖。正式运行前必须通过数据清单 SHA256、泄漏报告 `passed=true`、smoke 数量/解析/协议哈希门槛。

## 8. A/B 分流门槛

### A：允许进入复杂模块

B2 在 B1 的同协议比较中有稳定正向差值；打乱/空白/无目标显著降低具体诊断并提高拒答；prompt 换写不使可靠性崩溃；正样本定位仍有效；Macro-F1 改善不是只来自头部类。之后依次做 Evidence-First vs Diagnosis-First、null-evidence 消融、Evidence-Conditioned Adapter/Gating。

### B：返回数据与提示协议

B3 满分而 B2 中性模板显著下降；B1/B2 差异很小；空白/换图仍给具体类别；固定短语可预测 positive/null；或定位来自解析漏洞。此时修复 prompt 和正负构造，训练 Static QLoRA v2，重跑 Task 8，禁止动态 LoRA。
