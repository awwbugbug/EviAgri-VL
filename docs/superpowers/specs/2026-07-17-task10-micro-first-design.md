# Task 10 Micro-First 诊断—核验—定位分解实验设计

日期：2026-07-17  
状态：待用户审阅书面规格  
实验根目录：`/root/autodl-tmp/EviAgriDiag/experiments/task10_micro_first/2026-07-17`

## 1. 背景与已确定事实

Task 9D 已证明单体结构化 Static QLoRA 能学习输出格式和一部分候选核验行为，但没有证明开放害虫诊断；v2.2 又否定了“taxonomy value token 的 loss 干扰是主要病因”：TaxMask 相对 Control 的 Balanced Accuracy 均值下降 0.625pp，Overall Null FPR 恶化 3.33pp，且 pooled 95% CI 不支持改善。因此禁止通过增加训练步数、扩大数据或叠加模块继续抢救现有复合任务。

Task 10 将任务拆为：

`Open Diagnosis → Candidate Verification → Evidence Grounding → Abstention`

每一层先独立确认“数据可学、模型看图、指标可信”，再允许连接下一层。

## 2. 研究目标

1. 判断历史定位低分是否由 bbox 坐标协议错误造成。
2. 判断现有 verifier 的关键输出 token 是否真正依赖图像。
3. 判断 IP102 图像在 Qwen 视觉特征空间是否具有可学习的类别信息。
4. 判断简化后的 diagnosis-only QLoRA 是否能完成无候选提示的开放分类。
5. 仅在前述条件成立后，验证轻量两阶段闭环是否同时保持诊断能力并降低 null hallucination。

## 3. 明确非目标

- 不运行 Task 9E，不读取 Task 8 locked 186 families。
- 不下载7B或新backbone，不实现动态LoRA/Gating、SAM2、DPO或全量VCD训练融合。
- 不使用IP102官方test做训练、早停、模板选择或微实验评估。
- 不在Task 10中加入AGE、病害数据或新的大型数据集。
- 不修改、删除或覆盖Task 9D/v2.2已有产物。

## 4. 文献思想与实验映射

| 参考工作 | 采用的思想 | Task 10落点 |
|---|---|---|
| Qwen2.5-VL | 动态分辨率、实际输入尺寸上的绝对定位 | 10A-1 bbox坐标链与round-trip审计 |
| MMStar | image/no-image视觉增益、无图泄漏诊断 | 10B、10C所有模型的Visual Gain |
| M3ID | 有图/无图分布差异PDM，优先Hellinger距离 | 10A-2 teacher-forced token级PDM-H |
| HallusionBench | 原图/编辑图控制对、pair/figure consistency | 10A-3与10D的family级联合成功 |
| POPE | random/popular/adversarial负查询 | 10D三类semantic negatives |
| VCD | 原图/扰动图对比解码 | 仅在10D主协议通过后的32-family可选推理微实验 |
| AgMMU | 感知错误与知识错误分解 | 后续外部评测；本Task不下载、不使用 |

## 5. Micro-First全局硬门控

满足任一条件均视为“大型实验”：训练families超过1,000、预计GPU时间超过3小时、运行完整三种子全量矩阵、使用官方test、下载新backbone或加入新模型模块。大型实验必须满足：

1. 对应微实验通过预注册门槛；
2. 决策报告、输入哈希和代码哈希已冻结；
3. 再次获得用户明确批准。

未通过时禁止通过更换seed、增加步数或放宽阈值补救。任何门槛变更必须发生在结果产生前并留下新版本。

## 6. 数据与隔离协议

### 6.1 Task 10A输入

- 使用v2.2既有32个`task9_dev_audit` families及其非锁定manifest。
- 只读使用现有Control预测和图像；不训练、不写回原目录。

### 6.2 Task 10B/C输入

- 类别：从v2.2已冻结32类中，按官方trainval图像数量固定选择16类，head/medium/tail为6/5/5。
- 每类：20 train、5 val、10 dev，共320/80/160张。
- 所有样本只来自IP102官方detection `trainval`。
- 排除Task 9D/v2.2 train、val、dev使用过的source SHA256，避免复用旧实验图像。
- split按source SHA256和near-duplicate component隔离；任何重复组件只能进入一个split。
- 若任一类别无法满足精确配额则BLOCK；不得复制、移动、降级为近似匹配。
- 官方test和Task8 locked set仅检查“不被引用”的哈希/ID边界，不读取图像或标签内容。

### 6.3 Task 10D输入

- 从10C dev中每类固定取2个family，共32 families。
- 每个family构造：original、semantic-random、semantic-popular、semantic-adversarial、blank、blur、shuffle、source-visual-null，共256 rows。
- semantic negative必须在图像GT类集合外；popular/adversarial统计只由Task 10B train部分计算。
- blank颜色、blur半径、shuffle网格与未来confirmatory set保持不同并记录参数。

## 7. Task 10A：无训练法医审计

### 7.1 A1 Bbox坐标链

对32个正样本记录：原图W/H、processor实际像素W/H、`image_grid_thw`、GT frame、模型输出frame、转换矩阵。固定比较三种解释：原图绝对坐标、processor输入绝对坐标、0–1000归一化坐标，但主协议只采用Qwen官方语义与实际代码链一致的解释。

使用人工已知矩形执行`original → processed → original` round-trip：

- 最大坐标误差必须≤1 pixel；
- synthetic round-trip IoU必须≥0.999；
- 32/32样本必须得到合法尺寸和变换记录；
- prompt、训练target和evaluator必须声明同一frame。

任何一项失败则bbox状态为`BLOCKED_COORDINATE_PROTOCOL`，历史mIoU保留但标记为不可解释；只允许在新协议中修复，不改历史输出。

### 7.2 A2 PDM-H视觉依赖

同一assistant target分别进行有图与无图teacher-forced前向。对每个有效target token计算：

`H(P,Q) = sqrt(sum((sqrt(P_i)-sqrt(Q_i))^2)) / sqrt(2)`

其中P为有图下一token分布，Q为删除image content但保留完全相同文本prompt的分布。按JSON span分组：`evidence_present`、taxonomy value、bbox value、refusal/uncertain、其他assistant tokens。

质量门槛：

- 目标token映射覆盖率≥95%；
- 所有分布有限且归一化误差≤1e-5；
- 32 families全部同时具有conditioned/unconditioned结果。

视觉依赖门槛：original相对blank/shuffle，在taxonomy或`evidence_present`至少一组上的paired mean PDM-H差值>0，且1,000次family bootstrap 95% CI下界>0。未通过则现有verifier不得进入10D。

### 7.3 A3 Family/Pair指标

分别报告original positive TPR、各null FPR、original-to-intervention具体诊断下降率，以及：

- pair success：original接受正确且指定单个干预拒答；
- strict family success：original正确且五类null全部拒答；
- contradiction rate：同family出现因果相反输出的比例。

这部分是法医描述，不设置阻止10B/10C的阈值；它只决定能否复用现有verifier。

## 8. Task 10B：视觉特征可学习性微实验

### 8.1 特征与分类器

- Backbone固定为现有官方Qwen2.5-VL-3B-Instruct，不更新任何参数。
- 使用processor在200704–401408 pixels处理图像。
- 提取视觉塔post-merge visual tokens，mean pooling后L2归一化。
- 分类器为multinomial logistic regression：`C=1.0`、`class_weight=balanced`、`max_iter=2000`。
- seeds固定17/29/43；split完全相同，只改变分类器随机种子。
- 增加逐seed label-permutation control。

### 8.2 指标与通过条件

- Accuracy、Macro-F1、head/medium/tail Macro-F1；
- 1,000次family bootstrap 95% CI；
- 三seed均值、样本标准差、最差seed。

全部满足才PASS：

- mean Macro-F1≥25%；
- worst-seed Macro-F1≥20%；
- pooled Macro-F1 95% CI下界>12.5%（16类随机水平6.25%的两倍）；
- permutation-control mean Macro-F1≤10%；
- train/dev SHA与near-duplicate overlap均为0。

失败则停止QLoRA训练，优先调查分辨率、标签、类别可辨性和数据质量。

## 9. Task 10C：Diagnosis-Only Static QLoRA微实验

### 9.1 单一任务

输入prompt只要求识别图像中的害虫，不出现候选类别、positive/null、路径、文件名或非图像标签。assistant唯一输出：

```json
{"pest_id":"IP012"}
```

不包含bbox、evidence、reliability、stage或解释字段。JSON字段顺序固定，所有类别采用相同长度规范的canonical ID。

### 9.2 模型与训练

- D0：Base Qwen zero-shot，同一prompt和parser。
- D1：diagnosis-only Static QLoRA，NF4、r=16、alpha=32、dropout=0.05、targets=`q_proj,v_proj`。
- D2：10B线性探针，作为视觉可学习性参考上限。
- D1 seeds为17/29/43；每seed先执行8-step engineering smoke，随后固定64 optimizer steps。
- batch=1、gradient accumulation=8、lr=1e-4、per-example active-token mean then batch mean。
- 不早停、不根据dev选择checkpoint，固定最终step。

### 9.3 评估

- Top-1通过严格JSON解析。
- Top-k不要求模型生成列表，而是对16个canonical output字符串计算长度归一化条件log-likelihood并排序，避免自由生成格式混入候选能力。
- 有图与无图使用同一文本prompt；报告MMStar式项目Visual Gain=`MacroF1_image-MacroF1_no_image`。
- 对未见等价prompt改写报告prompt gap。

全部满足才PASS：

- D1 mean Macro-F1相对D0至少+5pp；
- paired family bootstrap差值95% CI下界>0；
- 至少2/3 seeds的Macro-F1高于D0；
- D1 image/no-image Visual Gain≥10pp；
- D1达到D2 mean Macro-F1的至少70%；
- worst-seed prompt gap<5pp；
- syntax/schema validity均≥99%；
- 无图text-only Macro-F1≤10%。

## 10. Task 10D：两阶段32-family闭环

仅当10B、10C通过，且10A-2证明现有verifier具有视觉依赖时执行。

1. D1按长度归一化条件似然生成Top-3候选。
2. 每个候选交给相同seed的v2.2 Control verifier（17↔17、29↔29、43↔43），避免事后挑选最佳seed。
3. verifier接受时才允许输出evidence region；全部拒绝则输出abstain。
4. 对三类POPE式semantic negatives与四类视觉反事实成对评估。

全部满足才PASS：

- 两阶段Macro-F1相对对应D1下降不超过3pp；
- overall、blank、blur Null FPR均<10%；
- Supported Diagnosis Rate相对D1提升，paired bootstrap 95% CI下界>0；
- strict family success相对单体verifier提升，95% CI下界>0；
- syntax、schema、semantic consistency、task compliance均≥99%；
- 三个matched seeds中至少2个方向一致。

VCD不是主门槛。只有10D普通greedy通过后，才允许在固定seed29、同一32 families上比较greedy与greedy-VCD；解码参数和parser保持不变，不训练新参数。

## 11. 分流决策

| 结果 | 病因判断 | 后续动作 |
|---|---|---|
| 10B失败 | 图像/标签/分辨率或类别可辨性不足 | 停止VLM训练，做数据与视觉特征审计 |
| 10B通过、10C失败 | 视觉特征可分，但生成式taxonomy接口失败 | 评估轻量分类/检索头；不进入动态LoRA |
| 10B/C通过、10A verifier失败 | 开放诊断可学，现有核验器不真正看图 | 重新设计verification-only微实验 |
| 10A/B/C/D全部通过 | 分解路线可行 | 冻结协议并申请扩大到32/97类 |

任何分支都不得自动进入全量实验。

## 12. 产物与失败安全

每个子任务独立目录，必须包含：

- `status.json`：`running/completed/failed/blocked`；
- `input_manifest.json`及SHA256；
- `config.snapshot.json`；
- `metrics.json`或正式report；
- `failure.json`（仅失败时）；
- `completion.sha256`（仅完整成功时）。

所有构建器默认拒绝已存在output；不覆盖、不删除、不自动重跑。BLOCK时写明不可行约束。服务器不关机。每个子任务完成后写入当天简短关键记忆。

## 13. 执行顺序与审查点

1. 10A-1 bbox audit；若BLOCK，仅阻止定位结论。
2. 10A-2 PDM-H与10A-3 pair audit；决定现有verifier是否可复用。
3. 10B线性探针；未通过则停止。
4. 10C diagnosis-only 8-step smoke；工程失败则修复后重新冻结。
5. 10C三seed 64-step微实验；未通过则停止。
6. 10D两阶段闭环；未通过则停止。
7. 汇总Task 10 decision report，再由用户决定是否扩大实验。

正式IP102 test、AGE、外部数据、VCD扩展和任何大型训练均不属于本规格的自动执行范围。

