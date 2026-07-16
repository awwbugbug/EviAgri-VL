# Task 10C Diagnosis-Only Static QLoRA 微实验设计

日期：2026-07-17
状态：设计已确认，待用户复核书面规格
上位规格：`docs/superpowers/specs/2026-07-17-task10-micro-first-design.md`

## 1. 唯一科学问题

Task 10B v2 已证明冻结 Qwen2.5-VL-3B 视觉表征在严格去重的16类 IP102 子集上可线性分离。Task 10C 只检验：移除候选核验、bbox、证据、可靠性和拒答等复合目标后，Static QLoRA 能否将同一视觉信息稳定映射为无候选提示下的 canonical pest ID。

本实验不检验 Evidence-First、null-evidence、定位或两阶段闭环，不允许同时改变模型、数据、输出schema或训练超参数。

## 2. 事实驱动修订记录

本规格补充并取代上位规格中与 Task 10C 冲突的旧细节，不回写或覆盖历史文件。

| 原始设想 | 本次冻结值 | 修改依据 |
|---|---|---|
| 每类20/5/10，合计320/80/160 | 每类12/3/5，合计192/48/80 | Task 10B v1 精确配额不可行；Task 10B v2 exact split 已通过类别、SHA和近重复组件审计 |
| 重新按规则构建10C数据 | 逐行复用10B v2 manifest | 保证D1与D2严格同数据比较，避免重新抽样造成数据漂移 |
| 8-step后进入3×64-step | 3×8-step完成后强制停止并汇报 | 用户要求任何较长训练前先进行独立可信度/可行度确认 |
| D2仅为抽象参考上限 | D2实测Macro-F1=0.8094315407，70%门槛=0.5666020785 | Task 10B v2正式结果及1000次source-image bootstrap已经通过 |
| 10C通过后可进入既有10D | 10C通过只授权规划新的verification-only微实验 | Task 10A虽证明视觉依赖，但旧verifier正确original TPR和strict family success均为0，pair contract失败 |

## 3. 数据冻结与隔离

- 输入必须逐行复用 `artifacts/2026-07-17_task10/10B_v2_linear_probe/protocol/manifest.jsonl` 对应的服务器原始manifest。
- manifest SHA256必须为 `84d2d1b20d4a781bc6fca8c4e9c41dd57051b6201287193681409451095edc90`。
- 类别固定为 `9,10,16,17,22,24,45,50,64,68,71,82,83,87,99,101`；head/medium/tail为6/5/5类。
- split固定为train/val/dev=`192/48/80`，每类分别为`12/3/5`。
- 仅train参与参数更新；val只报告管线完整性；dev只在协议冻结后评测，不选checkpoint。
- source SHA256与near-duplicate component跨split重叠必须均为0。
- 只使用IP102 detection官方trainval；不得读取Task 8 locked内容、官方test、AGE或新增数据。
- prompt和模型输入不得包含路径、文件名、类别目录、source ID、class band、positive/null或其他非像素标签。

## 4. 对话与输出契约

固定system消息：

```text
You are an agricultural pest image classifier. Follow the requested output format exactly.
```

训练/主评测user消息：

```text
Identify the insect pest shown in this image. Return exactly one JSON object containing its canonical IP102 class ID.
```

未见等价user消息：

```text
Which insect pest category is visible in this image? Reply with one JSON object containing only its canonical IP102 identifier.
```

唯一合法assistant target为紧凑JSON：

```json
{"pest_id":"IP009"}
```

- `pest_id`必须匹配`^IP\d{3}$`并属于冻结16类。
- 禁止物种名、bbox、evidence、stage、解释、markdown或额外字段。
- 所有类别使用完全相同的字段、顺序和字符长度规范。
- no-image条件仅删除image content，system与user文本逐字不变。
- 严格解析器拒绝前后缀、代码围栏、非法JSON、额外字段和非冻结ID，并保留raw output与失败类型。

## 5. 模型、图像与训练冻结

- Base固定为服务器现有Qwen2.5-VL-3B-Instruct，并记录模型文件SHA256。
- processor固定`min_pixels=200704`、`max_pixels=401408`。
- QLoRA固定NF4、`r=16`、`alpha=32`、`dropout=0.05`、targets=`q_proj,v_proj`。
- per-device batch size=`1`，gradient accumulation=`8`，learning rate=`1e-4`。
- loss reduction固定为per-example active-token mean，再做batch mean。
- seeds固定`17,29,43`；除seed与确定性shuffle外配置完全一致。
- 不早停、不按val选checkpoint、不解冻新模块、不从smoke续跑。
- greedy解码固定`do_sample=false`、`num_beams=1`、`max_new_tokens=32`，其余采用模型默认EOS契约。

## 6. 阶段C0：训练前协议门控

任何训练前必须生成带SHA256的preflight report，并同时满足：

1. manifest SHA、行数、类别和split精确匹配本规格；
2. source SHA和near-duplicate overlap均为0；
3. prompt、metadata和图像路径没有进入模型文本；
4. canonical target逐行可严格解析且ID与class_id一一对应；
5. train/val/dev角色不可写错，Task 8与官方test引用数为0；
6. 模型、processor、LoRA、loss、解码和seed配置已快照并签名。

任一失败写`blocked`与明确原因，禁止加载训练器。

## 7. 阶段C1：三种子8-step工程smoke

- smoke train固定每类4张，共64张；按source SHA排序后每类取前4张，三seed使用相同样本。
- 每个seed从同一Base独立训练8 optimizer steps，恰好64次样本暴露。
- smoke dev固定每类1张，共16张；按source SHA排序后每类取第一张。
- 每个adapter评测原图/无图 × 训练提示/未见提示，共64条预测。
- smoke结果不用于选seed、调超参数、修改阈值或宣称科学提升。

每个seed必须满足：

- 恰好8 optimizer steps、64 exposures；loss与梯度范数有限；
- 仅冻结配置中的LoRA参数可训练；
- adapter保存、重载和SHA256一致；
- 64条推理全部完成，raw output、解析结果和失败类型完整；
- 记录train loss、peak VRAM、耗时、实际样本数、状态、日志和completion SHA256；
- 无failure文件、数据越界或输入契约违规。

syntax/schema、类别多样性、Macro-F1、prompt gap与Visual Gain只作非约束性观察。三seed全部完成后必须停止并向用户汇报；未经再次批准不得运行阶段C2。

## 8. 阶段C2：三种子64-step正式微实验

仅在C1全部通过且用户再次批准后执行：

- D0：Base Qwen，同协议、同parser、同解码；
- D1：Diagnosis-only Static QLoRA，三seed均从Base独立训练64 optimizer steps；
- D2：Task 10B v2线性探针，作为视觉可学习性参考。

D1每seed固定64 steps、512 exposures，使用最终step adapter；不得早停、挑checkpoint或续跑smoke。

## 9. 冻结评测

- 80张dev均运行训练提示与未见提示；二者均运行image/no-image配对输入。
- primary Top-1来自greedy严格JSON解析。
- Top-k不要求自由生成列表；对16个合法assistant字符串计算active answer token的平均条件log-likelihood，报告Top-1/3/5。
- 报告Accuracy、Macro-F1、head/medium/tail Macro-F1、混淆矩阵、类别覆盖、syntax validity、schema validity和解析失败类型。
- Visual Gain=`MacroF1_image - MacroF1_no_image`。
- prompt gap为同seed有图条件下训练提示与未见提示Macro-F1差值的绝对值。
- D1-D0使用source image为单位的1000次paired bootstrap；报告每seed和pooled 95% CI。
- 报告三seed均值、样本标准差和最差seed；不得只报最佳seed。

## 10. 预注册决策

全部满足才`PASS`：

1. D1平均Macro-F1至少比D0高0.05；
2. D1-D0 pooled paired bootstrap 95% CI下界大于0；
3. 至少2/3 seeds的Macro-F1高于D0；
4. D1 image/no-image Visual Gain至少0.10；
5. D1平均Macro-F1至少0.5666020785；
6. 每个seed的绝对prompt gap小于0.05；
7. 每个seed、每种评测条件的syntax与schema validity均不低于0.99；
8. 最差seed的no-image Macro-F1不高于0.10；
9. source SHA与near-duplicate overlap保持0。

## 11. 失败分流与边界

- `PASS`：只证明16类生成式开放诊断接口可行；只授权规划新的verification-only微实验，不授权直接复用旧verifier。
- D1明显低于D2：保留冻结视觉特征，转向轻量分类/检索头；不扩大QLoRA。
- D1与D0接近或Visual Gain不足：停止增加训练步数，诊断视觉到语言接口。
- prompt gap或no-image失败：先修模板/语言先验协议，不加模型模块。
- 任一结果都不得自动扩大类别、使用官方test、进入动态LoRA/Gating、SAM2、7B或新backbone。

## 12. 产物与失败安全

每个阶段使用全新目录，拒绝覆盖已有输出。必须保存`status.json`、输入manifest及SHA、`config.snapshot.json`、日志、adapter SHA、predictions、metrics、failure（仅失败）和`completion.sha256`（仅完整成功）。服务器保持开机；任何异常先保留现场并只读诊断。
