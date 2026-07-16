# Task 10C C2 64-Step 学习曲线微实验设计

日期：2026-07-17  
状态：用户已批准方案 A，待书面规格复核  
上位规格：`docs/superpowers/specs/2026-07-17-task10c-diagnosis-only-design.md`

## 1. 唯一科学问题

C1 已证明训练、adapter 保存/重载、推理和严格评测链路可运行，但 8-step 三种子在四种条件下的严格 Macro-F1 均为 0。本实验只回答：在不改变数据、模型、提示、目标字符串、LoRA 配置和评测器的前提下，将独立训练增加到 64 optimizer steps，能否出现可复现的规范输出学习、类别学习和图像依赖趋势。

C2 不是正式全量训练，不检验 Evidence-First、null-evidence、定位、动态 LoRA/Gating、SAM2、7B 或新 backbone。

## 2. 单变量与不可变项

相对 C1 的唯一实验变量是每个 seed 的训练步数由 8 增加到 64，并在固定步数保存观察点。以下内容全部复用 C0 冻结协议：

- manifest SHA256：`84d2d1b20d4a781bc6fca8c4e9c41dd57051b6201287193681409451095edc90`；
- 16 类：`9,10,16,17,22,24,45,50,64,68,71,82,83,87,99,101`；
- train/val/dev=`192/48/80`，每类=`12/3/5`；
- Base：服务器现有 `Qwen2.5-VL-3B-Instruct` 及 C0 已签名模型文件；
- processor：`min_pixels=200704`、`max_pixels=401408`；
- QLoRA：NF4、`r=16`、`alpha=32`、`dropout=0.05`、targets=`q_proj,v_proj`；
- batch=`1`、gradient accumulation=`8`、learning rate=`1e-4`；
- loss reduction：per-example active-token mean，再做 batch mean；
- seeds=`17,29,43`；
- system/user 提示、紧凑 target `{"pest_id":"IP009"}`、严格 parser 和 greedy 解码全部不变。

每个 seed 必须从同一 Base 独立初始化；禁止续跑或加载 C1 adapter。64 steps 对应恰好 512 次样本 exposure。不得按 val、dev 或中间性能选择 checkpoint、seed 或超参数。

## 3. 数据边界

- 只读取 C0 已冻结的 train、val、dev 和由其确定的 smoke-dev；不重新抽样主 split。
- 只有 train 参与参数更新；val 只检查管线完整性；dev 只在设计与代码冻结后评测。
- Task 8 的 186 families、IP102 official test、AGE 和其他新增数据保持不可读。
- 模型消息不得包含路径、文件名、source ID、class band、目录类别或其他非像素标签。
- source SHA256 与 near-duplicate component 的跨 split 重叠必须继续为 0；任一不满足即在模型加载前 BLOCK。

## 4. 训练与 checkpoint

每个 seed 运行一次连续 64-step 训练，并保存 step=`8,16,32,64` 四个 adapter checkpoint。checkpoint 只承担学习曲线观察，不参与选择；正式 D1 始终定义为 step 64。

每个 checkpoint 必须记录：

- adapter 文件、字节数和 SHA256；
- global step、累计 exposure、逐步 finite loss/grad norm；
- peak allocated/reserved VRAM 与累计训练时长；
- trainable parameter 名称审计；
- Base 模型哈希、协议哈希、完整配置快照；
- `status.json`、`trainer_state.json` 和成功时的 `completion.sha256`。

只有工程异常可提前停止：数据边界违规、loss/grad 非有限、出现非 LoRA 可训练参数、checkpoint/hash 不完整、OOM 或进程异常。中间性能无论高低都不得提前停止、改协议或重跑。

## 5. 两级推理预算

### 5.1 中间趋势检查

step 8、16、32、64 在固定 16 张 smoke-dev 上运行四条件推理：

- image + train prompt；
- image + unseen prompt；
- no-image + train prompt；
- no-image + unseen prompt。

每 checkpoint 每 seed 恰好 64 条 prediction；只有同一 smoke-dev 上的四个 checkpoint 可以互相比较学习趋势。step 64 完成 smoke 推理后，再额外进入完整 dev 正式评测。中间结果只描述学习轨迹，不能授权选择或修改实验。

### 5.2 最终正式评测

step 64 在完整 80 张 dev 上运行相同四条件，共 320 条 prediction/seed。D0 Base Qwen 在相同 80 张图、提示、图像分辨率、解码参数、max tokens、parser 和评测脚本下运行一次 320 条预测。D2 使用已冻结的 Task 10B v2 线性探针结果作为视觉可学性参考，不重新调参。

最终才运行 1000 次 source-image paired bootstrap、三种子汇总和预注册决策。conditional log-likelihood Top-1/3/5 只在最终 step 64 的有图条件计算，不在中间 checkpoint 运行。

## 6. 主指标与非评分法医指标

### 6.1 严格主指标

决策只使用原冻结 parser 的结果：

- syntax validity、schema validity；
- Accuracy、Macro-F1、head/medium/tail Macro-F1；
- 混淆矩阵、预测类别覆盖；
- Visual Gain=`MacroF1_image - MacroF1_no_image`；
- train/unseen prompt gap；
- D1-D0 source-image paired bootstrap 95% CI；
- 三种子均值、样本标准差和最差 seed；
- 最终有图条件的 conditional log-likelihood Top-1/3/5。

### 6.2 非评分法医指标

为区分“没有学会类别”和“学会类别但格式错误”，额外只读报告：

- Markdown 围栏出现率；
- 去除单层代码围栏后、仍要求唯一字段 `pest_id` 和冻结 canonical ID 的 schema validity；
- raw text 中冻结 canonical ID 的 mention rate；
- 错误字段名、非 canonical 值、额外文本、截断和空输出的计数。

这些指标不得替代严格 parser、计入 Accuracy/Macro-F1、改变 PASS 阈值或用于选择 checkpoint。

## 7. 中间观察与最终决策

step 8/16/32/64 仅在相同 smoke-dev 上按 checkpoint、seed、条件报告严格 schema validity、严格 Macro-F1、canonical-ID mention rate、no-image Macro-F1 和 Visual Gain，不设置中间淘汰阈值。学习曲线中的 `image Macro-F1`、`no-image Macro-F1` 和 `image schema validity` 分别取两个对应 prompt 条件的算术平均，Visual Gain 为前两者之差。

最终科学 `PASS` 继续使用上位规格已冻结的全部九项规则，不因 C1 结果降低标准。除此之外增加解释性分流，但不改变 PASS：

- `LEARNING_SIGNAL_ONLY`：未满足科学 PASS，但至少 2/3 seeds 在同一 smoke-dev 上满足下列三项中的至少两项：step 64 相对 step 8 的 image schema validity 增加至少 0.25、image Macro-F1 增加至少 0.05、Visual Gain 增加至少 0.05 且 step 64 Visual Gain 大于 0。只允许设计一个新的单变量微实验，不允许扩大训练。
- `STRUCTURAL_FAILURE`：未满足科学 PASS，且少于 2 个 seed 满足上述学习信号规则。停止增加 QLoRA 步数，转向视觉到语言接口、目标表示和监督信号法医审计。
- `ENGINEERING_FAILURE`：运行完整性、数据边界、哈希或有限数门控失败。结果不用于科学判断，保留现场并只读诊断。

任何分流都不自动授权全量训练、官方 test、动态模块或新 backbone。

## 8. 产物、失败安全与停止点

C2 使用全新实验目录，拒绝覆盖 C1 或任何既有结果。训练、checkpoint 推理、最终推理和评估均分别写入状态、日志、配置、原始预测、指标、失败报告和完成哈希。

所有三个 seed 与最终评估完成后必须：

1. 核验服务器全部 completion 和 adapter SHA256；
2. 下载到本地 Git 忽略的 `artifacts/2026-07-17_task10/10C_c2_learning_curve/`；
3. 再次本地核验 SHA256；
4. 写入当日简短关键记忆并提交 GitHub；
5. 强制停止在 C2 结果汇报处，服务器保持开机，不自动进入任何后续训练。
