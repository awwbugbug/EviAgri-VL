# Task 10C C2 结果（64-step 微实验）

- 协议 SHA：`84d2d1b20d4a781bc6fca8c4e9c41dd57051b6201287193681409451095edc90`；同一 64-row smoke-train，seeds=17/29/43，checkpoints=8/16/32/64。
- 完整性：3 次训练、12 checkpoints、12 smoke 推理、4 full-dev 推理、4 candidate scoring；36/36 completion 与 12/12 adapter 本地复验通过，failure=0。
- 决策：`STRUCTURAL_FAILURE`；learning signal 0/3 seeds；九项科学门槛通过 4、失败 5；禁止扩大训练或自动进入下一实验。
- 严格分类：D0 Base image Macro-F1=0；D1 三种子均值=0.005145（17=0，29=0.0100，43=0.005435）；paired bootstrap 95% CI=[0, 0.009115]；D2 冻结视觉探针=0.809432。
- 输出可靠性：syntax=1.0，但 D1 image schema validity 仅 0.03125/0.10625/0.09375；visual gain 均值=0.005145；no-image Macro-F1=0；最大 prompt gap=0.01667。
- 非绑定候选似然 Top-1：Base=0.0625；D1=0.0500/0.05625/0.0625，接近 16 类随机水平；说明失败不只是 JSON parser，而是 canonical pest-ID 映射未学成。
- 学习曲线：step 8→64 几乎无 Macro-F1/visual-gain 增长；仅 seed43 step64 smoke Macro-F1=0.025，未达到预注册学习信号。
- 客观解释：视觉表征本身可用（D2 强），但当前 `q_proj/v_proj` diagnosis-only Static QLoRA、64 样本/64 steps 无法把视觉类别信息可靠写入自回归 canonical JSON 输出。否定的是当前桥接/监督方案，不是农业视觉方向或数据全部失效。
- 本地归档：`artifacts/2026-07-17_task10/10C_c2_learning_curve/`；代码分支：`task10c-c2-learning-curve`。服务器保持开机。
