# Task 10 Micro-First研究路线（长期关键记忆）

日期：2026-07-17

## 路线核心

Task9D说明单体结构化QLoRA未证明开放诊断；v2.2否定TaxMask病因。后续固定采用：

`Open Diagnosis → Candidate Verification → Evidence Grounding → Abstention`

禁止直接扩大Task9E、换7B或叠加动态模块。

## 文献落点

- Qwen2.5-VL：先审计动态resize与bbox坐标链。
- MMStar：所有诊断结果报告image/no-image Visual Gain。
- M3ID：用teacher-forced PDM-H判断taxonomy/evidence/bbox token是否看图。
- HallusionBench：按family报告原图正确且反事实拒答的联合成功。
- POPE：使用random/popular/adversarial语义负候选。
- VCD：只在两阶段普通greedy已通过后做32-family推理微实验。
- AgMMU：留作后期固定版本外部评测。

## 强制顺序

1. Task10A：32-family bbox/PDM-H/pair法医，无训练。
2. Task10B：16类冻结Qwen视觉特征线性探针，验证数据可学习。
3. Task10C：16类diagnosis-only QLoRA，3 seeds×64 steps。
4. Task10D：32-family Top-3诊断→候选核验→定位→拒答闭环。
5. 全部通过后才申请扩大到32/97类；官方test只在最终协议冻结后运行一次。

## 全局门控

超过1,000 train families、3 GPU小时、完整三seed全量矩阵、官方test、新backbone或新模型模块，均属于大型实验。必须先通过对应微实验、冻结报告与哈希，并再次获得用户批准。失败后禁止靠加步数、换seed或放宽阈值抢救。

## 关键分流

- 线性探针失败：数据/标签/分辨率问题，停止VLM训练。
- 线性探针通过而QLoRA失败：生成式taxonomy接口问题，考虑轻量分类/检索头。
- diagnosis通过而verifier无视觉依赖：重做verification-only微实验。
- A/B/C/D全过：分解路线成立，再做正式规模。

完整书面规格：`docs/superpowers/specs/2026-07-17-task10-micro-first-design.md`。

