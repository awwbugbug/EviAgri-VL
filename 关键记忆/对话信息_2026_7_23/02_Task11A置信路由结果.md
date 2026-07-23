# Task 11A 置信路由结果（2026-07-23）

- 正式产物：服务器 `task11a_r4/evaluation_v2`；本地 `artifacts/2026-07-23_task11/11A_confidence_router/`；全部 completion SHA256 通过。
- 输入：Task10B 签名原图特征 320x2048；新增 640 个 val/dev blank、blur、shuffle 冻结视觉特征；Qwen 参数全冻结，提取 119.9s，峰值显存 7.61GB。
- Router：同一 LogisticRegression；val-only 温度=`0.18887`、阈值=`0.63`；三 seed。
- 原图：强制 Macro-F1=`0.80943`；置信拒答后=`0.78566`，差值=`-2.38pp`，paired bootstrap 95% CI=`[-9.02pp,+2.52pp]`；coverage=`81.25%`，selective accuracy=`92.31%`。
- Null：blank FPR=`0`；blur FPR=`10.00%`，95% CI=`[3.75%,17.50%]`；shuffle FPR 均值=`7.08%`，CI=`[2.49%,12.50%]`；overall FPR=`5.42%–6.25%`。
- JSON：syntax/schema/semantic/task compliance 均为 1.0，但这是确定性 renderer 的工程保证，不等于语言模型生成能力。
- 预注册决策：`FAIL`，仅因 blur 要求严格 `<10%` 而观测值恰为 `10%`；其余 gates 三 seed 全通过。禁止大型训练，未授权原 Task11B。
- 科学解释：显式 router 已解决 canonical pest-ID/JSON 接口，但 softmax confidence 不能充分表示证据质量；下一步先法医审计 8 个 blur false positives，再设计单变量、极小的显式 evidence/OOD head。不得回到加步数 QLoRA。
- 编排记录：`task11a`/`r2` 是 screen 命令拆分产生的空壳；`r3` 在模型加载前因提取环境误引 sklearn 被拦截；均无 GPU 计算/科研输出，保留不覆盖。`r4/evaluation_v2` 为唯一正式结果。
