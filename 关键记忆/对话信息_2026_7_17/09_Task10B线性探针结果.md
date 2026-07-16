# Task 10B v2 冻结视觉线性探针

- 结论：`PASS`；只授权规划 Task 10C，不授权直接执行或启动大训练。
- 数据：IP102 detection 官方 trainval；16 类；train/val/dev=`192/48/80`；源图与近重复组件跨 split 重叠均为 `0`。
- 表征：冻结 Qwen2.5-VL-3B 视觉塔，320×2048 float32；全部参数冻结；特征 L2 范数约 1；峰值显存 7.61 GB；提取 37.70 s。
- Dev：Accuracy=`0.8250`，Macro-F1=`0.8094`；head/medium/tail F1=`0.8439/0.9167/0.8040`。
- 稳健性：3 seeds 最差/均值 Macro-F1 均为 `0.8094`；1000 次源图 bootstrap 95% CI=`[0.7043, 0.8714]`。
- 对照：标签置换 Macro-F1 均值=`0.0278`；无图基线=`0.0503–0.0689`；平均视觉增益=`0.7517`。
- 完整性：protocol、smoke、formal features、evaluation 的 `completion.sha256` 均通过；无 failure 文件。
- 环境：NumPy 1.26.3、SciPy 1.15.3、scikit-learn 1.6.1；LogisticRegression(C=1, balanced, lbfgs, max_iter=2000)。
- 科学解释：证明当前 Qwen 冻结视觉表征在严格去重的小规模 IP102 子集上含有强可分信息；尚不能证明端到端诊断、Evidence-First/null 可靠性或完整研究方法成立。
- 本地归档：`artifacts/2026-07-17_task10/10B_v2_linear_probe/`（30 文件，SHA256 复核通过，Git 忽略）。
