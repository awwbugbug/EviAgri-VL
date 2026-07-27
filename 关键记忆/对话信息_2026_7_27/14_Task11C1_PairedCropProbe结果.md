# Task 11C.1 原图 vs 证据裁剪冻结特征探针

- 协议：同一 16 个 IP102 正样本 + 16 个 PlantSeg real-null；冻结 Qwen 视觉塔、Task10B 训练分类器、3 seeds、温度 0.188874、阈值 0.63；唯一变量为原图/Task11C0局部输入。
- 工程记录：`attempt_01` 因日志父目录不存在，正式脚本未启动、模型/GPU未加载；保留 failure SHA `082a3460...c46f0b`。科学协议不变，`attempt_02` 完成且全部 SHA256 通过。
- 结果：`BLOCK`。正样本 Accuracy 0.8125→0.875，但 Supported Diagnosis 0.75→0.6875，真实类别概率均值下降 0.01380（95% CI `[-0.0742, 0.0266]`）。
- 可靠性恶化：PlantSeg FPR 0→0.125，平均置信度增加 0.09140，配对 bootstrap 95% CI `[0.0102, 0.1879]`；两张病斑裁剪被高置信接受（0.8190、0.6687）。
- 管线可信：identity fallback 特征 cosine 最小值 1.0、预测完全一致；effective crop 中位 cosine 0.8850，证明输入变量确实生效。
- 解释：直接用局部病斑替换全图会移除上下文，并把病斑纹理推向害虫类别；同时与“全图训练”的分类头产生分布偏移。不能扩大裁剪实验或训练。
- 本地证据：`artifacts/2026-07-27_task11/11C1_paired_crop_probe_result`。下一候选只能是新鲜样本上的 context-preserving 双视图微验证，禁止复用本批调参。
