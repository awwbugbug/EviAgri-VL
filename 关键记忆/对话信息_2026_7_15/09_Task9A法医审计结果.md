# Task 9A v1数据法医审计结果（2026-07-15）

- 状态：9A 完成；只读 `static_qlora_v1` train/val/test，`locked_set_read=false`；未读取 Task 8 predictions，未训练。
- 权威产物：`artifacts/2026-07-15_task9/9A_v1_forensics_r3/`；服务器 `.../task9/9A_v1_forensics/2026-07-15_r3/`；SHA256 全通过。
- 数据量：train `13652 pos + 6826 null`；val `1526+1526`；test `3798+3798`；总计 `31126`。
- Critical：train 拟合、val/test 评估的三路 text-only probe（user；system+user；prompt+metadata）均为 Balanced Accuracy `1.000`、AUROC `1.000`。
- 直接首因：positive 固定 `Identify...`，null 固定 `Is [class]...`；仅用户提示即可无图完美判别标签。
- Null 构造：`12150/12150` 全是 `prompt_conflict_null_evidence` synthetic null，real null=`0`；每个 null 与 positive 复用同一 image_id（train/val/test overlap=`6826/1526/3798`）。
- 输出捷径：null target 全集仅 `1` 个唯一答案、长度固定 `140`；positive `18976` 个唯一答案、长度 `151–183`。
- 元数据泄漏：task_type 与标签一一对应；`31126/31126` record ID 含 positive/null。当前 collator 只用 messages，故它们不是 v1 的直接模型输入，但属于高风险管线泄漏源。
- 路径检查：prompt 中图像路径/文件名暴露 `0`；精确 image_id 跨 split `0`。近重复不是精确 ID，9B 仍必须使用既有高相似簇按簇重分 split。
- JSON 四级质量：syntax validity、schema validity、semantic consistency、task compliance 均 `31126/31126=1.000`；说明失败不是 JSON 语法，而是监督信号设计。
- 长尾：positive 仅覆盖 `97` 类，聚合计数 min=`2`、median=`98`、max=`2933`，max/min=`1466.5`；null 查询覆盖 `102` 类且较均衡（`96–137`）。
- 结论：v1 数据协议不具备因果可识别性，不能复用；必须进入 9B 重构并冻结 v2 协议，之后再运行 9C shortcut gate。
- 验证：专项 `12 passed`；全量 `101 passed`。服务器保持开机。
