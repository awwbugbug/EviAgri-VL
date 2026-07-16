# Task 9B v2 数据协议冻结结果

- 结论：9B 构建与冻结审计通过；这只证明数据协议/工程完整性，尚未通过 9C shortcut gate。
- 规模：18,790 families / 56,370 rows；train 14,941、val 1,930、task9_dev_audit 1,919 families。
- 配比：positive / real semantic null / synthetic visual null 均为 18,790，严格 1:1:1。
- 锁定：Task 8 的 186 IDs + 186 SHA 仅作盲化排除；未读取预测/指标调参。
- 隔离：source ID、source SHA256、near-duplicate component 跨 split 均为零交叉。
- JSON：syntax/schema/semantic/task compliance 均为 1.0。
- 完整性：服务器 `sha256sum -c --quiet completion.sha256` 返回 0；manifest 37,592 files，completion 37,593 entries。
- 产物：服务器 `datasets/derived/static_qlora_v2_protocol/2026-07-15/`；本地紧凑报告 `artifacts/2026-07-15_task9/9B_v2_protocol_freeze/`。
- 下一步：停在 9C 前；分别对 user prompt、system+user、prompt+非图像元数据执行 BA/AUROC gate，任一 >0.55 禁止训练。
