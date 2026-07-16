# Task 9B v2.1 精确修复与 9C 复验

- v2.1 约束：每个 `split × template` 独立整数最大流；semantic query 逐类计数必须与 positive 完全一致，且不得命中图中任一有效类别；不可行即 BLOCK，无近似降级。
- 可行性：8 个分层全部满流；query 分布 max TV=0。
- family 完整性：18,790→18,790；rows 56,370→56,370；无删除、无复制、无跨 split/template 移动；37,580 图像以 hardlink 原样复用。
- 冻结审计：四级 JSON 质量均 1.0；source ID/SHA/near-duplicate component 零交叉；服务器 completion SHA256 通过。
- 同协议 9C：固定 TF-IDF + class-balanced LR、seed=20260715、60k features、阈值 0.55，未调参。
- 9C 结果：PASS，violations=0。user prompt BA/AUROC val=0.5/0.5、dev=0.5/0.5；system+user 同为约 0.5；metadata val=0.500/0.498、dev=0.501/0.495。
- 验证：v2.1 completion SHA=PASS；9C metrics SHA=PASS；本地全量 138 tests passed。
- 产物：服务器 `datasets/derived/static_qlora_v2_1_protocol/2026-07-15/`；本地 `artifacts/2026-07-15_task9/9B_v21_exact_repair/` 与 `9C_shortcut_gate/formal_v21/`。
- 下一步：可以进入 9D A/B/C 三随机种子小规模消融；尚未启动任何训练。
