# Task 8 同协议审计（2026-07-15）

- 实例保持开机；除非用户明确要求，禁止自动关机/定时关机。
- B0–B3、六条件反事实、严格语义 JSON schema、条件指标、family-level 1000 次 bootstrap、paired bootstrap、McNemar 已实现；本地 `89 passed`。
- 泄漏审计发现 IP102 官方 split 有近重复：2050 个 dHash 候选，二级复核 1781 对高相似；主审计排除 673 张污染 test 图。
- 清洗后覆盖 93/102 类、每类 2 张，共 186 families；9 类因无干净 test 图不进入主结论。正式清单选中污染图数=0。
- clean-v4 smoke：4 families/96 predictions，96/96 schema 有效，全部工程门禁通过，峰值显存 8.30 GB。
- smoke 信号：B1 正样本 Acc=0.75，B2=0.25；B2 null FPR=0.65、合规错误=0.65；B3 null 指标仍满分，疑似模板捷径，需正式统计确认。
- 正式运行目录：`/root/autodl-tmp/EviAgriDiag/experiments/task8_causal_audit/2026-07-15/formal_clean_v2`；186 families、1116 条条件记录、4464 个 B0–B3 作业，已启动可断点续跑推理，预计约 3 小时。
- v1 adapter SHA256：`162237f0141d0bfea6b01fe6d0bcc08c86efd4b2b06e8f525f5676fa6d425284`。
- 分流规则不变：正式结果若确认模板捷径则走 B（修数据/提示→Static QLoRA v2），禁止直接上动态 LoRA。

