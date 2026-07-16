# Task 9C Shortcut Gate 结果

- 结论：`BLOCK`，禁止训练 v2；9B 工程协议虽完整，但文本标签泄漏仍超阈值。
- 固定 gate：word+char TF-IDF + class-balanced Logistic Regression；train 拟合，val/dev 独立评估；BA、AUROC 均须 ≤0.55。
- user prompt：val BA/AUROC=0.619/0.655；dev=0.610/0.649。
- system+user：val=0.619/0.655；dev=0.612/0.650。
- prompt+非图像元数据：val=0.531/0.600；dev=0.511/0.582。
- 根因：模板单独 BA/AUROC=0.5；查询虫种名单独 AUROC=0.655/0.650。positive 遵循真实长尾，semantic negative 近似均匀抽类，二者 query 分布 TV=0.469，类别名暴露标签。
- 完整性：三份 probe SHA 与 9B manifest 一致；9C metrics SHA 校验通过；本地完整测试 132 passed。
- 产物：`artifacts/2026-07-15_task9/9C_shortcut_gate/formal_v1/metrics.json`。
- 下一步建议：保留失败冻结集不覆盖；回到 9B 建 v2.1，在每个 split 内用约束匹配/derangement 让 semantic-negative query 类别边际与 positive 完全相同，同时保证 query 不在图中；然后用同一 9C gate 一次复验。未通过前不得进入 9D/训练。
