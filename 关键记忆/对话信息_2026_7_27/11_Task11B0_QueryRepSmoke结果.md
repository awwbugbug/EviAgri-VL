# Task11B.0 QueryRep smoke 结果

- 目的：验证冻结 Qwen2.5-VL-3B 第 27 层 query-token 表征是否真正携带图像信息；不训练、不建 optimizer。
- 协议：`protocol_v1/attempt_01`；4 张 IP102 正样本 + 4 张 PlantSeg 真实 null，各做原图/灰色空白图，共 16 个 2048 维向量；另重复首图一次。
- 完整性：Git `eca8cdf`；代码、配置、输入清单和 8 张图 SHA 均在启动前通过；输出 `completion.sha256` 全通过。
- 结果：PASS；重复最大误差 0；原图-空白图余弦距离 median 0.017622（min 0.015280）；L2 min 0.174812。
- 约束：36 层、取 `hidden_states[27]`、中性问题 19 tokens；模型参数全冻结；未读 Task8；未启动 Task11B.1。
- 解释：语言侧 query-token 表征对像素有稳定响应，证明桥接候选可计算且非恒定；尚未证明它能提升 pest/null 判别或 IP102 分类。
- 下一步：Task11B.1 小规模同协议 RepProbe，只比较 vision-pooled 与 query-token；同 split、同线性分类器、同 seed/评估。若无显著增益则停止该桥接路线，再针对病因查阅论文。
- 产物：`artifacts/2026-07-27_task11/11B0_query_rep_smoke/`。
