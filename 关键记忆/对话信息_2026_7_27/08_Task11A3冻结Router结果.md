# Task11A.3 冻结 Router 外部真实空证据审计

- 数据：PlantSeg 人工双审后 290 张 KEEP；原始像素输入，唯一图 290，既有内容重叠 0。
- 协议：冻结 Qwen2.5-VL-3B 视觉特征 + Task10B 16 类 LogisticRegression；seeds 17/29/43，T=0.1888737，tau=0.63；未训练、未调阈值、未读 Task8。
- 结果：三 seed 一致，24/290 被错误接收；Real-null FPR=8.28%，拒答准确率=91.72%，JSON 合规率=100%。
- 区间：1000 次图像 bootstrap 95% CI=[5.17%,11.72%]；Clopper-Pearson 95% 上界=12.06%。
- 决策：PASS（三项预注册门禁均通过）。只证明 Router 对外部病害/损伤空证据有一定拒答能力，不证明害虫诊断准确率；24 个误接收需后续误差法医。
- 工程审计：formal_v1 因冻结 manifest SHA 手抄少 1 字符被门禁拦截；formal_v2 因上传脚本无执行权限未进入实验；均未运行 Python/GPU且保留证据。有效结果为 formal_v3，代码提交 `a5451b6`。
- 关键哈希：manifest `7196eb45259b851908c362dfbbb08e1b5b81f65f36dfe1a25d36692e63efc025`；features `e05f01467c70ec334656f1702e1e0ec8fd4c5d8a14a7dad1616b9d98fc62b618`。
- 本地结果：`artifacts/2026-07-27_task11/11A3_frozen_router_result`。停止在下一阶段前，不自动启动 Task11B。
