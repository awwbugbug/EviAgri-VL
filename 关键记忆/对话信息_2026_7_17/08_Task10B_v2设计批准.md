# Task 10B v2设计批准

- 日期：2026-07-17；用户明确批准v2配额修订。
- 类别池仍为v2.2冻结32类；选head/medium/tail=`6/5/5`。
- 每类改为`12 train + 3 val + 5 dev`，合计`192/48/80=320`张。
- 选类只依据官方trainval数量、class ID和预先冻结的component可行性；禁止按特征或结果挑类。
- 执行门控：exact-split协议与哈希 → 8图冻结视觉特征smoke → 320图特征 → 三seed线性探针。
- 任一门控失败即停止；无论PASS/FAIL均不得自动进入Task10C。
- 完整计划：`docs/superpowers/plans/2026-07-17-task10b-v2-linear-probe.md`。
