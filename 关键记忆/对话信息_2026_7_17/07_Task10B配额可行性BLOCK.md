# Task 10B v1配额可行性BLOCK

- 日期：2026-07-17；仅做服务器只读元数据预检，未加载模型、未占用GPU、未启动训练。
- 冻结要求：v2.2的32类中选head/medium/tail=`6/5/5`；每类`20 train + 5 val + 10 dev=35`；排除Task9D/v2.2已用source SHA。
- 实际满足35张的类别数：head=`11`、medium=`11`、tail=`1`；要求tail=`5`，故结论为`BLOCKED_CLASS_QUOTA`。
- 更严格地排除触及旧实验/官方test的near-duplicate component后，仍仅1个tail类满足35张；不能通过加强隔离解决。
- 禁止动作：不得近似匹配、减少tail类、换seed、读取Task8 locked内容或直接启动特征提取/10C。
- 推荐v2修订：保留冻结32类与`6/5/5`，把每类配额降为`12 train + 3 val + 5 dev=20`；元数据预检显示至少6个tail类拥有≥20个单类独立component，可进入正式exact-split gate。
- 备选修订：保留`20/5/10`，但在观察任何特征/指标前扩大预声明类别池；改动更大，不优先。
- 下一步：用户批准v2配额后，先冻结选择规则、manifest与哈希，再做8图特征smoke；通过后才提取全量320张特征并运行三seed线性探针。
