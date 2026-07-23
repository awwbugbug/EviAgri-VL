# Task 11A.1 v1 BLOCK 与 v2 修订（2026-07-23）

- v1 smoke：24/24 几何零交集且 SHA256 通过；但 `val_IP016000811` crop 含底部中文物种文字，属于像素标签泄漏。
- 决策：v1=`BLOCK_INVALID_NULL`；不删除该样本、不运行模型、不覆盖产物。
- v2 仅修订底部文字安全区：val=60px/margin5%/17-grid/排除底部10%；dev=72px/margin8%/19-grid/排除底部12%。
- v2 预审计可用 val=24、dev=33；两边 head/medium/tail 均足够各取4张 smoke。
- v2 视觉 gate 增加：不得出现虫体、局部虫体、类别文字或可读物种水印。
