# Task 11A.1 Background Null v2 修订

## 修订原因

v1 smoke 的 24 张在几何上全部合法，但 `val_IP016000811` crop 包含原图底部中文物种文字，构成像素层标签泄漏。v1 因此冻结为 `BLOCK_INVALID_NULL`，不得删掉单一样本后继续。

## v2 单一协议修订

- 所有候选 crop 必须位于底部文字安全线以上。
- val：crop `60x60`、bbox margin `5%`、`17x17` grid、排除图像底部 `10%`。
- dev：crop `72x72`、bbox margin `8%`、`19x19` grid、排除图像底部 `12%`。
- 其余 hash 选择、split、三频段平衡和 gates 不变。

修订后的几何预审计可用量：val=24（head/medium/tail=9/5/10），dev=33（14/5/14）。v2 仍先运行全新 24 张 smoke；视觉 gate 新增“无类别文字或可读物种水印”。

