# Task 9D v2.2 TaxMask 微实验结果

完成：2026-07-17 00:58（实验目录日期2026-07-16）

## 完整性

- Control/TaxMask × seeds 17/29/43 共6组；每组352 predictions，SHA256全部通过。
- 六组使用同一manifest与同一解码contract：greedy、512 tokens、200704–401408 pixels、同一parser。
- 1000次family paired bootstrap；未读取Task8 locked set；服务器保持开机。

## H1结论

- H1“taxonomy value loss materially impairs evidence learning”：**FAIL**。
- Balanced Accuracy：Control `65.10%`，TaxMask `64.48%`，均值差 `-0.625pp`；95% CI `[-3.54pp, +2.08pp]`。
- Positive TPR：Control `44.79%`，TaxMask `46.88%`，差 `+2.08pp`，未达到预注册`+10pp`。
- Overall Null FPR：Control `14.58%`，TaxMask `17.92%`，恶化`+3.33pp`；95% CI `[+1.04pp,+5.62pp]`。
- Visual Null FPR：Control `14.32%`，TaxMask `18.49%`；TaxMask方差更大。
- TaxMask schema validity `92.71%`、semantic consistency `82.95%`、task compliance `55.21%`，均未通过99%门槛。
- 三seed BA差：17 `-1.875pp`、29 `+1.875pp`、43 `-1.875pp`；仅1/3 seed改善。
- pooled McNemar：p=`0.0243`，且方向不利于TaxMask（Control-only correct 24，TaxMask-only correct 10）。

## 决策

- 不授权更大factorization ablation；不授权Task9E。
- TaxMask不是Task9D失败的主要修复方向；不能通过简单屏蔽taxonomy答案token恢复视觉诊断。
- 下一步按既定路线：bbox坐标链只读审计 → family/pair指标 → 32-family PDM-H双前向法医；同时将开放诊断与候选核验拆分。

## 本地正式产物

`artifacts/2026-07-16_task9/9D_v22_micro_result/`

- `v22_decision_report.json`
- `group_metrics.json`
- `completion.sha256`

