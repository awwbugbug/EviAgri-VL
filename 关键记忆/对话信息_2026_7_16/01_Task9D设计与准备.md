# Task 9D 设计与准备（2026-07-16）

## 冻结设计
- A：统一中性提示 + 精确匹配 semantic negative；训练暴露 768 positive / 768 semantic。
- B：A + 轮换视觉反事实；512 positive / 512 semantic / 512 visual。
- C：B + 三种正负对称模板；角色暴露同 B。
- 共用 train 512、val 192、dev 512 families；challenge 128；seeds=17/29/43。
- Qwen2.5-VL-3B，Static QLoRA q/v-only，r16/alpha32/dropout0.05，LR=1e-4，192 steps，accumulation=8。
- Task 8 的 186-family locked confirmatory set 未读取；禁止 9E/动态 LoRA/SAM2/7B。

## 准备结果
- 本地完整测试：162 passed（增加最终 preflight 测试后为 164，待正式结束再复核）。
- 服务器协议：`.../experiments/task9d/2026-07-16/preparation/protocol`，freeze passed。
- family：train 512 / val 192 / dev 512 / challenge 128；locked overlap=0；near-duplicate split crossing=0。
- 统一评估：每模型 2560 rows；blank RGB(91,107,123)，blur=0.16，shuffle=7x7；共 6 种 prompt views。
- A/B/C 三视图 shortcut gate 均 PASS（阈值 BA/AUROC <=0.55）。
- 6336 条 train/val 记录 schema、图像与角色审计通过。

## Smoke 与风险
- A/B/C seed17 各 3 steps：loss/grad 均有限；峰值 VRAM 4.71–4.95 GiB；72 个语言 q/v targets。
- 三 adapter 均保存、SHA256 校验、磁盘重载和确定性生成成功。
- 风险：3-step 自由生成仍为 Base 旧 fenced JSON，尚未学到 v2 schema；作为非阻断工程风险记录，正式评估的 schema/semantic 淘汰规则不放宽。
- `preparation/pretraining_gate.json`: passed=true，无 blocking failure。

## 当前状态
- 九运行矩阵已于 2026-07-16 启动，固定顺序 A17/A29/A43→B17/B29/B43→C17/C29/C43。
- 自动跟踪：`eviagri-task9d`，每 20 分钟；不重启失败任务、不关机。
