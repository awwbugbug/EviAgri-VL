# Task 11A.3 完整候选池模型盲离审计修订

## 原因

八宿主 smoke 已因拼图、水印和疑似虫体而阻断；不得换图继续评估。完整官方归档现已验证，只授权对完整 Validation 候选池做模型盲离来源质量审计，不读取 router 输出、不训练。

## 冻结筛选与审计单位

- Validation；短边 `>=224`；mask ratio `[0.02,0.40]`；license 为 `CC-BY-NC/CC0`；Metadata URL 的 URL path 必须直接以 `.jpg/.jpeg/.png` 结尾。
- 保留全部行级索引，同时按图像 SHA256 标记精确重复；已进入旧 smoke 的图像只标记排除，不重新审计。
- 人工审计单位为未见过的唯一图像内容；contact sheet 只显示 audit ID、原图和官方 mask overlay，不包含模型输出、病名、文件名或 URL。
- 人工字段冻结为：真实照片、病损可见、无可辨识虫体、无主导文字、非拼图、mask 有效；任一必要项失败则该图不能作为 strict real-null。

## 实测计数边界

- 严格按 URL path（不是 query 中出现的文件扩展名）判断直接图片后，冻结元数据条件产生344行；精确内容去重后341张；其中3张与旧 smoke 重叠；因此新盲审单位为338张。
- 一条代理页面 URL 在 query 参数中嵌入 `.jpg`，旧探索性正则曾将其误计为直接图片；正式构建已 fail-closed 排除并用测试冻结。
- 旧自动化中的“341”对应精确去重后的物理图像数，不是最终新审计数。禁止通过保留重复、重新使用 smoke 或放宽来源规则凑数。

## 分流

- 本阶段只生成索引、审计表和哈希签名，decision=`PENDING_MANUAL_AUDIT`。
- 人工审计完成前禁止运行 router、Task11B、QLoRA、动态模块、SAM2、7B 或 Task8 confirmatory。
