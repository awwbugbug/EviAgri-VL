# Task 11A.3：PlantSeg 病损但无虫 Real-Null 微评估

## 科学问题

Task11A.2 只证明冻结 router 在40张外部健康叶片上不凭空报 IP102 害虫。Task11A.3 进一步检验：真实病斑/损伤存在但无可见虫体时，router 是否会把“任何叶片异常”误当成害虫。

## 官方来源与许可

- 官方代码：`https://github.com/tqwei05/PlantSeg`。
- 官方数据：Zenodo record `17719108`，DOI `10.5281/zenodo.17719108`，文件 `plantseg.zip`，size=`1057281724`，MD5=`9358a66dff88cdd15c4fe009763c40a3`。
- record access=`open`，record license=`CC-BY-NC-4.0`；Metadata 中逐图 license 为 `CC-BY-NC` 或 `CC0`。只作非商业科研审计，图像不得提交 Git。
- archive 实测：7,774 张 disease image + 7,774 个 mask；train/val/test=`5367/846/1561`；无 healthy 条目。通过 HTTP Range 只取选中成员，不下载整包。

## 冻结候选与独立抽样

- 仅官方 `Validation`；从现在起选中图像 hash 永久保留为 locked router audit，不得进入后续 PlantSeg 训练、调参、早停或 evidence-head 选择。
- 宿主固定为：`Apple/Citrus/Corn/Grape/Rice/Soybean/Tomato/Wheat`。
- 资格条件：短边 `>=224`；官方 mask ratio 在 `[0.02,0.40]`；图像、mask、Metadata 三者必须一一匹配。
- 每宿主按 `SHA256(task11a3|record_id|archive_md5|plant|Name)` 排序。smoke 取第1张，共8张；正式集取前3张，共24张，包含 smoke，不得按模型输出或人工偏好换图。
- 与 Task10B、Task11A.2 的 image SHA256 必须零重叠；模型输入只含 pixels，不含文件名、病名、URL、split 或 mask。

## 构造有效性门

- smoke 8/8、正式24/24 必须：可解码、真实照片、病损可见、无可辨识成虫/幼虫/卵/蛹/虫网或疑似虫体、无占主导的文字/拼图/截图。
- mask 必须非空，实算 mask ratio 与 Metadata 误差在容差内。
- 任一失败则整阶段 `BLOCK_INVALID_DAMAGE_NULL`；不得删除该图并取下一张。只有在看模型输出前修订统一资格规则后，才能另建新版本。

## 冻结评估

- 沿用 Task10B Qwen2.5-VL-3B 冻结视觉特征与同一 16类 LogisticRegression。
- seeds=`17/29/43`，temperature=`0.18887372662036642`，tau=`0.63`；PlantSeg 不参与拟合或阈值选择。
- 报告 overall/per-host/per-disease FPR、refusal、最大置信度、accepted diagnosis distribution、JSON contract、mask-ratio/confidence 的探索性 Spearman；不据相关性调参。
- 以24个唯一图像为统计单位，报告1,000次 image bootstrap和 Clopper–Pearson 95% CI；三个相同 seed 不当成72个独立样本。
- PASS gate：FPR `<10%` 且 exact 95% upper `<25%`，JSON contract=`100%`。

## 分流

- PASS：与 Task11A.2 共同支持真实视觉拒答可行性；只授权规划 Task11B 的极小 evidence/localization head，不授权大型训练。
- FAIL：先法医分析病种、mask ratio、场景与高置信类别；不得训练 Task11B、QLoRA、动态模块、SAM2 或7B。

## Smoke 实施结果与来源边界

- v1 因既有环境缺 `aiohttp` 在下载前失败；未安装依赖。v2 改用标准库 HTTP RangeReader，8图+8mask 的结构/哈希/mask 校验通过。
- 人工视觉门 BLOCK：Rice 为双面板拼图；Soybean 有可读网站水印；Apple 枯萎结构具有疑似虫体歧义。未提取特征、未读取 router 输出。
- 事后只提出一个统一、模型盲离的来源质量候选规则：Metadata URL 必须直接指向 `.jpg/.jpeg/.png`。资格计数为 Apple19/Citrus5/Corn21/Grape18/Rice0/Soybean26/Tomato24/Wheat50。
- 因 Rice=0，八宿主严格协议判定 `BLOCK_INFEASIBLE_SOURCE_QUALITY_RULE`；不得删除 Rice、使用近似文章页规则或继续迭代到通过。
- 再继续需要独立数据扩展：对完整候选池做模型盲离人工质量标注，或采集可信真实无虫病损图；需另立协议。
