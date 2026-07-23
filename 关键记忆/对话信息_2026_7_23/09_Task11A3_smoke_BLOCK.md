# Task 11A.3 PlantSeg smoke BLOCK（2026-07-23）

- 官方源审计：record=`17719108`，CC-BY-NC-4.0，archive 1.057GB；Range-only 构建，无整包下载。archive 实际为7,774 disease image+mask（train/val/test=5367/846/1561），无 healthy。
- v1：fsspec 缺 aiohttp，在写图前失败，空目录保留；未安装依赖。r2 改标准库 HTTP RangeReader，8图+8mask、CRC/尺寸/mask ratio/零重叠/completion 全通过。
- 人工门 `BLOCK_INVALID_DAMAGE_NULL`：Rice 是双面板拼图；Soybean 有可读中文网站水印；Apple 枯萎结构具疑似虫体歧义。
- 未读取任何 router 输出，未提取模型特征；不得替换单图、构建正式24图或评估模型。
- 统一 v2 来源规则（URL直接为jpg/jpeg/png）资格计数：Apple19/Citrus5/Corn21/Grape18/Rice0/Soybean26/Tomato24/Wheat50；Rice=0，故八宿主协议 `BLOCK_INFEASIBLE_SOURCE_QUALITY_RULE`，不得删 Rice 或降级近似匹配。
- 科学含义：PlantSeg 的病损+mask 适合 evidence 研究，但网络抓取图必须先做与模型盲离的来源质量控制，不能直接充当严格 real-null。
- 下一步必须扩展数据工作：完整候选池模型盲离人工质量标注，或新增可信真实无虫病损图；仍禁 Task11B、QLoRA、大型训练、动态模块、SAM2、7B、Task8 confirmatory。
- 权威现场：server `datasets/raw/plantseg_damage_null_smoke_2026-07-23_r2`；local `artifacts/2026-07-23_task11/11A3_plantseg_damage_null_smoke/`。
