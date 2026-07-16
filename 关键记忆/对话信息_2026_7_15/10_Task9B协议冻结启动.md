# Task 9B 协议冻结启动

- 状态：2026-07-15 22:12 已在服务器后台启动；仅 CPU 数据构建/审计，不训练、不关机。
- 输出：`datasets/derived/static_qlora_v2_protocol/2026-07-15/`（新目录，禁止覆盖）。
- 协议：每族 `1 positive + 1 real semantic null + 1 rotating synthetic null`；模型侧仅 opaque id/messages，私有侧车保存标签与溯源。
- 隔离：Task 8 的 186 families 仅以 source ID/SHA 盲化排除；高置信近重复组件整体隔离；train/dev 模板与反事实参数物理分离。
- 校验：四级 JSON 质量、角色/模板/长度分布、locked/component 零交叉、三种 9C 文本探针、全文件 SHA256。
- 本地回归：127 tests passed；6 个上传脚本 SHA256 与服务器一致。
- 当前：后台会话 `task9b_build` 正常；完成 9B 后停在 9C gate 前等待确认。
