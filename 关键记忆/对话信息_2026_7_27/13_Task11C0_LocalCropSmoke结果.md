# Task 11C.0 局部证据裁剪烟雾测试

- `protocol_v1`：32 张均有效，但“每张必须严格局部”因 6 张合法大目标/大病斑样本阻断；不是数据损坏或工程失败。
- `protocol_v2` 单变量修订：裁剪面积 `<0.95` 用 `effective_crop`；`>=0.95` 用原始图像 `identity_full_frame`，不冒充定位收益，不删/移 family。
- 结果：`PASS`；effective crop 26/32（81.25%），identity fallback 6/32；裁剪面积中位数 0.47048；全部证据包含、模式和哈希门禁通过。
- 输入清单 SHA256：Task10B `84d2d1b2...95edc90`；PlantSeg `7196eb45...63efc025`。未训练、未读 Task8 locked set、服务器保持开机。
- 本地证据：`artifacts/2026-07-27_task11/11C0_local_crop_smoke_protocol_v2`；协议代码提交 `f62ccfb`。
- 决策：几何可行性成立；仅批准设计下一步 Task 11C.1 极小规模 paired crop-vs-full 特征验证，尚未批准任何大训练。
