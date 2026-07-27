# Task11C.0：Local-Crop/Mask 几何与视觉 smoke

- 目标仅验证局部证据裁剪合同，不训练、不提取模型特征、不读取Task8。
- IP102：Task10B dev每类按ID固定1张，共16张；解析官方XML，选面积最大的GT bbox，四边各扩展20% bbox宽高并裁剪。
- PlantSeg：按mask ratio排序等距固定16张；只取最大8连通病斑组件，四边各扩展25%组件宽高，禁止使用全mask并集框。
- 裁剪坐标均在原图像素系；必须边界合法、证据100%包含、图像与XML/mask尺寸一致、裁剪面积严格小于原图、SHA完整。
- 输出opaque crop、逐例JSONL、两张contact sheet与HTML。PASS只授权Task11C.1同协议vision-pooled微型比较；不授权任何训练或方法叠加。
- 路径采用`task11c0_local_crop/2026-07-27/protocol_v1/attempt_01`；工程故障只增加attempt。
