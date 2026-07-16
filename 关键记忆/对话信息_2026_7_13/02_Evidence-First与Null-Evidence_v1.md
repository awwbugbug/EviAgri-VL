# Evidence-First / Null-Evidence v1

- 版本：`eviagridiag-detection-v1`，固定 seed 同版本名。
- 强证据来源：仅 IP102 Detection 真实 bbox；不把 Age/IP102 分类标签伪造成局部框。
- 每张有效图生成：1 个最大有效框正样本 + 1 个确定不在该图标注中的错误类别 null 样本。正/null 分文件保存，训练时再控制比例。
- 输出顺序固定：`evidence_present → evidence_bbox → visible_attributes → diagnosis → reliability`。未标注的可视属性不猜测，用空列表。
- 数量：train 13,652 正 + 13,652 null；val 1,526+1,526；test 3,798+3,798；共 37,952 条唯一样本。
- 清洗：丢弃 1 个退化框；重复根 XML 只取第一个完整 annotation；18,976 张有 XML 图均保留了至少 1 个有效框。
- 验收：37,952 ID 全唯一，逐行 JSON 可解析，图像路径全存在，框全合法，null 查询与真实类无冲突，语义验收 0 错误。
- 服务器：`/root/autodl-tmp/EviAgriDiag/datasets/derived/eviagridiag_detection_v1`
- 本地构建摘要：`artifacts/2026-07-13_eviagridiag_detection_v1/build_summary.json`
- 下一门槛：冻结首轮静态 QLoRA 的正/null 抽样比例、超参数与评估协议；未冻结前不启动训练。
