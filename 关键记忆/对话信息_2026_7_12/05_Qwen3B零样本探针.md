# Qwen2.5-VL-3B 零样本探针

_日期：2026-07-12 | 状态：完成_

- 数据：MVP val子集，Age 8图 + IP102 8图；仅为流程探针，非论文最终指标
- 协议：102类闭集提示；结构化输出诊断/虫态/bbox/可见属性/可靠性
- 结果：16/16 JSON可解析；平均2.40秒/图；峰值显存7.63GB
- 诊断：Age 1/8，IP102 1/8；Age虫态3/8
- Schema：16/16字段与基本类型完整；15/16 bbox格式合法；11/16遵守闭集标签或uncertain
- 发现：zero-shot细粒度识别弱；会输出`unidentified/unanswerable`、枚举大小写偏差及越界bbox
- 元数据问题已修复：manifest仅含非空类导致Age候选101类；新增taxonomy后恢复Age/IP102各102类
- 决策：后续需LoRA/QLoRA、约束解码、bbox裁剪/校验、null-evidence训练与verifier
- 证据：`artifacts/2026-07-12_zero_shot_v1/`
