# IP102 官方 Detection 审计

_日期：2026-07-12 | 结论：正是所需 bbox 分支_

- 原包：`本地数据集/IP102_Detection-20260712T104348Z-2-001.zip`，852,827,550字节
- ZIP SHA256：`486afb19a27240a97162a29ebb23a1ec06c76ef87c7dd83aab6ceb64fc055d18`
- 结构：VOC2007；`JPEGImages.tar`、`Annotations.tar`、`trainval.txt`、`test.txt`
- 图片18,981；XML18,976；trainval/test=15,178/3,798；无划分重叠
- 5张额外图片无XML且不在split中；正式检测数据为18,976图
- XML共22,283个bbox；编号范围0–101，但只覆盖97类
- 未覆盖0-based类：59/60/63/75/80（Viteus vitifoliae等5类）
- 异常1：`IP087000986.xml`重复拼接两份相同annotation，位于trainval
- 异常2：`IP046000898.xml`首框 `[14,35,14,37]` 零宽，位于trainval
- 决策：原包只读保留；派生清洗版修复重复XML，剔除退化框并生成审计清单
- 用途：IP102 bbox证据定位、Energy-in-Box、Pointing Game及证据可靠性评估
