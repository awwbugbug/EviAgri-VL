# Task11B.1：vision-pooled vs query-token RepProbe

## 单变量与数据合同

- 唯一变量：冻结 Qwen2.5-VL-3B 表示。V=`post_merge_token_mean_then_l2`；Q=`hidden_states[27]` 中完整19-token中性问题 mean-pool 后 L2。
- 分类器、`C=1`、balanced、lbfgs、seeds 17/29/43、family/source split、温度拟合、阈值搜索、parser和评估脚本完全相同。
- 正样本：Task10B 原320图（train192/val48/dev80）。synthetic null：既有640条 blank/blur/shuffle，仅用于val阈值选择与dev测试。
- 外部测试：40张PlantDoc healthy null、290张人工审定PlantSeg damage-null；禁止参与训练、温度或阈值选择。PlantSeg已用于法医，只作开发性外部证据，不称confirmatory。
- 所有推理只有图像像素与同一句中性问题；路径、类别和mask元数据不进入模型。全模型冻结、无optimizer、不读Task8。

## 预注册统计与门禁

- 1000次source/image配对bootstrap，报告Q-V的Accuracy、Macro-F1、synthetic FPR、PlantDoc FPR和PlantSeg FPR差值95% CI。
- Q正样本Accuracy与forced Macro-F1相对V均不低于-3pp；Macro-F1配对CI下界不低于-5pp；coverage不低于70%且相对V不低于-5pp。
- Q的blank/blur FPR<10%，shuffle FPR<25%，synthetic overall FPR不高于V。
- PlantDoc Q FPR<10%且不高于V+2.5pp；PlantSeg Q FPR<10%、至少比V低2pp，且PlantSeg配对bootstrap差值CI上界<0。
- JSON四级一致性必须全为1。全部满足才授权规划极小evidence head；任一失败即停止query-token桥接，下一单变量转Local-Crop/Mask，不叠加方法。

## 执行边界

- `protocol_v1/attempt_01`；工程故障只增加attempt，不增加协议版本。
- 先测试、Git快照、完整preflight；使用`bash script.sh`启动。
- 不启动QLoRA、动态LoRA/Gating、SAM2、7B、新backbone或Task8确认集。
