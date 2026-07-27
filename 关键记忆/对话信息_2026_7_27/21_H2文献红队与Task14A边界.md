# H2文献红队与Task14A边界

- TokenLearner支持自适应token但需训练；TokenPacker支持global foundation+region cue但会改projector；DeCo支持保空间的参数无关pooling。当前均不直接实现。
- Qwen只读探针：`image_grid_thw=[1,28,38]`、merge=2、输出`266×2048`，可恢复为`14×19`全帧空间token。
- Task14A冻结为annotation-only oracle：单次全图编码，比较`G/R/GG/GR`，主比较`GR-GG`；正样本用GT bbox，PlantSeg只用mask做可靠性压力测试。
- 若oracle不能安全提升，learned selector缺乏上界依据；若通过，也只授权极小selector原型。禁止crop替换、第二编码器、大训练和Task8。
