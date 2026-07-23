# Task 11A.3 PlantSeg damage-null 计划（2026-07-23）

- Task11A.2 PASS 后仍不直接进 Task11B；下一门验证“有病损但无虫”是否被误报成 IP102 害虫。
- 官方 PlantSeg record=`17719108`，CC-BY-NC-4.0，archive 1.057GB/MD5=`9358a66d...40a3`；HTTP Range 可只取子集。实测7,774 disease images+mask，val=846，无 healthy 条目。
- 冻结8宿主：Apple/Citrus/Corn/Grape/Rice/Soybean/Tomato/Wheat；val、短边>=224、mask ratio 2%–40%；确定性排序。
- 先 smoke 8（每宿主1），视觉/虫体/mask 任一失败即整阶段 BLOCK；通过后才取正式24（每宿主3）。不得按模型结果换样本。
- router/temperature=`0.1888737266`/tau=`0.63` 全冻结；24图 exact CI，FPR<10%、upper<25%、JSON=100% 才 PASS。
- PASS 仅授权 Task11B 极小 evidence/localization head 规划；仍禁大型训练、动态 LoRA/Gating、SAM2、7B、Task8 confirmatory。
- 详细协议：`docs/superpowers/specs/2026-07-23-task11a3-plantseg-damage-null-design.md`。
