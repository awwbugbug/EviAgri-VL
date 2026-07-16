# Static QLoRA v1 客观阶段评价

- 训练链路健康：1279 steps 全完成，eval loss 0.2306→0.1838，无显著过拟合信号。
- 临时 val 快照（462 条）：schema=100%，presence F1=100%，null FPR/EBHR=0%；positive diagnosis acc≈66.5%，macro-F1≈45.0%；mean IoU≈0.588，IoU@0.5≈70.4%，pointing≈91.0%。
- 诊断 acc 95% Wilson CI（后续 247 positive 快照）约 60.3%–72.0%；定位 IoU 中位数≈0.646，但存在 IoU=0 且类别正确的案例。
- 不能宣称“改进”：尚无 base Qwen 同协议基线，且 val/test 全量未完成。
- 关键风险：positive/null 使用不同固定提示模板；presence/reliability/null 满分可能来自文本模板捷径，不足以证明真正抗幻觉。
- 必做对照：base Qwen zero-shot、图像打乱/空白图、positive/null 同模板 counterfactual query、官方 test 全量、按 image 聚类置信区间。
- 当前判断：工程成功，任务效果有明显潜力，论文级证据仍不足；先完成全量评估，不急于继续训练。
