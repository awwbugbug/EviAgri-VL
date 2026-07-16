# Static QLoRA v1 正式训练完成

- 训练完成：1,279/1,279 steps，1 epoch（epoch=0.9993），failure=false。
- 用时：20,159s（约 5.60h）；train loss=0.2956；128 个记录 loss 全为有限数。
- eval loss：step 250/500/750/1000/1250 = 0.2306/0.1988/0.1896/0.1853/0.1838。
- 峰值 reserved VRAM=13.158 GiB。
- 最终 adapter 与 checkpoint-1279 adapter SHA256 一致：`162237f0...25284`。
- 最终 checkpoint 包含 adapter、optimizer、scheduler、trainer state；训练目录约 131MB。
- 服务器：`/root/autodl-tmp/EviAgriDiag/experiments/static_qlora_v1/formal/`。
- 本地证据：`artifacts/2026-07-14_static_qlora_v1/formal/`。
- 注意：loss 下降只证明优化链路正常，尚不代表诊断/定位/抗幻觉指标改善；下一步执行 Task 7 val/test 生成评估。
