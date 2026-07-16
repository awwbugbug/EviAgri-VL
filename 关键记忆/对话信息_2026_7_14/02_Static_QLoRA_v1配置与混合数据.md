# Static QLoRA v1：配置与混合数据完成

- 方案1/设计已确认：Qwen2.5-VL-3B，4-bit NF4，仅语言 self-attention LoRA（r16/alpha32/dropout0.05），视觉塔冻结。
- 服务器依赖：`peft 0.15.2`、`bitsandbytes 0.45.5`；`torch 2.5.1+cu121`、`transformers 4.51.3`未升级，CUDA/4080 SUPER 正常。
- 正式配置：`/root/EviAgri-VL/server/configs/static_qlora_v1.json`；本地同步在 `server/configs/`。
- 混合数据：`/root/autodl-tmp/EviAgriDiag/datasets/derived/static_qlora_v1`。
- 计数：train=13,652 positive + 6,826 null = 20,478（2:1）；val=3,052；test=7,596（val/test 均 1:1）。
- 全量验证：37,952 条记录 ID 唯一、图片存在；`sha256sum -c` 全部通过。
- 本地证据：`artifacts/2026-07-14_static_qlora_v1/data_mix/`。
- 下一步：Qwen 多模态 collator、assistant-only loss mask 与 train/val 全量预检。
