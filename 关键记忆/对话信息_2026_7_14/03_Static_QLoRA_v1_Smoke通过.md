# Static QLoRA v1 Smoke 六项门槛通过

- 全量预检：train 20,478 + val 3,052 全通过；最长 639/637 tokens，assistant token 最少 35。
- 4-bit/LoRA 验收：144 个语言 self-attention 目标，288 个 LoRA 可训练张量（7,372,800 参数），视觉/merger/projector 可训练项=0。
- Smoke：24 positive + 8 null，2 optimizer steps；loss=1.4653/1.2451。
- 峰值 reserved VRAM=4.998 GiB，低于 30 GiB 门槛。
- adapter 重载成功；正样本与 null 样本均生成可解析 Evidence-First JSON。
- `smoke_gate.json`: 六项全 true；SHA256=`f3cd9def...fb5d`。
- 服务器：`/root/autodl-tmp/EviAgriDiag/experiments/static_qlora_v1/smoke/`。
- 本地证据：`artifacts/2026-07-14_static_qlora_v1/smoke/`。
- 下一步：只在 gate=true 前提下后台启动正式 1 epoch 训练。
