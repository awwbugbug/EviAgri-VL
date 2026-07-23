# 服务器重连与 Task 11A 路线（2026-07-23）

- 已用既有 SSH key 重连服务器；RTX 4090 空闲，无 screen/训练进程；未关机。
- 本地 Git：`main@9016ccc`，工作区起始干净；服务器 `/root/EviAgri-VL` 是部署镜像，不是 Git 仓库。
- Task10B 签名特征复核：320x2048、SHA256=`5c730b...accc`；正式 completion 全通过。发现的 `smoke_8_r2/failure.json` 是被后续 r3 替代的历史 smoke 失败，不影响正式结果。
- 已确认病因：视觉特征可分（Macro-F1=0.8094），生成式 canonical taxonomy/JSON 桥接失败；旧 verifier 也未通过 pair contract。
- 下一步冻结为 Task11A：线性 taxonomy router + 确定性 JSON + val-only 置信度拒答；对独立参数的 dev blank/blur/shuffle 做极小验证。
- 通过才规划轻量 evidence/localization head；失败则转显式 evidence/OOD head。禁止直接扩大 QLoRA、7B、动态 LoRA/Gating、SAM2 或读取 Task8 locked set。
