# Task 10C C0/C1 结果（2026-07-17）

## 结论

- C0 数据与工程预检通过；C1 判定为 `PASS_C1_ENGINEERING`。
- 仅证明 3B Static QLoRA 的训练、保存、重载、四条件推理与严格评测链路可运行；**不构成科学有效性证据**。
- C2 未启动、未授权，必须由用户另行确认。

## 关键事实

- 冻结清单 SHA256：`84d2d1b20d4a781bc6fca8c4e9c41dd57051b6201287193681409451095edc90`。
- 16 类；train/val/dev=`192/48/80`，smoke train/dev=`64/16`；source/component overlap 均为 0；未读 Task8 locked set，未引用 official test。
- seeds=`17/29/43`；每组 8 optimizer steps、64 exposures；loss 与 grad norm 全部有限，末步 loss=`1.8836/1.9015/1.9086`。
- 单组训练约 `36.2–39.8 s`；峰值 reserved VRAM 最大约 `8.64 GB`；3 个 adapter SHA256 均已记录并复核。
- 每个 seed 完成 64 条推理（有图/无图 × 训练/未见提示，各 16 条），adapter 重载与哈希验证通过；单组推理约 `61.2–61.6 s`。
- 服务器及本地归档的 8 份 completion 清单均通过；本地共 75 文件、60,081,072 bytes。

## 非绑定性能与法医判断

- 四条件三种子 Accuracy、Macro-F1 均为 0；训练提示 syntax/schema validity=`0/0`，未见提示仅偶有 1/16 syntax-valid，但 schema 仍为 0。
- 原始输出常带 Markdown 代码围栏，并使用错误字段/值，如 `{"IP102":"Cicadellidae"}`、`{"class_id":"102"}`；严格 schema 正确拒绝这些输出。
- 无图条件退化为固定答案（训练提示常为 `{"class_id":"IP102_003"}`，未见提示常为 `{"IP102":"C"}`）；说明 8-step 模型尚未学会规范类别输出，也不能证明视觉诊断能力。
- 因此当前不是“方向失败”，而是工程烟雾测试通过、性能证据不足；若进入 C2，应只把它视为冻结的 64-step 小规模可行性验证，且不得复用 C1 adapter 作为正式结果。

## 归档

- 本地：`artifacts/2026-07-17_task10/10C_c0_c1_smoke`（Git 忽略）。
- 服务器：`/root/autodl-tmp/EviAgriDiag/experiments/task10_micro_first/2026-07-17/task10c_c0_c1`。
- 服务器保持开机，任务已结束，GPU 空闲。
