# Task 10C 设计冻结

- 唯一问题：diagnosis-only Static QLoRA能否把Task10B已证实可分的视觉信息映射为无候选canonical pest ID。
- 数据修订：复用Task10B v2 exact manifest；16类，train/val/dev=`192/48/80`，SHA=`84d2d1...edc90`；旧`320/80/160`配额不再适用。
- 先做C0协议门控，再做seeds 17/29/43各8-step独立smoke；三组完成后强制停止汇报，不自动进入64-step。
- 正式C2仍为3 seeds×64 steps，从Base独立开始；D2实测Macro-F1=`0.8094`，70%门槛=`0.5666`。
- Task10A旧verifier虽有视觉依赖但pair contract失败；即使10C通过，也只授权规划新的verification-only微实验，不能直接进入原10D。
- 完整规格：`docs/superpowers/specs/2026-07-17-task10c-diagnosis-only-design.md`。
