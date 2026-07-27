# 远程实验部署与启动规范

## 1. 分离科学版本与工程尝试

- `protocol_vN`：只在数据、模型、训练/推理参数、阈值、评测指标或决策门禁发生科学变化时递增。
- `attempt_NN`：路径、权限、引号、网络、环境等工程启动失败时递增。
- 同一科学协议下的启动故障不得创建新的 `formal_vN`。
- 推荐目录：`<experiment>/protocol_v1/attempt_01`；成功后用只读元数据标记被采纳的 attempt，不移动或覆盖历史输出。

## 2. 上传规则

- 上传到独立 `.partial` 目录，逐文件核对 SHA256 后原子提升。
- Linux shell 使用 LF 换行。
- 优先以 `bash script.sh` 启动，不依赖可执行位；若需要直接执行，则部署时使用 `install -m 755`。
- 不覆盖已验证代码快照、日志或实验输出。

## 3. 一次性启动前门禁

在创建正式输出前一次完成以下检查：

1. 脚本和依赖代码存在，上传 SHA256 与本地一致。
2. `bash -n script.sh` 通过，且脚本为 LF 换行。
3. 所需 Python 环境存在且可执行，关键依赖 import smoke 通过。
4. 模型、数据、manifest 和配置哈希匹配冻结协议。
5. 目标 `attempt_NN` 输出目录与日志均不存在；不存在同名后台任务。
6. 磁盘空间和 GPU 状态满足任务要求。
7. 启动命令固定为 `bash script.sh > attempt_NN.log 2>&1`。

任一项失败即 BLOCK，不启动任务，也不改变科学协议版本。

## 4. 失败与成功记录

- 工程失败记录：`attempt_NN.log`、错误类别、是否进入 Python/GPU、是否产生有效输出。
- 科学失败记录：只有任务实际完成并按冻结评测协议得到未通过结论时才能使用。
- 不把 `Permission denied`、路径错误、引号错误或环境缺失归因于模型、数据或研究方向。

## 5. Task11A.3 历史映射

- 科学身份始终为同一个冻结 Router 协议 `protocol_v1`。
- 旧物理路径 `formal_v1` = `attempt_01`：manifest SHA 常量录入错误，未进入 Python/GPU。
- 旧物理路径 `formal_v2` = `attempt_02`：shell 执行权限错误，未进入 Python/GPU。
- 旧物理路径 `formal_v3` = `attempt_03`：完成并通过门禁，是 `protocol_v1` 的有效结果。
- 现有路径不重命名，避免破坏哈希、报告和引用；后续报告按上述逻辑身份表述。
