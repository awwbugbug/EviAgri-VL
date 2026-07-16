# Static QLoRA v1 正式训练已启动

- 启动前条件：全量预检通过，smoke 六项 gate 全 true，GPU 空闲，数据盘剩余约 31GB。
- 正式训练：1 epoch，1,279 optimizer steps，effective batch=16；输出 `.../experiments/static_qlora_v1/formal/`。
- 后台会话：`screen: static_qlora_v1`；启动日志为 `formal-launch.log`，完成后归位为 `formal/train.log`。
- 新鲜验收：step 22/1,279，latest loss=0.9611，GPU 11,582/32,760 MiB，failure=false。
- 状态指令：`bash /root/EviAgri-VL/server/check_static_qlora_status.sh`，会写 `formal/status.json`。
- 自动监控：`eviagri`，每 30 分钟只读检查；不重复启动，完成后接管 Task 7 评估并停用自身。
- 粗估：纯训练约 5 小时，加上定期 val 会更长；服务器在训练完成前不可关闭。
